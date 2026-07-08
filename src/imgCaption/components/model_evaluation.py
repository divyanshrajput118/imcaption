import os
import gc
import json
import numpy as np
from pathlib import Path
from tqdm import tqdm

import tensorflow as tf
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.translate.bleu_score import corpus_bleu

from imgCaption import logger
from imgCaption.utils.common import load_pkl_file, save_json
from imgCaption.entity.config_entity import ModelEvaluationConfig


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config
        self.load_feature_extractor()
        self.load_trained_model()
        self.load_vectorizer()

    def load_feature_extractor(self):
        self.feature_extractor = load_model(self.config.image_feex_path)

    def load_trained_model(self):
        self.trained_model = load_model(self.config.trained_model_path)
    
    def load_vectorizer(self):
        from_disk = load_pkl_file(self.config.vectorizer_path)
        self.vectorizer = TextVectorization.from_config(from_disk["config"])

        if "vocabulary" in from_disk and len(from_disk["vocabulary"]) > 2:
            vocab = [w for w in from_disk["vocabulary"] if w not in ('', '[UNK]')]
            self.vectorizer.set_vocabulary(vocab)
            logger.info(f"Vectorizer loaded via vocabulary list. Size: {len(self.vectorizer.get_vocabulary())}")
        elif from_disk.get("weights"):
            self.vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["dummy"]))
            self.vectorizer.set_weights(from_disk["weights"])
            logger.info(f"Vectorizer loaded via set_weights. Size: {len(self.vectorizer.get_vocabulary())}")
        else:
            raise ValueError(
                "Could not restore vectorizer vocabulary. "
                "Re-run Stage 2 (data_transformation) to regenerate the vectorizer_data.pkl with vocabulary saved."
            )
        self.vocab = self.vectorizer.get_vocabulary()
        self.word_to_index = {w: i for i, w in enumerate(self.vocab)}

    def features_ext_dict(self,image_caption_map):
        d={}
        images=list(image_caption_map.keys())
        batch_size = 32

        for start in tqdm(range(0, len(images), batch_size), desc="Extracting features", unit="batch"):
            batch_names = images[start : start + batch_size]
            batch_imgs = []
            for img_name in batch_names:
                img_path = os.path.join(self.config.images_dir, img_name)
                img = load_img(img_path, target_size=(224, 224))
                img = img_to_array(img)
                img = preprocess_input(img)
                batch_imgs.append(img) 
            batch_imgs = np.array(batch_imgs)
            features = self.feature_extractor.predict(batch_imgs, verbose=0)
            for i, image in enumerate(batch_names):
                d[image] = features[i].flatten()                
            del batch_imgs, features
        logger.info(f"Feature extraction complete: {len(d)} images processed")
        return d
    
    def encode_sentence(self, sentence: str) -> np.ndarray:
        token_ids = self.vectorizer([sentence]).numpy()[0]
        padded = pad_sequences([token_ids], maxlen=self.config.MAX_LENGTH, padding='post')
        return padded

    def decode_index(self, idx: int) -> str | None:
        if idx <= 1 or idx >= len(self.vocab):
            return None
        return self.vocab[idx]

    def save_score(self, results: dict):
        save_json(path=self.config.scores_path, data=results)
        logger.info(f"Scores saved to {self.config.scores_path}")

    def generate_caption(self, image_feature: np.ndarray | None = None, beam_width: int | None = None) -> str:
        beam_width = beam_width or self.config.BEAM_WIDTH
        max_length = self.config.MAX_LENGTH

        img_vec = np.reshape(image_feature, (1, 2048))

        sequences = [(['startseq'], 0.0)]

        for _ in range(max_length):
            all_candidates = []

            for caption_words, score in sequences:
                if caption_words[-1] == 'endseq':
                    all_candidates.append((caption_words, score))
                    continue

                current_sentence = " ".join(caption_words)
                seq = self.encode_sentence(current_sentence)

                yhat = self.trained_model(
                    [img_vec, seq], training=False
                ).numpy()[0]

                yhat[0] = 0.0
                top_indices = np.argsort(yhat)[-beam_width:]

                for idx in top_indices:
                    word = self.decode_index(int(idx))
                    if word is None:
                        continue
                    prob = float(yhat[idx])
                    new_score = score + np.log(prob + 1e-10)
                    all_candidates.append((caption_words + [word], new_score))

            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_width]

            if all(seq[-1] == 'endseq' for seq, _ in sequences):
                break

        best_words, _ = max(sequences, key=lambda x: x[1] / max(1, len(x[0]) - 1))

        result = [w for w in best_words if w not in ('startseq', 'endseq')]
        return " ".join(result).strip()

    def evaluate_bleu(
        self,
        test_features: dict,
        test_captions: dict,
        checkpoint_every: int = 100,
    ) -> dict:
        actual, predicted = [], []
        start_idx = 0

        ckpt_path = self.config.root_dir / "eval_checkpoint.json"
        if ckpt_path.exists():
            ckpt = json.loads(ckpt_path.read_text())
            actual    = ckpt["actual"]
            predicted = [p.split() for p in ckpt["predicted_str"]]
            start_idx = len(actual)
            logger.info(f"Resuming from checkpoint at {start_idx}/{len(test_captions)}")

        img_ids = list(test_captions.keys())

        for i in tqdm(range(start_idx, len(img_ids)), total=len(img_ids), initial=start_idx):
            img_id = img_ids[i]

            references = []
            for cap in test_captions[img_id]:
                words = [
                    w for w in cap.split()
                    if w not in ('startseq', 'endseq')
                ]
                if words:
                    references.append(words)
            actual.append(references)

            pred_caption = self.generate_caption(
                test_features[img_id],
                beam_width=self.config.BEAM_WIDTH,
            )
            predicted.append(pred_caption.split())

            if (i + 1) % checkpoint_every == 0:
                gc.collect()
                ckpt_path.write_text(json.dumps({
                    "actual": actual,
                    "predicted_str": [" ".join(p) for p in predicted],
                }))
                logger.info(f"Checkpoint saved at {i + 1}/{len(img_ids)}")

        bleu1 = corpus_bleu(actual, predicted, weights=(1.0, 0, 0, 0))
        bleu2 = corpus_bleu(actual, predicted, weights=(0.5, 0.5, 0, 0))
        bleu3 = corpus_bleu(actual, predicted, weights=(0.33, 0.33, 0.33, 0))
        bleu4 = corpus_bleu(actual, predicted, weights=(0.25, 0.25, 0.25, 0.25))

        logger.info(f"BLEU-1: {bleu1:.4f}")
        logger.info(f"BLEU-2: {bleu2:.4f}")
        logger.info(f"BLEU-3: {bleu3:.4f}")
        logger.info(f"BLEU-4: {bleu4:.4f}")

        results = {
            "bleu1": round(bleu1, 6),
            "bleu2": round(bleu2, 6),
            "bleu3": round(bleu3, 6),
            "bleu4": round(bleu4, 6),
        }

        self.save_score(results)

        if ckpt_path.exists():
            ckpt_path.unlink()

        return results

    def evaluate(self):
        logger.info("Loading test captions...")
        test_captions = load_pkl_file(self.config.test_images_captions_path)
        logger.info(f"Test images: {len(test_captions)}")

        logger.info("Extracting image features for test set...")
        test_features = self.features_ext_dict(test_captions)

        logger.info("Running BLEU evaluation...")
        results = self.evaluate_bleu(test_features, test_captions)

        return results
    
