import numpy as np
from pathlib import Path
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences

from imgCaption import logger
from imgCaption.utils.common import load_pkl_file
from imgCaption.config.configuration import ConfigurationManager


class PredictionPipeline:
    """
    Pipeline for generating a caption for a single input image.

    Attributes
    ----------
    image_feex_path : Path
        Path to the saved feature-extractor (ResNet-50) model.
    trained_model_path : Path
        Path to the saved image-captioning model.
    vectorizer_path : Path
        Path to the saved TextVectorization data (.pkl).
    MAX_LENGTH : int
        Maximum caption sequence length used during training.
    BEAM_WIDTH : int
        Beam width used for beam-search decoding.
    """

    def __init__(self):
        config_manager = ConfigurationManager()
        eval_cfg = config_manager.get_evaluation_config()

        self.image_feex_path: Path = eval_cfg.image_feex_path
        self.trained_model_path: Path = eval_cfg.trained_model_path
        self.vectorizer_path: Path = eval_cfg.vectorizer_path
        self.MAX_LENGTH: int = eval_cfg.MAX_LENGTH
        self.BEAM_WIDTH: int = eval_cfg.BEAM_WIDTH

        self._load_feature_extractor()
        self._load_trained_model()
        self._load_vectorizer()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _load_feature_extractor(self) -> None:
        """Load the ResNet-50 feature-extraction model."""
        logger.info(f"Loading feature extractor from: {self.image_feex_path}")
        self.feature_extractor = load_model(self.image_feex_path)
        logger.info("Feature extractor loaded successfully.")

    def _load_trained_model(self) -> None:
        """Load the trained image-captioning model."""
        logger.info(f"Loading trained model from: {self.trained_model_path}")
        self.trained_model = load_model(self.trained_model_path)
        logger.info("Trained model loaded successfully.")

    def _load_vectorizer(self) -> None:
        """Restore the TextVectorization layer from the saved .pkl file."""
        logger.info(f"Loading vectorizer from: {self.vectorizer_path}")
        from_disk = load_pkl_file(self.vectorizer_path)
        self.vectorizer = TextVectorization.from_config(from_disk["config"])

        if "vocabulary" in from_disk and len(from_disk["vocabulary"]) > 2:
            vocab = [w for w in from_disk["vocabulary"] if w not in ("", "[UNK]")]
            self.vectorizer.set_vocabulary(vocab)
            logger.info(
                f"Vectorizer restored via vocabulary list. "
                f"Size: {len(self.vectorizer.get_vocabulary())}"
            )
        elif from_disk.get("weights"):
            self.vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["dummy"]))
            self.vectorizer.set_weights(from_disk["weights"])
            logger.info(
                f"Vectorizer restored via set_weights. "
                f"Size: {len(self.vectorizer.get_vocabulary())}"
            )
        else:
            raise ValueError(
                "Could not restore vectorizer vocabulary. "
                "Re-run Stage 2 (data_transformation) to regenerate the vectorizer_data.pkl "
                "with vocabulary saved."
            )

        self.vocab = self.vectorizer.get_vocabulary()
        self.word_to_index = {w: i for i, w in enumerate(self.vocab)}
        logger.info("Vectorizer loaded successfully.")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def extract_features(self, image_path: str | Path) -> np.ndarray:
        """
        Pre-process a single image and extract its CNN feature vector.

        Parameters
        ----------
        image_path : str | Path
            Absolute or relative path to the image file.

        Returns
        -------
        np.ndarray
            Flattened feature vector of shape (2048,).
        """
        img = load_img(str(image_path), target_size=(224, 224))
        img_array = img_to_array(img)
        img_array = preprocess_input(img_array)
        img_array = np.expand_dims(img_array, axis=0)          # (1, 224, 224, 3)
        features = self.feature_extractor.predict(img_array, verbose=0)
        return features[0].flatten()                            # (2048,)

    def _encode_sentence(self, sentence: str) -> np.ndarray:
        """Vectorize and pad a partial caption sentence."""
        token_ids = self.vectorizer([sentence]).numpy()[0]
        padded = pad_sequences([token_ids], maxlen=self.MAX_LENGTH, padding="post")
        return padded

    def _decode_index(self, idx: int) -> str | None:
        """Convert a token index back to its word; returns None for special tokens."""
        if idx <= 1 or idx >= len(self.vocab):
            return None
        return self.vocab[idx]

    def generate_caption(
        self,
        image_feature: np.ndarray,
        beam_width: int | None = None,
    ) -> str:
        """
        Generate a caption for the given image feature vector using beam search.

        Parameters
        ----------
        image_feature : np.ndarray
            Flattened feature vector of shape (2048,).
        beam_width : int, optional
            Overrides the default beam width from config when provided.

        Returns
        -------
        str
            The generated caption string (without startseq / endseq tokens).
        """
        beam_width = beam_width or self.BEAM_WIDTH
        max_length = self.MAX_LENGTH

        img_vec = np.reshape(image_feature, (1, 2048))

        # Each entry: (list_of_words, cumulative_log_prob)
        sequences = [(['startseq'], 0.0)]

        for _ in range(max_length):
            all_candidates = []

            for caption_words, score in sequences:
                if caption_words[-1] == 'endseq':
                    all_candidates.append((caption_words, score))
                    continue

                current_sentence = " ".join(caption_words)
                seq = self._encode_sentence(current_sentence)

                yhat = self.trained_model(
                    [img_vec, seq], training=False
                ).numpy()[0]

                # Suppress the padding / unknown token
                yhat[0] = 0.0
                top_indices = np.argsort(yhat)[-beam_width:]

                for idx in top_indices:
                    word = self._decode_index(int(idx))
                    if word is None:
                        continue
                    prob = float(yhat[idx])
                    new_score = score + np.log(prob + 1e-10)
                    all_candidates.append((caption_words + [word], new_score))

            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            sequences = ordered[:beam_width]

            if all(seq[-1] == 'endseq' for seq, _ in sequences):
                break

        # Pick the best sequence (length-normalised score)
        best_words, _ = max(sequences, key=lambda x: x[1] / max(1, len(x[0]) - 1))
        result = [w for w in best_words if w not in ('startseq', 'endseq')]
        return " ".join(result).strip()

    def predict(self, image_path: str | Path) -> str:
        """
        End-to-end caption prediction for a single image.

        Parameters
        ----------
        image_path : str | Path
            Path to the input image.

        Returns
        -------
        str
            Generated caption for the image.
        """
        logger.info(f"Running prediction for image: {image_path}")
        features = self.extract_features(image_path)
        caption = self.generate_caption(features)
        logger.info(f"Generated caption: {caption}")
        return caption


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python prediction.py <path_to_image>")
        sys.exit(1)

    image_path = sys.argv[1]
    pipeline = PredictionPipeline()
    caption = pipeline.predict(image_path)
    print(f"\nGenerated Caption: {caption}")
