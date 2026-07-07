import pickle
import os
from imgCaption import logger
from imgCaption.constants import *
from imgCaption.utils.common import read_yaml,create_directories,load_pkl_file
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.preprocessing.image import load_img,img_to_array
from imgCaption.entity.config_entity import ModelTrainerConfig

class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config
        self.load_vectorizer()
        self.load_feature_extractor()
        self.load_main_model()

    def load_vectorizer(self):
        from_disk=load_pkl_file(self.config.vectorizer_path)
        self.vectorizer = TextVectorization.from_config(from_disk["config"])
        self.vectorizer.adapt(tf.data.Dataset.from_tensor_slices(["dummy"]))
        self.vectorizer.set_weights(from_disk["weights"])

    def load_feature_extractor(self):
        self.feature_extractor = load_model(self.config.image_feex_path)

    def load_main_model(self):
        self.main_model = load_model(self.config.main_model_path)

    @staticmethod
    def save_model(path: Path, model: tf.keras.Model):
        model.save(path)
        
    def split_dict_by_keys(self,image_ids):
        train,val=train_test_split(image_ids,test_size=self.config.SPLIT_SIZE,random_state=self.config.RANDOM_STATE)
        logger.info(f"Length of train_images is : {len(train)}")
        logger.info(f"Length of val_images is : {len(val)}")
        return train,val
        
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
    
    def data_generator(self,image_caption_map,features_map):
        while True:
            X1,X2,Y = list(),list(),list()
            cnt=0
            for image,captions in image_caption_map.items():
                for cap in captions:
                    cap=self.vectorizer([cap]).numpy()[0]
                    for j in range(1,len(cap)):
                        cur_seq=np.array([cap[:j]])
                        cur_seq = pad_sequences(cur_seq, maxlen=self.config.MAX_LENGTH, padding='post')[0]
                        next_word=cap[j]
                        X1.append(features_map[image])
                        X2.append(cur_seq)
                        Y.append(next_word)
                cnt+=1
                if cnt==self.config.BATCH_SIZE:
                    yield [np.array(X1),np.array(X2)],np.array(Y)
                    X1,X2,Y = list(),list(),list()
                    cnt=0   
            if len(X1) > 0:
                yield [np.array(X1), np.array(X2)], np.array(Y)

    def train(self):
        train_images_caption=load_pkl_file(self.config.train_images_captions_path)
        image_ids = list(train_images_caption.keys())
        train_keys, val_keys = self.split_dict_by_keys(image_ids)

        train_img_cap_map = {k: train_images_caption[k] for k in train_keys}
        val_img_cap_map   = {k: train_images_caption[k] for k in val_keys}
        train_features_map = self.features_ext_dict(train_images_caption)

        train_feat_map = {k: train_features_map[k] for k in train_keys}
        val_feat_map   = {k: train_features_map[k] for k in val_keys}

        train_generator = self.data_generator(train_img_cap_map, train_feat_map)
        val_generator = self.data_generator(val_img_cap_map, val_feat_map)


        self.main_model.compile(
            optimizer="adam",
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        logger.info("Model Training Started")
        steps_per_epoch = len(train_img_cap_map) // self.config.BATCH_SIZE
        validation_steps = len(val_img_cap_map) // self.config.BATCH_SIZE
        self.main_model.fit(
            train_generator,
            validation_data=val_generator,
            epochs=30,
            steps_per_epoch=steps_per_epoch,
            validation_steps=validation_steps
        )  
        self.save_model(
        path=self.config.trained_model_path,
        model=self.main_model
        ) 