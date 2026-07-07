from imgCaption.constants import *
from imgCaption.utils.common import read_yaml,create_directories,save_pkl_file
import os
import re
import json
import pickle
import yaml
import numpy as np
import pandas as pd
from imgCaption import logger
from sklearn.model_selection import train_test_split
from tensorflow.keras.layers import TextVectorization
from imgCaption.entity.config_entity import DataTransformationConfig

class DataTransformation:
    def __init__(self,config : DataTransformationConfig):
        self.config = config

    def captions_preprocessing(self):
        df = pd.read_csv(self.config.captions_file, sep=",")
        captions = list(df['caption'])
        captions = [
        re.sub(r'\s+', ' ', 
        re.sub(r'\d+', '', 
        re.sub(r'[^\w\s]', '', cap)
        )).strip().lower() 
        for cap in captions]
        captions = list(map(lambda cap: f"startseq {cap} endseq", captions))
        df['captions_tok'] = captions
        logger.info("Data frame created successfully")
        return df

    def split_images(self,df):
        images = list(set(df['image']))
        train_images,test_images=train_test_split(images,test_size=self.config.SPLIT_SIZE,random_state=self.config.RANDOM_STATE)
        logger.info(f"Train images : {len(train_images)}")
        logger.info(f"Test images : {len(test_images)}")
        return train_images,test_images

    def vectorizer(self,df):
        captions = list(df['captions_tok'])
        vectorizer=TextVectorization()
        vectorizer.adapt(captions)
        vocab_size=vectorizer.vocabulary_size()
        max_length=vectorizer(captions).shape[1]
        logger.info(f"Vocab Size : {vocab_size}")
        logger.info(f"Max length of captions : {max_length}")
        existing_params = read_yaml(PARAMS_FILE_PATH)
        existing_params.update({"VOCAB_SIZE": vocab_size, "MAX_LENGTH": max_length})
        with open(PARAMS_FILE_PATH, "w") as file:
            yaml.safe_dump(existing_params.to_dict(), file, sort_keys=False)
        logger.info("Params updated Sucessfully")
        vocabulary = vectorizer.get_vocabulary()
        vectorizer_data = {"config": vectorizer.get_config(),
                            "weights": vectorizer.get_weights(),
                            "vocabulary": vocabulary}
        save_pkl_file(vectorizer_data, self.config.vectorizer_path)
        logger.info("Vectorizer Saved Sucessfully")

    @staticmethod
    def build_image_caption_map(df,images,output_path:Path):
        image_caption_map = {}
        filtered_df = df[df["image"].isin(images)]
        image_caption_map=filtered_df.groupby("image")["captions_tok"].apply(list).to_dict()
        save_pkl_file(image_caption_map,output_path)
        logger.info("Image captions dict saved")
        return image_caption_map