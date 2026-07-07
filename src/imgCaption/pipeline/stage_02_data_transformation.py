from imgCaption.config.configuration import ConfigurationManager
from imgCaption.components.data_transformation import DataTransformation
from imgCaption import logger


STAGE_NAME = "Data_Transformation_Stage"


class DataTransformationTrainingPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        data_transformation_config = config.get_data_transformation_config()
        data_transformation = DataTransformation(config=data_transformation_config)
        df=data_transformation.captions_preprocessing()
        data_transformation.vectorizer(df)
        train_img,test_img=data_transformation.split_images(df)
        data_transformation.build_image_caption_map(df,train_img,data_transformation_config.train_images_captions_path)
        data_transformation.build_image_caption_map(df,test_img,data_transformation_config.test_images_captions_path)

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = DataTransformationTrainingPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e