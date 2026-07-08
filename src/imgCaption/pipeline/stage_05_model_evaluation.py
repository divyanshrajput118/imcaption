from imgCaption.config.configuration import ConfigurationManager
from imgCaption.components.model_evaluation import ModelEvaluation
from imgCaption import logger

STAGE_NAME = "Model_Evaluation_Stage"


class ModelEvaluationPipeline:
    def __init__(self):
        pass

    def main(self):
        config = ConfigurationManager()
        model_evaluation_config = config.get_evaluation_config()
        model_evaluation = ModelEvaluation(config=model_evaluation_config)
        results = model_evaluation.evaluate()

        logger.info(f"Final evaluation results: {results}")

if __name__ == '__main__':
    try:
        logger.info(f"*******************")
        logger.info(f">>>>>> stage {STAGE_NAME} started <<<<<<")
        obj = ModelEvaluationPipeline()
        obj.main()
        logger.info(f">>>>>> stage {STAGE_NAME} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise e
