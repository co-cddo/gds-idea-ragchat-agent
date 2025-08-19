import logging

from langchain_aws import ChatBedrock

logger = logging.getLogger(__name__)


class BedrockHandler:
    def __init__(self, config, model_kwargs: None | dict = None):
        logger.info("Initializing BedrockHandler")
        self.region = config.LLM_REGION
        self.model_id = config.LLM_MODEL

        if model_kwargs and not isinstance(model_kwargs, dict):
            logger.error("model_kwargs should be a dict[str]")
            raise ValueError("model_kwargs should be a dict[str]")

        self.model_kwargs = model_kwargs

        logger.debug(f"LLM Region: {self.region}")
        logger.debug(f"LLM Model ID: {self.model_id}")

    def get_llm(self):
        logger.info("Creating ChatBedrock LLM instance")
        try:
            llm = ChatBedrock(
                model_id=self.model_id,
                region_name=self.region,
                model_kwargs=self.model_kwargs,
            )
            logger.info("Successfully created ChatBedrock LLM instance")
            return llm
        except Exception as e:
            logger.error(f"Error creating ChatBedrock LLM instance: {str(e)}")
            raise
