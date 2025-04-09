import logging
import os
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class Config:
    DATA_REGION: str = os.getenv("DATA_REGION")
    """the region where the api gateway, lamdba etc reside"""
    LLM_REGION: str = os.getenv("LLM_REGION")
    """the region of the llm"""
    COLLECTION_URL: str = os.getenv("COLLECTION_URL")
    """the url of the collection"""
    INDEX_NAME: str = os.getenv("INDEX_NAME")
    """the name of the index in the collection"""
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL")
    """the name of the model used to embed queries to the collection"""
    LLM_MODEL: str = os.getenv("LLM_MODEL")
    """the name of the LLM used in the agent"""
    SESSION_HISTORY: str = os.getenv("SESSION_HISTORY")
    """session history table name"""
    RATING_HISTORY: str = os.getenv("RATING_HISTORY")
    """rating hsitory table name"""
    GUARDRAIL_ID: str = os.getenv("GUARDRAILS")
    """id of the guardrails to run queries through"""
    GUARDRAIL_VERSION: str = os.getenv("GUARDRAILS_VERSION")
    """version number of the guardrails"""

    def __post_init__(self):
        # Check is any of the values are None.
        missing_fields = [
            field_name
            for field_name, field_value in self.__dict__.items()
            if field_value is None
        ]
        # If they are raise a value error
        if missing_fields:
            logger.error(
                f"There were missing fields {missing_fields} when trying to initialise log for agent."
            )
            raise ValueError(
                f"Field{'s' if len(missing_fields) > 1 else ''} '{missing_fields}' cannot be None"
            )

        logger.info("Loaded config for agent successfully.")
        # TODO more checks to see if URL is an URL and if regions are real regions etc.
