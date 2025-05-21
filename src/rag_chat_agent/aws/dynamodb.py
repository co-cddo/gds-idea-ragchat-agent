import logging

import boto3
from botocore.exceptions import ClientError
from langchain_community.chat_message_histories import DynamoDBChatMessageHistory

logger = logging.getLogger(__name__)


class DynamoDBHandler:
    def __init__(self, config):
        logger.info("Initializing DynamoDBHandler")
        self.dynamodb = boto3.resource("dynamodb")
        self.dynamodb_client = boto3.client("dynamodb")
        self.session_history_table = config.SESSION_HISTORY
        self.rating_history_table = config.RATING_HISTORY
        self.tokenID = (
            None  # This is only needed for chat history, set with get_chat-history
        )
        self.collection_url = config.COLLECTION_URL
        self.index_name = config.INDEX_NAME
        self.embedding_model = config.EMBEDDING_MODEL
        self.llm_model = config.LLM_MODEL
        self.chat_history_length = config.CHAT_HISTORY_LENGTH
        logger.debug(f"Session History Table: {self.session_history_table}")
        logger.debug(f"Rating History Table: {self.rating_history_table}")

    def table_exists(self, table_name: str):
        logger.info(f"Checking if table '{table_name}' exists")
        try:
            self.dynamodb_client.describe_table(TableName=table_name)
            logger.info(f"Table '{table_name}' exists")
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "ResourceNotFoundException":
                logger.warning(f"Table '{table_name}' does not exist")
                return False
            else:
                logger.error(f"Error checking table '{table_name}': {str(e)}")
                raise

    def put_item(self, table_name: str, item: dict):
        table = self.dynamodb.Table(table_name)
        return table.put_item(Item=item)

    def get_chat_history(self, tokenID: str):
        logger.info(f"Getting chat history for tokenID: {tokenID}")
        if tokenID is None:
            logger.error("tokenID is None. Cannot retrieve chat history.")
            raise ValueError(
                "tokenID is not set. Ensure a tokenID is correctly passed to Agent."
            )

        self.tokenID = tokenID

        try:
            history = DynamoDBChatMessageHistory(
                table_name=self.session_history_table,
                session_id=self.tokenID,
                history_size=self.chat_history_length,
            )
            logger.info(f"Successfully retrieved chat history for tokenID: {tokenID}")
            return history
        except Exception as e:
            logger.error(f"Error getting chat history for tokenID {tokenID}: {str(e)}")
            raise

    def update_session_attributes(self):
        logger.info(f"Updating session attributes for tokenID: {self.tokenID}")
        table = self.dynamodb.Table(self.session_history_table)
        try:
            table.update_item(
                Key={"SessionId": self.tokenID},
                UpdateExpression="""
                    SET 
                    History = if_not_exists(History, :empty_list),
                    CollectionName = if_not_exists(CollectionName, :coll_value),
                    IndexName = if_not_exists(IndexName, :idx_value),
                    EmbeddingModel = if_not_exists(EmbeddingModel, :emb_value),
                    LLMModelVersion = if_not_exists(LLMModelVersion, :llm_value)
                """,
                ExpressionAttributeValues={
                    ":empty_list": [],
                    ":coll_value": self.collection_url,
                    ":idx_value": self.index_name,
                    ":emb_value": self.embedding_model,
                    ":llm_value": self.llm_model,
                },
                ConditionExpression="attribute_exists(SessionId)",
            )
            logger.info(
                f"Successfully updated session attributes for tokenID: {self.tokenID}"
            )
        except ClientError as e:
            if e.response["Error"]["Code"] == "ConditionalCheckFailedException":
                logger.warning(
                    f"Session {self.tokenID} does not exist. Unable to update attributes."
                )
            else:
                logger.error(
                    f"Error updating session attributes for tokenID {self.tokenID}: {str(e)}"
                )
            raise
        except Exception as e:
            logger.error(
                f"Unexpected error updating session attributes for tokenID {self.tokenID}: {str(e)}"
            )
            raise
