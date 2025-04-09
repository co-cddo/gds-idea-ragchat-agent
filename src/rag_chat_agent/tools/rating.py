import logging
import uuid
from datetime import datetime

from langchain_core.messages import messages_to_dict
from langchain_core.tools import StructuredTool

logger = logging.getLogger(__name__)


class RatingTool:
    def __init__(self, dynamodb_handler):
        logger.info("Initializing RatingTool")
        self.dynamodb = dynamodb_handler
        self.tool = StructuredTool.from_function(
            func=self.rate_conversation,
            name="rating_tool",
            description="""A tool designed to store user ratings in a DynamoDB database. Use this tool when you need to save or record a user's feedback and rating scores for a specific experience. The tool handles the process of sending the feedback information and rating score data to the database. Store all the complete user comments only. Do not add your own comments or interpretation of the user feedback.""",
        )
        logger.debug("Initialized RatingTool with StructuredTool")

    def rate_conversation(self, rating_string: str):
        rating_uuid = str(uuid.uuid4())[:4]
        logger.info(
            f"Rating conversation for session {self.dynamodb.tokenID}_{rating_uuid}."
        )
        try:
            conversation_history = self.dynamodb.get_chat_history(self.dynamodb.tokenID)
            conversation_content = conversation_history.messages
            conversation_str = messages_to_dict(conversation_content)
            logger.debug(
                f"Retrieved conversation history for session {self.dynamodb.tokenID}_{rating_uuid}."
            )

            item = {
                "ratingID": f"{self.dynamodb.tokenID}_{rating_uuid}",
                "sessionID": self.dynamodb.tokenID,
                "timestamp": int(datetime.now().strftime("%Y%m%d%H%M%S")),
                "rating": rating_string,
                "conversation_history": conversation_str,
            }
            logger.debug(
                f"Prepared rating item for DynamoDB {self.dynamodb.tokenID}_{rating_uuid}."
            )

            self.dynamodb.put_item(self.dynamodb.rating_history_table, item)
            logger.info(
                f"Successfully rated conversation for session {self.dynamodb.tokenID}_{rating_uuid}."
            )
            return f"Conversation successfully rated for session {self.dynamodb.tokenID}_{rating_uuid}."
        except Exception as e:
            logger.exception(
                f"Failed to rate conversation for session {self.dynamodb.tokenID}. Error: {str(e)}"
            )
            return f"Failed to rate conversation for session {self.dynamodb.tokenID}. An Error was raised by the tool."

    def get_tool(self):
        logger.debug("Returning RatingTool")
        return self.tool
