import logging
import os
from uuid import uuid4

import boto3
from dotenv import load_dotenv

from helpers import get_new_session_with_mfa
from rag_chat_agent.agent import RAGChatAgent


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    load_dotenv(override=True)

    get_new_session_with_mfa(os.getenv("AWS_PROFILE"))

    sts_client = boto3.client("sts")
    identity = sts_client.get_caller_identity()
    print("Running with the following role ARN:", identity.get("Arn"))

    session_id = str(uuid4())
    print("Session ID:", session_id)

    question = "What is the Shared Parental Leave policy?"

    agent = RAGChatAgent(tokenID=session_id)
    response = agent.invoke_agent(question)
    print(response)


if __name__ == "__main__":
    main()
