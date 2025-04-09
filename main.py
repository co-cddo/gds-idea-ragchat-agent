import datetime
import json
import logging
import os
from pathlib import Path
from uuid import uuid4

import boto3
import botocore.configloader
from dotenv import load_dotenv

from rag_chat_agent.agent import RAGChatAgent

AWS_CONFIG_PATH = os.path.expanduser("~/.aws/config")
PROFILE = "assume-ds-role"  # Change this to the profile you want to use


def get_profile_config(profile_name: str = "default"):
    """Load the AWS profile configuration from the config file."""
    raw_config = botocore.configloader.load_config(AWS_CONFIG_PATH)
    profile_config = raw_config.get("profiles", {}).get(profile_name, {})

    if not profile_config:
        raise ValueError(f"Profile '{profile_name}' not found in AWS config.")

    role_arn = profile_config.get("role_arn")
    mfa_serial = profile_config.get("mfa_serial")

    if not role_arn or not mfa_serial:
        raise ValueError(f"Profile '{profile_name}' is missing role_arn or mfa_serial.")

    return role_arn, mfa_serial


def get_new_session_with_mfa():
    """Assume a role and return a new session with MFA."""
    # Load profile configuration
    role_arn, mfa_serial = get_profile_config(PROFILE)
    base_session = boto3.Session(profile_name="default")
    sts = base_session.client("sts")

    # Ask the user for MFA code
    mfa_token = input(f"Enter MFA code for {mfa_serial}: ")

    # Assume the role
    response = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName="ScriptSession",
        SerialNumber=mfa_serial,
        TokenCode=mfa_token,
        DurationSeconds=8 * 60 * 60,  # Valid for 8 hours
    )

    credentials = response["Credentials"]

    # Set the temporary credentials as environment variables
    os.environ["AWS_ACCESS_KEY_ID"] = credentials["AccessKeyId"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["SecretAccessKey"]
    os.environ["AWS_SESSION_TOKEN"] = credentials["SessionToken"]


def main():
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    load_dotenv(override=True)

    # Try to load cached session, otherwise get a new one with MFA
    get_new_session_with_mfa()

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
