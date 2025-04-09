import os

import boto3
import botocore.configloader

AWS_CONFIG_PATH = os.path.expanduser("~/.aws/config")


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


def get_new_session_with_mfa(profile_name: str):
    """Assume a role and return a new session with MFA."""
    role_arn, mfa_serial = get_profile_config(profile_name)
    base_session = boto3.Session(profile_name="default")
    sts = base_session.client("sts")
    mfa_token = input(f"Enter MFA code for {mfa_serial}: ")

    response = sts.assume_role(
        RoleArn=role_arn,
        RoleSessionName="ScriptSession",
        SerialNumber=mfa_serial,
        TokenCode=mfa_token,
        DurationSeconds=8 * 60 * 60,  # aka 8 hours
    )

    credentials = response["Credentials"]

    # Set the temporary credentials as environment variables
    os.environ["AWS_ACCESS_KEY_ID"] = credentials["AccessKeyId"]
    os.environ["AWS_SECRET_ACCESS_KEY"] = credentials["SecretAccessKey"]
    os.environ["AWS_SESSION_TOKEN"] = credentials["SessionToken"]
