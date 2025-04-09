import json
import logging

import boto3

logger = logging.getLogger(__name__)


class GuardrailsHandler:
    def __init__(self, config):
        logger.info("Initializing GuardrailsHandler")
        self.guardrails_runtime = boto3.client("bedrock-runtime")
        self.guardrail_id = config.GUARDRAIL_ID
        self.guardrail_version = config.GUARDRAIL_VERSION
        logger.debug(f"Guardrail ID: {self.guardrail_id}")
        logger.debug(f"Guardrail Version: {self.guardrail_version}")

    def apply_guardrail(self, text, source):
        logger.info(f"Applying guardrail for {source}")
        guardrail_payload = [{"text": {"text": text}}]
        try:
            response = self.guardrails_runtime.apply_guardrail(
                guardrailIdentifier=self.guardrail_id,
                guardrailVersion=self.guardrail_version,
                source=source,
                content=guardrail_payload,
            )
            logger.info(f"Guardrail successfully applied for {source}")
            return response
        except Exception as e:
            logger.error(
                f"An error occurred while applying guardrail for {source}: {str(e)}"
            )
            return None

    def check_input(self, text):
        logger.info("Checking query with guardrail")
        return self.apply_guardrail(text, "INPUT")

    def check_output(self, text):
        logger.info("Checking LLM response with guardrail")
        return self.apply_guardrail(text, "OUTPUT")
