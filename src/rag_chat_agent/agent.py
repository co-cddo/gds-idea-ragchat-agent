import json
import logging
import time

from langchain.agents import AgentExecutor, create_structured_chat_agent
from langchain_core.runnables.history import RunnableWithMessageHistory

from .aws.bedrock import BedrockHandler
from .aws.dynamodb import DynamoDBHandler
from .aws.guardrails import GuardrailsHandler
from .aws.opensearch import OpenSearchHandler
from .config import Config
from .prompts.prompt_templates import get_agent_prompt
from .tools.python_repl import PythonREPLTool
from .tools.rating import RatingTool
from .tools.retriever import RetrieverTool

logger = logging.getLogger(__name__)


def normalise_response(response):
    """
    Returns the entire response object with a normalized `output` field.
    If the `output` field is a dictionary, it extracts the `response` key.
    If it's already a string, it remains unchanged.
    """
    # Create a copy of the response to avoid modifying the original
    normalized_response = response.copy()

    # Normalize the `output` field
    output = response.get("output")
    if isinstance(output, dict):
        # Extract the 'response' key if it exists, or fallback to an empty string
        normalized_response["output"] = output.get("response", "")
    elif isinstance(output, str):
        # Keep it as-is if it's already a string
        normalized_response["output"] = output
    else:
        # Handle unexpected types (e.g., raise an error or log)
        raise ValueError(
            f"Unexpected output type: {type(output)}. Expected string or dict."
        )

    return normalized_response


class RAGChatAgent:
    def __init__(self, tokenID: str, model_kwargs: None | dict = None):
        start_time = time.time()
        logger.info("AskOps Start")
        self.config = Config()
        self.model_kwargs = model_kwargs
        self.tokenID = tokenID

        # 1) DynamoDB
        init_start = time.time()
        self.dynamodb = DynamoDBHandler(self.config)
        logger.info(
            f"[TIMING] DynamoDBHandler init took {time.time() - init_start:.2f}s"
        )

        # 2) Bedrock
        init_start = time.time()
        self.bedrock = BedrockHandler(self.config, self.model_kwargs)
        logger.info(
            f"[TIMING] BedrockHandler init took {time.time() - init_start:.2f}s"
        )

        # 3) OpenSearch
        init_start = time.time()
        self.opensearch = OpenSearchHandler(self.config)
        logger.info(
            f"[TIMING] OpenSearchHandler init took {time.time() - init_start:.2f}s"
        )

        # 4) Tools
        init_start = time.time()
        self.retriever = RetrieverTool(self.opensearch)
        self.python_repl = PythonREPLTool()
        self.rating_tool = RatingTool(self.dynamodb)

        self.tools = [
            self.retriever.get_tool(),
            # self.python_repl.get_tool(),
            self.rating_tool.get_tool(),
        ]

        logger.info(f"[TIMING] Tools init took {time.time() - init_start:.2f}s")

        # 5) LLM
        init_start = time.time()
        self.llm = self.bedrock.get_llm()
        logger.info(f"[TIMING] get_llm took {time.time() - init_start:.2f}s")

        # 6) Agent Prompt
        init_start = time.time()
        self.prompt = get_agent_prompt(custom_prompt_path="./prompts/system_prompt.txt")
        logger.info(f"[TIMING] get_agent_prompt took {time.time() - init_start:.2f}s")

        # 7) Agent Executor
        init_start = time.time()
        self.agent_executor = self.set_agent_executor()
        logger.info(f"[TIMING] set_agent_executor took {time.time() - init_start:.2f}s")

        # 8) Guardrails
        init_start = time.time()
        self.guardrails = GuardrailsHandler(self.config)
        logger.info(
            f"[TIMING] GuardrailsHandler init took {time.time() - init_start:.2f}s"
        )

        logger.info(
            f"RAGChatAgent __init__ completed. Total init time: {time.time() - start_time:.2f}s"
        )

    def set_agent_executor(self, verbose=False, handle_parse=True):
        logger.info("Setting up agent executor")
        try:
            agent = create_structured_chat_agent(self.llm, self.tools, self.prompt)
            logger.debug("Structured chat agent created")

            executor = AgentExecutor(
                agent=agent,
                tools=self.tools,
                verbose=verbose,
                handle_parsing_errors=handle_parse,
            )
            logger.info("Agent executor successfully set up")
            return executor
        except Exception as e:
            logger.error(f"Error setting up agent executor: {str(e)}")
            raise

    def invoke_agent(self, query):
        overall_start = time.time()
        logger.info(f"Starting invoke_agent for session {self.tokenID}")

        if isinstance(query, dict):
            query = query.get("input", "")
        elif not isinstance(query, str):
            logger.error("Invalid query type")
            raise ValueError(
                "Query must be a string or a dictionary with an 'input' key."
            )
        logger.info(f"Processing query for session {self.tokenID}: {query}")

        # Guardrail check for input
        start_step = time.time()
        guardrail_check_input = self.guardrails.check_input(query)
        logger.info(f"[TIMING] guardrails.check_input: {time.time() - start_step:.2f}s")
        logger.debug(f"Guardrail input check result: {guardrail_check_input['action']}")

        if guardrail_check_input["action"] == "GUARDRAIL_INTERVENED":
            logger.info(f"Guardrail intervened on Input for session {self.tokenID}")
            save_start = time.time()
            chat_history = self.dynamodb.get_chat_history(self.tokenID)
            chat_history.add_user_message("GUARDRAILS_INPUT_TRIGGERED: " + query)
            chat_history.add_ai_message(json.dumps(guardrail_check_input, indent=2))
            self.dynamodb.update_session_attributes()
            logger.info(
                f"[TIMING] Logging guardrail-intervened input took {time.time() - save_start:.2f}s"
            )
            return guardrail_check_input.get("outputs", [])[0].get("text")

        else:
            config = {"configurable": {"session_id": self.tokenID}}
            step_start = time.time()

            agent_with_chat_history = RunnableWithMessageHistory(
                self.agent_executor,
                self.dynamodb.get_chat_history,
                input_messages_key="input",
                history_messages_key="chat_history",
            )

            try:
                logger.debug(f"Invoking agent for session {self.tokenID}")
                response = agent_with_chat_history.invoke(
                    {"input": query}, config=config
                )
                logger.info(f"{response=}")
                logger.info(
                    f"Response successfully invoked for session {self.tokenID}."
                )
            except Exception as e:
                logger.error(
                    f"Failed to invoke response for session {self.tokenID}. Error: {str(e)}"
                )
                raise
            logger.info(
                f"[TIMING] agent invocation took {time.time() - step_start:.2f}s"
            )

            response = normalise_response(response)

            # Guardrail check for output
            step_start = time.time()
            guardrail_check_output = self.guardrails.check_output(response["output"])
            logger.info(
                f"[TIMING] guardrails.check_output took {time.time() - step_start:.2f}s"
            )

            logger.debug(
                f"Guardrail output check result: {guardrail_check_output['action']}"
            )
            if guardrail_check_output["action"] == "GUARDRAIL_INTERVENED":
                logger.info(
                    f"Guardrail intervened on Output for session {self.tokenID}"
                )
                chat_history = self.dynamodb.get_chat_history(self.tokenID)
                chat_history.add_user_message(
                    "GUARDRAILS_OUTPUT_TRIGGERED: " + response.get("output")
                )
                chat_history.add_ai_message(
                    json.dumps(guardrail_check_output, indent=2)
                )
                self.dynamodb.update_session_attributes()
                return guardrail_check_output.get("outputs", [])[0].get("text")

            step_start = time.time()
            self.dynamodb.update_session_attributes()
            logger.info(
                f"[TIMING] dynamodb.update_session_attributes took {time.time() - step_start:.2f}s"
            )

            logger.info(
                f"--- Finished invoke_agent for session {self.tokenID}, total time: {time.time() - overall_start:.2f}s ---"
            )

            logger.info("AskOps Ends")
            return response.get("output")
