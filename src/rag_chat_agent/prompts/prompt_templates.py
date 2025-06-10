import logging
from importlib.resources import files

from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)

logger = logging.getLogger(__name__)


def get_agent_prompt(custom_prompt_path: str | None = None):
    logger.info("Configuring agent prompt template")
    system_template = """Respond to the human as helpfully and accurately as possible. You have access to the following tools:

    {tools}

    Use a json blob to specify a tool by providing an action key (tool name) and an action_input key (tool input).

    Valid "action" values: "Final Answer" or {tool_names}

    Provide only ONE action per $JSON_BLOB, as shown:

    ```
    {{
      "action": $TOOL_NAME,
      "action_input": $INPUT
    }}
    ```

    Follow this format:

    Question: input question to answer
    Thought: consider previous and subsequent steps
    Action:
    ```
    $JSON_BLOB
    ```
    Observation: action result
    ... (repeat Thought/Action/Observation N times)
    Thought: I know what to respond
    Action:
    ```
    {{
      "action": "Final Answer",
      "action_input": "Final response to human"
    }}

    Begin! Reminder to ALWAYS respond with a valid json blob of a single action. Use tools if necessary. Respond directly if appropriate.
    Format is Action:```$JSON_BLOB```then Observation
    """
    if custom_prompt_path:
        logger.info(f"Attempting to read custom prompt from {custom_prompt_path}")
        try:
            with open(custom_prompt_path, "r") as file:
                custom_content = file.read()
            system_template += f"\n\n{custom_content}"
            logger.info("Custom prompt successfully added to system template")
        except FileNotFoundError:
            logger.warning(f"Custom prompt file not found: {custom_prompt_path}")
        except Exception as e:
            logger.error(
                f"An unexpected error occurred while reading {custom_prompt_path}: {str(e)}"
            )
            raise
    else:
        logger.info("Using default prompt")
        default_prompt_path = files("rag_chat_agent.prompts").joinpath(
            "system_prompt.txt"
        )
        default_prompt = default_prompt_path.read_text()
        system_template += f"\n\n{default_prompt}"

    system_message_prompt = SystemMessagePromptTemplate.from_template(system_template)
    logger.debug("System message prompt created")

    human_template = (
        "{input}{agent_scratchpad}\n(reminder to respond in a JSON blob no matter what)"
    )
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    logger.debug("Human message prompt created")

    prompt_agent = ChatPromptTemplate.from_messages(
        [
            system_message_prompt,
            MessagesPlaceholder(variable_name="chat_history", optional=True),
            human_message_prompt,
        ]
    )
    logger.debug("Chat prompt template created")

    # Set the input variables. These need to be aligned to what the system and human templates
    # expect or they will fail.
    prompt_agent.input_variables = ["agent_scratchpad", "input", "tool_names", "tools"]
    prompt_agent.partial_variables = {"chat_history": []}

    logger.info("Agent prompt template configured successfully")
    return prompt_agent
