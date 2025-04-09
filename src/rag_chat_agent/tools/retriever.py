from langchain.tools.retriever import create_retriever_tool

import logging

logger = logging.getLogger(__name__)


class RetrieverTool:
    def __init__(self, opensearch):
        logger.info("Initializing RetrieverTool")
        try:
            self.retriever = opensearch.get_retriever()
            logger.info("Successfully created retriever")
            self.tool = create_retriever_tool(
                retriever=self.retriever,
                name="guidance-retriever",
                description="Searches and returns potentially relevant guidance or policy documents.",
            )
            logger.info("Successfully created retriever tool")
        except Exception as e:
            logger.error(f"Error initializing RetrieverTool: {str(e)}")
            raise

    def get_tool(self):
        logger.debug("Returning retriever tool")
        return self.tool
