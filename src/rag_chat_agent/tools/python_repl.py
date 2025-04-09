import logging

from langchain_core.tools import StructuredTool

from .safer_repl import SaferREPL

logger = logging.getLogger(__name__)

SaferREPL()


class PythonREPLTool:
    def __init__(self):
        logger.info("Initialising PythonREPLTool")
        self.repl = SaferREPL()
        logger.debug("Underlying SaferREPL instance created")

        self.tool = StructuredTool.from_function(
            func=self.repl.run,
            name="python_repl",
            description="A Python shell for calculations. Input should be a valid python command. If you want to see the output of a value, you should assign it as a string to 'result'.",
        )
        logger.debug("StructuredTool created for PythonREPLTool")

    def get_tool(self):
        logger.debug("Returning PythonREPLTool")
        return self.tool
