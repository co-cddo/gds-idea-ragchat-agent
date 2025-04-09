import logging
import os
import platform
import re
import signal
from typing import Any, Dict

SAFE_GLOBALS = {
    "abs": abs,
    "all": all,
    "any": any,
    "ascii": ascii,
    "bin": bin,
    "bool": bool,
    "bytearray": bytearray,
    "bytes": bytes,
    "chr": chr,
    "complex": complex,
    "dict": dict,
    "divmod": divmod,
    "enumerate": enumerate,
    "filter": filter,
    "float": float,
    "frozenset": frozenset,
    "hex": hex,
    "int": int,
    "isinstance": isinstance,
    "issubclass": issubclass,
    "len": len,
    "list": list,
    "map": map,
    "max": max,
    "min": min,
    "oct": oct,
    "ord": ord,
    "pow": pow,
    "range": range,
    "repr": repr,
    "reversed": reversed,
    "round": round,
    "set": set,
    "slice": slice,
    "sorted": sorted,
    "str": str,
    "sum": sum,
    "tuple": tuple,
    "type": type,
    "zip": zip,
}


class TimeoutException(Exception):
    pass


class SaferREPL:
    """Simulates a standalone Python REPL.

    Based on https://api.python.langchain.com/en/latest/_modules/langchain_experimental/utilities/python.html#PythonREPL

    Intended as a tool for a langchain agent. Instructions for the tool should include
    'return and input as a string assigned to "result"'

    Works in a lambda as it used no mutliprocessing. It also attempts to be slightly
    safer by limiting exec to a list of globals and an empty name space.

    Doesn't work in non-unix envs due to the use of signal.
    """

    def __init__(self, timeout=1):
        logging.info("Launching SafeREPL")
        self.ensure_unix_environment()
        self.timeout = timeout

    @staticmethod
    def ensure_unix_environment():
        """
        Check if the current environment is Unix-like.
        Raises an EnvironmentError if not in a Unix-like environment.
        """
        if os.name != "posix" or platform.system() == "Windows":
            raise EnvironmentError("SaferREPL requires a Unix-like environment to run.")

    @staticmethod
    def sanitize_input(query: str) -> str:
        """Sanitize input to the python REPL.

        Remove whitespace, backtick & python
        (if llm mistakes python console as terminal)

        Args:
            query: The query to sanitize

        Returns:
            str: The sanitized query
        """
        query = re.sub(r"^(\s|`)*(?i:python)?\s*", "", query)
        query = re.sub(r"(\s|`)*$", "", query)
        logging.info(f"Sanitized query: {query}")
        return query

    def _execute_in_restricted_environment(self, code_string: str) -> Dict[str, Any]:
        # Create a safe globals dictionary

        # give the exec no access to local variables.
        namespace_for_exec = {}
        exec(code_string, SAFE_GLOBALS, namespace_for_exec)
        logging.debug("namespace_for_exec after exec: {print(namespace_for_exec)}")
        return namespace_for_exec

    def _check_result(self, namespace_for_exec):
        if "result" in namespace_for_exec:
            return str(namespace_for_exec["result"])
        return (
            'No valid output returned, the output must be a string assigned to "result"'
        )

    def run(self, code_string):
        logging.info(f"Running in REPL: {code_string}")

        def timeout_handler(signum, frame):
            raise TimeoutException()

        # Set the signal handler for timeout
        signal.signal(signal.SIGALRM, timeout_handler)

        # Set an alarm for the specified timeout value
        signal.alarm(self.timeout)

        try:
            namespace_for_exec = self._execute_in_restricted_environment(code_string)
            if isinstance(namespace_for_exec, dict):
                return self._check_result(namespace_for_exec)
        except TimeoutException:
            return "Error: Execution timed out"
        except NameError as e:
            return f"Error: {str(e)}"
        except SyntaxError as e:
            return f"Error: {str(e)}"
        except Exception as e:
            return f"Error: An unexpected error occurred: {str(e)}"
        finally:
            signal.alarm(0)


def main():
    repl = SaferREPL(timeout=5)
    assert repl.run("import time\ntime.sleep(2)\nresult='pass'") == "pass"
    assert (
        repl.run("import time\ntime.sleep(2)\nnot_result='pass'")
        == 'No valid output returned, the output must be a string assigned to "result"'
    )
    assert (
        repl.run("import time\ntime.sleep(6)\nresult='pass'")
        == "Error: Execution timed out"
    )
    # TODO add some more assert to check other cases
    # TODO move these to an actual test suite.


if __name__ == "__main__":
    main()
