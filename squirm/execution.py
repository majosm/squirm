"""
:mod:`squirm.execution` wraps non-submitting executors (mpiexec, srun, etc.).

.. autoclass:: Executor
.. autoclass:: ExecParams

.. autoclass:: ProcessError
.. autoclass:: ExecParamError

.. autoclass:: BasicExecutor
.. autoclass:: SlurmExecutor
.. autoclass:: LCLSFExecutor

.. autofunction:: make_executor
.. autofunction:: get_some_executor
"""

__copyright__ = """
Copyright (C) 2020 University of Illinois Board of Trustees
"""

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import abc
import os
import subprocess
import sys


class Executor(metaclass=abc.ABCMeta):
    """
    Base class for a general MPI executor.

    .. automethod:: get_command
    .. automethod:: __call__
    .. automethod:: run
    .. automethod:: call
    """

    @abc.abstractmethod
    def get_command(self, command, exec_params=None):
        """
        Return the full command that will be used to launch a command with MPI.

        :arg command: A `list` of strings representing the command to execute
            with MPI.
        :arg exec_params: An instance of :class:`ExecParams`.
        """
        pass

    @abc.abstractmethod
    def __call__(self, command, exec_params=None):
        """
        Execute a command with MPI.

        :arg command: A `list` of strings representing the command to execute
            with MPI.
        :arg exec_params: An instance of :class:`ExecParams`.
        """
        pass

    def run(self, code_string, exec_params=None):
        """
        Run Python code with MPI.

        :arg code_string: A string containing the Python code to execute with MPI.
        :arg exec_params: An instance of :class:`ExecParams`.
        """
        self.__call__([sys.executable, "-m", "mpi4py", "-c", "\'"
                    + code_string + "\'"], exec_params)

    def call(self, func, *args, exec_params=None, **kwargs):
        r"""
        Call a function with MPI.

        :arg func: The function to execute with MPI. Must be picklable.
        :arg \*args: Positional arguments to pass to *func*. Must be picklable.
        :arg \*\*kwargs: Keyword arguments to pass to *func*. Must be picklable.
        :arg exec_params: An instance of :class:`ExecParams`.
        """
        def embed(obj):
            import base64
            import pickle
            obj_string = base64.b64encode(pickle.dumps(obj)).decode("ascii")
            return ('pickle.loads(base64.b64decode("' + obj_string
                        + '".encode("ascii")))')
        calling_code = "import pickle; import base64; " + embed(func) + "("
        if len(args) > 0:
            calling_code += "*" + embed(args) + ", "
        if len(kwargs) > 0:
            calling_code += "**" + embed(kwargs)
        calling_code += ")"
        self.run(calling_code, exec_params)


class ExecParams:
    """
    Collection of parameters to pass to the executor.

    .. automethod:: __init__
    """

    def __init__(self, num_tasks=None, num_nodes=None, tasks_per_node=None,
                gpus_per_task=None):  # noqa: D205
        """
        :arg num_tasks: The number of MPI tasks to launch.
        :arg num_nodes: The number of nodes on which to run.
        :arg tasks_per_node: The number of MPI tasks to launch per node.
        :arg gpus_per_task: The number of GPUs to assign per task.

        .. note::
            A given executor may not support all of these arguments. If it is
            passed an unsupported argument, it will raise an instance of
            :class:`ExecParamError`.
        """
        self._create_param_dict(num_tasks=num_tasks, num_nodes=num_nodes,
                    tasks_per_node=tasks_per_node, gpus_per_task=gpus_per_task)

    def _create_param_dict(self, **kwargs):
        self.param_dict = {}
        for name, value in kwargs.items():
            if value is not None:
                self.param_dict[name] = value


class ProcessError(RuntimeError):
    """Error raised when a command executed with MPI fails."""

    def __init__(self, exit_code):
        self.exit_code = exit_code
        super().__init__(f"Execution failed with exit code {exit_code}.")


class ExecParamError(RuntimeError):
    """Error raised when an executor is passed an unsupported parameter."""

    def __init__(self, param_name):
        self.param_name = param_name
        super().__init__(f"Executor does not support parameter '{self.param_name}'.")


class BasicExecutor(Executor):
    """Simple `mpiexec` executor."""

    def get_command(self, command, exec_params=None):  # noqa: D102
        exec_command = ["mpiexec"]
        param_dict = {}
        if exec_params is not None:
            param_dict = exec_params.param_dict
        for name, value in param_dict.items():
            if name == "num_tasks":
                exec_command += ["-n", str(value)]
            else:
                raise ExecParamError(name)
        exec_command += command
        return exec_command

    def __call__(self, command, exec_params=None):  # noqa: D102
        exec_command = self.get_command(command, exec_params)
        exit_code = subprocess.call(" ".join(exec_command), shell=True)
        if exit_code != 0:
            raise ProcessError(exit_code)


class SlurmExecutor(Executor):
    """Executor for Slurm."""

    def get_command(self, command, exec_params=None):  # noqa: D102
        exec_command = ["srun"]
        param_dict = {}
        if exec_params is not None:
            param_dict = exec_params.param_dict
        for name, value in param_dict.items():
            if name == "num_tasks":
                exec_command += ["-n", str(value)]
            elif name == "num_nodes":
                exec_command += ["-N", str(value)]
            elif name == "tasks_per_node":
                exec_command += [f"--ntasks-per-node={value}"]
            else:
                raise ExecParamError(name)
        exec_command += command
        return exec_command

    def __call__(self, command, exec_params=None):  # noqa: D102
        exec_command = self.get_command(command, exec_params)
        exit_code = subprocess.call(" ".join(exec_command), shell=True)
        if exit_code != 0:
            raise ProcessError(exit_code)


class LCLSFExecutor(Executor):
    """Executor for Livermore wrapper around IBM LSF."""

    def get_command(self, command, exec_params=None):  # noqa: D102
        exec_command = ["lrun"]
        param_dict = {}
        if exec_params is not None:
            param_dict = exec_params.param_dict
        for name, value in param_dict.items():
            if name == "num_tasks":
                exec_command += ["-n", str(value)]
            elif name == "num_nodes":
                exec_command += ["-N", str(value)]
            elif name == "tasks_per_node":
                exec_command += ["-T", str(value)]
            elif name == "gpus_per_task":
                exec_command += ["-g", str(value)]
            else:
                raise ExecParamError(name)
        exec_command += command
        return exec_command

    def __call__(self, command, exec_params=None):  # noqa: D102
        exec_command = self.get_command(command, exec_params)
        exit_code = subprocess.call(" ".join(exec_command), shell=True)
        if exit_code != 0:
            raise ProcessError(exit_code)


def make_executor(executor_type_name):
    """
    Create an executor of some type.

    :arg executor_type_name: The executor type name. Can be one of `'basic'`,
        `'slurm'`, or `'lclsf'`.

    :return: An instance of a class derived from :class:`Executor`.
    """
    type_name_map = {
        "basic": BasicExecutor,
        "slurm": SlurmExecutor,
        "lclsf": LCLSFExecutor
    }
    return type_name_map[executor_type_name]()


def get_some_executor():
    """
    Create an executor compatible with the current environment.

    Uses the environment variable `SQUIRM_EXECUTOR_TYPE` if it's set. Otherwise
    tries to guess.

    :return: An instance of a class derived from :class:`Executor`.
    """
    executor_type_name = os.environ.get("SQUIRM_EXECUTOR_TYPE", None)
    if executor_type_name is not None:
        return make_executor(executor_type_name)
    else:
        executor_types_in_ascending_order = [
            ("basic", BasicExecutor),
            ("slurm", SlurmExecutor),
            ("lclsf", LCLSFExecutor)
        ]
        guessed_executor_name, guessed_executor_type = None, None
        for name, executor_type in executor_types_in_ascending_order:
            executable = executor_type().get_command("")[0]
            if subprocess.call(f"which {executable}", stdout=subprocess.DEVNULL,
                        shell=True) == 0:
                guessed_executor_name, guessed_executor_type = name, executor_type
        if guessed_executor_type is not None:
            from warnings import warn
            warn("SQUIRM_EXECUTOR_TYPE has not been set; guessed executor type '"
                        + f"{guessed_executor_name}'.")
        else:
            raise RuntimeError("Unable to detect a valid MPI executor.")
        return guessed_executor_type()
