""":mod:`squirm.execution` wraps non-submitting executors (mpiexec, srun, etc.)

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
import pickle
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
        Returns a list of strings representing the full command that will be executed
        to launch *command*.
        """
        pass

    @abc.abstractmethod
    def __call__(self, command, exec_params=None):
        """Executes *command* with MPI."""
        pass

    def run(self, code_string, exec_params=None):
        """Runs Python code stored in *code_string* with MPI."""
        self.__call__([sys.executable, "-m", "mpi4py", "-c", "\'"
                    + code_string + "\'"], exec_params)

    def call(self, func, exec_params=None):
        """Calls *func* with MPI. Note: *func* must be picklable."""
        calling_code = ('import sys; import pickle; pickle.loads(bytes.fromhex("'
                    + pickle.dumps(func).hex() + '"))()')
        self.run(calling_code, exec_params)


class ExecParams:
    """
    Collection of parameters to pass to the executor.

    .. automethod:: __init__
    """
    def __init__(self, num_tasks=None, num_nodes=None, tasks_per_node=None,
                gpus_per_task=None):
        """
        Possible arguments are:

        :arg num_tasks: The number of MPI tasks to launch.
        :arg num_nodes: The number of nodes on which to run.
        :arg tasks_per_node: The number of MPI tasks to launch per node.
        :arg gpus_per_task: The number of GPUs to assign per task.

        Note: A given executor may not support all of these arguments. If it is
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
    def __init__(self, exit_code):
        self.exit_code = exit_code
        super().__init__(f"Execution failed with exit code {exit_code}.")


class ExecParamError(RuntimeError):
    def __init__(self, param_name):
        self.param_name = param_name
        super().__init__(f"Executor does not support parameter '{self.param_name}'.")


class BasicExecutor(Executor):
    """Simple `mpiexec` executor."""
    def get_command(self, command, exec_params=None):
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

    def __call__(self, command, exec_params=None):
        exec_command = self.get_command(command, exec_params)
        exit_code = subprocess.call(" ".join(exec_command), shell=True)
        if exit_code != 0:
            raise ProcessError(exit_code)


class SlurmExecutor(Executor):
    """Executor for Slurm."""
    def get_command(self, command, exec_params=None):
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

    def __call__(self, command, exec_params=None):
        exec_command = self.get_command(command, exec_params)
        exit_code = subprocess.call(" ".join(exec_command), shell=True)
        if exit_code != 0:
            raise ProcessError(exit_code)


class LCLSFExecutor(Executor):
    """Executor for Livermore wrapper around IBM LSF."""
    def get_command(self, command, exec_params=None):
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

    def __call__(self, command, exec_params=None):
        exec_command = self.get_command(command, exec_params)
        exit_code = subprocess.call(" ".join(exec_command), shell=True)
        if exit_code != 0:
            raise ProcessError(exit_code)


def make_executor(executor_type_name):
    """
    Returns an instance of a class derived from :class:`Executor` given an executor
    type name as input.

    :arg executor_type_name: The executor type name. Can be one of `'basic'`,
        `'slurm'`, or `'lclsf'`.
    """
    type_name_map = {
        "basic": BasicExecutor,
        "slurm": SlurmExecutor,
        "lclsf": LCLSFExecutor
    }
    return type_name_map[executor_type_name]()


def get_some_executor():
    """
    Returns an instance of a class derived from :class:`Executor` based on the
    environment variable `SQUIRM_EXECUTOR_TYPE` if it's set (and a guess, otherwise).
    """
    executor_type_name = os.environ.get("SQUIRM_EXECUTOR_TYPE", None)
    if executor_type_name is not None:
        return make_executor(executor_type_name)
    else:
        from warnings import warn
        executor_types_in_ascending_order = [
            BasicExecutor,
            SlurmExecutor,
            LCLSFExecutor
        ]
        guessed_executor_type = None
        for executor_type in executor_types_in_ascending_order:
            executable = executor_type().get_command("")[0]
            if subprocess.call(f"which {executable}", stdout=subprocess.DEVNULL,
                        shell=True) == 0:
                guessed_executor_type = executor_type
        if guessed_executor_type is not None:
            warn("SQUIRM_EXECUTOR_TYPE has not been set; guessed executor type '"
                        + f"{guessed_executor_type.__name__}'.")
        else:
            raise RuntimeError("Unable to detect a valid MPI executor.")
        return guessed_executor_type()
