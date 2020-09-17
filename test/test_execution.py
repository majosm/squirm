from squirm.execution import (  # noqa
        make_executor,
        ProcessError,
        ExecParams,
        ExecParamError)

import pytest
import os
from functools import partial


def get_test_executor():
    executor_type_name = os.environ.get("MPI_EXECUTOR_TYPE", None)
    if executor_type_name is None:
        from warnings import warn
        warn("MPI_EXECUTOR_TYPE has not been set; guessing executor type.")
    return make_executor(executor_type_name)


@pytest.mark.parametrize("num_tasks", [1, 2])
def test_execute_success(num_tasks):
    pytest.importorskip("mpi4py")
    mpi_exec = get_test_executor()
    mpi_exec(["true"], exec_params=ExecParams(num_tasks=num_tasks))


@pytest.mark.parametrize("num_tasks", [1, 2])
def test_execute_fail(num_tasks):
    pytest.importorskip("mpi4py")
    mpi_exec = get_test_executor()
    with pytest.raises(ProcessError):
        mpi_exec(["false"], exec_params=ExecParams(num_tasks=num_tasks))


@pytest.mark.parametrize("num_tasks", [1, 2])
def test_run_success(num_tasks):
    pytest.importorskip("mpi4py")
    mpi_exec = get_test_executor()
    mpi_exec.run("from mpi4py import MPI; MPI.COMM_WORLD.Barrier(); assert True",
                exec_params=ExecParams(num_tasks=num_tasks))


@pytest.mark.parametrize("num_tasks", [1, 2])
def test_run_fail(num_tasks):
    pytest.importorskip("mpi4py")
    mpi_exec = get_test_executor()
    with pytest.raises(ProcessError):
        mpi_exec.run("from mpi4py import MPI; MPI.COMM_WORLD.Barrier(); "
                    + "assert False", exec_params=ExecParams(num_tasks=num_tasks))


def _test_mpi_func(arg):
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()
    assert arg == "hello"


@pytest.mark.parametrize("num_tasks", [1, 2])
def test_call_success(num_tasks):
    pytest.importorskip("mpi4py")
    mpi_exec = get_test_executor()
    mpi_exec.call(partial(_test_mpi_func, "hello"), exec_params=ExecParams(
                num_tasks=num_tasks))


@pytest.mark.parametrize("num_tasks", [1, 2])
def test_call_fail(num_tasks):
    pytest.importorskip("mpi4py")
    mpi_exec = get_test_executor()
    with pytest.raises(ProcessError):
        mpi_exec.call(partial(_test_mpi_func, "goodbye"), exec_params=ExecParams(
                    num_tasks=num_tasks))


def test_unsupported_param():
    pytest.importorskip("mpi4py")
    mpi_exec = get_some_executor()
    try:
        mpi_exec.call(partial(_test_mpi_func, "hello"), exec_params=ExecParams(
                    num_tasks=2, gpus_per_task=1))
        pytest.skip("Oops. Unsupported param is actually supported.")
    except ExecParamError as e:
        assert e.param_name == "gpus_per_task"


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:
        exec(sys.argv[1])
    else:
        from pytest import main
        main([__file__])
