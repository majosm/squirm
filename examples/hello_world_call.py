from squirm.execution import get_some_executor, ExecParams


def print_greeting(greeting):
    from mpi4py import MPI
    MPI.COMM_WORLD.Barrier()
    print(greeting)


if __name__ == "__main__":
    # Can't pickle directly from __main__
    from hello_world_call import print_greeting as _print_greeting

    mpi_exec = get_some_executor()
    mpi_exec.call(_print_greeting, "Hello, world!", exec_params=ExecParams(
                num_tasks=4))
