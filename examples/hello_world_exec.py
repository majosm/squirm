from squirm.execution import get_some_executor, ExecParams

if __name__ == "__main__":
    mpi_exec = get_some_executor()
    mpi_exec(["echo", "Hello, world!"], exec_params=ExecParams(num_tasks=4))
