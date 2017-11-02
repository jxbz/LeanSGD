from pprint import pprint
import numpy as np
from mpi4py import MPI
comm = MPI.COMM_WORLD

rank = comm.Get_rank()
data = [rank] * rank
send = {'rank': rank, 'data': data}
random_send = {i: np.random.choice(10, size=rank) for i in range(3)}
send.update(random_send)

recv = comm.allgather(send)
pprint(f"rank {rank} received {recv}")
