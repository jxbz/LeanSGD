from pprint import pprint
import numpy as np
import sys
from mpi4py import MPI
import io
comm = MPI.COMM_WORLD

rank = comm.Get_rank()
data = [rank] * rank
send = {'rank': rank, 'data': data}
random_send = {i: np.random.choice(10, size=rank) for i in range(3)}
send.update(random_send)

#  recv = b' ' * sys.getsizeof(send) * 10
#  recv = io.BytesIO()
#  #  recv = io.BufferedWriter()
#  with recv as f:
recv = comm.allgather(send)
print(f"rank {rank} received")
pprint(recv)
print(type(recv), type(recv[0]))
