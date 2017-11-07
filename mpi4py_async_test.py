from pprint import pprint
import numpy as np
import sys
import pickle
from mpi4py import MPI
import io
import numpy as np
import time
comm = MPI.COMM_WORLD

rank = comm.Get_rank()

send_obj = {'rank': rank}
recv_obj = {'rank': rank}

send = bytes("rank = {}".format(rank))
recv = bytes("rank = {}".format(rank))
req = comm.Iallgather()

#  send_msg = pickle.dumps(send_obj)
#  recv_msg = pickle.dumps(recv_obj)
#  send = io.BytesIO(send_msg)
#  recv = io.BytesIO(recv_msg)
#  req = comm.Iallgather(send.getbuffer(), recv.getbuffer())
#  req.wait()

#  print(type(recv))
#  print(bytes(recv.getbuffer()))
#  bytes_obs = bytes(recv.getbuffer())
#  recv_obj = pickle.loads(bytes_obs, encoding='bytes')
#  print(recv)
#  print(recv_obj, rank)

#  obj = {'rank': rank}
#  obj = comm.Ibcast(obj)
#  print(obj)

#  send = np.ones(8, dtype=int) * (rank+1)
#  recv = np.empty(8, np.int)

#  req = comm.Iallgather(send, recv)
#  req.wait()
#  print(f"rank {rank} received")
#  pprint(recv)
