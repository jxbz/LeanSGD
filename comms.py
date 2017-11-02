import numpy as np
import mpi4py
from mpi4py import MPI
import torch

comm = MPI.COMM_WORLD
rank = comm.Get_rank()


def _make_tags(model):
    return {name: i for i, (name, param) in enumerate(model.named_parameters())}


def _fmt_send(tensor, dtype='float64'):
    to_send = tensor.data.cpu().numpy().astype(dtype)
    return to_send


def _fmt_recv(tensor, dtype='float64'):
    empty = 0 * np.zeros(tensor.grad.size()).astype(dtype)
    return empty


def send_grads_(model):
    tags = _make_tags(model)
    for name, param in model.named_parameters():
        to_send = _fmt_send(param)
        comm.Isend(to_send, dest=0, tag=tags[name])


def _make_grads(model):
    grads = {}
    for name, param in model.named_parameters():
        grads[name] = _fmt_recv(param)
    return grads


def _to_tensor(x):
    return torch.Tensor(x)


def reduce_grads(grads, model):
    # grads is dict with {name: {tag: grad}}
    for rank in grads:
        grads[rank] = {k: _to_tensor(v) for k, v in grads[rank].items()}
    out = _make_grads(model)
    for k in out:
        out[k] = sum([grads[rank][k] for rank in grads])
    for k in out:
        out[k] = out[k].cuda()
    return out

def recv_all_grads(model, comm):
    size = comm.Get_size()
    grads = {}
    for i in range(1, size):
        grads[i] = recv_grads(model, source=i)
    # TODO: put barrier here to make sure all gradients received
    return grads


def recv_grads(model, source=1):
    grads = _make_grads(model)
    tags = _make_tags(model)
    reqs = []
    for name in grads:
        reqs += [comm.Irecv(grads[name], source=source, tag=tags[name])]
    for req in reqs:
        req.wait()
    return grads


def barrier(comm):
    comm.barrier()


def set_grads(model, grads):
    for name, param in model.named_parameters():
        param.grad.data = grads[name]
    return model


def send_model(model, to=1):
    to = list(to)
    offset = len(list(model.named_parameters()))
    reqs = []
    for dest in to:
        reqs += [comm.isend(model, dest=dest, tag=offset + 1)]
    return reqs
        #  comm.send(model, dest=dest, tag=offset + 1)
    #  return True


def recv_model(model):
    offset = len(list(model.named_parameters()))
    model = comm.recv(source=0, tag=offset + 1)
    #  req = comm.irecv(source=0, tag=offset + 1)
    #  model = req.wait()
    return model
