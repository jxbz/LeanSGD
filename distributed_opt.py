import sys
import time
import torch
from mpi4py import MPI
import zlib

import svd_comms


def _bytes_of(obj):
    # BUG: for 2D arrays doesn't return the number of bytes
    # that is, when sizes printed, only 1D sizes printed
    if isinstance(obj, torch.autograd.Variable):
        print('autograd variable')
        return _bytes_of(obj.grad) + obj.element_size()*obj.numel()
    cuda_tensor = getattr(obj, 'cuda', False)
    if isinstance(obj, torch.Tensor) or cuda_tensor:
        # t_size is a lower bound; only the number of elements
        t_size = obj.element_size() * obj.numel()
        #  py_size = sys.getsizeof(obj)
        return t_size

    if isinstance(obj, dict):
        return sum([_bytes_of(v) for k, v in obj.items()])
    if isinstance(obj, tuple) or isinstance(obj, list):
        return sum([_bytes_of(v) for v in obj])

    return sys.getsizeof(obj)  # only counting tensors as stores


class MiniBatchSGD(torch.optim.SGD):
    def __init__(self, *args, **kwargs):
        self.compress = kwargs.pop('compress', False)
        self.encode_kwargs = {key: getattr(self, key) for key in ['compress']}
        super(MiniBatchSGD, self).__init__(*args, **kwargs)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.encode = svd_comms.encode
        self.decode = svd_comms.decode
        print(f"dist_opt.py: rank {self.rank} of {self.size}")

    def collect(self, grad):
        """
        Function that takes the input from one node and returns the outputs from all
        nodes.

        This function distributes parameters to every machine correctly
        [param1(machine1) == param1(machine2)] if MPI is installed correctly
        """
        t0 = time.time()
        send = self.encode(grad, **self.encode_kwargs)
        t1 = time.time()
        recv = self.comm.allgather(send)
        t2 = time.time()
        grads = [self.decode(msg) for msg in recv]
        t3 = time.time()

        times = {'encode_time': t1 - t0, 'comm_time': t2 - t1,
                 'decode_time': t3 - t2, 'rank': self.rank, 'size': self.size,
                 'msg_bytes': _bytes_of(recv)}
        return sum(grads), times

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        data = []
        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            #  reqs = self.collect_all(group['params'])
            for p in group['params']:
                if p.grad is None:
                    continue
                # reqs[i].wait()
                # d_p = self._format_request(reqs[i].get_data())
                d_p, times = self.collect(p.grad.data)
                data += [times]
                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = p.data.new().resize_as_(p.data).zero_()
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        times = {key: val for key, val in data[-1].items()}
        for key in data[-1]:
            if 'time' not in key:
                continue
            times[key] = sum([item[key] for item in data])
        return loss, times
