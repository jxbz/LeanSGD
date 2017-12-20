# Wide Residual Networks (WideResNets) in PyTorch
WideResNets for CIFAR10/100 implemented in PyTorch. This implementation requires less GPU memory than what is required by the official Torch implementation: https://github.com/szagoruyko/wide-residual-networks.

Example:
```
python train.py --dataset cifar100 --layers 40 --widen-factor 4
```

# Distributed training
``` shell
mpirun -n 3 -hostfile hosts --map-by ppr:1:node python train.py
```

A quick speed test with 2 p2.xlarges and 34 layers:

* qsgd: 1.33it/s (with the 1 bit compression)
* normal: 2.33it/s (no compression)
* svd: 1.77it/s (theoretically sound compression)

And with 100 layers:

* svd: 1.65s/it
* norm: 1.36s/it
* qsgd: 2.22s/it

How will this change as the number of workers increase?

## Async comms
``` python
def encode(grad):
    u, s, v = svd(grad)
    return concat(h, u, s, v, w)

def decode(encode_output):
    h, w = encode_output[...]
    u = encode_out[...]
    s = encode_out[...]
    v = encode_out[...]
    grad = u @ s @ v

def _decode(recv):
    grads = [decode(recv[i]) for i in range(len(recv))]
    return sum(grads)

def _encode(model):
    for model in named_params:
        send = encode(param)
        recv = np.array([send] * workers)
        req = Ialltoall(send, recv)
    return recvs

def step():
    self.svd_rank = known

    recvs = _encode(model)
    for param, recv in training_loop:
        req.wait()
        p.grad.data = _decode(recv)
```

# Acknowledgement
- [densenet-pytorch](https://github.com/andreasveit/densenet-pytorch)
- Wide Residual Networks (BMVC 2016) http://arxiv.org/abs/1605.07146 by Sergey Zagoruyko and Nikos Komodakis.
