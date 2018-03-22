# Wide Residual Networks (WideResNets) in PyTorch
WideResNets for CIFAR10/100 implemented in PyTorch. This implementation requires less GPU memory than what is required by the official Torch implementation: https://github.com/szagoruyko/wide-residual-networks.

Example:
```
python train.py --dataset cifar100 --layers 40 --widen-factor 4
```

# How to generate results

The script `run.py` will run `train.py` and write summary CSVs into
`output/{today`. The `run.py` script runs the commands

``` python
python train.py --qsgd=1  # use QSGD coding
python train.py --compress=1 --svd_rank=0 --svd_rescale=1  # use SVD coding
python train.py --compress=0  # use normal SGD with the param server
```

Note that extra arguments are added to each of these commands.

# File structure
```
pytorch_ps_mpi/
    ps.py
    mpi_comms.py
codings/
    coding.py
    ...
train.py
run.py
```

* `pytorch_ps_mpi`: The package that manages distributed training. It is a
  separate Git repository at https://github.com/stsievert/pytorch_ps_mpi (and
  is unfortunately named).
  * `ps.py`: The main file in this package, which holds the different
    optimizers. These are slightly modified from `torch.optim` optimizers. In
    this, we code the gradients asynchrously with gradient computation. We then
    wait for all codings to finish before sending them.
  * `mpi_comms.py`: This is the script that serializes the gradients and sends
    them.
* `codings`: The package with different coding schemes. The base coding class
  is in `coding.py`.
* `train.py`: The main training script. `run.py` calls this with an `os.system`
  call.


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

# Acknowledgement
- [densenet-pytorch](https://github.com/andreasveit/densenet-pytorch)
- Wide Residual Networks (BMVC 2016) http://arxiv.org/abs/1605.07146 by Sergey Zagoruyko and Nikos Komodakis.
