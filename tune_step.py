import os
import sys
import pickle
import subprocess
import numpy as np
from distributed import Client, LocalCluster
import dask
import distributed
from mpi4py import MPI
from multiprocessing import Pool
from scipy.optimize import curve_fit
from pprint import pprint


# TODO
# - get dask (or some parallelism) working
# - launch on cluster

def map(f, args, **kwargs):
    client = Client()
    result = client.map(f, args, **kwargs)
    client.gather(result)
    return result


def _find_best_acc(lines):
    for line in reversed(lines):
        if 'best accuracy' in line.lower():
            return float(line.split()[-1])
    raise Exception('best accuracy not found in these lines')


def macc(stepsize, cmd=''):
    #  return (stepsize - 0.1)**2
    run = cmd.format(lr=stepsize)
    print("About to run this command:")
    print(run)
    out = subprocess.check_output(run.split(' '))
    print("Done with check_output")
    lines = out.split(b'\n')
    lines = [l.decode() for l in lines]
    try:
        _acc = _find_best_acc(lines)
    except Exception:
        _acc = 0.0
    print('stepsize={:0.4f}, acc={}'.format(stepsize, _acc))
    return -1 * _acc


def find_step_size(macc, space, k=0, history=None, get_client=False, ip=None,
                   **kwargs):
    if history is None:
        history = {}
    if get_client and ip:
        print(f"Trying to connect to {ip}")
        client = distributed.get_client(address=ip)
        jobs = []
        params = []
        for param in space:
            if param not in history:
                jobs += [dask.compute(dask.delayed(macc), param, **kwargs)]
                #  jobs += [client.submit(macc, param, **kwargs)]
                params += [param]
        #  output = [job.result() for job in jobs]
        #  distributed.secede()
        output = client.gather(jobs)
        #  distributed.rejoin()
        for _k, _v in zip(params, output):
            history[_k] = _v
    else:
        for param in space:
            if param not in history:
                history[param] = macc(param, **kwargs)

    optimal = min(history, key=history.get)
    if k == 3:
        return optimal, history
    rand = np.random.rand
    factor = 10 / (k + 1)
    new_space = np.logspace(np.log10(optimal/factor) - rand() / 4,
                            np.log10(optimal*factor) + rand() / 4, num=4)
    return find_step_size(macc, new_space, k=k + 1, history=history,
                          get_client=get_client, **kwargs)


def test_find_step_size():
    space = [10**i for i in [-2, -1, 0]]
    rel_error = []
    log_rel_error = []
    for m in np.logspace(-3, 1.0, num=1000):
        for f in [lambda x: (x-m)**2, lambda x: np.exp(((x-m)/30)**2)]:
            x_hat, _ = find_step_size(f, space)
            diff = np.abs(m - x_hat) / m
            rel_error += [diff]
            log_rel_error += [np.log10(np.abs(m - x_hat) / m)]
    for diffs in [rel_error, log_rel_error]:
        avg_diff = sum(diffs) / len(diffs)
        print("avg =", avg_diff)
        print("median =", np.median(diffs))
        print("max =", max(diffs))
        print("min =", min(diffs))
        print('-' * 30)
    assert np.median(rel_error) < 0.15
    assert sum(rel_error) / len(rel_error) < 0.15
    assert max(rel_error) < 0.4
    return x_hat, _


if __name__ == "__main__":
    #  opt, hist = test_find_step_size()
    #  pprint(sorted(list(hist.keys())))
    #  print(len(hist))
    #  sys.exit(0)
    #  client = Client(LocalCluster(n_workers=4))
    ip = '172.31.10.71:8786'
    client = Client(ip)
    #  client = Client()
    #  client = Client()
    #  client = None
    #  args = ['--compress=0']
    #  args += [f'--compress=1 --svd_rank={rank} --svd_rescale={rescale}'
             #  for rank in [1, 2, 4, 8] for rescale in [0, 1]]
    #  args += ['--compress=1 --svd_rank=0 --svd_rescale=1']
    #  args += ['--compress=1 --svd_rank=-1 --svd_rescale=0']
    args = ['--compress=0', '--qsgd=1', '--compress=1 --svd_rank=0 --svd_rescale=1']
    args = [arg + f' --use_mpi={use_mpi}'
            for use_mpi in [0, 1] for arg in args]

    print(f"running len(args) = {len(args)} jobs")

    layers = 34
    n_workers = 1
    home = '/home/ec2-user'
    cmd = (f'mpirun -n {n_workers} sudo {home}/anaconda3/bin/python '
           f'{home}/WideResNet-pytorch/train.py --layers={layers} '
           '--lr={lr} --epochs=6 --num_workers=1 '
           '--nesterov=0 --weight-decay=0 --momentum=0')
    cmds = []
    jobs = []
    for arg in args:
        run = cmd + ' ' + arg
        print(run)
        kwargs = {'cmd': run}

        space = [10**i for i in [-2, -1, 0]]
        #  jobs += [client.submit(find_step_size, macc, space, **kwargs)]
        jobs += [client.submit(macc, 0.1, **kwargs)]
        cmds += [run]
        #  x_hat, history = find_step_size(macc, space, client=client, **kwargs)
    output = client.gather(jobs)
    print("Done with all jobs")
