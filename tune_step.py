import os
import sys
import pickle
import subprocess
import numpy as np
from distributed import Client, LocalCluster
import dask
import distributed
import pandas as pd
from mpi4py import MPI
from multiprocessing import Pool
from scipy.optimize import curve_fit
from pprint import pprint
from distributed import worker_client
from distributed import get_client, secede, rejoin


def _find_lowest_loss(lines):
    losses_str = filter(lambda line: 'min_train_loss' in line, lines)
    losses = map(lambda line: {'loss': float(line.split(' ')[1]),
                               'step': int(line.split(' ')[2])}, losses_str)
    df = pd.DataFrame(list(losses)).groupby(by='step').mean()
    avg_loss = df.to_dict()['loss']

    if len(avg_loss) == 0:
        raise Exception('min_train_loss not found in these lines')
    return min(v for _, v in avg_loss.items())


def loss(stepsize, cmd=''):
    #  return (stepsize - 0.1)**2
    #  run = cmd + ''
    run = cmd.format(lr=stepsize)
    print("About to run this command:")
    print(run)
    try:
        out = subprocess.check_output(run.split(' '))
    except:
        print(run)
        raise
    print("Done with check_output")
    lines = out.split(b'\n')
    lines = [l.decode() for l in lines]
    try:
        _loss = _find_lowest_loss(lines)
    except Exception:
        _loss = np.inf
    print('stepsize={:0.4f}, loss={}'.format(stepsize, _loss))
    return _loss


def find_step_size(f, space, k=0, history=None, dask=True,
                   get_client_=False, ip=None,
                   **kwargs):
    if history is None:
        history = {}
    if get_client_:
        print(f"Trying get_client()")
        jobs = []
        params = []
        client = get_client()
        for param in space:
            if param not in history:
                jobs += [client.submit(f, param, **kwargs)]
                params += [param]
        secede()
        output = client.gather(jobs)
        rejoin()
        for _k, _v in zip(params, output):
            history[_k] = _v
    else:
        for param in space:
            if param not in history:
                history[param] = f(param, **kwargs)

    optimal = min(history, key=history.get)
    if k == 3:
        return optimal, history
    rand = np.random.rand() / 4
    factor = 10 / (k + 1)
    new_space = np.logspace(np.log10(optimal/factor) - rand,
                            np.log10(optimal*factor) + rand, num=4)
    return find_step_size(f, new_space, k=k + 1, history=history,
                          get_client_=get_client_, **kwargs)


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
    ip = '172.31.13.19:8786'
    client = Client(ip)
    #  client = Client()
    #  client = None
    #  args = ['--compress=0', '--qsgd=1', '--compress=1 --svd_rank=0 --svd_rescale=1']
    #  args = [arg + f' --use_mpi={use_mpi}'
            #  for use_mpi in [0, 1] for arg in args]
    args = ['--qsgd=1', '--compress=1 --svd_rank=0 --svd_rescale=1',
            '--compress=0']
    args += [f'--compress=1 --svd_rank={rank} --svd_rescale={rescale}'
             for rank in [2, 4, 8] for rescale in [0, 1]]
    #  args = ['--compress=1 --svd_rank=0 --svd_rescale=1']

    print(f"running len(args) = {len(args)} jobs (each jobs spawns more adaptively)")

    layers = 94
    n_worker = 1
    home = '/home/ec2-user'
    pre = 'mpirun -n 1'
    cmd = (f'sudo {home}/anaconda3/bin/python '
           f'{home}/WideResNet-pytorch/train.py --layers={layers} '
           '--epochs=4 --num_workers=1 '
           '--nesterov=0 --weight-decay=0 --momentum=0')
    cmds = [pre + ' ' + cmd + ' ' + arg for arg in args]
    cmds = [cmd + ' --lr={lr}' for cmd in cmds]
    print(cmds)
    #  cmds = []
    jobs = []
    for run in cmds:
        print(run)
        kwargs = {'cmd': run, 'get_client_': True}

        space = [10**i for i in [-2, -1, 0]]
        jobs += [client.submit(find_step_size, loss, space, **kwargs)]
        #  jobs += [client.submit(loss, 0.1, **kwargs)]
        #  x_hat, history = find_step_size(macc, space, client=client, **kwargs)
    output = client.gather(jobs)
    print("Done with all jobs")
