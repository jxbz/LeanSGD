import os
import sys
import pickle
import subprocess
import numpy as np
from distributed import Client, LocalCluster
from dask import delayed
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
    _acc = _find_best_acc(lines)
    print('stepsize={:0.4f}, acc={}'.format(stepsize, _acc))
    return -1 * _acc


def find_step_size(macc, space, k=0, history=None, get_client=False, **kwargs):
    if history is None:
        history = {}
    if get_client:
        client = distributed.get_client()
        jobs = []
        params = []
        for param in space:
            if param not in history:
                jobs += [client.compute(delayed(macc), param, **kwargs)]
                params += [param]
        #  output = [job.result() for job in jobs]
        output = client.gather(jobs)
        for _k, _v in zip(params, output):
            history[_k] = _v
    else:
        for param in space:
            if param not in history:
                history[param] = macc(param, **kwargs)

    optimal = min(history, key=history.get)
    if k == 2:
        return optimal, history
    new_space = np.logspace(np.log10(optimal/10),
                            np.log10(optimal*10), num=7)
    return find_step_size(macc, new_space, k=k + 1, history=history,
                          get_client=get_client, **kwargs)


def test_find_step_size():
    space = [10**i for i in [-2, -1, 0]]
    diffs = []
    for m in np.logspace(-3, 1, num=400):
        for f in [lambda x: (x-m)**2, lambda x: np.exp(((x-m)/10)**2)]:
            x_hat, _ = find_step_size(f, space)
            diff = np.abs(m - x_hat) / m
            diffs += [diff]
    avg_diff = sum(diffs) / len(diffs)
    print(avg_diff)
    print(np.median(diffs))
    print(max(diffs))
    assert np.median(diffs) < 0.3
    assert avg_diff < 0.3
    assert max(diffs) < 0.7
    return x_hat, _


if __name__ == "__main__":
    #  opt, hist = test_find_step_size()
    #  pprint(list(hist.keys()))
    #  print(len(hist))
    #  sys.exit(0)
    #  client = Client(LocalCluster(n_workers=4))
    client = Client('172.31.14.142:8786')
    #  client = Client()
    #  client = None
    args = ['--compress=0']
    args += [f'--compress=1 --svd_rank={rank} --svd_rescale={rescale}'
             for rank in [1, 2, 4, 8] for rescale in [0, 1]]
    args += ['--compress=1 --svd_rank=0 --svd_rescale=1']
    print("running len(args) = {len(args)} jobs (each spawns more jobs adaptively)")

    layers = 50
    n_workers = 1
    home = '/home/ec2-user'
    cmd = (f'mpirun -n {n_workers} sudo {home}/anaconda3/bin/python '
           f'{home}/WideResNet-pytorch/train.py --layers={layers} '
           '--lr={lr} --epochs=2 --num_workers=1 '
           '--nesterov=0 --weight-decay=0 --momentum=0')
    cmds = []
    jobs = []
    for arg in args:
        cmd += ' ' + arg
        print(cmd)
        kwargs = {'cmd': cmd}

        space = [10**i for i in [-2.5, -1.5, -1, 0]]
        jobs += [client.submit(find_step_size, macc, space, **kwargs)]
        cmds += [cmd]
        #  x_hat, history = find_step_size(macc, space, client=client, **kwargs)
    output = client.gather(jobs)
    for cmd, (x_hat, history) in zip(cmds, output):
        print(f"{cmd} found stepsize_est = {x_hat} with history = {history}")

        uid = cmd.replace(' ', '')
        with open('output/2017-11-20/' + uid + '.pkl', 'wb') as f:
            pickle.dump({'stepsize_est': x_hat, 'cmd': cmd, 'history': history}, f)
