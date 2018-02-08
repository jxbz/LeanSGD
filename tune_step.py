import os
import sys
import pickle
import subprocess

import numpy as np
import pandas as pd

import distributed
from distributed import Client, get_client, secede, rejoin, Future

from pprint import pprint


def _find_lowest_loss(lines):
    losses_str = filter(lambda line: 'min_train_loss' in line, lines)
    losses = map(lambda line: {'loss': float(line.split(' ')[1]),
                               'step': int(line.split(' ')[2])}, losses_str)
    #  import ipdb as pdb; pdb.set_trace()
    df = pd.DataFrame(list(losses)).groupby(by='step').mean()
    avg_loss = df.to_dict()['loss']

    if len(avg_loss) == 0:
        raise Exception('min_train_loss not found in these lines')
    return min(v for _, v in avg_loss.items())


def loss(stepsize, cmd=''):
    #  return (stepsize - 20.001)**2
    #  run = cmd + ''
    run = cmd.format(lr=stepsize)
    print("About to run this command:")
    print(run)
    #  try:
    out = subprocess.check_output('which mpirun'.split(' '))
    try:
        out = subprocess.check_output(run.split(' '))
    except subprocess.CalledProcessError as e:
        print(e.cmd)
        print(e.stdout)
        print(e.output)
        raise
    print("Done with check_output")
    lines = out.split(b'\n')
    lines = [l.decode() for l in lines]
    #  try:
    _loss = _find_lowest_loss(lines)
    #  except Exception:
        #  _loss = np.inf
    print('stepsize={:0.4f}, loss={}'.format(stepsize, _loss))
    return _loss


def find_step(step_loss, cmd=''):
    client = get_client()
    for step in step_loss:
        if step_loss[step] is None:
            step_loss[step] = client.submit(loss, step, cmd=cmd)
    secede()
    for step, value in step_loss.items():
        if isinstance(value, Future):
            step_loss[step] = value.result()
    rejoin()

    step_w_min_loss = min(step_loss, key=step_loss.get)

    if step_w_min_loss == min(step_loss):
        step_loss[step_w_min_loss / 2] = None
        return find_step(step_loss, cmd=cmd)
    if step_w_min_loss == max(step_loss):
        step_loss[step_w_min_loss * 2] = None
        return find_step(step_loss, cmd=cmd)

    return step_w_min_loss, step_loss, cmd


if __name__ == "__main__":
    layers = 10
    n_workers = 1
    widen_factor = 1
    home = '/home/ec2-user'
    pre = (f'mpirun -n {n_workers} -hostfile {home}/WideResNet-pytorch/hosts '
           f'--map-by ppr:1:node')
    pre = 'mpirun -n 1'
    cmd = (f'{home}/anaconda3/bin/python '
           f'{home}/WideResNet-pytorch/train.py --layers={layers} '
           '--epochs=0 --num_workers=1 '
           '--nesterov=0 --weight-decay=0 --momentum=0 '
           f'--widen-factor={widen_factor}')
    cmd = pre + ' ' + cmd
    args = ['--qsgd=1', '--compress=1 --svd_rank=0 --svd_rescale=1',
            '--compress=0']
    #  args += [f'--compress=1 --svd_rank={rank} --svd_rescale={rescale}'
    #  for rank in [2, 4, 8]]
    cmds = [cmd + ' ' + arg + ' --lr={lr}' for arg in args]
    cmds = [cmds[0]]

    step_loss = {0.1: None, 0.2: None, 0.05: None}

    client = Client()
    futures = []
    for cmd in cmds:
        futures += [client.submit(find_step, step_loss, cmd=cmd)]

    results = client.gather(futures)
    results = [{'step': step, 'history': history, 'cmd': cmd}
               for step, history, cmd in results]
    pprint(results)
    with open('tune_step.pkl', 'wb') as f:
        pickle.dump(results, f)
