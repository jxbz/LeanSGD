import os
import sys
import pickle
import subprocess

import numpy as np
import pandas as pd

from pprint import pprint


def _find_lowest_loss(lines):
    losses_str = filter(lambda line: 'min_train_loss' in line, lines)
    losses = map(lambda line: {'loss': float(line.split(' ')[1]),
                               'step': int(line.split(' ')[2])}, losses_str)
    #  import ipdb as pdb; pdb.set_trace()
    df = pd.DataFrame(list(losses)).groupby(by='step').mean()
    avg_loss = df.to_dict()['loss']
    print(avg_loss)

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
    _loss = _find_lowest_loss(lines)
    print('## stepsize={:0.4f}, loss={}'.format(stepsize, _loss))
    return _loss


def find_step(step_loss, cmd=''):
    for step in step_loss:
        if step_loss[step] is None:
            step_loss[step] = loss(step, cmd=cmd)

    step_w_min_loss = min(step_loss, key=step_loss.get)

    if step_w_min_loss == min(step_loss):
        step_loss[step_w_min_loss / 2] = None
        return find_step(step_loss, cmd=cmd)

    if step_w_min_loss == max(step_loss):
        step_loss[step_w_min_loss * 2] = None
        return find_step(step_loss, cmd=cmd)

    return step_w_min_loss, step_loss, cmd


# 4 => 1: 16, 8, 4, 2 machines
# 6 => 1: 6 algs
# 6: 6 combos of params (3 factors, 2 freq)
# 39: 39 epochs
# 50: 50 steps/epoch
# 2: 2 s/step

from concurrent.futures import ThreadPoolExecutor, as_completed
if __name__ == "__main__":
    args = ['--code=svd --svd_rank=1 --svd_rescale=1',
            '--code=qsvd --scheme=qsgd --svd_rank=1',
            '--code=qsvd --scheme=terngrad --svd_rank=1']

    divs = {'--code=qsgd': [1, 2, 4], '--code=terngrad': [1, 2, 4],
            '--code=svd --svd_rank=3 --svd_rescale=1': [1, 2, 4],
            '--code=svd --svd_rank=6 --svd_rescale=1': [1, 2, 4],
            '--code=qsvd --scheme=terngrad': [8, 16, 32],
            '--code=qsvd --scheme=qsgd': [8, 16, 32]}

    results = []
    worker_idx = int(os.environ.get('WORKER_IDX', 0))
    cluster_size = int(os.environ.get('CLUSTER_SIZE', 1))
    idx = 0
    for n_workers in [4, 2]:
        for layers, widen_factor, init_lr in [(142, 1, 2.0), (52, 4, 0.1250)]:
            for arg in args:
                idx += 1
                print("idx =", idx, arg, layers)
                if not ((idx % cluster_size) == worker_idx):
                    continue
                home = '/home/ec2-user'
                pre = (f'mpirun -n {n_workers} -hostfile '
                       f'{home}/WideResNet-pytorch/hosts{worker_idx} '
                       f'--map-by ppr:1:node')
                #  pre = 'mpirun -n 1'
                cmd = (f'{home}/anaconda3/bin/python '
                       f'{home}/WideResNet-pytorch/train.py --layers={layers} '
                       '--epochs=6 '
                       '--nesterov=0 --weight-decay=0 --momentum=0 '
                       f'--widen-factor={widen_factor}')
                cmd = pre + ' ' + cmd + ' ' + arg + ' --lr={lr}'
                print('arg in divs?', arg in divs)
                lrs = [init_lr / (i * np.sqrt(n_workers)) for i in divs.get(arg, [4, 8, 16])]
                print(lrs)
                print([type(lr) for lr in lrs])
                step_loss = {lr: None for lr in lrs}

                step, history, cmd = find_step(step_loss, cmd=cmd)
                results += [{'step': step, 'history': history, 'cmd': cmd}]
                pprint(results[-1])

                with open('tune_step.pkl', 'wb') as f:
                    pickle.dump(results, f)
