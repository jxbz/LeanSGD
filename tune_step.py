import subprocess
import numpy as np
from distributed import Client
from multiprocessing import Pool
from scipy.optimize import curve_fit



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


def objective(stepsize, cmd=''):
    return (stepsize - 0.1)**2
    #  run = cmd.format(lr=stepsize)
    #  out = subprocess.check_output(run.split(' '))
    #  lines = out.split(b'\n')
    #  lines = [l.decode() for l in lines]
    #  acc = _find_best_acc(lines)
    #  print('stepsize={:0.4f}, acc={}'.format(stepsize, acc))
    #  return acc


def find_step_size(acc, space, k=0, history=None):
    if history is None:
        history = {}
    for param in space:
        if param not in history:
            history[param] = acc(param)
    optimal = min(history, key=history.get)
    if k == 1:
        return optimal, history
    new_space = np.logspace(np.log10(optimal/10),
                            np.log10(optimal*10), num=4)
    return find_step_size(acc, new_space, k=k + 1, history=history)


def test_find_step_size():
    space = [10**i for i in [-2, -1, 0]]
    ratios = []
    for m in np.logspace(-3, 1, num=200):
        for f in [lambda x: (x-m)**2, lambda x: np.exp(((x-m)/10)**2)]:
            x_hat, _ = find_step_size(f, space)
            ratio = x_hat / m
            ratio = 1 / ratio if ratio < 1 else ratio
            ratios += [ratio]
    avg_ratio = sum(ratios) / len(ratios)
    print(avg_ratio)
    print(np.median(ratios))
    print(max(ratios))
    assert np.median(ratios) < 2
    assert avg_ratio < 2
    assert max(ratios) < 3


if __name__ == "__main__":
    test_find_step_size()

    f = lambda x: (x - 0.29) ** 2
    cmd = ('mpirun -n 1 python train.py --layers=10 --lr={lr} --epochs=1 '
           '--num_workers=3')
    space = [10**i for i in [-2, -1, 0]]
    x_hat, history = find_step_size(f, space)

# 9 params
# 4 algs
# 40s/it
# 100it
