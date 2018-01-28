from functools import reduce
import numpy as np
from scipy import stats
import torch
import time


def encode(v, **kwargs):
    norm = torch.norm(v)
    w = v.view(-1).numpy()

    signs = np.sign(w).astype('int') #.int()
    probs = np.abs(w) / norm
    mask = stats.bernoulli.rvs(probs).astype('bool')
    idx = np.arange(len(w))

    selected = idx[mask].astype('uint32')
    signs = signs[mask].astype('int8')
    signs = ((signs + 1) / 2).astype('bool')

    code = {'signs': signs, 'size': v.size(), 'selected': selected,
            'norm': norm}

    if kwargs.pop('timings', False):
        data = {}
        return code, data
    return code


def decode(code, cuda=False):
    v = np.zeros(code['size'])
    signs = np.array(code['signs'], dtype='int8')
    signs = signs*2 - 1
    selected = np.array(code['selected'], dtype='int16')
    #  selected = torch.LongTensor(selected)

    if len(selected) > 0:
        v.flat[selected] = code['norm'] * signs
    v = torch.Tensor(v)
    if cuda:
        v = v.cuda()
    return v

if __name__ == "__main__":
    n = 50
    x = torch.rand(n)
    repeats = int(10e3)

    codes = [encode(x) for _ in range(repeats)]
    approxs = [decode(code) for code in codes]
    data = map(lambda arg: {'y': arg[1], 'norm(y)**2': torch.norm(arg[1])**2,
                            'len(signs)': len(arg[0]['signs'])},
               zip(codes, approxs))
    sums = reduce(lambda x, y: {k: x[k] + y[k] for k in x}, data)
    avg = {k: v / len(codes) for k, v in sums.items()}
    assert avg['norm(y)**2'] <= np.sqrt(n) * torch.norm(x)**2
    assert avg['len(signs)'] <= np.sqrt(n)
    rel_error = torch.norm(avg['y'] - x) / torch.norm(x)
    print(rel_error)
    assert rel_error < 0.2
