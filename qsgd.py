from functools import reduce
import numpy as np
from scipy import stats
import torch
import time


def encode(v, scheme='qsgd', **kwargs):
    w = v.view(-1).numpy()
    if scheme == 'qsgd':
        norm = torch.norm(v)
    elif scheme == 'terngrad':
        norm = np.linalg.norm(w, ord=np.inf)
        limit = grad_clip_limit(w, clip_factor=2.5)
        w = np.clip(w, -limit, limit)

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


def decode(code, cuda=False, **kwargs):
    if 'max_norm' in kwargs.keys():
        code['norm'] = kwargs['max_norm']

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


def pre_decode(codes):
    scalars = [code['norm'] for code in codes]
    return {'max_norm': max(scalars)}


def grad_clip_limit(grad, clip_factor=2.5):
    """ Get the scalers."""
    if clip_factor > 1.0e-5:
        return clip_factor * np.std(grad.flat[:])
    return np.max(np.abs(grad.flat[:]))


if __name__ == "__main__":
    n = 50
    x = torch.rand(n)
    for scheme in ['terngrad', 'qsgd']:
        repeats = int(10e3)
        codes = [encode(x, scheme=scheme) for _ in range(repeats)]

        max_scalar = pre_decode(codes)
        approxs = [decode(code, **max_scalar) for code in codes]

        data = map(lambda arg: {'y': arg[1], 'norm(y)**2': torch.norm(arg[1])**2,
                                'len(signs)': len(arg[0]['signs'])},
                   zip(codes, approxs))
        sums = reduce(lambda x, y: {k: x[k] + y[k] for k in x}, data)
        avg = {k: v / len(codes) for k, v in sums.items()}
        assert avg['norm(y)**2'] <= np.sqrt(n) * torch.norm(x)**2
        if scheme == 'qsgd':
            assert avg['len(signs)'] <= np.sqrt(n)
        rel_error = torch.norm(avg['y'] - x) / torch.norm(x)
        print(scheme, rel_error)
        assert rel_error < 0.25
