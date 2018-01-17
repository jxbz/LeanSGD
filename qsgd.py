from functools import reduce
import numpy as np
from scipy import stats
import torch
import time


def encode(v, **kwargs):
    norm = torch.norm(v)
    w = v.view(-1)

    t = [time.time()]
    signs = torch.sign(w).int()
    probs = torch.abs(w) / norm
    mask = torch.distributions.Bernoulli(probs).sample().byte()
    t += [time.time()]
    idx = torch.arange(0, len(w))
    t += [time.time()]
    if v.is_cuda:
        idx = idx.cuda()
        mask = mask.cuda()
    t += [time.time()]

    selected = torch.masked_select(idx, mask).long()
    signs = torch.masked_select(signs, mask)
    t += [time.time()]
    data = {'masking_time': t[-1] - t[-2], 'gen_mask_time': t[1] - t[0],
            'to_gpu_time': t[-2] - t[-3]}
    return {'signs': signs, 'size': v.size(), 'selected': selected,
            'norm': norm}#, data


def decode(code, cuda=False):
    v = torch.zeros(code['size'])
    signs = code['signs']
    print("QSGD cuda =", cuda)
    if cuda:
        v = v.cuda()
        signs = signs.cuda()
    flat = v.view(-1)
    if len(code['selected']) > 0:
        r = code['norm'] * signs.float()
        flat[code['selected']] = r
    return v

if __name__ == "__main__":
    n = 500
    x = torch.rand(n)
    repeats = 1000

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
    assert rel_error < 0.2
