import torch
import numpy as np


def _resize_to_2d(x):
    """
    x.shape > 2
    If x.shape = (a, b, *c), assumed that each one of (a, b) pairs has relevant information in c.
    """
    size = x.size()
    if all([s == 1 for s in size[2:]]):
        return x.view(size[0], size[1])
    # each of (a, b) has related features
    x = x.view(size[0], size[1], -1)
    # stack those related features into a tall matrix
    return x.view(size[0]*size[1], -1)


def _sample_svd(s, rank=0):
    probs = s / s[0] if rank == 0 else rank * s / s.sum()
    sampled_idx = []
    for i, p in enumerate(probs):
        if np.random.rand() < p:
            sampled_idx += [i]
    rank_hat = len(sampled_idx)
    if rank_hat == 0 or (rank != 0 and np.abs(rank_hat - rank) >= 2):
        return _sample_svd(s, rank=rank)
    return sampled_idx


def encode(grad, compress=True, svd_rank=0, random_sample=True):
    if not compress:
        size = list(grad.size())
        return {'grad': grad, 'encode': False}

    orig_size = list(grad.size())
    ndims = len(grad.size())
    reshaped_flag = False
    if ndims > 2:
        grad = _resize_to_2d(grad)
        size = list(grad.size())
        ndims = len(size)
        reshaped_flag = True

    if ndims == 2:
        u, s, v = torch.svd(grad, some=True)
        if random_sample:
            i = _sample_svd(s, rank=svd_rank)
            i = torch.LongTensor(i)
            if torch.cuda.is_available():
                i = i.cuda()
            u = u[:, i]
            s = s[i]
            v = v[:, i]
        else:
            u = u[:, :svd_rank]
            s = s[:svd_rank]
            v = v[:, :svd_rank]

        return {'u': u, 's': s, 'v': v, 'orig_size': orig_size,
                'reshaped': reshaped_flag, 'encode': True}
    return {'grad': grad, 'encode': False}


def decode(encode_output, rescale=True):
    encode = encode_output.get('encode', False)
    if not encode:
        return encode_output['grad']

    u, s, v = (encode_output[key] for key in ['u', 's', 'v'])
    if rescale:
        s = s[0] / s
    grad_approx = u @ torch.diag(s) @ v.t()
    if encode_output.get('reshaped', False):
        grad_approx = grad_approx.view(encode_output['orig_size'])
    return grad_approx
