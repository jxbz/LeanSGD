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


def _sample_svd(s):
    probs = s / s[0]
    sampled_idx = []
    for i, p in enumerate(probs):
        if np.random.rand() < p:
            sampled_idx += [i]
    return sampled_idx


def encode(grad, compress=True):
    if not compress:
        size = list(grad.size())
        info = [0, len(size)] + size
        info = torch.Tensor(info)
        return torch.cat((info, grad.view(-1)), 0)

    orig_size = list(grad.size())
    ndims = len(grad.size())
    if ndims > 2:
        grad = _resize_to_2d(grad)
        size = list(grad.size())
        ndims = len(size)
        reshaped = [1, len(size), size]
        reshaped_flag = True

    if ndims == 2:
        u, s, v = torch.svd(grad, some=True)
        #  i = _sample_svd(s)
        #  u = u[:, i]
        #  s = s[torch.cuda.LongTensor(i)]
        #  v = v[:, i]
        u = u[:, :compress]
        s = s[:compress]
        v = v[:, :compress]

        info = [1, len(orig_size)] + orig_size
        if reshaped_flag:
            info += reshaped
        else:
            info += [0]

        info = torch.Tensor(info)
        return torch.cat((info, u.view(-1), s.view(-1), v.view(-1)), 0)

    info = [0, len(orig_size)] + orig_size
    info = torch.Tensor(info)
    return torch.cat((info, grad.view(-1)), 0)


def decode(encode_output):
    encode = encode_output[0]
    if not encode:
        ndims = encode_output[1]
        size = encode_output[2:ndims]
        grad = encode_output[ndims:].view(size)
        return grad
    #  reshaped =

    u, s, v = (encode_output[key] for key in ['u', 's', 'v'])
    s = s[0] / s
    grad_approx = u @ torch.diag(s) @ v.t()
    if encode_output.get('reshaped', False):
        grad_approx = grad_approx.view(encode_output['orig_size'])
    return grad_approx
