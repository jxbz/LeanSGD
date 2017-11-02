import torch

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


def encode(grad, svd_rank=3):
    orig_size = grad.size()
    ndims = len(grad.size())
    reshaped = False
    if ndims > 2:
        grad = _resize_to_2d(grad)
        ndims = len(grad.size())
        reshaped = True

    if ndims == 2:
        u, s, v = torch.svd(grad, some=True)
        u = u[:, :svd_rank]
        s = s[:svd_rank]
        v = v[:, :svd_rank]
        return {'u': u, 's': s, 'v': v, 'encode': True, 'orig_size': orig_size,
                'reshaped': reshaped}
    return {'grad': grad, 'encode': False}


def decode(encode_output):
    if not encode_output.get('encode', False):
        return encode_output['grad']
    u, s, v = (encode_output[key] for key in ['u', 's', 'v'])
    grad_approx = u @ torch.diag(s) @ v.t()
    if encode_output.get('reshaped', False):
        grad_approx = grad_approx.view(encode_output['orig_size'])
    return grad_approx
