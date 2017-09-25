import numpy as np
import torch
# TODO:
# *
#   * time / profile


storage = {}

verbose = True
comm_bytes = 0


def _get_size(d, verbose=True):
    print(d)
    sizes = [np.prod(v['size']) for k, v in d.items()]
    numel = sum(sizes)
    n_bytes = numel * 4
    if verbose:
        print("storage takes up {mb}MB".format(mb=n_bytes / 1024**2))
    return n_bytes


def _resize_tensor(x):
    """
    x.shape > 2

    If x.shape = (a, b, *c), assumed that each one of (a, b) pairs has relevant information in c.
    """
    size = x.size()
    #  print(tuple(size))
    if all([s == 1 for s in size[2:]]):
        # if (a, b, 1, 1)
        return x.view(size[0], size[1])
    # each of (a, b) has related features
    x = x.view(size[0], size[1], -1)
    # stack those related features into a tall matrix
    return x.view(-1, size[0]*size[1])


def encode(name=None):
    assert name is not None, "name cannot be none"

    def hook(grad, verbose=False):
        if verbose:
            print("set keys =", storage.keys())

        storage[name] = storage.get(name, {})
        grad = grad.data
        storage[name]['size'] = grad.size()
        size = grad.size()

        if len(grad.size()) > 2:
            grad = _resize_tensor(grad)
            storage[name]['reshaped'] = True

        # relative error when decoding names with 'shortcut' in name is >=30
        take_svd = len(grad.size()) == 2 and 'shortcut' not in name.lower()
        if storage[name].get('initialize', True):
            storage[name]['initialize'] = False
            storage[name]['grad'] = grad

            if take_svd:
                storage[name]['encode'] = True
                (u, s, v) = torch.svd(grad)
                storage[name]['svd'] = {'u': u, 's': s, 'v': v}
        else:
            storage[name]['grad'] += grad
            if take_svd:
                (u, s, v) = torch.svd(grad)
                for key, value in {'u': u, 's': s, 'v': v}.items():
                    storage[name]['svd'][key] += value

    return hook


def decode(name, verbose=False):
    """
    Returns gradient as torch Tensor
    """
    storage[name]['initialize'] = True

    if verbose:
        print("get keys =", storage.keys())
    if not storage[name].get('encode', False):
        return storage[name]['grad']

    u, s, v = [storage[name]['svd'][k] for k in ['u', 's', 'v']]
    grad = storage[name]['grad']
    grad_approx = u @ torch.diag(s) @ v.t()

    if storage[name].get('reshaped', False):
        grad_approx = grad_approx.view(storage[name]['size'])

    rel_error = torch.norm(grad_approx - grad) / torch.norm(grad)
    del storage[name]['svd']
    del storage[name]['grad']
    return grad_approx
