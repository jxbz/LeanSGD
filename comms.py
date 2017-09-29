import numpy as np
import torch


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
    return x.view(size[0]*size[1], -1)


def encode(name=None):
    assert name is not None, "name cannot be none"

    def hook(grad, verbose=False):
        if verbose:
            print("set keys =", storage.keys())

        storage[name] = storage.get(name, {})
        grad = grad.data
        #  storage[name]['original_grad'] = grad
        storage[name]['size'] = grad.size()

        resize = len(grad.size()) > 2
        if resize:
            grad = _resize_tensor(grad)
            storage[name]['reshaped'] = True
        #  print(resize, grad.size(), name)

        # relative error when decoding names with 'shortcut' in name is >=30
        take_svd = len(grad.size()) == 2
        if storage[name].get('initialize', True):
            storage[name]['initialize'] = False

            if take_svd:
                storage[name]['encode'] = True
                #  storage[name]['grad'] = grad
                (u, s, v) = torch.svd(grad, some=True)
                del grad
                r = s.size()[0] // 2
                #  r = 3
                u = u[:, :r]
                s = s[:r]
                v = v[:, :r]
                storage[name]['svd'] = {'u': u, 's': s, 'v': v}
            else:
                storage[name]['grad'] = grad
        else:
            if take_svd:
                (u, s, v) = torch.svd(grad)
                for key, value in {'u': u, 's': s, 'v': v}.items():
                    storage[name]['svd'][key] += value
            else:
                storage[name]['grad'] += grad

    return hook


def decode(name, verbose=False):
    """
    Returns gradient as torch Tensor
    """
    if name not in storage.keys():
        print(name)
        #  return None
    storage[name]['initialize'] = True

    if verbose:
        print("get keys =", storage.keys())
    if not storage[name].get('encode', False):
        grad = storage[name]['grad']
        return grad

    u, s, v = [storage[name]['svd'][k] for k in ['u', 's', 'v']]
    #  grad = storage[name]['grad']
    #  orig_grad = storage[name]['original_grad']
    grad_approx = u @ torch.diag(s) @ v.t()

    if storage[name].get('reshaped', False):
        grad_approx = grad_approx.view(storage[name]['size'])
        #  print(storage[name]['size'], grad_approx.size())

    #  grad = storage[name]['grad']
    #  orig_grad = storage[name]['original_grad']
    #  rel_error1 = torch.norm(grad_approx - grad) / torch.norm(grad)
    #  rel_error2 = torch.norm(grad_approx - orig_grad) / torch.norm(orig_grad)
    #  print(name, rel_error1, rel_error2)  # prints on order of 1e-7
    #  convShortcut print rel_error2 == small, rel_error1 == big. this is a
    #  size difference; (a, b, 1, 1) vs (a, b)
    del storage[name]['svd']
    return grad_approx
