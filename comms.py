import sys
from pprint import pprint
import numpy as np
import torch


storage = {}

verbose = False
comm_bytes = 0


def _bytes_of(obj):
    # BUG: for 2D arrays doesn't return the number of bytes
    # that is, when sizes printed, only 1D sizes printed
    if isinstance(obj, torch.autograd.Variable):
        print('autograd variable')
        return _bytes_of(obj.grad) + obj.element_size()*obj.numel()
    cuda_tensor = getattr(obj, 'cuda', False)
    #  print('type(obj) =', type(obj), 'cuda_tensor =', cuda_tensor)
    if isinstance(obj, torch.Tensor) or cuda_tensor:
        # t_size is a lower bound; only the number of elements
        t_size = obj.element_size() * obj.numel()
        # this is a better bound; it has all the overhead too
        #  py_size = sys.getsizeof(obj)
        return t_size

    if isinstance(obj, dict):
        # I'm not sure why I'm not trusting this function with dict. But it's
        # called infrequently enough that I'm okay with it.
        return sum(_bytes_of(v) for k, v in obj.items())
    return 0  # sys.getsizeof(obj)  # only counting tensors as stores

def _size_of(obj):
    try:
        return obj.size()
    except:
        pass
    #  if isinstance(obj, torch.autograd.Variable):
        #  return obj.size()
    #  cuda_tensor = getattr(obj, 'cuda', False)
    #  if isinstance(obj, torch.Tensor) or cuda_tensor:
        #  return obj.size()

    if isinstance(obj, dict):
        return {k: _size_of(v) for k, v in obj.items()}
    #  if type(obj) in [bool, int, str]:
        #  return obj

def _clean(d):
    ret = {}
    for k, v in d.items():
        if v is None:
            continue
        if isinstance(v, dict):
            ret[k] = _clean(v)
        else:
            ret[k] = v
    return ret

def _get_size(verbose=True):
    n_bytes = sum([_bytes_of(v) for k, v in storage.items()])
    sizes = {k: _size_of(v) for k, v in storage.items()}
    #  pprint(_clean(sizes))
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


def encode(name=None, compress=True, rank=None):
    assert name is not None, "name cannot be none"
    assert rank is not None, "name cannot be none"
    assert type(rank) == int, "rank must be an int"

    def hook(grad, verbose=False):
        if verbose:
            print("set keys =", storage.keys())

        storage[name] = storage.get(name, {})
        grad = grad.data
        if not compress:
            storage[name]['grad'] = grad
            storage[name]['encode'] = False
            return

        #  storage[name]['original_grad'] = grad
        storage[name]['size'] = grad.size()

        resize = len(grad.size()) > 2
        if resize:
            grad = _resize_tensor(grad)
            storage[name]['reshaped'] = True

        take_svd = (len(grad.size()) == 2) and name[:2] != 'fc'
        #  print(f"{name}: take_svd = {take_svd}")
        if storage[name].get('initialize', True):
            storage[name]['initialize'] = False

            if take_svd:
                storage[name]['encode'] = True
                #  storage[name]['grad'] = grad
                (u, s, v) = torch.svd(grad, some=True)
                del grad
                u = u[:, :rank]
                s = s[:rank]
                v = v[:, :rank]
                storage[name]['svd'] = {'u': u, 's': s, 'v': v}
            else:
                storage[name]['grad'] = grad
        else:
            if take_svd:
                storage[name]['encode'] = True
                (u, s, v) = torch.svd(grad)
                for key, value in {'u': u, 's': s, 'v': v}.items():
                    storage[name]['svd'][key] += value
            else:
                storage[name]['grad'] += grad

    return hook


def decode(name, verbose=False, compress=True):
    """
    Returns gradient as torch Tensor
    """
    meta = {'n_bytes': _get_size()}
    if not compress:
        return storage[name]['grad'], meta
    if name not in storage.keys():
        print(name)
        #  return None
    storage[name]['initialize'] = True

    if verbose:
        print("get keys =", storage.keys())
    if not storage[name].get('encode', False):
        grad = storage[name]['grad']
        return grad, meta

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
    #  del storage[name]['svd']
    return grad_approx, meta
