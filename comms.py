import torch
import numpy as np
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

def encode(name=None):
    assert name is not None, "name cannot be none"

    def hook(grad, verbose=False):
        if verbose:
            print("set keys =", storage.keys())

        storage[name] = storage.get(name, {})
        storage[name]['size'] = grad.data.size()

        if storage[name].get('zero', True):
            storage[name]['zero'] = False
            if len(grad.size()) == 2:
                print(f'Encoding SVD of {name}')
                storage[name]['encode'] = True
                (u, s, v) = torch.svd(grad.data)
                storage[name]['svd'] = {'u':u, 's':s, 'v':v}
                storage[name]['grad'] = grad.data
            else:
                storage[name]['grad'] = grad.data
        else:
            if len(grad.size()) == 2:
                print(f'Adding to encoding SVD of {name}')
                (u, s, v) = torch.svd(grad.data)

                for key, value in {'u':u, 's':s, 'v':v}.items():
                    storage[name]['svd'][key] += value
                storage[name]['grad'] += grad.data
            else:
                storage[name]['grad'] += grad.data

        #  storage[name]['grad'] += grad.data

    return hook


def decode(name, verbose=False):
    """
    Returns gradient as torch Tensor
    """
    #  _get_size(storage, verbose=verbose)
    storage[name]['zero'] = True

    if verbose:
        print("get keys =", storage.keys())
    if not storage[name].get('encode', False):
        return storage[name]['grad']

    print(f'Decoding SVD of {name}')
    u, s, v = [storage[name]['svd'][k] for k in ['u', 's', 'v']]
    del storage[name]['svd']
    grad = storage[name]['grad']
    grad_approx = u @ torch.diag(s) @ v.t()

    rel_error = torch.norm(grad_approx - grad) / torch.norm(grad)
    return grad_approx
