import torch
# TODO:
#   * see how gradient computed; is it passing the entire gradient? Can we take
#     the SVD of the gradient with each example? See how Variable.register_hook
#     is called
#   * time / profile


storage = {}


def encode(name=None):
    assert name is not None, "name cannot be none"

    def hook(grad, verbose=False):
        if verbose:
            print("set keys =", storage.keys())
        if len(grad.size()) == 2:
            (u, s, v) = torch.svd(grad.data)
            storage[name] = {'svd': (u, s, v), 'grad': grad.data}
        else:
            storage[name] = grad
    return hook


def decode(name, verbose=False):
    """
    Returns gradient as torch Tensor
    """
    if verbose:
        print("get keys =", storage.keys())
    if not isinstance(storage[name], dict):
        grad = storage[name]
        return grad.data

    (u, s, v) = storage[name]['svd']
    grad = storage[name]['grad']
    grad_approx = u @ torch.diag(s) @ v.t()

    rel_error = torch.norm(grad_approx - grad) / torch.norm(grad)
    return grad_approx
