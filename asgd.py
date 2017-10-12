import math
import time
import torch

from torch.optim import Optimizer


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

def encode(grad):
    ret = {}
    resize = len(grad.size()) > 2
    ret['resize'] = resize
    if resize:
        ret['size'] = grad.size()
        grad = _resize_tensor(grad)
    if len(grad.size()) == 2:
        ret['svd'] = torch.svd(grad, some=True)
        ret['encode'] = True
    else:
        ret['grad'] = grad
        ret['encode'] = False
    return ret


def decode(ret):
    if ret['encode']:
        grad = ret['svd'][0] @ torch.diag(ret['svd'][1]) @ ret['svd'][2].t()
    else:
        grad = ret['grad']
    if ret['resize']:
        grad = grad.view(ret['size'])
    return grad


class ASGD(Optimizer):
    """Implements Averaged Stochastic Gradient Descent.

    It has been proposed in `Acceleration of stochastic approximation by
    averaging`_.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lambd (float, optional): decay term (default: 1e-4)
        alpha (float, optional): power for eta update (default: 0.75)
        t0 (float, optional): point at which to start averaging (default: 1e6)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)

    .. _Acceleration of stochastic approximation by averaging:
        http://dl.acm.org/citation.cfm?id=131098
    """

    def __init__(self, params, lr=1e-2, lambd=1e-4, alpha=0.75, t0=1e6, weight_decay=0):
        defaults = dict(lr=lr, lambd=lambd, alpha=alpha, t0=t0,
                        weight_decay=weight_decay)
        super(ASGD, self).__init__(params, defaults)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        data = {'encode_time': 0, 'decode_time': 0, 'step_time': 0}
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                tmp1 = p.grad.data.cpu()
                #  grad = p.grad.cpu()

                start = time.time()
                tmp2 = encode(tmp1)
                data['encode_time'] += time.time() - start

                start = time.time()
                tmp3 = decode(tmp2)
                data['decode_time'] += time.time() - start

                grad = p.grad.data

                start = time.time()
                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['eta'] = group['lr']
                    state['mu'] = 1
                    state['ax'] = grad.new().resize_as_(grad).zero_()

                state['step'] += 1

                if group['weight_decay'] != 0:
                    grad = grad.add(group['weight_decay'], p.data)

                # decay term
                p.data.mul_(1 - group['lambd'] * state['eta'])

                # update parameter
                p.data.add_(-state['eta'], grad)

                # averaging
                if state['mu'] != 1:
                    state['ax'].add_(p.data.sub(state['ax']).mul(state['mu']))
                else:
                    state['ax'].copy_(p.data)

                # update eta and mu
                state['eta'] = (group['lr'] /
                                math.pow((1 + group['lambd'] * group['lr'] * state['step']), group['alpha']))
                state['mu'] = 1 / max(1, state['step'] - group['t0'])
                data['step_time'] += time.time() - start


        return loss, data
