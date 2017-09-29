import torch.optim as optim
import math
import comms

class LowCommSGD(optim.SGD):
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = comms.decode(p.grad)

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = d_p.clone()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss

class LowCommASGD(optim.ASGD):
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                #  grad = p.grad.data
                grad = comms.decode(group['name'])
                if grad is None:
                    continue
                state = self.state[p]
                #  print([type(x) for x in [grad, p.data]])
                #  print([x.size() for x in [grad, p.data]])

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
                                math.pow((1 + group['lambd'] * group['lr'] *
                                          state['step']), group['alpha']))
                state['mu'] = 1 / max(1, state['step'] - group['t0'])

        return loss
