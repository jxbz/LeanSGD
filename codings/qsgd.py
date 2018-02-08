from functools import reduce
import numpy as np
import numpy.linalg as LA
from scipy import stats
import torch
import time
from .coding import Coding


class QSGD(Coding):

    def __init__(self, *args, scheme='qsgd', **kwargs):
        self.scheme = scheme
        super().__init__(*args, **kwargs)

    def encode(self, v, **kwargs):
        if isinstance(v, (torch.Tensor, torch.cuda.FloatTensor)):
            w = v.cpu().numpy().flat[:]
        elif isinstance(v, np.ndarray):
            w = v.flat[:]
        else:
            raise ValueError("Object passed to encode not ndarray or torch.Tensor")

        if self.scheme == 'qsgd':
            norm = LA.norm(v)
        elif self.scheme == 'terngrad':
            norm = np.linalg.norm(w, ord=np.inf)
            limit = grad_clip_limit(w, clip_factor=2.5)
            w = np.clip(w, -limit, limit)

        signs = np.sign(w).astype('int')
        probs = np.abs(w) / norm
        #  print(probs.shape, probs.mean(), probs.min(), probs.max())
        mask = stats.bernoulli.rvs(probs).astype('bool')
        idx = np.arange(len(w))

        selected = idx[mask].astype('uint32')
        signs = signs[mask].astype('int8')
        signs = ((signs + 1) / 2).astype('bool')

        code = {'signs': signs, 'shape': v.shape, 'selected': selected,
                'norm': norm}

        if kwargs.pop('timings', False):
            data = {}
            return code, data
        return code

    def decode(self, code, cuda=False, codes=[], **kwargs):
        if self.scheme == 'terngrad' and len(codes) > 0:
            code['norm'] = self._get_max_norm(codes)

        v = np.zeros(code['shape'])
        signs = np.array(code['signs'], dtype='int8')
        signs = signs*2 - 1
        selected = np.array(code['selected'], dtype='int16')
        #  selected = torch.LongTensor(selected)

        if len(selected) > 0:
            v.flat[selected] = code['norm'] * signs
        v = torch.Tensor(v)
        if cuda:
            v = v.cuda()
        return v

    def _get_max_norm(self, codes):
        scalars = [code['norm'] for code in codes]
        return max(scalars)


def grad_clip_limit(grad, clip_factor=2.5):
    """ Get the scalers."""
    if clip_factor > 1.0e-5:
        return clip_factor * np.std(grad.flat[:])
    return np.max(np.abs(grad.flat[:]))


