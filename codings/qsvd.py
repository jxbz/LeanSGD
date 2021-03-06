import copy
import numpy as np
import torch
from toolz import reduce, partial

import codings


class QSVD(codings.Coding):
    def __init__(self, scheme='qsgd', rank=0, *args, **kwargs):
        self.scheme = scheme
        self.rank = rank
        if scheme not in ['qsgd', 'terngrad']:
            raise ValueError(f'Illegal value for scheme: {scheme} not in '
                             '["qsgd", "terngrad"]')
        self.qsgd = codings.QSGD(scheme=scheme)
        self.svd = codings.SVD(rank=self.rank)
        super().__init__(self, *args, **kwargs)

    def encode(self, grad, **kwargs):
        svd_code = self.svd.encode(grad, **kwargs)
        vT = self.qsgd.encode(svd_code['vT'], **kwargs)
        u = self.qsgd.encode(svd_code['u'], **kwargs)
        svd_code.update({'u': u, 'vT': vT})
        return svd_code

    def decode(self, code, codes=[], **kwargs):
        u = self.qsgd.decode(code['u'], codes=[code['u']
                                               for code in codes], **kwargs)
        vT = self.qsgd.decode(code['vT'], codes=[code['vT']
                                                 for code in codes], **kwargs)
        new_code = copy.copy(code)
        new_code.update({'u': u, 'vT': vT})
        return self.svd.decode(new_code, **kwargs)
