import torch
import numpy as np
import numpy.linalg as LA
from toolz import reduce
import codings


def test_qsgd_and_terngrad():
    n = 50
    x = np.random.rand(n)
    code = codings.QSGD()
    codes = [codings.QSGD(scheme=scheme) for scheme in ['terngrad', 'qsgd']]
    for code in codes:
        repeats = int(10e3)
        codes = [code.encode(x, scheme=code.scheme) for _ in range(repeats)]
        code.codes = codes

        approxs = [code.decode(x) for x in codes]

        data = map(lambda arg: {'y': arg[1], 'norm(y)**2': LA.norm(arg[1])**2,
                                'len(signs)': len(arg[0]['signs'])},
                   zip(codes, approxs))
        sums = reduce(lambda x, y: {k: x[k] + y[k] for k in x}, data)
        avg = {k: v / len(codes) for k, v in sums.items()}
        assert avg['norm(y)**2'] <= np.sqrt(n) * LA.norm(x)**2
        if code.scheme == 'qsgd':
            assert avg['len(signs)'] <= np.sqrt(n)
        rel_error = LA.norm(avg['y'] - x) / LA.norm(x)
        print(code.scheme, rel_error)
        assert rel_error < 0.25

def test_svd(n=64*32, r=9):
    x = np.random.rand(n, r)
    u, s, vT = np.linalg.svd(x, full_matrices=False)
    s = np.exp(-1 * np.linspace(0, 5, num=len(s)))
    x = u @ np.diag(s) @ vT

    code = codings.SVD(rank=0, compress=True, random_sample=True)
    repeats = int(10e3)

    codes = [code.encode(x) for _ in range(repeats)]
    approxs = [code.decode(x) for x in codes]
    norms = [LA.norm(g, ord='fro')**2 for g in approxs]
    est = {'mean': np.mean(approxs, axis=0), 'norm**2': np.mean(norms)}
    rel_error = LA.norm(est['mean'] - x) / LA.norm(x)
    print('svd', rel_error)
    assert rel_error < 0.05
    assert est['norm**2'] < np.sqrt(r) * LA.norm(x)**2


def test_qsvd(n=64*32, r=9):
    n = 50
    x = np.random.rand(n, n//10)

    for scheme in ['terngrad', 'qsgd']:
        qsvd = codings.QSVD(scheme=scheme)
        repeats = int(1000)
        codes = [qsvd.encode(x, scheme=scheme) for _ in range(repeats)]
        approxs = [qsvd.decode(code, codes=codes) for code in codes]

        est = np.mean(approxs, axis=0)
        rel_error = LA.norm(est - x) / LA.norm(x)
        print('qsvd', scheme, rel_error)
        assert rel_error < 0.25

if __name__ == "__main__":
    test_qsvd()
    test_svd()
    test_qsgd_and_terngrad()
