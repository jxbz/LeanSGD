import torch


def rand(p=0.5):
    return 1 - torch.round(torch.rand() + 0.5 - p)


def prob(a, s, l):
    return a*s - l


def encode(v, s=4):
    """
    Parameters
    ==========
    v : torch.Tensor
        Array of the gradient to encode

    Returns
    =======
    coding : dict
        The dictionary that encodes the gradient

    Notes
    =====
    Q_s(v) = ||v||_2 sign(v) rand(v, s)

    * where rand(v, s) are independent random variables. That is, let
      0 <= l < s such that | v_i | / | |v | |_2 in [l / s, (l + 1) / s]. It's
      the quantization level of v_i.
    * where rand(v, s) = l / s wp 1 - p(| v_i | / | |v||_2, s), (l+1)/s o/w
    * where p(a, s) = a * s - l


    from section 3.2 of QSGD paper by Alistar et. al.
    """
    norm = torch.norm(v)
    signs = []
    rand_vars = []
    for vi in v.view(-1):
        abs_vi = torch.abs(vi)
        l = torch.floor(abs_vi * s / norm)
        r = rand(p=1 - prob(abs_vi / norm, s, l))
        signs += [torch.sign(vi)]
        rand_vars += [(r, l)]
    return {'rand_vars': rand_vars, 'signs': signs, 'shape': v.size(),
            'norm': norm, 's': s}


def decode(out):
    v = torch.zeros(out['shape'])
    flat = v.view(-1)
    norm = out['norm']
    s = out['s']
    for i, (sign, (r, l)) in enumerate(zip(out['signs'], out['rand_vars'])):
        r = l/s if r == 1 else (l + 1)/s
        flat[i] = sign * r * norm
    return v
