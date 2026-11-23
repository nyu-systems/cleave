"""Single-reduction-group attention-backward programs ("D given": L [H,SQ,1] fp32, D [H,SQ,1]).

P = exp(q k^T c - L), dS = P * (dO v^T - D); dV = P^T dO, dQ = c dS k, dK = c dS^T q.
"""

import cleave.lang as cl
from cleave.frontend import trace

C = 64 ** -0.5
SMALL = dict(H=2, SQ=256, SKV=256, D=64)


def _p(q, k, L):
    return ((q @ k.mT) * C - L).exp()


def dv(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"], dO: cl.Tensor["H", "SQ", "D"],
       L: cl.Tensor["H", "SQ", 1, cl.f32]):
    return _p(q, k, L).mT @ dO


def dq(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"], v: cl.Tensor["H", "SKV", "D"],
       dO: cl.Tensor["H", "SQ", "D"], L: cl.Tensor["H", "SQ", 1, cl.f32], D: cl.Tensor["H", "SQ", 1]):
    return (_p(q, k, L) * (dO @ v.mT - D) * C) @ k


def dk(q: cl.Tensor["H", "SQ", "D"], k: cl.Tensor["H", "SKV", "D"], v: cl.Tensor["H", "SKV", "D"],
       dO: cl.Tensor["H", "SQ", "D"], L: cl.Tensor["H", "SQ", 1, cl.f32], D: cl.Tensor["H", "SQ", 1]):
    return (_p(q, k, L) * (dO @ v.mT - D) * C).mT @ q


PROGRAMS = {"dv": dv, "dq": dq, "dk": dk}


def program(name, shapes=SMALL):
    return trace(PROGRAMS[name], shapes, cl.f16)
