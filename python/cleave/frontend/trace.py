"""``cleave.trace(fn, shapes, dtype)``: run an annotated function on proxies -> ``Program``."""

from __future__ import annotations

import inspect

from ..core.errors import SpecError
from .program import Program
from .tensor import Dim, ScalarSpec, Tensor, TensorSpec, as_dtype, as_splits, f16


def signature_dims(fn) -> dict[str, Dim]:
    """Named dims appearing in ``fn``'s ``cl.Tensor[...]`` annotations, in order."""
    out = {}
    for p in inspect.signature(fn, eval_str=True).parameters.values():
        if isinstance(p.annotation, TensorSpec):
            for d in p.annotation.dims:
                if isinstance(d, Dim):
                    out.setdefault(d.name, d)
    return out


def trace(fn, shapes: dict | None = None, dtype=None, *, split=None, name=None):
    """-> ``Program``, or a ``PagedProgram`` for a function with ``cl.Scalar`` / optional inputs."""
    from .jit import JitFunction
    from .paged import is_paged, trace_paged

    if is_paged(fn):
        if shapes:
            raise SpecError(f"{getattr(fn, '__name__', fn)}: paged dims are dynamic; drop shapes=")
        return trace_paged(fn, dtype, name=name, split=split)
    if isinstance(fn, JitFunction):
        dtype = dtype or fn.dtype
        split = fn.split if split is None else split
        name = name or fn.__name__
        fn = fn.fn
    shapes = dict(shapes or {})
    known = signature_dims(fn)
    unknown = set(shapes) - set(known)
    if unknown:
        raise SpecError(f"{fn.__name__}: shapes {sorted(unknown)} are not dims of the signature "
                        f"(dims: {list(known)})")
    prog = Program(name or fn.__name__, as_dtype(dtype) or f16)
    prog.split = as_splits(split)
    args = []
    for p in inspect.signature(fn, eval_str=True).parameters.values():
        spec = p.annotation
        if isinstance(spec, TensorSpec):
            ds = []
            for d in spec.dims:
                if isinstance(d, Dim):
                    d = d.bind(shapes[d.name]) if d.name in shapes else d
                    if d.size is None and d.hint is None:
                        raise SpecError(f"{fn.__name__}: dim {d.name!r} of {p.name!r} unbound; "
                                        f"pass shapes={{{d.name!r}: ...}}")
                ds.append(d)
            args.append(prog.input(p.name, ds, spec.dtype))
        elif isinstance(spec, ScalarSpec):
            raise NotImplementedError(f"{p.name}: runtime cl.Scalar arguments are not in the tile IR yet")
        elif p.default is not inspect.Parameter.empty:
            args.append(p.default)
        else:
            raise SpecError(f"{fn.__name__}: annotate {p.name!r} with cl.Tensor[...] or give it a default")
    out = fn(*args)
    if isinstance(out, dict):
        items = list(out.items())
    else:
        outs = out if isinstance(out, (tuple, list)) else (out,)
        items = [(None, o) for o in outs]
    for n, o in items:
        if not isinstance(o, Tensor):
            raise SpecError(f"{fn.__name__} must return cl.Tensor(s), got {type(o).__name__}")
        prog.output(o, n)
    return prog
