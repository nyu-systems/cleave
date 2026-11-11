"""``@cleave.jit``: an annotated function that traces to a ``Program`` and compiles lazily."""

from __future__ import annotations

import functools
import inspect

from .tensor import Tensor, TensorSpec, as_dtype
from .trace import signature_dims, trace


class JitFunction:
    def __init__(self, fn, *, split=None, name=None, dtype=None, **options):
        functools.update_wrapper(self, fn)
        self.fn, self.split, self.dtype, self.options = fn, split, as_dtype(dtype), options
        if name:
            self.__name__ = name
        self.dim_names = tuple(signature_dims(fn))
        self._kernels: dict = {}

    @functools.cached_property
    def paged(self) -> bool:
        from .paged import is_paged

        return is_paged(self.fn)

    def trace(self, dtype=None, **shapes):
        return trace(self, shapes, dtype)

    def compile(self, **kwargs):
        """``f.compile(M=128, K=4096, preset="fast")``: dim sizes + ``cleave.compile`` options."""
        from cleave.pipeline import compile as _compile

        shapes = {k: kwargs.pop(k) for k in list(kwargs) if k in self.dim_names}
        dtype = kwargs.pop("dtype", None)
        opts = {**self.options, **kwargs}
        if self.split is not None:
            opts.setdefault("split", self.split)
        if self.paged:
            if shapes:
                raise TypeError(f"{self.__name__}: paged dims are dynamic; drop {sorted(shapes)}")
            kernel = self._kernels[None] = _compile(self, dtype=dtype, **opts)
            return kernel
        prog = self.trace(dtype, **shapes)
        kernel = _compile(prog, **opts)
        self._kernels[_key(prog.sizes, [s.dtype.name for s in prog.inputs])] = kernel
        return kernel

    def _tensors(self, args, kwargs) -> list:
        """The torch tensors bound to the ``cl.Tensor`` parameters, in signature order."""
        sig = inspect.signature(self.fn, eval_str=True)
        bound = sig.bind(*args, **kwargs).arguments
        return [(p, bound[p.name]) for p in sig.parameters.values()
                if isinstance(p.annotation, TensorSpec) and p.name in bound]

    def bind_shapes(self, *args, **kwargs) -> dict:
        """Dim sizes implied by concrete tensors."""
        sizes = {}
        for p, x in self._tensors(args, kwargs):
            for d, n in zip(p.annotation.dims, x.shape):
                if not isinstance(d, int) and sizes.setdefault(d.name, int(n)) != int(n):
                    raise ValueError(f"dim {d.name!r}: inconsistent sizes {sizes[d.name]} vs {n}")
        return sizes

    def __call__(self, *args, **kwargs):
        """Inside a trace: inline. On torch tensors: the kernel ``.compile`` made for these
        shapes and dtypes, else the function itself, eagerly (the reference)."""
        if any(isinstance(a, Tensor) for a in (*args, *kwargs.values())):
            return self.fn(*args, **kwargs)
        if self.paged:
            kernel = self._kernels.get(None)
            return kernel(*args, **kwargs) if kernel is not None else self.fn(*args, **kwargs)
        tensors = [x for _, x in self._tensors(args, kwargs)]
        dtypes = [str(x.dtype).removeprefix("torch.") for x in tensors]
        kernel = self._kernels.get(_key(self.bind_shapes(*args, **kwargs), dtypes))
        return kernel(*tensors) if kernel is not None else self.fn(*args, **kwargs)

    def __repr__(self):
        return f"<cleave.jit {self.__name__} dims={list(self.dim_names)}>"


def _key(sizes: dict, dtypes) -> tuple:
    return tuple(sorted(sizes.items())), tuple(dtypes)


def jit(fn=None, **kwargs):
    """``@cleave.jit`` or ``@cleave.jit(split=cl.Split("K"), name=..., dtype=cl.bf16)``."""
    if fn is None:
        return lambda f: JitFunction(f, **kwargs)
    return JitFunction(fn, **kwargs)
