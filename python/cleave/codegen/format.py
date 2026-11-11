import black

IMPORT_CODE = """from math import gcd
from typing import Optional
import tilelang as tl
import tilelang.language as T
import tvm
"""

UTIL_CODE = """VEC_WIDTH = 4
DEBUG = False
def _S(A: tvm.tirx.stmt.BufferRegion, new_offset_factor: Optional[int]=None, squeeze_dim=None): # convert a sliced bufferregion to a tensor
    ranges = A.region
    shape = [r.extent for r in ranges]
    strides = A.buffer.strides
    if len(strides) == 0: # not specified
        # reversed cumprod on shapes. e.g., shape=[4, 2], strides=[2, 1]
        strides = A.buffer.shape[1:] + [1]
        for i in reversed(range(len(strides) - 1)):
            strides[i] *= strides[i + 1]
    offset = sum([r.min * s for s, r in zip(strides, ranges)])
    # TODO: infer buffer_type from original tensor
    # elem_offset: the starting offset of the first element.
    # offset_factor: a factor that divides elem_offset (though not sure where it's used in TVM?)
    if new_offset_factor is not None:
        offset_factor = gcd(A.buffer.offset_factor, new_offset_factor)
    else:
        offset_factor = A.buffer.offset_factor

    if squeeze_dim is not None:
        for i in squeeze_dim:
            assert shape[i] == 1, f"Can only squeeze dimension of size 1, but got shape {shape} and squeeze_dim {squeeze_dim}"
        shape = [s for i, s in enumerate(shape) if i not in squeeze_dim]
        strides = [s for i, s in enumerate(strides) if i not in squeeze_dim]
        if DEBUG:
            print(f"Squeezed shape: {shape}, strides: {strides}")

    return T.Buffer(shape,
                    dtype=A.buffer.dtype,
                    strides=strides,
                    elem_offset=A.buffer.elem_offset + offset, offset_factor=offset_factor,
                    buffer_type="default", align=A.buffer.data_alignment, data=A.buffer.data,
                    scope=A.buffer.scope())
"""


def format_source(debug: bool, *sections: str) -> str:
    util_code = UTIL_CODE.replace("DEBUG = False", f"DEBUG = {debug}")
    mode = black.Mode(target_versions={black.TargetVersion.PY311}, line_length=100)
    return black.format_str(IMPORT_CODE + util_code + "".join(sections), mode=mode)
