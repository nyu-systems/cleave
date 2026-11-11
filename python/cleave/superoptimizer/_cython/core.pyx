
from CCore cimport *
from cpython cimport array
import builtins
import ctypes
import array
import numpy as np
from cython.operator cimport dereference as deref, preincrement as inc
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector


class dtype:
    SINT_TYPES = ['int8', 'int16', 'int32', 'int64']
    UINT_TYPES = ['uint8', 'uint16', 'uint32', 'uint64']
    FLOAT_TYPES = ['float16', 'bfloat16', 'float32', 'float64']

    def __init__(self, name):
        self.name = name
        assert name in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FLOAT_TYPES, name

    def is_float16(self):
        return self.name == 'float16'

    def is_bfloat16(self):
        return self.name == 'bfloat16'

    def is_float32(self):
        return self.name == 'float32'

    def is_float64(self):
        return self.name == 'float64'

    def is_int1(self):
        return self.name == 'int1'

    def is_int8(self):
        return self.name == 'int8'

    def is_int16(self):
        return self.name == 'int16'

    def is_int32(self):
        return self.name == 'int32'

    def is_int64(self):
        return self.name == 'int64'

    def is_uint8(self):
        return self.name == 'uint8'

    def is_uint16(self):
        return self.name == 'uint16'

    def is_uint32(self):
        return self.name == 'uint32'

    def is_uint64(self):
        return self.name == 'uint64'

    def __eq__(self, other: dtype):
        if not isinstance(other, dtype):
            return False
        return self.name == other.name

    def __ne__(self, other: dtype):
        return not self.__eq__(other)

    def __hash__(self):
        return hash((self.name, ))

    def __str__(self):
        return self.name

    def is_dtype(type_str):
        return type_str in dtype.SINT_TYPES + dtype.UINT_TYPES + dtype.FLOAT_TYPES

# data types
int8 = dtype('int8')
int16 = dtype('int16')
int32 = dtype('int32')
int64 = dtype('int64')
uint8 = dtype('uint8')
uint16 = dtype('uint16')
uint32 = dtype('uint32')
uint64 = dtype('uint64')
float16 = dtype('float16')
bfloat16 = dtype('bfloat16')
float32 = dtype('float32')
float64 = dtype('float64')

def get_kn_operator_type_string(int op_type):
    if op_type == KN_UNKOWN:
        return "kn_unknown"
    cdef const char* name = op_type_name(<KNOperatorType>op_type)
    if name == NULL:
        return "unknown_op_type" + str(op_type)
    return name.decode("utf-8")


def op_type_table():
    """``{serialized name: enum value}`` for every op in op_types.def."""
    cdef int i
    return {kOpTypes[i].name.decode("utf-8"): int(kOpTypes[i].type) for i in range(kNumOpTypes)}


_pybool = builtins.bool

_DTYPES = {d.name: d for d in (
    int8, int16, int32, int64, uint8, uint16, uint32, uint64,
    float16, bfloat16, float32, float64,
)}

# Bump when the checkpoint format or search semantics change.
_CHECKPOINT_SCHEMA = "cleave-checkpoint-v1"

_BUILD_ID = None


def _build_id():
    """Content hash of this extension (the C++ search is linked in statically),
    so any rebuild with changed C++ or Cython invalidates checkpoint keys."""
    global _BUILD_ID
    if _BUILD_ID is None:
        import hashlib
        h = hashlib.sha256()
        with open(__file__, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        _BUILD_ID = h.hexdigest()[:16]
    return _BUILD_ID


cdef DataType _to_cpp_dtype(object value) except *:
    if value is None:
        return DT_UNKNOWN
    return data_type_from_string(str(value).encode("utf-8"))


cdef object _from_cpp_dtype(DataType value):
    name = data_type_to_string(value).decode("utf-8")
    return _DTYPES.get(name) if name else None


cdef vector[int] _to_dims(object dims) except *:
    if len(dims) > MAX_TENSOR_DIMS:
        raise ValueError(
            f"Input tensors support at most {MAX_TENSOR_DIMS} dims, got {len(dims)}"
        )
    cdef vector[int] cdims = vector[int](len(dims))
    cdef size_t i
    for i in range(len(dims)):
        cdims[i] = dims[i]
    return cdims


cdef class DTensor:
    cdef CppDTensor* c_ptr
    # Keeps the graph that owns *c_ptr alive.
    cdef object _owner

    cdef inline _set_tensor(self, tensor):
        cdef unsigned long long ptr
        if tensor is None:
            self.c_ptr = <CppDTensor*>(NULL)
        else:
            ptr = ctypes.cast(tensor, ctypes.c_void_p).value
            self.c_ptr = <CppDTensor*>(ptr)

    property guid:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return self.c_ptr.guid

    property tensor:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return ctypes.cast(<unsigned long long>self.c_ptr, ctypes.c_void_p)

        def __set__(self, value):
            self._set_tensor(value)
            self._owner = None

    property num_dims:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return self.c_ptr.num_dims

    property dtype:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            return _from_cpp_dtype(self.c_ptr.dtype)
        def __set__(self, value):
            if self.c_ptr == NULL:
                raise ValueError("tensor is None")
            self.c_ptr.dtype = _to_cpp_dtype(value)

    def __cinit__(self, tensor):
        self._owner = None
        self._set_tensor(tensor)

    def dim(self, int idx):
        if self.c_ptr == NULL:
            raise ValueError("tensor is None")
        if idx < 0 or idx >= self.c_ptr.num_dims:
            raise IndexError(f"dim index {idx} out of range for {self.c_ptr.num_dims}-d tensor")
        return self.c_ptr.dim[idx]


cdef CppDTensor* _cptr(DTensor t) except NULL:
    # Typed DTensor args accept None; dereferencing it (or a null tensor) would segfault.
    if t is None or t.c_ptr == NULL:
        raise ValueError("expected a non-null DTensor")
    return t.c_ptr


cdef DTensor _make_dtensor(CppDTensor* ptr, object owner):
    cdef DTensor result = DTensor.__new__(DTensor, None)
    result.c_ptr = ptr
    result._owner = owner
    return result


cdef class CyKNOperator:
    cdef CppKNOperator* c_ptr
    cdef object _owner

    cdef inline _set_operator(self, op):
        cdef unsigned long long ptr
        if op is None:
            self.c_ptr = <CppKNOperator*>(NULL)
        else:
            ptr = ctypes.cast(op, ctypes.c_void_p).value
            self.c_ptr = <CppKNOperator*>(ptr)

    def get_input_dtensors(self):
        cdef size_t i
        if self.c_ptr == NULL:
            raise ValueError("operator is None")
        return [
            _make_dtensor(&self.c_ptr.input_tensors[i], self._owner)
            for i in range(self.c_ptr.input_tensors.size())
        ]

    def get_output_dtensors(self):
        cdef size_t i
        if self.c_ptr == NULL:
            raise ValueError("operator is None")
        return [
            _make_dtensor(&self.c_ptr.output_tensors[i], self._owner)
            for i in range(self.c_ptr.output_tensors.size())
        ]

    property op_type:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return get_kn_operator_type_string(int(self.c_ptr.op_type))

    def __cinit__(self, op, owner=None):
        # owner: object keeping the graph that holds *op alive.
        self._owner = owner
        self._set_operator(op)


cdef class CyKNGraph:
    cdef CppKNGraph *p_kgraph
    cdef bint _owns
    # User-built graphs report every tensor's dtype; searched graphs only
    # report input dtypes so conversion picks accum/out dtypes for the rest.
    cdef bint _all_dtypes
    cdef object _owner

    def __cinit__(self, graph = None, owner = None):
        cdef unsigned long long ptr
        # owner: object keeping a wrapped raw graph alive (unused when owning).
        self._owner = owner
        self._all_dtypes = True
        if graph is None:
            self.p_kgraph = new CppKNGraph()
            self._owns = True
        else:
            # Wrapping a raw pointer does not take ownership.
            ptr = ctypes.cast(graph, ctypes.c_void_p).value
            self.p_kgraph = <CppKNGraph*>(ptr)
            self._owns = False

    def __dealloc__(self):
        if self._owns and self.p_kgraph != NULL:
            del self.p_kgraph
        self.p_kgraph = NULL

    cdef inline DTensor _wrap(self, CppDTensor* ptr):
        return _make_dtensor(ptr, self)

    def new_input(self, tuple dims, dtype : dtype = float16):
        cdef vector[int] cdims = _to_dims(dims)
        cdef vector[bool] empty
        return self._wrap(self.p_kgraph.new_input_ptr(cdims, empty, empty, _to_cpp_dtype(dtype)))

    def mark_output(self, DTensor A):
        self.p_kgraph.mark_output(_cptr(A)[0])

    def matmul(self, DTensor A, DTensor B, bint trans_A = False, bint trans_B = False):
        return self._wrap(self.p_kgraph.matmul(_cptr(A), _cptr(B), trans_A, trans_B))

    def reduce_sum(self, DTensor input, int dim):
        return self._wrap(self.p_kgraph.reduce_sum(_cptr(input), dim, 1))

    def rms_norm(self, DTensor input, tuple normalized_shape):
        cdef vector[int] cshape = vector[int](len(normalized_shape))
        cdef size_t i
        for i in range(len(normalized_shape)):
            cshape[i] = normalized_shape[i]
        return self._wrap(self.p_kgraph.rms_norm(_cptr(input), cshape))

    def exp(self, DTensor input):
        return self._wrap(self.p_kgraph.exp(_cptr(input)))

    def silu(self, DTensor input):
        return self._wrap(self.p_kgraph.silu(_cptr(input)))

    def gelu(self, DTensor input):
        return self._wrap(self.p_kgraph.gelu(_cptr(input)))

    def relu(self, DTensor input):
        return self._wrap(self.p_kgraph.relu(_cptr(input)))

    def clamp(self, DTensor input, float min_val, float max_val):
        return self._wrap(self.p_kgraph.clamp(_cptr(input), min_val, max_val))

    def add(self, DTensor A, DTensor B):
        return self._wrap(self.p_kgraph.add(_cptr(A), _cptr(B)))

    def mul(self, DTensor A, DTensor B):
        return self._wrap(self.p_kgraph.mul(_cptr(A), _cptr(B)))

    def div(self, DTensor A, DTensor B):
        return self._wrap(self.p_kgraph.div(_cptr(A), _cptr(B)))

    def sub(self, DTensor A, DTensor B):
        return self._wrap(self.p_kgraph.sub(_cptr(A), _cptr(B)))

    def reduce_max(self, DTensor input, int dim):
        return self._wrap(self.p_kgraph.reduce_max(_cptr(input), dim))

    def get_input_dtensors(self):
        cdef size_t i
        cdef CppKNOperator* op
        inputs = []
        for i in range(self.p_kgraph.operators.size()):
            op = self.p_kgraph.operators[i]
            if op.op_type == KN_INPUT_OP:
                inputs.append(self._wrap(&op.output_tensors[0]))
        return inputs

    # visualizer utils

    def _kn_tensor_to_dict(self, DTensor t, input_guids=None):
        info = {
            "num_dims": t.num_dims,
            "dim": [t.dim(i) for i in range(t.num_dims)],
            "guid": t.guid
        }
        if self._all_dtypes or input_guids is None or t.guid in input_guids:
            tensor_dtype = t.dtype
            if tensor_dtype is not None:
                info["dtype"] = str(tensor_dtype)
        return info

    def _get_kn_operator_info(self, CyKNOperator op, input_guids=None):
        cdef CppKNElementBinaryOp* bin_op
        cdef CppKNMatmulOp* matmul_op
        info = {
            "op_type": op.op_type,
            "input_tensors": [self._kn_tensor_to_dict(t, input_guids) for t in op.get_input_dtensors()],
            "output_tensors": [self._kn_tensor_to_dict(t, input_guids) for t in op.get_output_dtensors()],
        }
        if op.op_type == "kn_matmul_op":
            matmul_op = <CppKNMatmulOp*> op.c_ptr
            info["trans_A"] = True if matmul_op.trans_A else False
            info["trans_B"] = True if matmul_op.trans_B else False
        if op.op_type in ("kn_add_op", "kn_div_op", "kn_mul_op"):
            bin_op = <CppKNElementBinaryOp*> op.c_ptr
            if bin_op.has_scalar:
                info["scalar"] = float(bin_op.scalar)
                info["scalar_on_left"] = True if bin_op.scalar_on_left else False
        return info

    def get_graph_structure(self):
        input_guids = {t.guid for t in self.get_input_dtensors()}
        operators = []
        ops = self.p_kgraph.operators
        for i in range(ops.size()):
            op = CyKNOperator(None, self)
            op.c_ptr = ops[i]
            operators.append(self._get_kn_operator_info(op, input_guids))
        return operators


cdef CyKNGraph _take_kn_graph(CppKNGraph* graph):
    cdef CyKNGraph result = CyKNGraph.__new__(CyKNGraph, ctypes.c_void_p(<unsigned long long>graph))
    result._owns = True
    result._all_dtypes = False
    return result


cdef class CyGeneratorConfig:
    cdef GeneratorConfig config

    def __cinit__(self):
        self.config = GeneratorConfig.get_default_config()

    property max_num_kernel_graph_op:
        def __get__(self):
            return self.config.max_num_kernel_graph_op
        def __set__(self, value):
            self.config.max_num_kernel_graph_op = value

    property search_thread:
        def __get__(self):
            return self.config.search_thread
        def __set__(self, value):
            self.config.search_thread = value

    def show(self):
        self.config.show()

    def enable_attention_specific_optimization(self):
        self.config.enable_attention_specific_optimization()

    def enable_concat_matmul_transformation(self):
        self.config.enable_concat_matmul_transformation()

    def _key_fields(self):
        return {
            "max_num_kernel_graph_op": int(self.config.max_num_kernel_graph_op),
            "knop_to_explore": [int(k) for k in self.config.knop_to_explore],
            "randomized_branches": _pybool(self.config.randomized_branches),
            "attention": _pybool(self.config._enable_attention_specific_optimization),
            "concat_matmul": _pybool(self.config._enable_concat_matmul_transformation),
        }


cdef class CyTileGraph:
    cdef CppTileGraph *p_tgraph
    cdef vector[CppDTensor*] owned_dtensors

    def __cinit__(self, split_enabled=False, symbolic=False):
        cdef bint csplit_enabled = False
        cdef bint csymbolic = symbolic
        if split_enabled is True or split_enabled is False:
            csplit_enabled = split_enabled
        else:
            try:
                csplit_enabled = any(split_enabled)
            except TypeError:
                csplit_enabled = 1 if split_enabled else 0
        self.p_tgraph = new CppTileGraph(csplit_enabled, csymbolic)

    def __dealloc__(self):
        cdef size_t i = 0
        for i in range(self.owned_dtensors.size()):
            if self.owned_dtensors[i] != NULL:
                del self.owned_dtensors[i]
        self.owned_dtensors.clear()
        if self.p_tgraph != NULL:
            del self.p_tgraph

    cdef inline DTensor _wrap_dtensor(self, CppDTensor tensor):
        cdef CppDTensor* ptr = new CppDTensor(tensor)
        self.owned_dtensors.push_back(ptr)
        return _make_dtensor(ptr, self)

    def create_tiled_input(self, tuple dims, can_split = None, dtype : dtype = float16):
        cdef vector[int] cdims
        cdef vector[int] ccan_split

        if isinstance(can_split, type(float16)):
            dtype = can_split
            can_split = None

        cdims = _to_dims(dims)
        cdef DataType cdtype = _to_cpp_dtype(dtype)

        if can_split is None:
            return self._wrap_dtensor(self.p_tgraph.create_tiled_input(cdims, cdtype))
        if len(can_split) != len(dims):
            raise ValueError("can_split must match dims length")

        ccan_split.resize(len(can_split))
        cdef size_t i
        for i in range(len(can_split)):
            # No ternary here: Cython binds it via FakeReference to a dead temporary.
            if can_split[i]:
                ccan_split[i] = 1

        return self._wrap_dtensor(
            self.p_tgraph.create_tiled_input(cdims, ccan_split, cdtype)
        )

    def create_tiled_output(self, DTensor O):
        self.p_tgraph.create_tiled_output(_cptr(O)[0])

    def matmul(self, DTensor A, DTensor B, bint trans_A = False, bint trans_B = False):
        return self._wrap_dtensor(self.p_tgraph.matmul(_cptr(A)[0], _cptr(B)[0], trans_A, trans_B))

    def add(self, A, B):
        cdef DTensor dt_a
        cdef DTensor dt_b
        cdef float scalar
        if isinstance(A, DTensor) and isinstance(B, DTensor):
            dt_a = <DTensor>A
            dt_b = <DTensor>B
            return self._wrap_dtensor(self.p_tgraph.add(_cptr(dt_a)[0], _cptr(dt_b)[0]))
        if isinstance(A, DTensor):
            dt_a = <DTensor>A
            if not isinstance(B, (int, float)):
                raise TypeError("Expected a real scalar for constant operand")
            scalar = <float>B
            return self._wrap_dtensor(self.p_tgraph.add(_cptr(dt_a)[0], scalar))
        raise TypeError("add expects the first argument to be a DTensor")

    def mul(self, A, B):
        cdef DTensor dt_a
        cdef DTensor dt_b
        cdef float scalar
        if isinstance(A, DTensor) and isinstance(B, DTensor):
            dt_a = <DTensor>A
            dt_b = <DTensor>B
            return self._wrap_dtensor(self.p_tgraph.mul(_cptr(dt_a)[0], _cptr(dt_b)[0]))
        if isinstance(A, DTensor):
            dt_a = <DTensor>A
            if not isinstance(B, (int, float)):
                raise TypeError("Expected a real scalar for constant operand")
            scalar = <float>B
            return self._wrap_dtensor(self.p_tgraph.mul(_cptr(dt_a)[0], scalar))
        raise TypeError("mul expects the first argument to be a DTensor")

    def div(self, A, B):
        cdef DTensor dt_a
        cdef DTensor dt_b
        cdef float scalar
        if isinstance(A, DTensor) and isinstance(B, DTensor):
            dt_a = <DTensor>A
            dt_b = <DTensor>B
            return self._wrap_dtensor(self.p_tgraph.div(_cptr(dt_a)[0], _cptr(dt_b)[0]))
        if isinstance(A, DTensor):
            dt_a = <DTensor>A
            if not isinstance(B, (int, float)):
                raise TypeError("Expected a real scalar for constant operand")
            scalar = <float>B
            return self._wrap_dtensor(self.p_tgraph.div(_cptr(dt_a)[0], scalar))
        if isinstance(B, DTensor):
            dt_b = <DTensor>B
            if not isinstance(A, (int, float)):
                raise TypeError("Expected a real scalar for constant operand")
            scalar = <float>A
            return self._wrap_dtensor(self.p_tgraph.div(scalar, _cptr(dt_b)[0]))
        raise TypeError("div expects at least one DTensor argument")

    def exp(self, DTensor A):
        return self._wrap_dtensor(self.p_tgraph.exp(_cptr(A)[0]))

    def silu(self, DTensor A):
        return self._wrap_dtensor(self.p_tgraph.silu(_cptr(A)[0]))

    def sqrt(self, DTensor A):
        return self._wrap_dtensor(self.p_tgraph.sqrt(_cptr(A)[0]))

    def reduce_sum(self, DTensor A, int dim):
        return self._wrap_dtensor(self.p_tgraph.reduce_sum(_cptr(A)[0], dim))

    def sub(self, A, B):
        """Elementwise A - B; a scalar on either side becomes an add (and mul by -1 on the left)."""
        cdef DTensor dt_a
        cdef DTensor dt_b
        cdef float scalar
        if isinstance(A, DTensor) and isinstance(B, DTensor):
            dt_a = <DTensor>A
            dt_b = <DTensor>B
            return self._wrap_dtensor(self.p_tgraph.sub(_cptr(dt_a)[0], _cptr(dt_b)[0]))
        if isinstance(A, DTensor):
            dt_a = <DTensor>A
            if not isinstance(B, (int, float)):
                raise TypeError("Expected a real scalar for constant operand")
            scalar = <float>B
            return self._wrap_dtensor(self.p_tgraph.sub(_cptr(dt_a)[0], scalar))
        if isinstance(B, DTensor):
            dt_b = <DTensor>B
            if not isinstance(A, (int, float)):
                raise TypeError("Expected a real scalar for constant operand")
            scalar = <float>A
            return self._wrap_dtensor(self.p_tgraph.sub(scalar, _cptr(dt_b)[0]))
        raise TypeError("sub expects at least one DTensor argument")

    def reduce_max(self, DTensor A, int dim):
        """Max over ``dim`` (keepdim)."""
        return self._wrap_dtensor(self.p_tgraph.reduce_max(_cptr(A)[0], dim))

    def kn_graph(self):
        """Non-owning CyKNGraph view of this tile graph (keeps ``self`` alive)."""
        return CyKNGraph(ctypes.c_void_p(<unsigned long long>&self.p_tgraph.tile_graph), self)

    def get_graph_structure(self):
        return self.kn_graph().get_graph_structure()

    def setup_search_config(self):
        cdef GeneratorConfig cfg = self.p_tgraph.setup_search_config()
        cdef CyGeneratorConfig py_cfg = CyGeneratorConfig.__new__(CyGeneratorConfig)
        py_cfg.config = cfg
        return py_cfg

    def _structure_key(self):
        """Canonical, guid-free description of the graph (shapes, flags, dtypes, op attrs)."""
        cdef size_t i, k
        cdef int d
        cdef CppKNOperator* op
        cdef CppDTensor* t
        cdef CppKNMatmulOp* matmul_op
        cdef CppKNElementBinaryOp* bin_op
        index = {}
        ops = []
        for i in range(self.p_tgraph.tile_graph.operators.size()):
            op = self.p_tgraph.tile_graph.operators[i]
            entry = [int(op.op_type),
                     [index.get(op.input_tensors[k].guid, -1) for k in range(op.input_tensors.size())]]
            outs = []
            for k in range(op.output_tensors.size()):
                t = &op.output_tensors[k]
                index[t.guid] = len(index)
                outs.append([[t.dim[d] for d in range(t.num_dims)],
                             [_pybool(t.can_split[d]) for d in range(t.num_dims)],
                             [_pybool(t.can_reduce[d]) for d in range(t.num_dims)],
                             data_type_to_string(t.dtype).decode("utf-8")])
            entry.append(outs)
            if op.op_type == KN_MATMUL_OP:
                matmul_op = <CppKNMatmulOp*> op
                entry.append([_pybool(matmul_op.trans_A), _pybool(matmul_op.trans_B)])
            elif op.op_type in (KN_ADD_OP, KN_MUL_OP, KN_DIV_OP):
                bin_op = <CppKNElementBinaryOp*> op
                if bin_op.has_scalar:
                    entry.append([float(bin_op.scalar), _pybool(bin_op.scalar_on_left)])
            ops.append(entry)
        return ops

    def checkpoint_key(self, bint enable_concat_matmul_transformation = False,
                       CyGeneratorConfig external_config = None):
        import hashlib
        import json
        cfg = external_config if external_config is not None else self.setup_search_config()
        payload = {
            "schema": _CHECKPOINT_SCHEMA,
            "build": _build_id(),
            "symbolic": _pybool(self.p_tgraph.symbolic),
            "graph": self._structure_key(),
            "config": cfg._key_fields(),
            "enable_concat_matmul_transformation": _pybool(enable_concat_matmul_transformation),
        }
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def search_tile_graph(self, str checkpoint_file_name = "",
                          bint enable_concat_matmul_transformation = False,
                          bint show_config = False,
                          CyGeneratorConfig external_config = None):
        """Search; a checkpoint is reused only if its sidecar ``<path>.key`` matches this graph+config."""
        import os
        cdef const GeneratorConfig* cfg_ptr = NULL
        if external_config is not None:
            cfg_ptr = &external_config.config
        if checkpoint_file_name is None:
            checkpoint_file_name = ""
        key = None
        key_path = None
        cdef bint load = False
        if checkpoint_file_name:
            key = self.checkpoint_key(enable_concat_matmul_transformation, external_config)
            key_path = checkpoint_file_name + ".key"
            if os.path.exists(checkpoint_file_name):
                try:
                    with open(key_path) as f:
                        load = f.read().strip() == key
                except OSError:
                    load = False
                if not load:
                    print(f"[cleave] ignoring stale checkpoint {checkpoint_file_name} (key mismatch); re-searching")
        cdef string ccheckpoint = checkpoint_file_name.encode('utf-8')
        cdef bint cconcat = enable_concat_matmul_transformation
        cdef bint cshow = show_config
        cdef vector[CppKNGraph*] graphs
        with nogil:
            graphs = self.p_tgraph.search_graphs(ccheckpoint, cconcat, cshow, cfg_ptr, load)
        new_graphs = [_take_kn_graph(graphs[i]) for i in range(graphs.size())]
        if key is not None and not load and os.path.exists(checkpoint_file_name):
            with open(key_path, "w") as f:
                f.write(key + "\n")
        return new_graphs


def create_symbolic_tile_graph(CyTileGraph concrete_graph not None):
    cdef unordered_map[int, int] c_symbol_mapping
    cdef CppTileGraph* symbolic_ptr = CppTileGraph.create_symbolic(
        concrete_graph.p_tgraph.tile_graph,
        &c_symbol_mapping
    )
    cdef CyTileGraph symbolic_graph = CyTileGraph.__new__(CyTileGraph)
    del symbolic_graph.p_tgraph
    symbolic_graph.p_tgraph = symbolic_ptr

    cdef unordered_map[int, int].iterator it = c_symbol_mapping.begin()
    symbol_map = {}
    while it != c_symbol_mapping.end():
        symbol_map[int(deref(it).first)] = int(deref(it).second)
        inc(it)

    return symbolic_graph, symbol_map


def search(CyKNGraph input_graph not None, *, int max_num_new_graphs = 1024, str previous_checkpoint = None, str default_config = None):
    if max_num_new_graphs <= 0:
        raise ValueError("max_num_new_graphs must be positive")
    cdef const char* cprevious_checkpoint = NULL
    cdef const char* cconfig = NULL
    checkpoint_bytes = previous_checkpoint.encode('UTF-8') if previous_checkpoint is not None else None
    config_bytes = default_config.encode('UTF-8') if default_config is not None else None
    if checkpoint_bytes is not None:
        cprevious_checkpoint = checkpoint_bytes
    if config_bytes is not None:
        cconfig = config_bytes
    cdef vector[CppKNGraph*] graphs
    with nogil:
        graphs = search_graphs(input_graph.p_kgraph, cprevious_checkpoint, cconfig)
    new_graphs = [_take_kn_graph(graphs[i]) for i in range(graphs.size())]
    return new_graphs[:max_num_new_graphs]
