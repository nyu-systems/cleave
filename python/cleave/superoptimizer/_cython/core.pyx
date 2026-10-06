
from CCore cimport *
from cpython cimport array
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
    elif op_type == KN_INPUT_OP:
        return "kn_input_op"
    elif op_type == KN_OUTPUT_OP:
        return "kn_output_op"
    elif op_type == KN_MATMUL_OP:
        return "kn_matmul_op"
    elif op_type == KN_EXP_OP:
        return "kn_exp_op"
    elif op_type == KN_SQUARE_OP:
        return "kn_square_op"
    elif op_type == KN_SQRT_OP:
        return "kn_sqrt_op"
    elif op_type == KN_SILU_OP:
        return "kn_silu_op"
    elif op_type == KN_GELU_OP:
        return "kn_gelu_op"
    elif op_type == KN_SIGMOID_OP:
        return "kn_sigmoid_op"
    elif op_type == KN_RELU_OP:
        return "kn_relu_op"
    elif op_type == KN_CLAMP_OP:
        return "kn_clamp_op"
    elif op_type == KN_LOG_OP:
        return "kn_log_op"
    elif op_type == KN_ADD_OP:
        return "kn_add_op"
    elif op_type == KN_MUL_OP:
        return "kn_mul_op"
    elif op_type == KN_DIV_OP:
        return "kn_div_op"
    elif op_type == KN_REDUCTION_0_OP:
        return "kn_reduction_0_op"
    elif op_type == KN_REDUCTION_1_OP:
        return "kn_reduction_1_op"
    elif op_type == KN_REDUCTION_2_OP:
        return "kn_reduction_2_op"
    elif op_type == KN_REDUCTION_3_OP:
        return "kn_reduction_3_op"
    elif op_type == KN_REDUCTION_4_OP:
        return "kn_reduction_4_op"
    elif op_type == KN_RMS_NORM_OP:
        return "kn_rms_norm_op"
    elif op_type == KN_SPLIT_0_OP:
        return "kn_split_0_op"
    elif op_type == KN_SPLIT_1_OP:
        return "kn_split_1_op"
    elif op_type == KN_SPLIT_2_OP:
        return "kn_split_2_op"
    elif op_type == KN_SPLIT_3_OP:
        return "kn_split_3_op"
    elif op_type == KN_SPLIT_4_OP:
        return "kn_split_4_op"
    else:
        return "unknown_op_type" + str(op_type)


cdef class DTensor:
    cdef CppDTensor* c_ptr # Hold a Tensor instance
    cdef object _dtype

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

    property num_dims:
        def __get__(self):
            if self.c_ptr == NULL:
                print("Error: tensor is None in num_dims property")
                return None
            else:
                return self.c_ptr.num_dims

    property dtype:
        def __get__(self):
            return self._dtype
        def __set__(self, value):
            self._dtype = value

    def __cinit__(self, tensor):
        self._dtype = None
        self._set_tensor(tensor)

    def dim(self, int idx):
        if (idx < self.c_ptr.num_dims):
            return self.c_ptr.dim[idx]
        else:
            assert False , "Error: index out of range"
            return None

cdef class CyKNOperator:
    cdef CppKNOperator* c_ptr # Hold a CppKNOperator instance

    cdef inline _set_operator(self, op):
        cdef unsigned long long ptr
        if op is None:
            self.c_ptr = <CppKNOperator*>(NULL)
        else:
            ptr = ctypes.cast(op, ctypes.c_void_p).value
            self.c_ptr = <CppKNOperator*>(ptr)
    
    def get_input_dtensors(self):
        cdef CppDTensor* cinputs[1024]
        num = self.c_ptr.get_input_dtensors(cinputs)
        inputs = list()
        for i in range(num):
            ptr = ctypes.cast(<unsigned long long>cinputs[i], ctypes.c_void_p)
            inputs.append(DTensor(ptr))
        return inputs

    def get_output_dtensors(self):
        cdef CppDTensor* coutputs[1024]
        num = self.c_ptr.get_output_dtensors(coutputs)
        outputs = list()
        for i in range(num):
            ptr = ctypes.cast(<unsigned long long>coutputs[i], ctypes.c_void_p)
            outputs.append(DTensor(ptr))
        return outputs

    property op_type:
        def __get__(self):
            if self.c_ptr == NULL:
                return None
            else:
                return get_kn_operator_type_string(int(self.c_ptr.op_type))

    def __cinit__(self, op):
        self._set_operator(op)

cdef class CyKNGraph:
    cdef CppKNGraph *p_kgraph #Hold a CppKNGraph instance
    cdef object tensor_dtypes

    def __cinit__(self, graph = None):
        cdef unsigned long long ptr
        self.tensor_dtypes = {}
        if graph is None:
            self.p_kgraph = new CppKNGraph()
        else:
            ptr = ctypes.cast(graph, ctypes.c_void_p).value
            self.p_kgraph = <CppKNGraph*>(ptr)

    cdef inline DTensor _wrap_dtensor_ptr(self, CppDTensor* ptr, object tensor_dtype=None):
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        result = DTensor(t)
        if tensor_dtype is None and ptr != NULL:
            tensor_dtype = self.tensor_dtypes.get(ptr.guid)
        result._dtype = tensor_dtype
        if tensor_dtype is not None and ptr != NULL:
            self.tensor_dtypes[result.guid] = tensor_dtype
        return result

    def new_input(self, tuple dims, dtype : dtype = float16):
        cdef vector[int] cdims
        cdims.resize(len(dims))
        for i in range(len(dims)):
            cdims[i] = dims[i]

        cdef CppDTensor* ptr = self.p_kgraph.new_input_ptr(cdims)
        return self._wrap_dtensor_ptr(ptr, dtype)

    def mark_output(self, DTensor A):
        self.p_kgraph.mark_output(A.c_ptr[0])

    def matmul(self, DTensor A, DTensor B, bint trans_A = False, bint trans_B = False):
        cdef CppDTensor* ptr = self.p_kgraph.matmul(A.c_ptr, B.c_ptr, trans_A, trans_B)
        return self._wrap_dtensor_ptr(ptr, A.dtype)

    def reduction(self, DTensor input, int dim):
        cdef CppDTensor* ptr = self.p_kgraph.reduction(input.c_ptr, dim, 1)
        return self._wrap_dtensor_ptr(ptr, input.dtype)

    def rms_norm(self, DTensor input, tuple normalized_shape):
        cdef vector[int] cshape
        cshape.resize(len(normalized_shape))
        for i in range(len(normalized_shape)):
            cshape[i] = normalized_shape[i]
        cdef CppDTensor* ptr = self.p_kgraph.rms_norm(input.c_ptr, cshape)
        return self._wrap_dtensor_ptr(ptr, input.dtype)

    def exp(self, DTensor input):
        cdef CppDTensor* ptr = self.p_kgraph.exp(input.c_ptr)
        return self._wrap_dtensor_ptr(ptr, input.dtype)

    def silu(self, DTensor input):
        cdef CppDTensor* ptr = self.p_kgraph.silu(input.c_ptr)
        return self._wrap_dtensor_ptr(ptr, input.dtype)
    
    def gelu(self, DTensor input):
        cdef CppDTensor* ptr = self.p_kgraph.gelu(input.c_ptr)
        return self._wrap_dtensor_ptr(ptr, input.dtype)
    
    def relu(self, DTensor input):
        cdef CppDTensor* ptr = self.p_kgraph.relu(input.c_ptr)
        return self._wrap_dtensor_ptr(ptr, input.dtype)
    
    def clamp(self, DTensor input, float min_val, float max_val):
        cdef CppDTensor* ptr = self.p_kgraph.clamp(input.c_ptr, min_val, max_val)
        return self._wrap_dtensor_ptr(ptr, input.dtype)

    def add(self, DTensor A, DTensor B):
        cdef CppDTensor* ptr = self.p_kgraph.add(A.c_ptr, B.c_ptr)
        return self._wrap_dtensor_ptr(ptr, A.dtype)

    def mul(self, DTensor A, DTensor B):
        cdef CppDTensor* ptr = self.p_kgraph.mul(A.c_ptr, B.c_ptr)
        return self._wrap_dtensor_ptr(ptr, A.dtype)

    def div(self, DTensor A, DTensor B):
        cdef CppDTensor* ptr = self.p_kgraph.div(A.c_ptr, B.c_ptr)
        return self._wrap_dtensor_ptr(ptr, A.dtype)

    def get_input_dtensors(self):
        cdef CppDTensor* cinputs[1024]
        num = self.p_kgraph.get_input_dtensors(cinputs)
        inputs = list()
        for i in range(num):
            inputs.append(self._wrap_dtensor_ptr(cinputs[i]))
        return inputs
    
    # visualizer utils

    def _kn_tensor_to_dict(self, DTensor t):
        tensor_dtype = t.dtype
        if tensor_dtype is None:
            tensor_dtype = self.tensor_dtypes.get(t.guid)
        info = {
            "num_dims": t.num_dims,
            "dim": [t.dim(i) for i in range(t.num_dims)],
            "guid": t.guid
        }
        if tensor_dtype is not None:
            info["dtype"] = str(tensor_dtype)
        return info

    def _get_kn_operator_info(self, CyKNOperator op):
        cdef CppKNElementBinaryOp* bin_op
        cdef CppKNMatmulOp* matmul_op
        info = {
            "op_type": op.op_type,
            "input_tensors": [self._kn_tensor_to_dict(t) for t in op.get_input_dtensors()],
            "output_tensors": [self._kn_tensor_to_dict(t) for t in op.get_output_dtensors()],
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
        operators = []
        ops = self.p_kgraph.operators
        for i in range(ops.size()):
            op = CyKNOperator(None)
            op.c_ptr = ops[i]
            operators.append(self._get_kn_operator_info(op))
        return operators

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

cdef class CyTileGraph:
    cdef CppTileGraph *p_tgraph
    cdef vector[CppDTensor*] owned_dtensors
    cdef object tensor_dtypes

    def __cinit__(self, split_enabled=False, symbolic=False):
        cdef bint csplit_enabled = False
        cdef bint csymbolic = symbolic
        self.tensor_dtypes = {}
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

    cdef inline DTensor _wrap_dtensor(self, CppDTensor tensor, object tensor_dtype=None):
        cdef CppDTensor* ptr = new CppDTensor(tensor)
        self.owned_dtensors.push_back(ptr)
        t = ctypes.cast(<unsigned long long>ptr, ctypes.c_void_p)
        result = DTensor(t)
        if tensor_dtype is None:
            tensor_dtype = self.tensor_dtypes.get(result.guid)
        result._dtype = tensor_dtype
        if tensor_dtype is not None:
            self.tensor_dtypes[result.guid] = tensor_dtype
        return result

    def create_tiled_input(self, tuple dims, can_split = None, dtype : dtype = float16):
        cdef vector[int] cdims
        cdef vector[int] ccan_split

        if isinstance(can_split, type(float16)):
            dtype = can_split
            can_split = None

        cdims.resize(len(dims))
        for i in range(len(dims)):
            cdims[i] = dims[i]

        if can_split is None:
            return self._wrap_dtensor(self.p_tgraph.create_tiled_input(cdims), dtype)
        if len(can_split) != len(dims):
            raise ValueError("can_split must match dims length")
        
        ccan_split.resize(len(can_split))
        for i in range(len(can_split)):
            ccan_split[i] = can_split[i]

        return self._wrap_dtensor(
            self.p_tgraph.create_tiled_input(cdims, ccan_split),
            dtype
        )

    def create_tiled_output(self, DTensor O):
        self.p_tgraph.create_tiled_output(O.c_ptr[0])

    def matmul(self, DTensor A, DTensor B, bint trans_A = False, bint trans_B = False):
        return self._wrap_dtensor(self.p_tgraph.matmul(A.c_ptr[0], B.c_ptr[0], trans_A, trans_B), A.dtype)

    def add(self, A, B):
        cdef DTensor dt_a
        cdef DTensor dt_b
        cdef float scalar
        if isinstance(A, DTensor) and isinstance(B, DTensor):
            dt_a = <DTensor>A
            dt_b = <DTensor>B
            return self._wrap_dtensor(self.p_tgraph.add(dt_a.c_ptr[0], dt_b.c_ptr[0]), dt_a.dtype)
        if isinstance(A, DTensor):
            dt_a = <DTensor>A
            if not isinstance(B, (int, float)):
                raise TypeError("Expected a real scalar for constant operand")
            scalar = <float>B
            return self._wrap_dtensor(self.p_tgraph.add(dt_a.c_ptr[0], scalar), dt_a.dtype)
        raise TypeError("add expects the first argument to be a DTensor")

    def mul(self, A, B):
        cdef DTensor dt_a
        cdef DTensor dt_b
        cdef float scalar
        if isinstance(A, DTensor) and isinstance(B, DTensor):
            dt_a = <DTensor>A
            dt_b = <DTensor>B
            return self._wrap_dtensor(self.p_tgraph.mul(dt_a.c_ptr[0], dt_b.c_ptr[0]), dt_a.dtype)
        if isinstance(A, DTensor):
            dt_a = <DTensor>A
            if not isinstance(B, (int, float)):
                raise TypeError("Expected a real scalar for constant operand")
            scalar = <float>B
            return self._wrap_dtensor(self.p_tgraph.mul(dt_a.c_ptr[0], scalar), dt_a.dtype)
        raise TypeError("mul expects the first argument to be a DTensor")

    def div(self, A, B):
        cdef DTensor dt_a
        cdef DTensor dt_b
        cdef float scalar
        if isinstance(A, DTensor) and isinstance(B, DTensor):
            dt_a = <DTensor>A
            dt_b = <DTensor>B
            return self._wrap_dtensor(self.p_tgraph.div(dt_a.c_ptr[0], dt_b.c_ptr[0]), dt_a.dtype)
        if isinstance(A, DTensor):
            dt_a = <DTensor>A
            if not isinstance(B, (int, float)):
                raise TypeError("Expected a real scalar for constant operand")
            scalar = <float>B
            return self._wrap_dtensor(self.p_tgraph.div(dt_a.c_ptr[0], scalar), dt_a.dtype)
        if isinstance(B, DTensor):
            dt_b = <DTensor>B
            if not isinstance(A, (int, float)):
                raise TypeError("Expected a real scalar for constant operand")
            scalar = <float>A
            return self._wrap_dtensor(self.p_tgraph.div(scalar, dt_b.c_ptr[0]), dt_b.dtype)
        raise TypeError("div expects at least one DTensor argument")

    def exp(self, DTensor A):
        return self._wrap_dtensor(self.p_tgraph.exp(A.c_ptr[0]), A.dtype)

    def silu(self, DTensor A):
        return self._wrap_dtensor(self.p_tgraph.silu(A.c_ptr[0]), A.dtype)

    def sqrt(self, DTensor A):
        return self._wrap_dtensor(self.p_tgraph.sqrt(A.c_ptr[0]), A.dtype)

    def reduction(self, DTensor A, int dim):
        return self._wrap_dtensor(self.p_tgraph.reduction(A.c_ptr[0], dim), A.dtype)

    def setup_search_config(self):
        cdef GeneratorConfig cfg = self.p_tgraph.setup_search_config()
        cdef CyGeneratorConfig py_cfg = CyGeneratorConfig.__new__(CyGeneratorConfig)
        py_cfg.config = cfg
        return py_cfg

    def search_tile_graph(self, str checkpoint_file_name = "checkpoint.json",
                          bool enable_concat_matmul_transformation = False,
                          bool show_config = False,
                          CyGeneratorConfig external_config = None):
        cdef GeneratorConfig* cfg_ptr = NULL
        if external_config is not None:
            cfg_ptr = &external_config.config
        cdef string ccheckpoint = checkpoint_file_name.encode('utf-8')
        cdef CppKNGraph* cnewgraphs[1024]
        cdef int max_num_new_graphs = 1024
        cdef int num = self.p_tgraph.search_tile_graph(
            ccheckpoint,
            enable_concat_matmul_transformation,
            show_config,
            cfg_ptr,
            max_num_new_graphs,
            cnewgraphs
        )
        new_graphs = list()
        for i in range(num):
            ptr = ctypes.cast(<unsigned long long>cnewgraphs[i], ctypes.c_void_p)
            new_graphs.append(CyKNGraph(ptr))
        return new_graphs

def create_symbolic_tile_graph(CyTileGraph concrete_graph):
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

def search(CyKNGraph input_graph, *, int max_num_new_graphs = 1024, str previous_checkpoint = None, str default_config = None):
    # allocate new graphs
    # currently support up to 1024 new graphs
    assert max_num_new_graphs <= 1024
    cdef CppKNGraph* cnewgraphs[1024]
    # set previous_checkpoint
    cdef char* cprevious_checkpoint = NULL
    if previous_checkpoint is not None:
        py_byte_string = previous_checkpoint.encode('UTF-8')
        cprevious_checkpoint = py_byte_string
    # convert config description
    cdef char* cconfig = NULL
    if default_config is not None:
        py_byte_string = default_config.encode('UTF-8')
        cconfig = py_byte_string
    num = cython_search(input_graph.p_kgraph, max_num_new_graphs, cnewgraphs, cprevious_checkpoint, cconfig)
    new_graphs = list()
    for i in range(num):
        ptr = ctypes.cast(<unsigned long long>cnewgraphs[i], ctypes.c_void_p)
        new_graphs.append(CyKNGraph(ptr))

    return new_graphs
