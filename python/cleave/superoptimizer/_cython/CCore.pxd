
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp cimport bool

ctypedef unsigned long int size_t

cdef extern from "cleave/superoptimizer/utils/vector_types.h":
    ctypedef struct int3:
        int x
        int y
        int z

cdef extern from "cleave/superoptimizer/type.h" namespace "cleave_superoptimizer::type":
    # Values come from op_types.def; only names used by core.pyx are declared.
    cdef enum KNOperatorType:
        KN_UNKOWN
        KN_INPUT_OP
        KN_MATMUL_OP
        KN_ADD_OP
        KN_MUL_OP
        KN_DIV_OP
    cdef struct OpTypeInfo:
        KNOperatorType type
        const char* name
    const OpTypeInfo kOpTypes[]
    const int kNumOpTypes
    const char* op_type_name(KNOperatorType t)

cdef extern from "cleave/superoptimizer/type.h" namespace "cleave_superoptimizer::type":
    cdef enum DataType:
        DT_UNKNOWN = 0
    DataType data_type_from_string(const string& name) except +
    string data_type_to_string(DataType dtype)

cdef extern from "cleave/superoptimizer/kernel/device_tensor.h" namespace "cleave_superoptimizer::kernel":
    cdef int MAX_TENSOR_DIMS
    cdef cppclass CppDTensor "cleave_superoptimizer::kernel::DTensor":
        CppDTensor()
        CppDTensor(const CppDTensor&)
        int num_dims
        int dim[5]
        bool can_reduce[5]
        bool can_split[5]
        long long guid
        void* owner_op
        int owner_ts_idx
        DataType dtype

cdef extern from "cleave/superoptimizer/kernel/graph.h" namespace "cleave_superoptimizer::kernel":

    cdef cppclass CppKNOperator "cleave_superoptimizer::kernel::KNOperator":
        KNOperatorType op_type
        vector[CppDTensor] input_tensors
        vector[CppDTensor] output_tensors

    cdef cppclass CppKNGraph "cleave_superoptimizer::kernel::Graph":
        CppKNGraph()
        CppDTensor* new_input_ptr(vector[int] dims) except +
        CppDTensor* new_input_ptr(vector[int] dims,
                                  vector[bool] can_reduce,
                                  vector[bool] can_split,
                                  DataType dtype) except +
        void mark_output(const CppDTensor& A) except +
        CppDTensor* matmul(const CppDTensor* A,
                           const CppDTensor* B,
                           bint trans_A,
                           bint trans_B) except +
        CppDTensor* reduce_sum(const CppDTensor* input, int dim, int size) except +
        CppDTensor* rms_norm(const CppDTensor* input, vector[int]) except +
        CppDTensor* exp(const CppDTensor* input) except +
        CppDTensor* silu(const CppDTensor* input) except +
        CppDTensor* gelu(const CppDTensor* input) except +
        CppDTensor* relu(const CppDTensor* input) except +
        CppDTensor* clamp(const CppDTensor* input, float min_val, float max_val) except +
        CppDTensor* add(const CppDTensor* op1, const CppDTensor* op2) except +
        CppDTensor* mul(const CppDTensor* op1, const CppDTensor* op2) except +
        CppDTensor* div(const CppDTensor* op1, const CppDTensor* op2) except +
        CppDTensor* sub(const CppDTensor* op1, const CppDTensor* op2) except +
        CppDTensor* reduce_max(const CppDTensor* input, int dim) except +
        vector[CppKNOperator*] operators

cdef extern from "cleave/superoptimizer/kernel/element_binary.h" namespace "cleave_superoptimizer::kernel":
    cdef cppclass CppKNElementBinaryOp "cleave_superoptimizer::kernel::KNElementBinaryOp"(CppKNOperator):
        bool has_scalar
        float scalar
        bool scalar_on_left

cdef extern from "cleave/superoptimizer/kernel/matmul.h" namespace "cleave_superoptimizer::kernel":
    cdef cppclass CppKNMatmulOp "cleave_superoptimizer::kernel::KNMatmulOp"(CppKNOperator):
        bool trans_A
        bool trans_B

cdef extern from "cleave/superoptimizer/search/search_c.h" namespace "cleave_superoptimizer::search_c":
    cdef vector[CppKNGraph*] search_graphs(const CppKNGraph *input_graph,
                                           const char * filename,
                                           const char * default_config) except + nogil

cdef extern from "cleave/superoptimizer/search/config.h" namespace "cleave_superoptimizer::search":
    cdef cppclass GeneratorConfig:
        size_t max_num_kernel_graph_op
        size_t search_thread
        vector[KNOperatorType] knop_to_explore
        bool randomized_branches
        bool _enable_attention_specific_optimization
        bool _enable_concat_matmul_transformation
        void show() const
        void enable_attention_specific_optimization()
        void enable_concat_matmul_transformation()
        @staticmethod
        GeneratorConfig get_default_config()

cdef extern from "cleave/superoptimizer/tile/graph.h" namespace "cleave_superoptimizer::tile":
    cdef cppclass CppTileGraph "cleave_superoptimizer::tile::TileGraph":
        CppTileGraph(bool split_enabled, bool symbolic) except +
        @staticmethod
        CppTileGraph* create_symbolic(
            const CppKNGraph& concrete_graph,
            unordered_map[int, int]* symbol_map) except +
        CppKNGraph tile_graph
        bool symbolic
        CppDTensor create_tiled_input(vector[int] dims, DataType dtype) except +
        CppDTensor create_tiled_input(vector[int] dims,
                                      vector[int] can_split,
                                      DataType dtype) except +
        void create_tiled_output(const CppDTensor& O) except +
        CppDTensor matmul(const CppDTensor& A,
                          const CppDTensor& B,
                          bint trans_A,
                          bint trans_B) except +
        CppDTensor add(const CppDTensor& A, const CppDTensor& B) except +
        CppDTensor add(const CppDTensor& A, float scalar) except +
        CppDTensor mul(const CppDTensor& A, const CppDTensor& B) except +
        CppDTensor mul(const CppDTensor& A, float scalar) except +
        CppDTensor div(const CppDTensor& A, const CppDTensor& B) except +
        CppDTensor div(const CppDTensor& A, float scalar) except +
        CppDTensor div(float scalar, const CppDTensor& A) except +
        CppDTensor exp(const CppDTensor& I) except +
        CppDTensor silu(const CppDTensor& I) except +
        CppDTensor sqrt(const CppDTensor& I) except +
        CppDTensor reduce_sum(const CppDTensor& I, const int& dim) except +
        CppDTensor sub(const CppDTensor& A, const CppDTensor& B) except +
        CppDTensor sub(const CppDTensor& A, float scalar) except +
        CppDTensor sub(float scalar, const CppDTensor& A) except +
        CppDTensor reduce_max(const CppDTensor& I, int dim) except +
        GeneratorConfig setup_search_config() except +
        vector[CppKNGraph*] search_graphs(
            const string& checkpoint_file_name,
            bool enable_concat_matmul_transformation,
            bool show_config,
            const GeneratorConfig* external_config,
            bool load_checkpoint) except + nogil
