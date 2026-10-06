
from libcpp.memory cimport shared_ptr, unique_ptr
from libcpp.string cimport string
from libcpp.unordered_map cimport unordered_map
from libcpp.vector cimport vector
from libcpp cimport bool

ctypedef unsigned long int size_t

cdef extern from "cleave/superoptimizer/utils/vector_types.h":
    ctypedef struct dim3:
        unsigned int x
        unsigned int y
        unsigned int z
    ctypedef struct int3:
        int x
        int y
        int z

cdef extern from "cleave/superoptimizer/type.h" namespace "cleave_superoptimizer::type":
    # This must be consistent with cleave/superoptimizer/type.h
    cdef enum KNOperatorType:
        KN_UNKOWN = 1000,
        KN_INPUT_OP = 1001,
        KN_OUTPUT_OP = 1002,
        KN_MATMUL_OP = 1003,
        # ElementUnary
        KN_EXP_OP = 1100,
        KN_SQUARE_OP = 1101,
        KN_SQRT_OP = 1102,
        KN_SILU_OP = 1104,
        KN_SIGMOID_OP = 1105,
        KN_GELU_OP = 1106,
        KN_RELU_OP = 1150,
        KN_CLAMP_OP = 1151,
        KN_LOG_OP = 1160,
        # ElementBinary
        KN_ADD_OP = 1200,
        KN_MUL_OP = 1201,
        KN_DIV_OP = 1202,
        # Reduction & Normalization
        KN_REDUCTION_0_OP = 1300,
        KN_REDUCTION_1_OP = 1301,
        KN_REDUCTION_2_OP = 1302,
        KN_REDUCTION_3_OP = 1303,
        KN_REDUCTION_4_OP = 1304,
        KN_RMS_NORM_OP = 1350,
        # Concat & Split
        KN_CONCAT_FIRST_OP_ID = 1400,
        KN_CONCAT_0_OP = 1400,
        KN_CONCAT_1_OP = 1401,
        KN_CONCAT_2_OP = 1402,
        KN_CONCAT_LAST_OP_ID = 1409,
        KN_SPLIT_FIRST_OP_ID = 1420,
        KN_SPLIT_0_OP = 1420,
        KN_SPLIT_1_OP = 1421,
        KN_SPLIT_2_OP = 1422,
        KN_SPLIT_3_OP = 1423,
        KN_SPLIT_4_OP = 1424,
        KN_SPLIT_LAST_OP_ID = 1424,

cdef extern from "cleave/superoptimizer/kernel/device_tensor.h" namespace "cleave_superoptimizer::kernel":
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

cdef extern from "cleave/superoptimizer/kernel/graph.h" namespace "cleave_superoptimizer::kernel":

    cdef cppclass CppKNOperator "cleave_superoptimizer::kernel::KNOperator":
        KNOperatorType op_type
        vector[CppDTensor] input_tensors
        vector[CppDTensor] output_tensors
        int get_input_dtensors(CppDTensor** cinputs)
        int get_output_dtensors(CppDTensor** cinputs)
 
    cdef cppclass CppKNGraph "cleave_superoptimizer::kernel::Graph":
        CppKNGraph()
        CppDTensor* new_input_ptr(vector[int] dims)
        void mark_output(const CppDTensor& A)
        CppDTensor* matmul(const CppDTensor* A,
                           const CppDTensor* B,
                           bint trans_A,
                           bint trans_B)
        CppDTensor* reduction(const CppDTensor* input, int dim, int size)
        CppDTensor* rms_norm(const CppDTensor* input, vector[int])
        CppDTensor* exp(const CppDTensor* input)
        CppDTensor* silu(const CppDTensor* input)
        CppDTensor* gelu(const CppDTensor* input)
        CppDTensor* relu(const CppDTensor* input)
        CppDTensor* clamp(const CppDTensor* input, float min_val, float max_val)
        CppDTensor* add(const CppDTensor* op1, const CppDTensor* op2)
        CppDTensor* mul(const CppDTensor* op1, const CppDTensor* op2)
        CppDTensor* div(const CppDTensor* op1, const CppDTensor* op2)
        int get_input_dtensors(CppDTensor** cinputs)
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
    cdef int cython_search(const CppKNGraph *input_graph,
                           int max_num_new_graphs,
                           CppKNGraph** new_graphs,
                           const char * filename,
                           const char * default_config)

cdef extern from "cleave/superoptimizer/search/config.h" namespace "cleave_superoptimizer::search":
    cdef enum VerifierType:
        PROBABILISTIC_VERIFIER
        FORMAL_VERIFIER

    cdef cppclass GeneratorConfig:
        size_t max_num_kernel_graph_op
        size_t search_thread
        VerifierType verifier_type
        vector[KNOperatorType] knop_to_explore
        bool randomized_branches
        bool _enable_attention_specific_optimization
        bool _enable_concat_matmul_transformation
        void show() const
        void enable_attention_specific_optimization()
        void enable_concat_matmul_transformation()
        @staticmethod
        GeneratorConfig get_default_config()

cdef extern from "cleave/superoptimizer/search/search.h" namespace "cleave_superoptimizer::search":
    cdef cppclass CppKernelGraphGenerator "cleave_superoptimizer::search::KernelGraphGenerator":
        void generate_kernel_graphs()

cdef extern from "cleave/superoptimizer/tile/graph.h" namespace "cleave_superoptimizer::tile":
    cdef cppclass CppTileGraph "cleave_superoptimizer::tile::TileGraph":
        CppTileGraph(bool split_enabled, bool symbolic)
        @staticmethod
        CppTileGraph* create_symbolic(
            const CppKNGraph& concrete_graph,
            unordered_map[int, int]* symbol_map)
        CppKNGraph tile_graph
        CppDTensor create_tiled_input(vector[int] dims)
        CppDTensor create_tiled_input(vector[int] dims,
                                      vector[int] can_split)
        void create_tiled_output(const CppDTensor& O)
        CppDTensor matmul(const CppDTensor& A,
                          const CppDTensor& B,
                          bint trans_A,
                          bint trans_B)
        CppDTensor add(const CppDTensor& A, const CppDTensor& B)
        CppDTensor add(const CppDTensor& A, float scalar)
        CppDTensor mul(const CppDTensor& A, const CppDTensor& B)
        CppDTensor mul(const CppDTensor& A, float scalar)
        CppDTensor div(const CppDTensor& A, const CppDTensor& B)
        CppDTensor div(const CppDTensor& A, float scalar)
        CppDTensor div(float scalar, const CppDTensor& A)
        CppDTensor exp(const CppDTensor& I)
        CppDTensor silu(const CppDTensor& I)
        CppDTensor sqrt(const CppDTensor& I)
        CppDTensor reduction(const CppDTensor& I, const int& dim)
        GeneratorConfig setup_search_config()
        unique_ptr[CppKernelGraphGenerator] search_tile_graph(
            const string& checkpoint_file_name,
            const bool& enable_concat_matmul_transformation,
            const bool& show_config,
            GeneratorConfig* external_config)
        int search_tile_graph(
            const string& checkpoint_file_name,
            const bool& enable_concat_matmul_transformation,
            const bool& show_config,
            GeneratorConfig* external_config,
            int max_num_graphs,
            CppKNGraph** new_graphs)
