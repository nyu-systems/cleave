#pragma once

#include "cleave/superoptimizer/kernel/device_tensor.h"
#include "cleave/superoptimizer/utils/json_utils.h"
#include <memory>
#include <vector>

namespace cleave_superoptimizer {
namespace kernel {

class Graph;

// An input that IS a function of earlier inputs of its graph: `definition`
// is a small graph whose inputs stand, in order, for the host graph's inputs
// at positions `sources` (positions among the host's input operators), and
// whose one output is the declared input. Immutable once made, so clones
// (and every candidate graph of a search) share it.
struct InputDeclaration {
  std::shared_ptr<Graph const> definition;
  std::vector<int> sources;
  // The definition's result: the tensor its output operator reads.
  DTensor const &result() const;
};

class KNOperator {
public:
  KNOperator(Graph *graph, cleave_superoptimizer::type::KNOperatorType _type);
  KNOperator(Graph *graph,
             cleave_superoptimizer::type::KNOperatorType _type,
             DTensor const &input1);
  KNOperator(Graph *graph,
             cleave_superoptimizer::type::KNOperatorType _type,
             DTensor const &input1,
             DTensor const &input2);
  KNOperator(Graph *graph,
             cleave_superoptimizer::type::KNOperatorType _type,
             std::vector<DTensor> const &inputs);
  int get_input_dtensors(DTensor **inputs);
  int get_output_dtensors(DTensor **inputs);

  virtual ~KNOperator();
  virtual bool fingerprint(void) = 0;
  virtual operator json() const = 0;

public:
  Graph *kgraph;
  cleave_superoptimizer::type::KNOperatorType op_type;
  std::vector<DTensor> input_tensors;
  std::vector<DTensor> output_tensors;
};

class KNInputOp : public KNOperator {
public:
  KNInputOp(Graph *_graph,
            std::vector<int> const &dims,
            std::vector<bool> const &can_reduce,
            std::vector<bool> const &can_split,
            int3 input_map = {-1, -1, -1});
  ~KNInputOp();
  bool fingerprint(void);
  bool fingerprint(int input_id);

  operator json() const override;

public:
  int3 input_map;
  bool infer_can_split = false;
  // Set for a declared input only.
  std::shared_ptr<InputDeclaration const> declaration;
};

class KNOutputOp : public KNOperator {
public:
  KNOutputOp(Graph *_graph,
             DTensor const &A,
             int3 output_map = {-1, -1, -1});
  ~KNOutputOp();
  bool fingerprint(void);

  operator json() const override;

public:
  int3 output_map;
};

} // namespace kernel
} // namespace cleave_superoptimizer
