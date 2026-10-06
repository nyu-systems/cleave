#include "cleave/superoptimizer/tile/graph.h"

namespace cleave_superoptimizer {
namespace tile {

DTensor TileGraph::matmul(DTensor const &A,
                          DTensor const &B,
                          bool trans_A,
                          bool trans_B) {
  tile_config.update_config(cleave_superoptimizer::type::KN_MATMUL_OP);

  DTensor tensor = tile_graph.matmul(A, B, trans_A, trans_B);
  return tensor;
}

DTensor TileGraph::add(DTensor const &A, DTensor const &B){
  tile_config.update_config(cleave_superoptimizer::type::KN_ADD_OP);

  DTensor tensor = tile_graph.add(A, B);
  return tensor;
}

DTensor TileGraph::add(DTensor const &A, float scalar){
  tile_config.update_config(cleave_superoptimizer::type::KN_ADD_OP);

  DTensor tensor = tile_graph.add(A, scalar);
  return tensor;
}

DTensor TileGraph::mul(DTensor const &A, DTensor const &B){
  tile_config.update_config(cleave_superoptimizer::type::KN_MUL_OP);

  DTensor tensor = tile_graph.mul(A, B);
  return tensor;
}

DTensor TileGraph::mul(DTensor const &A, float scalar){
  tile_config.update_config(cleave_superoptimizer::type::KN_MUL_OP);

  DTensor tensor = tile_graph.mul(A, scalar);
  return tensor;
}

DTensor TileGraph::div(DTensor const &A, DTensor const &B){
  tile_config.update_config(cleave_superoptimizer::type::KN_DIV_OP);

  DTensor tensor = tile_graph.div(A, B);
  return tensor;
}

DTensor TileGraph::div(DTensor const &A, float scalar){
  tile_config.update_config(cleave_superoptimizer::type::KN_DIV_OP);

  DTensor tensor = tile_graph.div(A, scalar);
  return tensor;
}

DTensor TileGraph::div(float scalar, DTensor const &A){
  tile_config.update_config(cleave_superoptimizer::type::KN_DIV_OP);

  DTensor tensor = tile_graph.div(scalar, A);
  return tensor;
}

DTensor TileGraph::exp(DTensor const &I){
  tile_config.update_config(cleave_superoptimizer::type::KN_EXP_OP);

  DTensor tensor = tile_graph.exp(I);
  return tensor;
}

DTensor TileGraph::silu(DTensor const &I){
  tile_config.update_config(cleave_superoptimizer::type::KN_SILU_OP);

  DTensor tensor = tile_graph.silu(I);
  return tensor;
}

DTensor TileGraph::sqrt(DTensor const &I){
  tile_config.update_config(cleave_superoptimizer::type::KN_SQRT_OP);

  DTensor tensor = tile_graph.sqrt(I);
  return tensor;
}

DTensor TileGraph::reduction(DTensor const &I, int const &dim){
  tile_config.update_config(cleave_superoptimizer::type::KN_REDUCTION_1_OP); // TODO: check op type

  DTensor tensor = tile_graph.reduction(I, dim);
  return tensor;
}

} // namespace tile
} // namespace cleave_superoptimizer
