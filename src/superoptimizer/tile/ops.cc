#include "cleave/superoptimizer/tile/graph.h"

namespace cleave_superoptimizer {
namespace tile {

DTensor TileGraph::matmul(DTensor const &A,
                          DTensor const &B,
                          bool trans_A,
                          bool trans_B) {
  DTensor tensor = tile_graph.matmul(A, B, trans_A, trans_B);
  tile_config.update_config(cleave_superoptimizer::type::KN_MATMUL_OP);
  return tensor;
}

DTensor TileGraph::add(DTensor const &A, DTensor const &B){
  DTensor tensor = tile_graph.add(A, B);
  tile_config.update_config(cleave_superoptimizer::type::KN_ADD_OP);
  return tensor;
}

DTensor TileGraph::add(DTensor const &A, float scalar){
  DTensor tensor = tile_graph.add(A, scalar);
  tile_config.update_config(cleave_superoptimizer::type::KN_ADD_OP);
  return tensor;
}

DTensor TileGraph::mul(DTensor const &A, DTensor const &B){
  DTensor tensor = tile_graph.mul(A, B);
  tile_config.update_config(cleave_superoptimizer::type::KN_MUL_OP);
  return tensor;
}

DTensor TileGraph::mul(DTensor const &A, float scalar){
  DTensor tensor = tile_graph.mul(A, scalar);
  tile_config.update_config(cleave_superoptimizer::type::KN_MUL_OP);
  return tensor;
}

DTensor TileGraph::div(DTensor const &A, DTensor const &B){
  DTensor tensor = tile_graph.div(A, B);
  tile_config.update_config(cleave_superoptimizer::type::KN_DIV_OP);
  return tensor;
}

DTensor TileGraph::div(DTensor const &A, float scalar){
  DTensor tensor = tile_graph.div(A, scalar);
  tile_config.update_config(cleave_superoptimizer::type::KN_DIV_OP);
  return tensor;
}

DTensor TileGraph::div(float scalar, DTensor const &A){
  DTensor tensor = tile_graph.div(scalar, A);
  tile_config.update_config(cleave_superoptimizer::type::KN_DIV_OP);
  return tensor;
}

DTensor TileGraph::exp(DTensor const &I){
  DTensor tensor = tile_graph.exp(I);
  tile_config.update_config(cleave_superoptimizer::type::KN_EXP_OP);
  return tensor;
}

DTensor TileGraph::silu(DTensor const &I){
  DTensor tensor = tile_graph.silu(I);
  tile_config.update_config(cleave_superoptimizer::type::KN_SILU_OP);
  return tensor;
}

DTensor TileGraph::sqrt(DTensor const &I){
  DTensor tensor = tile_graph.sqrt(I);
  tile_config.update_config(cleave_superoptimizer::type::KN_SQRT_OP);
  return tensor;
}

DTensor TileGraph::reduce_sum(DTensor const &I, int const &dim){
  DTensor tensor = tile_graph.reduce_sum(I, dim);
  tile_config.update_config(cleave_superoptimizer::type::KN_REDUCE_SUM_1_OP); // TODO: check op type
  return tensor;
}

DTensor TileGraph::sub(DTensor const &A, DTensor const &B) {
  DTensor tensor = tile_graph.sub(A, B);
  tile_config.update_config(cleave_superoptimizer::type::KN_SUB_OP);
  return tensor;
}

DTensor TileGraph::sub(DTensor const &A, float scalar) {
  return add(A, -scalar);
}

DTensor TileGraph::sub(float scalar, DTensor const &A) {
  return add(mul(A, -1.0f), scalar);
}

DTensor TileGraph::reduce_max(DTensor const &I, int dim) {
  DTensor tensor = tile_graph.reduce_max(I, dim);
  tile_config.update_config(tile_graph.operators.back()->op_type);
  return tensor;
}

} // namespace tile
} // namespace cleave_superoptimizer
