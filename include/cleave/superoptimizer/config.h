#pragma once
#include <cstddef>
#include <cstdint>

namespace cleave_superoptimizer {
namespace config {

uint16_t const FINGERPRINT_P = 167;
uint16_t const FINGERPRINT_Q = 83;
uint32_t const FINGERPRINT_EXP_BASE = 3;
uint16_t const FINGERPRINT_PQ = 13861;
// FINGERPRINT_P_MUL_Q_MOD_1 is a multiplier of P and is 1 module Q
uint16_t const FINGERPRINT_P_MUL_Q_MOD_1 = 167;
// FINGERPRINT_Q_MUL_P_MOD_1 is a multiplier of Q and is 1 module P
uint16_t const FINGERPRINT_Q_MUL_P_MOD_1 = 13695;

} // namespace config
} // namespace cleave_superoptimizer
