#include "cleave/superoptimizer/utils/json_utils.h"

void to_json(json &j, int3 const &i) {
  j["x"] = i.x;
  j["y"] = i.y;
  j["z"] = i.z;
}
