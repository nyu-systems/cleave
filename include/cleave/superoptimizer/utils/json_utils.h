#pragma once

#include "containers.h"
#include "cleave/superoptimizer/utils/vector_types.h"
#include <nlohmann/json.hpp>

using json = nlohmann::json;

void to_json(json &j, int3 const &i);
