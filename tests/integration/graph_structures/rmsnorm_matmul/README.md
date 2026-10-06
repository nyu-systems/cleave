# Invalid Graphs:

## Invalid 1

![](./images/before_split_rmsnorm_matmul_graph_structure_invalid_1.png)

path: "./before_split_rmsnorm_matmul_graph_structure_invalid_1.graph"

Reason: This is invalid because the split boundary is not clear. the reduction consumes the split path, but the matmul follows a non split path. This would make it hard to horizontally fuse the split from the matmul with the non-split path from the reduction. if they are horizontal dependencies (in the future will merge, but do not depend on each other), they need to share the same splitting logic to eventually merge them together.

## Invalid 2

![](./images/before_split_rmsnorm_matmul_graph_structure_invalid_2.png)

path: "./before_split_rmsnorm_matmul_graph_structure_invalid_2.graph"

Reason: This is invalid because the split boundary is not clear. the reduction consumes the split path, but the matmul follows a non split path. This would make it hard to horizontally fuse the split from the matmul with the non-split path from the reduction. if they are horizontal dependencies (in the future will merge, but do not depend on each other), they need to share the same splitting logic to eventually merge them together.

## Invalid 3

![](./images/before_split_rmsnorm_matmul_graph_structure_invalid_3.png)

path: "./before_split_rmsnorm_matmul_graph_structure_invalid_3.graph"

Reason: This is invalid because the split boundary is not clear. the matmul consumes the split path, but the "5: Reduction 1" follows a non split path. This would make it hard to horizontally fuse the split from the matmul with the non-split path from the reduction. if they are horizontal dependencies (in the future will merge, but do not depend on each other), they need to share the same splitting logic to eventually merge them together.


## Invalid 4

![](./images/before_split_rmsnorm_matmul_graph_structure_invalid_4.png)

path: "./before_split_rmsnorm_matmul_graph_structure_invalid_4.graph"

Reason: Althouth this is legit structure, it is invalid because this would require at least 3 kernels. 1 kernel up to the multiply op. 1 kernel for the split. 1 kernel for the combine. We only support up to 2 graphs (normal graph (1 kernel) or split+combine (2 kernels))


## Invalid 5

![](./images/before_split_rmsnorm_matmul_graph_structure_invalid_5.png)

path: "./before_split_rmsnorm_matmul_graph_structure_invalid_5.graph"

Reason: The node 0 crosses the split boundary. invalid.

## Invalid 6

![](./images/before_split_rmsnorm_matmul_graph_structure_invalid_6.png)

path: "./before_split_rmsnorm_matmul_graph_structure_invalid_6.graph"

Reason: The node 0 crosses the split boundary. invalid.


# Valid Graphs:

## Valid 1

![](./images/before_split_rmsnorm_matmul_graph_structure_valid_1.png)

path: "./before_split_rmsnorm_matmul_graph_structure_valid_1.graph"

Reason: Valid. No splitting exists. single kernel.

## Valid 2

![](./images/before_split_rmsnorm_matmul_graph_structure_valid_2.png)

path: "./before_split_rmsnorm_matmul_graph_structure_valid_2.graph"

Reason: Valid. Boundary is clear on reductions 11 and 12. 2 kernels only.

## Valid 3

![](./images/before_split_rmsnorm_matmul_graph_structure_valid_3.png)

path: "./before_split_rmsnorm_matmul_graph_structure_valid_3.graph"

Reason: Valid. No splitting exists. single kernel.