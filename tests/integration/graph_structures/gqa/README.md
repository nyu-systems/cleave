# Invalid Graphs:

## Invalid 1

![](./images/before_split_gqa_graph_structure_invalid_1.png)

path: "./before_split_gqa_graph_structure_invalid_1.graph"

Reason: Althouth this is legit structure, it is invalid because this would require at least 3 kernels. 1 kernel up to the div op. 1 kernel for the split. 1 kernel for the combine. We only support up to 2 graphs (normal graph (1 kernel) or split+combine (2 kernels))

## Invalid 2

![](./images/before_split_gqa_graph_structure_invalid_2.png)

path: "./before_split_gqa_graph_structure_invalid_2.graph"

Reason: This is invalid because the split boundary is not clear. the matmul uses split, but the "8: Reduction 2" follows a non split path. This would make it hard to horizontally fuse the split from the matmul with the non-split path from the reduction. if they are horizontal dependencies (in the future will merge, but do not depend on each other), they need to share the same splitting logic to eventually merge them together.


## Invalid 3

![](./images/before_split_gqa_graph_structure_invalid_3.png)

path: "./before_split_gqa_graph_structure_invalid_3.graph"

Reason: This is invalid because the split boundary is not clear. the reduction consumes the split, but the matmul follows a non split path. This would make it hard to horizontally fuse the split from the matmul with the non-split path from the reduction. if they are horizontal dependencies (in the future will merge, but do not depend on each other), they need to share the same splitting logic to eventually merge them together.

## Invalid 4

![](./images/before_split_gqa_graph_structure_invalid_4.png)

path: "./before_split_gqa_graph_structure_invalid_4.graph"

Reason: The split is redudanct, it duplicates the Exp op.

## Invalid 5

![](./images/before_split_gqa_graph_structure_invalid_5.png)

path: "./before_split_gqa_graph_structure_invalid_5.graph"

Reason: The split is redudanct, it duplicates the Exp op. On top of that, the split boundary is not clear. the matmul uses split, but the "9: Reduction 2" follows a non split path. This would make it hard to horizontally fuse the split from the matmul with the non-split path from the reduction. if they are horizontal dependencies (in the future will merge, but do not depend on each other), they need to share the same splitting logic to eventually merge them together.

## Invalid 6

![](./images/before_split_gqa_graph_structure_invalid_6.png)

path: "./before_split_gqa_graph_structure_invalid_6.graph"

Reason: The split consumes Exp, but then Exp is used again as non split which makes it redudanct.

## Invalid 7

![](./images/before_split_gqa_graph_structure_invalid_7.png)

path: "./before_split_gqa_graph_structure_invalid_7.graph"

Reason: This is invalid because the split boundary is not clear. the reduction consumes the split, but the matmul follows a non split path. This would make it hard to horizontally fuse the split from the matmul with the non-split path from the reduction. if they are horizontal dependencies (in the future will merge, but do not depend on each other), they need to share the same splitting logic to eventually merge them together.


## Invalid 8

![](./images/before_split_gqa_graph_structure_invalid_8.png)

path: "./before_split_gqa_graph_structure_invalid_8.graph"

Reason: The split is redudanct, it duplicates the Add and Exp op.


## Invalid 9

![](./images/before_split_gqa_graph_structure_invalid_9.png)

path: "./before_split_gqa_graph_structure_invalid_9.graph"

Reason: The split is redudanct, it duplicates the Exp op.


# Valid Graphs:

## Valid 1

![](./images/before_split_gqa_graph_structure_valid_1.png)

path: "./before_split_gqa_graph_structure_valid_1.graph"

Reason: Valid. Boundary is clear on reductions 11 and 12. 2 kernels only.


## Valid 2

![](./images/before_split_gqa_graph_structure_valid_2.png)

path: "./before_split_gqa_graph_structure_valid_2.graph"

Reason: Valid. Boundary is clear on reductions 11 and 12. 2 kernels only.


## Valid 3

![](./images/before_split_gqa_graph_structure_valid_3.png)

path: "./before_split_gqa_graph_structure_valid_3.graph"

Reason: Valid. Boundary is clear on reductions 11 and 12. 2 kernels only.


## Valid 4

![](./images/before_split_gqa_graph_structure_valid_4.png)

path: "./before_split_gqa_graph_structure_valid_4.graph"

Reason: Valid. Boundary is clear on reductions 11 and 12. 2 kernels only.


## Valid 5

![](./images/before_split_gqa_graph_structure_valid_5.png)

path: "./before_split_gqa_graph_structure_valid_5.graph"

Reason: Valid. Boundary is clear on reductions 11 and 12. 2 kernels only.


## Valid 6

![](./images/before_split_gqa_graph_structure_valid_6.png)

path: "./before_split_gqa_graph_structure_valid_6.graph"

Reason: Valid. No splitting exists. single kernel.


## Valid 7

![](./images/before_split_gqa_graph_structure_valid_7.png)

path: "./before_split_gqa_graph_structure_valid_7.graph"

Reason: Valid. No splitting exists. single kernel.
