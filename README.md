<p align="center">
  <img src="assets/logo.png" alt="Cleave logo" width="300">
</p>

Cleave is an ML compiler that automatically generates high-performance
GPU kernels. It separates symbolic algebraic optimization from operator
scheduling, making the search for efficient fused kernels more scalable.

## Quick Installation

Install from source with submodules, following the same clone style used by
Mirage MPK:

```bash
git clone --recursive https://github.com/davidpissarra/cleave
cd cleave
pip install -e . -v
```

If the repository was cloned without `--recursive`, initialize submodules before
installing:

```bash
git submodule update --init --recursive
```

For resource-constrained machines, cap build parallelism with `MAX_JOBS`:

```bash
MAX_JOBS=4 pip install -e . -v
```

`MAX_JOBS` is forwarded to the CMake superoptimizer build and `MAKEFLAGS`, so
the native build does not spawn more compiler jobs than the machine can
comfortably handle.

## Requirements

- Python 3.8+
- CMake 3.24+
- A C++17 compiler with OpenMP support
- CUDA-capable environment supported by TileLang
- Git submodules for vendored dependencies

The root package installs the vendored TileLang package and builds
`cleave.superoptimizer.core`. Z3 is provided through the `z3-solver` Python
package; nlohmann JSON headers come from `3rdparty/json`.

## Quickstart

Run a small matmul search and benchmark:

```bash
python demo/matmul.py --topk 100 --max_parallel 8
```

Other demos cover fused RMSNorm+MatMul, SwiGLU, Split-K matmul, MHA, MQA, GQA,
and MLA patterns:

```bash
ls demo
```

Use `--max_parallel` on demos to limit concurrent candidate compilation and GPU
execution.

## Search-to-Kernel Path

Cleave follows a graph-first flow: describe a tensor graph, ask the
superoptimizer for equivalent graph structures, convert them into Cleave graphs,
search schedules, then compile and benchmark generated TileLang kernels.

```python
import cleave.superoptimizer as csopt

from cleave.integration import graph_structure_to_cleave
from cleave.schedule import search_tiling
from cleave.util import compile_and_run_candidates_parallel

M, N, K = 128, 128, 4096

graph = csopt.new_tile_graph(split_enabled=[True, True])
x = graph.create_tiled_input((M, K), (False, True))
w = graph.create_tiled_input((K, N), (True, False))
out = graph.matmul(x, w)
graph.create_tiled_output(out)

symbolic_graph, symbol_map = csopt.create_symbolic_tile_graph(graph)
config = symbolic_graph.setup_search_config()
config.search_thread = 16

optimized = symbolic_graph.search_tile_graph(
    "checkpoint_matmul.json",
    False,
    True,
    config,
)

candidates = []
for opt_graph in optimized:
    graph_structure = opt_graph.get_graph_structure()
    cleave_graph = graph_structure_to_cleave(graph_structure, symbol_map=symbol_map)
    candidates.extend(search_tiling(cleave_graph, topk=100))

success_info, error_graphs, errors = [], [], []
compile_and_run_candidates_parallel(
    candidates,
    [M, N, K],
    "float16",
    "matmul",
    max_parallel=8,
    success_info=success_info,
    error_graphs=error_graphs,
    errors=errors,
)
```

See `demo/matmul.py` and `demo/rmsnorm_matmul.py` for complete runnable
versions, including benchmarking.

## Development

Run the Python test suite:

```bash
pip install pytest pytest-rerunfailures
pytest
```

Many end-to-end tests compile and run CUDA kernels. For local iteration, prefer
targeted tests first:

```bash
pytest tests/integration
pytest tests/superoptimizer/python
```

Useful environment variables:

- `MAX_JOBS`: limits source-build parallelism during installation.
- `CLEAVE_MAX_PARALLEL`: used by several schedule tests to cap candidate runs.
- `CLEAVE_CACHE_ROOT`: overrides Cleave's generated runtime/cache directory.
- `COMPILATION_PROGRESS_LOG=1`: prints long-running candidate compilation
  progress.

## Project Layout

- `python/cleave/`: Python graph, scheduling, codegen, integration, and runtime
  helpers.
- `src/superoptimizer/`: C++ superoptimizer implementation.
- `include/cleave/`: C++ public headers.
- `demo/`: runnable search/codegen examples.
- `tests/`: integration, reference, schedule, runtime, and superoptimizer tests.
- `3rdparty/`: vendored dependencies.

## Contributing

Issues, performance reports, and focused pull requests are welcome. Please keep
changes scoped, include tests for behavioral changes, and note any CUDA or GPU
assumptions needed to reproduce results.

## Citation

If you use Cleave in your research, please cite our paper:

```bibtex
@misc{pissarra2026cleave,
  title         = {Cleave: Scaling Tensor Program Optimization via Decoupled Algebraic Search and Operator Scheduling},
  author        = {David Pissarra and Jinkun Lin and Haitian Jiang and Aurojit Panda and Jinyang Li},
  year          = {2026},
  eprint        = {2610.07742},
  archivePrefix = {arXiv},
  primaryClass  = {cs.PL},
  url           = {https://arxiv.org/abs/2610.07742}
}
```

## License

Cleave is licensed under the [Apache-2.0](LICENSE) license.
