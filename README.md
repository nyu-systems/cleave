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

Search, compile and benchmark a matmul kernel:

```bash
python demo/matmul.py --preset fast --max-parallel 8
python demo/matmul.py --split K:4,8,16     # also search split-K partial/combine plans
```

Every demo is a few lines: the operator written as a `@cleave.jit` function over
`cleave.lang`, then `cleave.examples.main(...)` with its shapes, dtype, split and presets.
All demos share the same flags (`python demo/<name>.py --help`):

- one flag per dimension (`--M 256 --K 8192`), `--preset fast|default|exhaustive` (plus
  demo-specific presets), `--topk`, `--split DIM:f1,f2` (`none` disables), `--max-ops`,
  `--fusion auto|force|off`, `--max-parallel`, `--warmup/--iters`, `--no-cache`,
  `--save DIR`, `-v`, `--json`.

| Demo | Operator |
| --- | --- |
| `matmul.py` | `X @ W`; `--split K:4,8,16` for split-K |
| `rmsnorm_matmul.py` | `RMSNorm(X) @ W`; `--preset split-k` for the split-K shapes |
| `swiglu.py`, `lora.py`, `matmul_rowsum.py` | SwiGLU, LoRA (`cleave.Program` builder), row-sum |
| `matmul_matmul.py` | `(Q K^T) V`; `--preset split` for the split-KV decode shapes |
| `{mha,gqa,mqa}_decode.py` | decode attention, split along `SKV` by default |
| `mha_decode_exact.py` | decode attention with the exact `exp(s - max(s)) / sum` softmax |
| `{mha,gqa,mqa,mla}_prefill.py`, `mla_decode.py`, `qknorm.py` | prefill / MLA / QK-norm attention |
| `staged.py` | the pipeline one stage at a time (research view) |

`demo/dynamic/` holds the paged-KV dynamic-shape demos.

## Search-to-Kernel Path

Cleave follows a graph-first flow: describe a tensor program, ask the superoptimizer for
equivalent graph structures, search schedules for each, then compile, validate and
benchmark the generated TileLang kernels. `cleave.compile` runs the whole pipeline; each
stage is also a public function:

```python
import cleave
import cleave.lang as cl
from cleave.pipeline import schedule


@cleave.jit
def matmul(x: cl.Tensor["M", "K"], w: cl.Tensor["K", "N"]):
    return x @ w


def main():
    kernel = cleave.compile(matmul, shapes=dict(M=128, N=128, K=4096), dtype=cl.f16, preset="fast")
    print(kernel.report.summary())

    # The same compile, stage by stage.
    prog = cleave.trace(matmul, dict(M=128, N=128, K=4096), cl.f16)
    graphs = cleave.superoptimize(prog, cleave.SuperoptConfig(split=("K",)))
    plans = schedule(graphs, cleave.ScheduleConfig(topk=100, split_k=(4, 8, 16)))
    report = cleave.evaluate(cleave.build(plans, workers=8))
    print(report.table(top=5, by="variant"))
    print(report.best())


if __name__ == "__main__":  # candidate workers are spawned processes
    main()
```

See `demo/staged.py` for a complete runnable version.

## Development

Run the Python test suite:

```bash
pip install pytest pytest-xdist
pytest
```

Many end-to-end tests compile and run CUDA kernels. For local iteration, prefer
targeted tests first:

```bash
pytest tests/integration
pytest tests/superoptimizer/python
pytest -m "not gpu"   # markers: gpu, expected, search, slow
```

Generated code is pinned two ways: `tests/graph/handwritten/expected_kernels`
(hand-written schedules, checked by `test_expected_codegen.py`) and
`tests/pipeline/snapshots` (hashes of every candidate from search through
codegen). Regenerate them after an intended change with `--update-expected` or
`--update-snapshots`, and review the diff.

Useful environment variables:

- `MAX_JOBS`: limits source-build parallelism during installation.
- `CLEAVE_MAX_PARALLEL`: used by several schedule tests to cap candidate runs.
- `CLEAVE_CACHE_ROOT`: overrides Cleave's generated runtime/cache directory.
- `COMPILATION_PROGRESS_LOG=1`: prints long-running candidate compilation
  progress.

## Project Layout

- `python/cleave/`: the Python package: `frontend`/`lang` (tracing), `ir` (graph, ops),
  `lower` (superoptimizer graphs to IR), `schedule`, `codegen`, `runtime` (compile, validate,
  time), `pipeline` (the staged API), `dynamic` (paged/ragged kernels) and `core` (settings,
  target, `Session`).
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
