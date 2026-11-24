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
| `{mha,gqa,mqa}_decode.py` | decode attention, split along `SKV` (the split-KV of FlashDecoding; below) |
| `{mha,gqa,mqa,mla}_prefill.py`, `mla_decode.py`, `qknorm.py` | prefill / MLA / QK-norm attention |
| `mha_backward.py` | attention backward: `dQ, dK, dV` in one fused kernel (declared inputs `L`, `o`) |
| `staged.py` | the pipeline one stage at a time (research view) |

`--split` is parsed by every demo, but not every demo has a split run: `mla_prefill.py` and
`qknorm.py` declare a single tolerance and offer none (their docstrings say why),
`mla_decode.py` asks for none, and `mha_backward.py` runs unsplit (a program searched one
output at a time takes no split). The search does not split a graph whose max reduces the split
dim (`--split SKV` of an attention demo); the split is built from the loop state of each solo
graph instead: each partial returns the accumulators of its split and the max they are
relative to, and the combine takes the max over the splits, repairs every partial by the
proven update rule, sums and divides. A graph that has no such pair (the traced order, which
needs two passes over the row) is reported with the reason.

`demo/dynamic/` holds the paged-KV dynamic-shape demos.

An input can say what it is. `cl.Tensor[..., cl.Equals(fn)]` declares an input that equals `fn`
of earlier inputs: the program need not read it, and the search may read it through its
definition (the forward output `o` of `mha_backward.py`, which lets it replace
`sum_SKV(P * dP)` by `D = sum_D(dO * o)`). `cl.Tensor[..., cl.Sample(fn)]` only says how
validation draws the input (the forward's log-sum-exp `L`, from the drawn `q` and `k`); the
search treats it as free. A program with an `Equals` input and several outputs is searched
one output at a time (`SuperoptConfig(per_output="auto")`, the default) and one graph per
output is merged into one, equal ops shared. Where the outputs reduce over orthogonal axes
(the backward's `dQ` over `S_KV`, `dK` and `dV` over `S_Q`), a thread block owns a tile of one
axis and loops over the other: the outputs of that loop accumulate in registers, the others
across thread blocks with atomic adds into a float32 buffer, and both assignments are searched
and ranked. A value the loop does not change (`D`) is computed on its first iteration and read
by the others.

`cl.softmax` is the exact softmax, `exp(s - max(s)) / sum`, and records its max: the max is an
ordinary reduction of the flat graph the superoptimizer searches
(`cl.exp(s - cl.max(s, -1, keepdim=True))` written by hand is the same program). Where a tile
holds the row, the row max is computed in place. Where a tiling loops over the row
(FlashAttention), the loop carries the max across its iterations, and every sum and matmul
it accumulates under the max is repaired before it accumulates, `acc = acc * F(m, m') + tile`
(`m` the max before the iteration, `m'` after it). No `F` is written anywhere: once per loop
structure, `cleave.pipeline.update` evaluates the program's own flat graph on prefixes of the
looped dim, enumerates small update expressions per accumulator, proves every one that
matches over the reals with z3 (`exp` with its law `exp(a + b) = exp(a) exp(b)`), and ranks
the proven ones by accuracy, stability and cost; for attention it finds `exp(m - m')` for
the row sum and for the value matmul alike. The repair runs at every iteration. A loop whose
accumulators get no proven rule keeps the row in one block, and the refusal gives the
reason. The paged kernels carry a hand-placed running max. Where the softmax is contracted
over its row (a sum or a matmul, as in attention's `softmax(...) @ V`) this holds at any row
length. A program that returns the softmax itself (a standalone softmax, attention weights)
is exact too, but is served only while its row fits one tile: its exp is read by the
division, which is not summed over the row, so the max is not carried across a loop (the
refusal says so). A float16 softmax of 64 rows that is returned has plans up to 8192 columns
and none at 16384, where the only tilings that hold a row in one block hold one row.
Shapes that have no kernel yet (each is refused with its reason, or fails in TileLang, never
served wrongly):

- attention with a single query row (`S_Q = 1`): a unit dim that is not a batch dim has no tile
  (and the value matmul would read its operand from the exp's registers with one row a warp,
  which TileLang cannot lay out: the refusal names that, since no thread count is proposed);
- an output that reduces every dim to one element (`sum(x, 0)` of a vector, the total of a
  matrix): its tile has no dim, for the same reason;
- a matmul that reads its first operand from another matmul's accumulator (attention's
  `P @ V`) with fewer than 16 rows a warp, or over two warp groups at another M than 128
  unless the operand is staged through shared memory: 64 query rows have no 256-thread tiling
  (M is the rows of one iteration: for the backward's `dQ` summed across thread blocks, one
  step of the `S_Q` loop);
- a sum or max over the rows of a matmul's accumulator (a softmax over the queries,
  `dim=-2`) that would give a warp more than 16 of those rows, which TileLang's reduction
  cannot lower: attention normalized over 64 queries has plans, over 256 none;
- for 16-bit dtypes, a matmul that contracts a length no multiple of 16 divides (float16
  attention at `S_KV = 1000`), and for float32 one no multiple of 8 divides;
- a row that a tile has to hold whole and whose length is not a power of two (a standalone
  softmax or a row sum over 96 or 257 columns): the last tile of a reduction is not padded;
- a matmul output axis with no step a tensor-core tile takes (`M = 100`, whose steps are 1,
  2, 4 and 100): an output tile is not padded. (`M = 96` starts at 32 rows, `N = 48` at 16
  columns.) Tilings TileLang cannot partition or lay out are not proposed: fewer than 16 rows
  or 8 columns a warp, more than 64 rows that 64 does not divide, or, on 256 threads, fewer
  than 128 rows by columns 32 does not divide; attention over 96 queries has no tiling left;
- a reduction over a dim of one element (`sum(x, -1)` of a `64 x 1` tensor): the search
  builds no reduction over a single element, so `superoptimize` finds no graph and raises
  `SuperoptError` with that reason (as it does for any search that finds none, such as a
  `max_ops` below the program's own op count);
- one value returned twice, or returned and also read by another op (`x @ w` next to
  `exp(x @ w)`): the search counts as outputs only the tensors nothing reads and adds no op
  twice to the same operands, so it finds such a program only where it computes the value a
  second time in another form, and otherwise raises `SuperoptError` saying so.

A program with an input that no op reads is a `SpecError` in `superoptimize` (an `Equals`
input aside: the search reads it through its definition): the search counts a tensor that
nothing reads as an output, so none of its graphs could equal the program.
So is a program that returns one of its inputs as it is: no op writes that output.

`cl.softmax(x)` next to another `exp` of the same tensor is served under the limit of a
softmax that is returned: the softmax's exp reads `x - max(x)`, a value of its own that the
search keeps apart from `exp(x)`, and the row has to fit one tile. `cl.softmax(x) + cl.exp(x)`
has plans for float16 rows of 4096 and none at 8192 (where a softmax alone still has one).
Two softmaxes of `x` over different axes need a row and a column in one tile, which is the
whole tensor: 16x32 and 64x128 have plans, 64x256 has none.

Kernels of a graph whose softmax is exact (an `exp` that reads `x - max(x)`, whoever wrote it)
are validated on adversarial logits as well as on normal ones, wherever the logits can be
shaped through an input: every attention demo but
`qknorm.py`, whose logits are computed from the normalized `q` and `k` (its report says so,
as it does for any softmax of a program that no case shapes). A paged program gets its
adversarial workloads when it is one of the registry operators (`cleave.dynamic.registry`)
compiled on its registry workloads; any other paged program, and any `workloads=` of the
caller (the paged demos' `--synthetic`), is validated on the workloads as given.

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
    # Every candidate is validated against a torch reference at the tolerance the operator
    # declares (row-relative; there is no default).
    tolerance = cl.Tolerance(rtol=5e-3)
    kernel = cleave.compile(matmul, shapes=dict(M=128, N=128, K=4096), dtype=cl.f16, preset="fast",
                            tolerance=tolerance)
    print(kernel.report.summary())

    # The same compile, stage by stage.
    prog = cleave.trace(matmul, dict(M=128, N=128, K=4096), cl.f16)
    graphs = cleave.superoptimize(prog, cleave.SuperoptConfig(split=("K",)))
    plans = schedule(graphs, cleave.ScheduleConfig(topk=100, split_k=(4, 8, 16)))
    report = cleave.evaluate(cleave.build(plans, workers=8,
                                          config=cleave.EvalConfig(tolerance=tolerance)))
    print(report.table(top=5, by="variant"))
    print(report.best())


if __name__ == "__main__":  # candidate workers are spawned processes
    main()
```

See `demo/staged.py` for a complete runnable version.

### Validation

A candidate kernel is kept only if its output matches a same-dtype torch reference:
`|actual - ref| <= rtol * max|ref| + atol`, where the max is taken over the entry's *row* (the
last non-unit axis), so a row of small values is not excused by an absolute slack and an
all-zero row must be reproduced exactly. Any NaN fails, an infinity must be exactly the one
the reference has, and shapes and dtypes must agree.

There is no default tolerance. Each operator declares its own `cl.Tolerance(rtol=..., atol=...)`
(one, or one per output when every plan has that many: a program with split plans declares
one, because a split partial has more outputs than its combine; a split partial that also
returns a float32 max gets a bound of its own by boundary role,
`{"": cl.Tolerance(...), "shift": cl.Tolerance(...)}`, which the MHA, GQA and MQA decode and
prefill demos declare for the split of their softmax), set from its measured error
(the next 1-2-5 step at or above 4x the largest error of its valid kernels, 2x for a bf16
output, whose single ulp is already 0.8% of the row max; valid kernels sit within a few ulp of
the reference, wrong kernels above 10% of the row max): `cleave.compile(..., tolerance=...)`,
`@cleave.jit(tolerance=...)`, `cleave.EvalConfig(tolerance=...)`; the demos pass theirs to
`cleave.examples.main`, the paged operators declare `(out, lse)` in `cleave.dynamic.registry`.
The report carries the error every candidate measured (`report.to_dict()["validation"]`,
`--json`) and counts validation failures as `OutputMismatch`, `NonFiniteOutput` or
`LaunchCrash`. `python -m tests._measure_tolerances --gpus 0,1 --out DIR` re-measures every
operator and prints the tolerance that follows; `--pin` writes the measurement to
`tests/measured_errors.json`, which the tests hold every declared tolerance to.

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

### Rebuilding the native search

`cleave.superoptimizer.core` is a compiled extension (`python/cleave/superoptimizer/core*.so`,
not tracked): the C++ search under `src/` and `include/`, bound by
`python/cleave/superoptimizer/_cython/core.pyx` and built as `CMakeLists.txt` says. It is
built once by the install and is not rebuilt on import, so after any change or pull that
touches those files, run the install again from the repository root:

```bash
pip install -e . -v
```

(the command of Quick Installation; `MAX_JOBS` applies). An editable install always relinks
the extension.

The extension needs `libz3.so.4.15` and records, as the place to look for it, the `z3/lib`
directory of the `z3-solver` it was built against. Which one that is depends on the command:

- `pip install -e . -v` builds in a temporary environment that pip fills with the build
  requirements of `pyproject.toml` and deletes afterwards. The recorded directory is gone
  with it, so `ldd python/cleave/superoptimizer/core*.so` prints
  `libz3.so.4.15 => not found` (unless the loader finds a copy somewhere else). This is not
  a broken install: when the extension does not find the library,
  `python/cleave/superoptimizer/__init__.py` loads the `libz3.so.4.15` of the installed
  `z3-solver` and imports the extension again.
- `pip install -e . -v --no-build-isolation`, in an environment that already holds those
  build requirements (`setuptools>=77.0`, `wheel`, `cmake>=3.24`, `cython>=0.28`,
  `z3-solver==4.15.4`), builds against that environment's z3. `ldd` then resolves
  `libz3.so.4.15` to `.../site-packages/z3/lib/libz3.so.4.15` of that environment.

The check that holds after either command is the import, in the environment the tests run in:

```bash
python -c "import cleave.superoptimizer.core as core; print(core.__file__)"
```

It prints the path of the extension. It fails with an `ImportError` that names
`libz3.so.4.15` when the `z3-solver` of that environment is missing or is not a 4.15 release
(the library name carries the minor version; the install pins `z3-solver==4.15.4`).

With a library older than the sources the search still runs, with the behaviour of the old
sources: it is not reproducible across processes and saved checkpoints are searched again.
Check the library that is loaded with

```bash
pytest -n 0 tests/superoptimizer/python/search/test_search_reproducible.py
```

(each failure names the file it loaded). Every cache key holds a digest of the compiled
extension, so nothing saved with another build is reused.

The C++ unit tests (`tests/superoptimizer/cpp`) are not part of the pytest run:

```bash
cmake -S . -B build/native_tests -DBUILD_TESTS=ON
cmake --build build/native_tests --parallel
ctest --test-dir build/native_tests --output-on-failure
```

CMake takes the z3 headers and library from the `z3-solver` of the Python interpreter it
finds (`find_package(Python3)`); `-DZ3_CXX_INCLUDE_DIRS=...` and `-DZ3_LIBRARIES=...` name
another.

Useful environment variables:

- `MAX_JOBS`: limits source-build parallelism during installation.
- `CLEAVE_MAX_PARALLEL`: used by several schedule tests to cap candidate runs.
- `CLEAVE_CACHE_ROOT`: overrides Cleave's generated runtime/cache directory.
- `COMPILATION_PROGRESS_LOG=1`: prints long-running candidate compilation
  progress.

## Project Layout

- `python/cleave/`: the Python package: `frontend`/`lang` (tracing), `ir` (graph, ops),
  `lower` (superoptimizer graphs to IR), `schedule`, `codegen`, `runtime` (compile, validate,
  time), `pipeline` (the staged API; `pipeline/update` searches and proves the update rules
  of a carried max), `dynamic` (paged/ragged kernels) and `core` (settings, target,
  `Session`).
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
