from __future__ import annotations

import os
import shutil

from cleave.runtime.compile import CompilerPassOptions
from cleave.runtime.evaluate import run_candidates


MAX_PARALLEL = max(1, int(os.getenv("CLEAVE_MAX_PARALLEL", "32")))


def run_end2end_candidates(
    name: str,
    candidates,
    dtype: str,
    *,
    min_success: int,
    log: bool = True,
    compiler_pass_options: CompilerPassOptions | None = None,
):
    """Compile/validate scheduled candidates and assert a minimum success count.

    Correctness is checked by the graph executor against ``candidates[0].flat_graph``
    (the pre-tiling source graph), so no ``op_name``/``dims`` are needed.
    """
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "log", name)
    if os.path.exists(log_dir):
        shutil.rmtree(log_dir)

    results = run_candidates(candidates, dtype, max_parallel=MAX_PARALLEL,
                             compiler_pass_options=compiler_pass_options)
    ok = [r for r in results if r.ok]
    failed = [r for r in results if not r.ok]

    if failed and log:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "errors.err"), "w") as f:
            f.write(f"# {len(failed)} graphs failed\n\n")
            for i, r in enumerate(failed):
                f.write(f"--- Error {i + 1} ---\n")
                f.write(f"RStep Map: {r.cand.rstep_map}\nOutput tile: {r.cand.output_tile}\n")
                f.write(f"Candidate:\n{r.cand}\n\nError:\n{r.error}\n\n")

    if ok and log:
        os.makedirs(log_dir, exist_ok=True)
        with open(os.path.join(log_dir, "success.ok"), "w") as f:
            f.write(f"# {len(ok)} graphs succeeded\n\n")
            for i, r in enumerate(ok):
                f.write(f"--- Success {i + 1} ---\n")
                f.write(f"RStep Map: {r.cand.rstep_map}\nOutput tile: {r.cand.output_tile}\n")
                f.write(f"Candidate:\n{r.cand}\n\nTilelang Code:\n{r.code}\n\nCUDA Code:\n{r.cuda_src}\n\n")

    assert len(ok) >= min_success, (
        f"{len(ok)} scheduled graphs compiled and ran successfully, expected >= {min_success}."
    )
    return results
