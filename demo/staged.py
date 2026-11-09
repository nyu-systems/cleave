"""The compile of rmsnorm_matmul.py, one stage at a time (the research view of Cleave).

``cleave.compile(fn, ...)`` is ``evaluate(build(schedule(superoptimize(trace(fn))))).best()``;
every stage is a public function with a printable result.

    python demo/staged.py --topk 50 --max-parallel 8
"""
import argparse

import cleave
import cleave.lang as cl
from cleave.pipeline import schedule
from rmsnorm_matmul import rmsnorm_matmul


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--topk", type=int, default=200)
    ap.add_argument("--max-parallel", type=int, default=8)
    ap.add_argument("--save", default=None, metavar="DIR")
    args = ap.parse_args()

    prog = cleave.trace(rmsnorm_matmul, dict(M=128, N=6144, K=4096), cl.f16)
    print(prog)  # inputs, dims, op list

    # 1. algebraic search (C++ superoptimizer), cached by content hash
    graphs = cleave.superoptimize(prog, cleave.SuperoptConfig(split=("K",)))
    print(graphs)

    # 2. tile search: solo and split-K come out as uniform Plans
    plans = schedule(graphs, cleave.ScheduleConfig(topk=args.topk, split_k=(4, 8, 16, 32)))
    print(plans.summary())

    # 3. codegen + compile + validate in worker processes; 4. benchmark
    built = cleave.build(plans, workers=args.max_parallel)
    print(built)
    report = cleave.evaluate(built, cleave.EvalConfig(warmup=16, iters=1000))
    print(report.summary())
    print(report.table(top=5, by="variant"))
    best = report.best()
    print(best)
    if args.save:
        print("saved:", best.save(args.save))


if __name__ == "__main__":  # candidate workers are spawned processes that re-import __main__
    main()
