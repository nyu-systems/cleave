"""The prover's controls (DESIGN 7, 13.2 a): every update of the table is put to z3 on the
contributions read off three real flat graphs -- the row sum ``l`` and the value matmul
``acc`` of the flash form, and an accumulator of the split combine -- and a wrong one fails
the obligation that names what is wrong with it. CPU only, no search.
"""

import functools
from fractions import Fraction

import pytest

z3 = pytest.importorskip("z3")
pytest.importorskip("torch")

from cleave.ir.ops import Matmul, ReduceSum
from cleave.lower.state_split import combine_graph
from cleave.pipeline.update import proof
from cleave.pipeline.update.proof import OBLIGATIONS, contribution, prove

from ._attention import FACTOR, record_by_hand
from ._programs import flash_attention, identity, lowered, scaled_rows, tenth_rows

D, ZERO, ONE, TWO = OBLIGATIONS  # (D) defined, (0) accumulate form, (i) one element, (ii)


@functools.lru_cache(maxsize=None)
def flash_ir():
    ir = lowered(identity(flash_attention, dict(H=2, SQ=3, SKV=8, D=4)))
    full, record = record_by_hand(ir)
    return ir, full.name, record


@functools.lru_cache(maxsize=None)
def combine_ir():
    ir, full, record = flash_ir()
    return combine_graph(ir, full, record.rules, 4, split_pos=1, out_dtype="float16")


def site(which):
    """``(flat graph, full, acc)`` of one of the three contributions."""
    ir, full, record = flash_ir()
    by_type = {type(n): n.name for n in ir.nodes if n.name in record.loops}
    if which == "l":
        return ir, full, by_type[ReduceSum]
    if which == "acc":
        return ir, full, by_type[Matmul]
    combined = combine_ir()
    acc = next(n.name for n in combined.nodes if n.name == f"{by_type[Matmul]}_combined")
    return combined, "state_max", acc


SITES = ("l", "acc", "comb")

e = ("exp", ("sub", "m", "m'"))
einv = ("exp", ("sub", "m'", "m"))
RIGHT = {
    "canonical      r + a*exp(m-m')": ("add", "r", ("mul", "a", e)),
    "commuted       exp(m-m')*a + r": ("add", ("mul", e, "a"), "r"),
    "unstable       r + a/exp(m'-m)": ("add", "r", ("div", "a", einv)),
    "ratio          r + a*(exp(m)/exp(m'))": (
        "add", "r", ("mul", "a", ("div", ("exp", "m"), ("exp", "m'")))),
    "ratio          r + (a*exp(m))/exp(m')": (
        "add", "r", ("div", ("mul", "a", ("exp", "m")), ("exp", "m'"))),
    "ratio          r + exp(m)*(a/exp(m'))": (
        "add", "r", ("mul", ("exp", "m"), ("div", "a", ("exp", "m'")))),
    "needs L3       r + a*exp(m)*exp((m-m)-m')": (
        "add", "r", ("mul", ("mul", "a", ("exp", "m")), ("exp", ("sub", ("sub", "m", "m"), "m'")))),
    "deeper         r + a*e*e*exp(m'-m)": ("add", "r", ("mul", ("mul", ("mul", "a", e), e), einv)),
    "deeper         r + a*exp((m+m)-m')/exp(m)": (
        "add", "r", ("div", ("mul", "a", ("exp", ("sub", ("add", "m", "m"), "m'"))), ("exp", "m"))),
    "zero term      r + a*exp(m-m') + (m - m)": ("add", "r", ("add", ("mul", "a", e), ("sub", "m", "m"))),
}
# update, obligation it must fail (on every contribution)
WRONG = {
    "no repair      r + a": (("add", "r", "a"), ONE),
    "wrong sign     r + a*exp(m'-m)": (("add", "r", ("mul", "a", einv)), ONE),
    "tile only      r": ("r", ONE),
    "not separable  (r + a)*exp(m-m')": (("mul", ("add", "r", "a"), e), ZERO),
    "0/0 at a=0     a*((r/a) + exp(m-m'))": (("mul", "a", ("add", ("div", "r", "a"), e)), D),
    "0/0 at a=0     r + exp(m)/(exp(m')/a)": (
        ("add", "r", ("div", ("exp", "m"), ("div", ("exp", "m'"), "a"))), D),
    "0/0 at a=0     r + (a/a)*(a*exp(m-m'))": (
        ("add", "r", ("mul", ("div", "a", "a"), ("mul", "a", e))), D),
    "0/0 at rest    r + a*exp(m-m')*((m'-m)/(m'-m))": (
        ("add", "r", ("mul", ("mul", "a", e), ("div", ("sub", "m'", "m"), ("sub", "m'", "m")))), D),
    "0/0 at r=0     r + a*exp(m-m')*(r/r)": (("add", "r", ("mul", ("mul", "a", e), ("div", "r", "r"))), D),
    "non-additive   r + a*a*exp(m-m')": (("add", "r", ("mul", ("mul", "a", "a"), e)), TWO),
    "square         r + a*exp(m-m')*exp(m-m')": (("add", "r", ("mul", ("mul", "a", e), e)), ONE),
    "affine         r + a*exp(m-m') + (m'-m)": (("add", ("add", "r", ("mul", "a", e)), ("sub", "m'", "m")), ONE),
}


def test_the_contributions_are_read_off_the_flat_graphs():
    """Nobody writes them down: ``exp(s - m)`` for the row sum, ``exp(s - m) * v`` for the
    value matmul, ``x_l * exp(x_m - M)`` for a combined accumulator."""
    m = z3.Real("m")
    exp = proof._exp()
    built = {}
    for which in SITES:
        build, leaves = contribution(*site(which))
        built[which] = (build(m), {name: z3.Real(f"x_{name}") for name in leaves})
    (score,) = built["l"][1].values()
    assert proof.valid(built["l"][0] == exp(score - m)) == "unsat"
    others = [x for name, x in built["acc"][1].items() if not x.eq(score)]
    assert len(built["acc"][1]) == 2 and len(others) == 1
    assert proof.valid(built["acc"][0] == exp(score - m) * others[0]) == "unsat"
    xs = built["comb"][1]
    assert set(xs) == {"PlaceHolder_state_max", next(n for n in xs if n != "PlaceHolder_state_max")}
    (part,) = [x for n, x in xs.items() if n != "PlaceHolder_state_max"]
    assert proof.valid(built["comb"][0] == part * exp(xs["PlaceHolder_state_max"] - m)) == "unsat"
    # ... and it is not something else
    assert proof.valid(built["l"][0] == exp(m - score)) == "sat"


@pytest.mark.parametrize("which", SITES)
@pytest.mark.parametrize("label", list(RIGHT))
def test_a_right_update_is_proved(which, label):
    verdict = prove(RIGHT[label], *site(which))
    assert verdict.proved, verdict
    assert verdict.results == {name: "unsat" for name in OBLIGATIONS}


@pytest.mark.parametrize("which", SITES)
@pytest.mark.parametrize("label", list(WRONG))
def test_a_wrong_update_fails_the_obligation_that_names_it(which, label):
    update, obligation = WRONG[label]
    verdict = prove(update, *site(which))
    assert not verdict.proved
    assert verdict.results[obligation] == "sat", verdict.results


@pytest.mark.parametrize("label", [k for k, (_, o) in WRONG.items() if o == D and "a=0" in k])
def test_definedness_is_the_only_thing_wrong_with_a_zero_over_zero(label):
    """z3's division is total: ``r + (a/a)*(a*exp(m - m'))`` passes (0), (i) and (ii) on
    the row sum -- it is right wherever it is defined -- and only (D) rejects it."""
    update, _ = WRONG[label]
    verdict = prove(update, *site("l"))
    if label.startswith("0/0 at a=0     r + (a/a)"):
        assert verdict.failed() == [D]
    assert verdict.results[D] == "sat"


def test_the_decomposition_lemmas_are_needed(monkeypatch):
    """``r + a*exp(m)*exp((m - m) - m')`` is proved only with the L3 instances."""
    update = RIGHT["needs L3       r + a*exp(m)*exp((m-m)-m')"]
    assert prove(update, *site("l")).proved
    monkeypatch.setattr(proof, "linear", lambda term: None)
    verdict = prove(update, *site("l"))
    assert not verdict.proved and verdict.results[ONE] != "unsat"
    assert prove(RIGHT["canonical      r + a*exp(m-m')"], *site("l")).proved  # L1 suffices


def test_a_constant_of_the_flat_graph_is_the_exact_rational_it_is():
    """``exp((x - m) / 8)``: the update must divide by 8 as well; ``0.1`` is not ``1/10``."""
    ir = lowered(identity(scaled_rows, dict(R=3, C=8, D=2)))
    full, record = record_by_hand(ir)
    for acc in record.loops:
        eighth = ("exp", ("div", ("sub", "m", "m'"), ("const", 8.0)))
        assert prove(("add", "r", ("mul", "a", eighth)), ir, full.name, acc).proved
        wrong = prove(("add", "r", ("mul", "a", e)), ir, full.name, acc)
        assert not wrong.proved and wrong.results[ONE] == "sat"

    assert Fraction(float(0.1)) != Fraction(1, 10)
    assert proof.const(0.1).eq(z3.Q(3602879701896397, 36028797018963968))
    assert not proof.const(0.1).eq(z3.Q(1, 10))
    ir = lowered(identity(tenth_rows, dict(R=3, C=8, D=2)))
    full, record = record_by_hand(ir)
    for acc in record.loops:
        build, _ = contribution(ir, full.name, acc)
        assert "3602879701896397.0 36028797018963968.0" in build(z3.Real("m")).sexpr()
        tenth = ("exp", ("mul", ("sub", "m", "m'"), ("const", 0.1)))
        assert prove(("add", "r", ("mul", "a", tenth)), ir, full.name, acc).proved


def test_a_factor_written_by_hand_for_the_record_is_proved_too():
    """``record_by_hand``'s factor is the one the engine derives: ``a * F + r`` holds."""
    from cleave.ir.update_rule import update_of

    for which in SITES:
        assert prove(update_of(FACTOR), *site(which)).proved


def test_stability_is_a_fact_about_the_exp_arguments():
    assert proof.nonpositive_exp_args(("add", "r", ("mul", "a", e)), "max")
    assert not proof.nonpositive_exp_args(("add", "r", ("div", "a", einv)), "max")
    # without what the reducer gives (m <= m') nothing is known about the sign
    assert not proof.nonpositive_exp_args(("add", "r", ("mul", "a", e)), "sum")
