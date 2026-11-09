import importlib
import inspect
import pkgutil
import re

import pytest

import cleave.ir.ops as ops_pkg
from cleave.ir.graph import ComputeOpNode, Node, TensorOp, ViewOpNode
from cleave.ir.ops import MacroOp, PrologueOp
from cleave.ir.ops.elementwise import BinaryElementwise, UnaryElementwise
from cleave.ir.ops.registry import AXIS_MAP_EXEMPT, NO_REFERENCE, OPS, SUPEROPT_KINDS
from cleave.ir import analysis
from cleave.runtime import reference as graph_executor

_BASES = {ComputeOpNode, ViewOpNode, MacroOp, PrologueOp, BinaryElementwise, UnaryElementwise}


def _op_classes():
    found = set()
    for info in pkgutil.iter_modules(ops_pkg.__path__):
        mod = importlib.import_module(f"cleave.ir.ops.{info.name}")
        for _, obj in inspect.getmembers(mod, inspect.isclass):
            if obj.__module__ == mod.__name__ and issubclass(obj, (Node, MacroOp)):
                found.add(obj)
    return found - _BASES


def _kn_types():
    mod = importlib.import_module("cleave.lower.convert")
    rules = getattr(mod, "_RULES", None)
    if rules is None:  # converter written as an if/elif chain
        return set(re.findall(r'"(kn_\w+_op)"', inspect.getsource(mod)))
    from cleave.core.schema import format_op_type

    found = set()
    for kind in getattr(mod, "_SOURCE_KINDS", {k for k, _ in rules}):
        if kind in mod._AXIS_KINDS:
            found |= {format_op_type(kind, a) for a in range(mod._MAX_AXIS + 1)}
        else:
            found.add(format_op_type(kind))
    return found


def _has_reference(cls):
    """Probe graph_executor._eval_node with a bare instance: a refusal means no reference."""
    import torch

    node = cls.__new__(cls)
    node.__dict__.update(dtype="float32", out_dtype="float32", name="probe")
    try:
        graph_executor._eval_node(node, [torch.ones(2, 2)])
    except NotImplementedError:
        return False
    except TypeError as e:
        if "no torch reference" in str(e):
            return False
    except Exception:
        pass  # reached a real evaluator that needs attributes the probe lacks
    return True


def test_every_op_class_registered():
    names = {c.__name__ for c in _op_classes()}
    assert names == set(OPS), (names - set(OPS), set(OPS) - names)
    assert {n for n in ops_pkg.__all__ if n in names} == names, "op class not exported"
    for name, spec in OPS.items():
        assert getattr(ops_pkg, name) is spec.cls


@pytest.mark.parametrize("name", sorted(OPS))
def test_reference_matches_graph_executor(name):
    spec = OPS[name]
    assert spec.reference == _has_reference(spec.cls)
    assert spec.reference == (name not in NO_REFERENCE)
    assert spec.reference == (spec.cls in graph_executor.REFERENCES)
    if spec.kn_types:
        assert spec.reference, f"{name} is reachable from the superoptimizer but has no reference"


@pytest.mark.parametrize("name", sorted(OPS))
def test_axis_map_registered_or_exempt(name):
    spec = OPS[name]
    assert spec.axis_map == analysis.axis_map_kind(spec.cls)
    assert (spec.axis_map is None) == (name in AXIS_MAP_EXEMPT), name
    assert not (spec.kn_types and spec.axis_map is None)


def test_axis_map_dispatch_is_table_driven():
    assert set(analysis.AXIS_MAP.values()) <= set(analysis.AXIS_MAPPERS)
    assert set(analysis.AXIS_MAP) <= {s.cls for s in OPS.values()}
    src = inspect.getsource(analysis._map_input_axes_to_consumer_axes)
    assert "isinstance" not in src


def test_references_are_type_exact():
    assert set(graph_executor.REFERENCES) <= {s.cls for s in OPS.values()}
    src = inspect.getsource(graph_executor._eval_node)
    assert "isinstance" not in src


_CAPABILITIES = (
    "scalar_args",
    "source_args",
    "index_inputs",
    "stage_fragment_operands_in_shared",
    "resolve_operand_layout",
    "is_batch_dim",
    "input_is_batch_dim",
)


@pytest.mark.parametrize("name", sorted(OPS))
def test_capabilities_have_class_defaults(name):
    cls = OPS[name].cls
    caps = _CAPABILITIES if issubclass(cls, TensorOp) else ("scalar_args", "min_size")
    for attr in caps:
        assert attr in dir(cls), (name, attr)
    assert callable(getattr(cls, "macro_key"))
    for attr in cls._macro_attrs:
        assert isinstance(attr, str)


def test_kn_types_cover_converter():
    known = _kn_types() - {"kn_input_op", "kn_output_op"}
    registered = {t for s in OPS.values() for t in s.kn_types}
    assert known == registered, (known - registered, registered - known)
    for spec in OPS.values():
        assert spec.superopt in SUPEROPT_KINDS + (None,)
        assert bool(spec.kn_types) == (spec.superopt is not None)


def test_exemption_tables_only_name_registered_ops():
    assert set(NO_REFERENCE) <= set(OPS)
    assert set(AXIS_MAP_EXEMPT) <= set(OPS)
    assert all(NO_REFERENCE.values()) and all(AXIS_MAP_EXEMPT.values())
