import json

import pytest

pytest.importorskip("tilelang")

from tests.pipeline import _pipeline as P


@pytest.mark.parametrize("name", P.fixture_names())
def test_pipeline_snapshot(name, request):
    path = P.snapshot_path(name)
    actual = P.snapshot(name)
    if request.config.getoption("--update-snapshots"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(P.dumps(actual))
        return
    assert path.exists(), f"missing {path}; run pytest --update-snapshots"
    expected = json.loads(path.read_text())
    assert actual == expected, (
        f"pipeline output for {name} changed; if intended, regenerate with "
        "--update-snapshots and review the diff"
    )
