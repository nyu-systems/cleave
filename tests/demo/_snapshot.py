"""The demos' recorded csopt builder calls (``snapshots/builder_calls.json``); ``--update-snapshots`` rewrites entries."""
import fcntl
import json
import tempfile
from pathlib import Path

PATH = Path(__file__).resolve().parent / "snapshots" / "builder_calls.json"
SNAPSHOT = json.loads(PATH.read_text())


def updating(request) -> bool:
    return request.config.getoption("--update-snapshots")


def record(section: str, key: str, value) -> None:
    with open(Path(tempfile.gettempdir()) / "cleave_builder_calls.lock", "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(PATH.read_text())
        data.setdefault(section, {})[key] = value
        PATH.write_text(json.dumps(data, indent=1, sort_keys=True) + "\n")
        SNAPSHOT.setdefault(section, {})[key] = value
