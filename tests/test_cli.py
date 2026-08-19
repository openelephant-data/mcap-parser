import json
import subprocess
import sys
from pathlib import Path


def test_cli_runs(synth_mcap, tmp_path):
    out = tmp_path / "out"
    r = subprocess.run(
        [sys.executable, "-m", "mcap_parser", str(synth_mcap),
         "--out_dir", str(out), "--overwrite", "true"],
        capture_output=True, text=True,
    )
    assert r.returncode == 0, r.stderr
    m = json.loads((out / "manifest.json").read_text())
    assert m["frame_count"] == 3
