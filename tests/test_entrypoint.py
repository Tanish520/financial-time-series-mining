"""Exercise the hosted script launch without pytest's repository import path."""

from pathlib import Path
import subprocess
import sys


ENTRY = Path(__file__).resolve().parents[1] / "app/streamlit_app.py"


def test_hosted_entrypoint_imports_from_an_unrelated_working_directory(tmp_path):
    probe = """
import runpy
import sys
from pathlib import Path

entry = Path(sys.argv[1])
# A console-script launch supplies the app directory, not the repository root.
sys.path.insert(0, str(entry.parent))
namespace = runpy.run_path(str(entry), run_name="hosted_entrypoint_probe")
bundle = namespace["load_bundle"](namespace["default_bundle"]())
assert bundle.is_walk_forward
assert len(bundle.tickers) == 10
assert bundle.manifest["fold_count"] == 44
assert "src.mining" not in sys.modules
assert "yfinance" not in sys.modules
print("Hosted entrypoint and saved results loaded successfully")
"""
    result = subprocess.run(
        [sys.executable, "-I", "-c", probe, str(ENTRY)],
        cwd=tmp_path, capture_output=True, text=True, timeout=40,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "Hosted entrypoint and saved results loaded successfully" in result.stdout
