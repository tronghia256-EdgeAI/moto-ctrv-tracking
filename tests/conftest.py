"""Pytest configuration — make `import code.*` resolve from the repo root.

The project package is named `code`, which collides with the stdlib `code`
module. pytest imports pdb at startup, and pdb imports the stdlib `code`
module — so by the time tests run, sys.modules['code'] already holds the
stdlib module and `import code.models` fails with "'code' is not a package".
Purge it so the project package (repo root is first on sys.path) wins.
"""

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

_stale = sys.modules.get("code")
if _stale is not None and not hasattr(_stale, "__path__"):
    del sys.modules["code"]
