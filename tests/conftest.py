"""
conftest.py

Session-level test setup for repolens.

frontend/dist/index.html must exist before the TestClient in test_api.py is
initialised. TestClient(app) is a module-level expression in test_api.py, so it
runs when the module is imported. Pytest imports conftest.py before importing
test modules, meaning module-level code here executes first — before api.py is
loaded and before app.mount() is attempted.

We only create the stub if the directory does not already exist, so a real
`npm run build` output is never overwritten.
"""

from pathlib import Path

_DIST = Path(__file__).parent.parent / "frontend" / "dist"

if not _DIST.exists():
    _DIST.mkdir(parents=True, exist_ok=True)
    (_DIST / "index.html").write_text(
        "<!DOCTYPE html><html><body>repolens test stub</body></html>\n"
    )
