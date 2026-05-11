"""Diagnostic: probe cv_quixer import chain, printing full tracebacks."""
import sys
import traceback

print("=== sys.version ===")
print(sys.version)

print("\n=== sys.path ===")
for p in sys.path:
    print(" ", p)

steps = [
    "import cv_quixer",
    "import cv_quixer.config",
    "import cv_quixer.config.schema",
    "import cv_quixer.data.transforms",
    "import cv_quixer.data.mnist",
    "import cv_quixer.data",
    "from cv_quixer.data.mnist import PatchedDataset",
    "import cv_quixer.models",
    "from cv_quixer.models import build_model",
]

for stmt in steps:
    try:
        exec(stmt)
        print(f"OK   {stmt}")
    except Exception:
        print(f"FAIL {stmt}")
        traceback.print_exc()
        print()
