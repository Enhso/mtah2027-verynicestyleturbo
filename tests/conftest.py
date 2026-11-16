import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for path in (ROOT, HERE, os.path.join(ROOT, "starter"), os.path.join(ROOT, "evaluation")):
    if path not in sys.path:
        sys.path.insert(0, path)
