"""Write the submission CSV, run the official validator on it, and hash it."""

import hashlib
import os
import subprocess
import sys

from . import ingest, match as mt
from baseline import write_submission  # noqa: E402  (starter/, on the path through ingest)


def write(codes, path):
    """frame,label CSV with LF line endings. Returns (validator output, sha256)."""
    write_submission(path, mt.LETTERS[codes])
    return validate(path), sha256(path)


def validate(path):
    """Run evaluation/validate_submission.py on the file and return its output; raise unless OK."""
    run = subprocess.run(
        [sys.executable, os.path.join(ingest.ROOT, "evaluation", "validate_submission.py"), path],
        capture_output=True, text=True)
    if run.returncode != 0 or not run.stdout.startswith("OK"):
        raise RuntimeError("validator rejected the file: " + run.stdout.strip()[:200])
    return run.stdout


def sha256(path):
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()
