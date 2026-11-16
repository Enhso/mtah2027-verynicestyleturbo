"""python -m possession [--no-smoke]: write evaluation/results/Verynicestyleturbo.csv, validate it,
print its sha256 and the release gates."""

import argparse
import os

from . import gates, ingest, write
from .params import Params
from .predict import predict

TEAM = "Verynicestyleturbo"


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-smoke", action="store_true",
                        help="skip the public-window smoke gate (use while developing)")
    parser.add_argument("--out", default=os.path.join(ingest.ROOT, "evaluation", "results",
                                                      TEAM + ".csv"))
    parser.add_argument("--cache-dir", default=None,
                        help="parsed-data cache folder (default $MTAH_DATA_DIR/cache); an empty "
                             "folder makes the run parse the raw ATD and XML first")
    args = parser.parse_args()

    match = ingest.ensure(cache_dir=args.cache_dir)
    info = {}
    codes = predict(match, Params(), info=info)
    output, digest = write.write(codes, args.out)
    print(output.splitlines()[0])
    print("guard 1 merged {} short D runs".format(info["merged_dead_runs"]))
    print("sha256 {}  {}".format(digest, os.path.relpath(args.out, ingest.ROOT)))
    print("\nrelease gates")
    print(gates.format_rows(gates.compute(match, codes, True, smoke=not args.no_smoke)))


if __name__ == "__main__":
    main()
