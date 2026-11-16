"""One-off comparisons on the proxy objective: python -m calibration.explore "<name>=<v1>,<v2>;..." [base changes as name=value ...]

Each ';'-separated spec is a one-dimensional grid applied to the base. Prints J per value and per variant.
"""
import dataclasses
import sys

from calibration import common as c, search
from possession.params import Params


def parse(v):
    if v in ("True", "False"):
        return v == "True"
    return float(v) if "." in v or "e" in v else int(v)


def cast(name, v):
    t = type(getattr(Params(), name))
    return t(v) if t is not bool else bool(v)


if __name__ == "__main__":
    base = Params()
    for kv in sys.argv[2:]:
        k, v = kv.split("=")
        base = dataclasses.replace(base, **{k: cast(k, parse(v))})
    keys = search.setup()
    r0 = search.evaluate([base], keys)[0]
    print("base J {:.4f}/{:.4f} [{}]".format(*search.objective(r0, keys), search.by_variant(r0, keys)))
    for spec in sys.argv[1].split(";"):
        if not spec:
            continue
        name, vals = spec.split("=")
        vals = [cast(name, parse(v)) for v in vals.split(",")]
        res = search.evaluate([dataclasses.replace(base, **{name: v}) for v in vals], keys)
        for v, r in zip(vals, res):
            print("  {:<16} {:<8} J {:.4f}/{:.4f} [{}] n_pred {:.0f}".format(
                name, str(v), *search.objective(r, keys), search.by_variant(r, keys),
                sum(x[2] for x in r.values()) / len(r)))
