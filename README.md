# Possession labelling for FC Barcelona vs Athletic Club

Entry of team **Verynicestyleturbo** to the ball-possession challenge of More than a Hack 2027.
The program labels every frame of the tactical video as home (`H`), away (`A`), dead ball (`D`) or ungraded (`X`). It uses a three-state sequence model decoded with Viterbi, scored from carry, touch, restart, tag and blackout evidence on the tracking data. It has no learned model. Thresholds were calibrated on public Metrica and DFL sample matches and transferred to this match by quantile. The write-up is in [docs/technical.md](docs/technical.md).

Author: Verynicestyleturbo

## Setup

Python 3.12.

```
python -m venv .venv
.venv/bin/pip install -r requirements-submission.txt   # numpy and pandas, pinned
.venv/bin/pip install -r requirements-dev.txt          # only for the tests
```

## Data

The match data is not in this repository and cannot be redistributed. Download it as described in `DATA.md` and put it in `./data`, or point `MTAH_DATA_DIR` at the folder that holds it (see the top of `games.py`).

## Run

```
python -m possession
```

This writes `evaluation/results/Verynicestyleturbo.csv`, runs the organisers' validator and prints the sha256 and a table of release gates. The expected sha256 is

```
e09133425dce724a8ddfb62054b6e1e10272343f0308b6a36946aee5457f1f88
```

The first run parses the tracking and tagging files and caches them (about 11 s). `--cache-dir <empty dir>` forces a parse from the raw files. `--no-smoke` skips the public-window check. Byte-identical output depends on the pinned numpy and pandas versions.

## Tests

```
.venv/bin/python -m pytest tests/
```

The properties the tests encode are listed in `tests/PROPERTIES.md`.

## Calibration

Calibration is optional and not needed to regenerate the file. The exporters in `harness/` read the public Metrica and DFL data and use a separate environment (`harness/requirements.txt`: kloppy, newer numpy and pandas). Then run, from the repository root:

```
python -m calibration.report
```

This prints the scores of the current parameters on the proxy set. `python -m possession.params` prints every parameter with its source.

## Layout

- `possession/`: the submission. Features, evidence channels, Viterbi decoder, guards, parameters (`params.py`), writer and release gates. `FROZEN.txt` records the frozen version.
- `calibration/`: proxy variants, baselines, coordinate descent, quantile transfer, reports.
- `harness/`: exporters and scorers for the external Metrica and DFL sets.
- `analysis/`: label-free statistics of this match (no labels read).
- `broadcast/`: probe that aligns the broadcast recording to the tactical frames. Not used in v1.
- `tests/`: property tests.
- `docs/`: the technical document.

## Provenance

The challenge files (`CHALLENGE.md`, `RULES.md`, `DATA.md`, `EVALUATION.md`, `FAQ.md`, `OVERVIEW.md`, `SUBMISSION.md`, `games.py`, `starter/`, `evaluation/`) come from the organisers' repository `metrica-sports/more-than-a-hack-2027` under its MIT licence (`LICENSE`). The match data is not included and is not redistributable.
