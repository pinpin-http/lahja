"""Command line: python -m lahja <step> [--corpus open|madar]."""

from __future__ import annotations

import argparse
import logging
import sys

from .config import CORPUS, ENCODERS


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="lahja")
    p.add_argument("--corpus", default=CORPUS, choices=["open", "madar"])
    sub = p.add_subparsers(dest="cmd", required=True)
    for name in (
        "ingest",
        "normalize",
        "label",
        "split",
        "data",
        "baseline",
        "export",
        "evaluate",
        "report",
        "publish",
    ):
        sub.add_parser(name)
    ft = sub.add_parser("finetune")
    ft.add_argument("--model", required=True, choices=list(ENCODERS))
    ft.add_argument("--epochs", type=int, default=None)
    ft.add_argument("--limit", type=int, default=None, help="train on N random rows (smoke test)")
    ft.add_argument("--seed", type=int, default=None, help="extra run with another seed (predictions only)")
    args = p.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s", datefmt="%H:%M:%S")
    for noisy in ("httpx", "httpx2", "huggingface_hub", "urllib3"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    c = args.corpus

    if args.cmd in ("ingest", "normalize", "label", "split", "data"):
        from . import data

        steps = {
            "ingest": [data.step_ingest],
            "normalize": [data.step_normalize],
            "label": [data.step_label],
            "split": [data.step_split],
            "data": [data.step_ingest, data.step_normalize, data.step_label, data.step_split],
        }[args.cmd]
        for step in steps:
            step(c)
    elif args.cmd == "baseline":
        from . import baseline

        baseline.train(c)
    elif args.cmd == "finetune":
        from . import finetune
        from .config import SEED

        finetune.train(c, args.model, epochs=args.epochs, limit=args.limit, seed=args.seed or SEED)
    elif args.cmd == "export":
        from . import export

        export.run(c)
    elif args.cmd == "evaluate":
        from . import evaluate

        evaluate.run(c)
    elif args.cmd == "report":
        from . import report

        report.run(c)
    elif args.cmd == "publish":
        from . import publish

        return publish.run(c)
    return 0


if __name__ == "__main__":
    sys.exit(main())
