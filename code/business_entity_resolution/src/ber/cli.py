"""Command line entry point: ``ber <stage> [--config ...] [--set k=v] [--run ID]``.

Each pipeline stage is one sub-command, which reads the previous stage's
artifacts from ``artifacts/<run>/`` and writes its own. Stages that haven't
been implemented yet are listed with the task that will add them.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from dataclasses import dataclass

import yaml

from .artifacts import StageRun
from .config import Config, ConfigError, load_config
from .log import get_logger, setup_logging

log = get_logger("cli")


@dataclass(frozen=True)
class Command:
    func: Callable[[Config, argparse.Namespace], None]
    help: str
    task: str | None = None  # plan task that implements it; None = implemented


def cmd_config(cfg: Config, args: argparse.Namespace) -> None:
    """Print the fully resolved config (after extends and overrides)."""
    print(f"# run_id={cfg.run_id} config_hash={cfg.hash()} root={cfg.root}")
    print(yaml.safe_dump(cfg.to_dict(), sort_keys=False), end="")


def cmd_noop(cfg: Config, args: argparse.Namespace) -> None:
    """Exercise the stage machinery: writes a manifest, a log and a small artifact."""
    with StageRun(cfg, "noop") as st:
        st.write_json({"hello": "world"}, "result")
        st.metric("answer", 42)


def _planned(task: str) -> Callable[[Config, argparse.Namespace], None]:
    def run(cfg: Config, args: argparse.Namespace) -> None:
        raise SystemExit(f"not implemented yet (planned in {task}; see plan.md)")
    return run


COMMANDS: dict[str, Command] = {
    "config": Command(cmd_config, "print the resolved config"),
    "noop": Command(cmd_noop, "no-op stage (tests the artifact/manifest machinery)"),
    "split": Command(_planned("T0.3"), "geographic train/val split + dev slice", "T0.3"),
    "normalize": Command(_planned("T1.4"), "stage 0: normalise train + test records", "T1.4"),
    "candidates": Command(_planned("T2.2"), "stage 1: TF-IDF top-k candidates + recall", "T2.2"),
    "features": Command(_planned("T3.3"), "stage 2: pair + context features", "T3.3"),
    "scorer": Command(_planned("T3.4"), "stage 2: train/apply LightGBM scorer", "T3.4"),
    "route": Command(_planned("T3.6"), "stage 2: send the uncertain band to the LLM queue", "T3.6"),
    "llm-export": Command(_planned("T5.3"), "stage 3: export prompts for Kaggle", "T5.3"),
    "llm-import": Command(_planned("T5.5"), "stage 3: import Kaggle LLM scores", "T5.5"),
    "decide": Command(_planned("T4.1"), "stage 4: one-owner assignment + expected-F0.5", "T4.1"),
    "submit": Command(_planned("T3.5"), "write output/*.tsv and validate", "T3.5"),
    "evaluate": Command(_planned("T3.5"), "score a run on validation", "T3.5"),
    "explain": Command(_planned("T4.3"), "trace one entity through every stage", "T4.3"),
}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="ber", description="Business entity resolution pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="stages marked (T..) are planned; see plan.md")
    sub = p.add_subparsers(dest="command", required=True, metavar="<stage>")
    for name, c in COMMANDS.items():
        label = c.help if c.task is None else f"{c.help} ({c.task})"
        sp = sub.add_parser(name, help=label, description=label)
        sp.add_argument("--config", "-c", action="append", default=None,
                        help="YAML config file(s), merged in order "
                             "(default: configs/default.yaml)")
        sp.add_argument("--set", "-s", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config value, e.g. --set candidates.k_name=20")
        sp.add_argument("--run", "-r", default=None, help="run id (artifacts/<run>/)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, args.set, args.run)
    except ConfigError as e:
        print(f"config error: {e}", file=sys.stderr)
        return 2
    setup_logging(cfg.log_level)
    COMMANDS[args.command].func(cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
