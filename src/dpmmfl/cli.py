"""Shared CLI flags: --config (repeatable), --set key=value (repeatable), --smoke."""
import argparse
from .utils import load_config


def parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--config", action="append", default=[], help="extra YAML merged over configs/base.yaml")
    ap.add_argument("--set", action="append", default=[], help="override, e.g. dp.lr=0.1")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end check (cache/smoke, results/smoke, no TEST)")
    return ap


def config(a):
    return load_config(*a.config, smoke=a.smoke, sets=a.set)
