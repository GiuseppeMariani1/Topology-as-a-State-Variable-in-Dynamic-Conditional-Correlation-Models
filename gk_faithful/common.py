# -*- coding: utf-8 -*-
"""
Shared helpers for the gk_faithful add-on: config loading and output paths.

Every output path is built here, under config_gk.yaml's output_dir, and
_guard_output() refuses any path that resolves into the original thesis
data folder. That is the mechanical guarantee behind "this add-on cannot
overwrite thesis results".
"""

import os
import sys

import yaml

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
GK_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'config_gk.yaml')

if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

# The original loader resolves config/config.yaml relative to the working
# directory, so every entry point runs from the repo root.
os.chdir(REPO_ROOT)

from src.data.loader import load_config  # noqa: E402  (original, read-only use)


def load_gk_config(path=GK_CONFIG_PATH):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def load_thesis_paths():
    """Original thesis paths. Used for READING inputs only."""
    return load_config()['paths']


def _guard_output(path):
    thesis_dir = os.path.realpath(os.path.join(REPO_ROOT, 'data', 'processed'))
    target = os.path.realpath(path)
    if target == thesis_dir or target.startswith(thesis_dir + os.sep):
        raise RuntimeError(
            f"Refusing to write {path}: it is inside the thesis data folder. "
            "gk_faithful outputs must go under output_dir in config_gk.yaml."
        )
    os.makedirs(os.path.dirname(target), exist_ok=True)
    return path


def out_dir(gk_cfg=None):
    gk_cfg = gk_cfg or load_gk_config()
    return os.path.join(REPO_ROOT, gk_cfg['output_dir'])


def norms_path(w, gk_cfg=None):
    """Per-day landscape norms for point-cloud window w."""
    return _guard_output(os.path.join(out_dir(gk_cfg), f'norms_w{w}.parquet'))


def spectral_path(w, m, gk_cfg=None):
    """Variance / low-frequency spectral density, point cloud w, spectral window m."""
    return _guard_output(os.path.join(out_dir(gk_cfg), f'spectral_w{w}_m{m}.parquet'))


def result_path(kind, tag, gk_cfg=None):
    """kind in {'oos', 'perm', 'perm_reg', 'diagnostic'}; tag identifies the run."""
    ext = {'diagnostic': 'png'}.get(kind, 'npy')
    return _guard_output(os.path.join(out_dir(gk_cfg), kind, f'{tag}.{ext}'))


def run_tag(features, w, m, split_name):
    return f'{features}_w{w}_m{m}_{split_name}'
