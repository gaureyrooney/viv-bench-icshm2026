"""
Reproducibility metadata: capture everything needed to replay a run later
without the operator having to remember what flags they used - resolved CLI
args, git commit + dirty-tree status, installed package versions, and the RNG
seed used to make cv2's RANSAC-based estimators (estimateAffinePartial2D,
estimateAffine2D, findHomography) deterministic.
"""
import platform
import subprocess
import time


def get_git_state(cwd=None):
    """Returns {'commit': <hash or None>, 'dirty': <bool or None>}."""
    try:
        commit = subprocess.run(
            ['git', 'rev-parse', 'HEAD'], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return {'commit': None, 'dirty': None}

    try:
        status = subprocess.run(
            ['git', 'status', '--porcelain'], cwd=cwd, capture_output=True, text=True, check=True
        ).stdout
        dirty = len(status.strip()) > 0
    except Exception:
        dirty = None

    return {'commit': commit, 'dirty': dirty}


def get_package_versions():
    """Versions of the packages whose behaviour can change results, recorded in
        every run's config.json so a number can be traced to an environment."""
    versions = {}
    for pkg in ['cv2', 'numpy', 'scipy', 'pandas', 'openpyxl']:
        try:
            mod = __import__(pkg)
            versions[pkg] = getattr(mod, '__version__', 'unknown')
        except ImportError:
            versions[pkg] = None
    return versions


def set_deterministic_seed(seed):
    """Seed every RNG this pipeline's RANSAC steps could touch."""
    import numpy as np
    import cv2
    np.random.seed(seed)
    cv2.setRNGSeed(seed)


def build_run_config(args, repo_dir):
    """
    Args:
        args: argparse.Namespace (or anything vars()-able) of resolved CLI args
        repo_dir: path used to resolve the git commit/dirty state

    Returns a dict, self-contained enough that saving it alongside metrics.json
    lets a future run be replayed by feeding the same resolved_args back into
    run_experiment.py's argument parser (or reading them off directly).
    """
    return {
        'resolved_args': dict(vars(args)),
        'git': get_git_state(cwd=repo_dir),
        'python_version': platform.python_version(),
        'package_versions': get_package_versions(),
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
