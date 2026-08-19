"""
Central resolution of the (large, un-committed) input data paths.

The video and laser-sensor recordings are far too large to belong in version
control, so the repository never contains them - it only refers to them. Every
script therefore resolves them through this module instead of hardcoding a
path, which is what makes the repository runnable on a machine other than the
one it was developed on.

Resolution order for each path, first match wins:

  1. an environment variable  (VIVBENCH_VIDEO, VIVBENCH_LDS, ...)
  2. configs/paths.json       (copy configs/paths.example.json and edit)

If neither is set, the error message names the variable and the JSON key, so a
new user is told exactly what to do rather than seeing a bare FileNotFoundError
from somewhere deep in a tracker.

Usage:
    from paths import video_path, lds_path
    cap = cv2.VideoCapture(video_path())
"""
import json
import os

_HERE = os.path.dirname(os.path.abspath(__file__))
_CONFIG = os.path.join(_HERE, 'configs', 'paths.json')
_EXAMPLE = os.path.join('configs', 'paths.example.json')

# key in configs/paths.json -> (environment variable, human description)
_SPEC = {
    'video': ('VIVBENCH_VIDEO', 'the UAV video recording (Video.MP4)'),
    'lds': ('VIVBENCH_LDS', 'the laser displacement sensor workbook (LDS data.xlsx)'),
    'templatematch_affine_script': ('VIVBENCH_TM_AFFINE_SCRIPT',
                           'the frozen template-match + affine-flow configuration script '
                           '(full-frame NCC template matching, RANSAC similarity compensation), '
                           'kept unchanged as the regression fixed point'),
    'templatematch_affine_config_json': ('VIVBENCH_TM_AFFINE_CONFIG',
                           'that configuration\'s saved settings JSON, which supplies its ROI values'),
}


def _load_config():
    if not os.path.exists(_CONFIG):
        return {}
    try:
        with open(_CONFIG, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"{_CONFIG} exists but could not be read as JSON: {exc}") from exc


def resolve(key, required=True):
    """Resolve one configured path. Returns None if not set and required=False."""
    if key not in _SPEC:
        raise KeyError(f"unknown path key {key!r}; known keys: {sorted(_SPEC)}")
    env_var, desc = _SPEC[key]
    value = os.environ.get(env_var) or _load_config().get(key)

    if not value:
        if not required:
            return None
        raise FileNotFoundError(
            f"Path for {key!r} ({desc}) is not configured.\n"
            f"  Set it either way:\n"
            f"    - environment variable {env_var}=<path>\n"
            f"    - or copy {_EXAMPLE} to configs/paths.json and set \"{key}\"\n"
            f"  configs/paths.json is git-ignored, so local paths never get committed."
        )

    value = os.path.expanduser(os.path.expandvars(value))
    if not os.path.exists(value):
        raise FileNotFoundError(
            f"Path for {key!r} ({desc}) is configured but does not exist:\n  {value}\n"
            f"  Fix {env_var} or the \"{key}\" entry in configs/paths.json."
        )
    return value


def video_path():
    """UAV video recording."""
    return resolve('video')


def lds_path():
    """Laser displacement sensor workbook (ground truth)."""
    return resolve('lds')


def templatematch_affine_script(required=True):
    """Frozen template-match + affine-flow configuration script (regression fixed point)."""
    return resolve('templatematch_affine_script', required=required)


def templatematch_affine_config_json(required=True):
    """That configuration's saved settings, which supply its exact ROI values."""
    return resolve('templatematch_affine_config_json', required=required)


def repo_root():
    return _HERE


if __name__ == '__main__':
    print(f"repo root: {repo_root()}")
    for key, (env_var, desc) in _SPEC.items():
        try:
            print(f"  [ok]      {key:22s} {resolve(key)}")
        except FileNotFoundError:
            print(f"  [not set] {key:22s} ({env_var}) - {desc}")
