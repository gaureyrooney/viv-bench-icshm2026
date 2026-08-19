"""
Central registries for run_experiment.py: trackers, compensators, filters,
ROI-generation methods, and named recipes (known-good flag combinations).

Adding a new method means adding one file (e.g. trackers/my_tracker.py) and
one entry here - run_experiment.py's CLI parsing never needs to change, since
its --tracker/--compensator/--filter/--roi-method choices are all derived
from these dicts' keys.
"""
import os

# --------------------------------------------------------------------------
# Trackers: name -> factory(**kwargs) -> Tracker instance
# --------------------------------------------------------------------------

def _make_dft_tracker(**kwargs):
    from trackers.dft_reg import DFTRegistrationTracker
    return DFTRegistrationTracker(upsample_factor=kwargs.get('upsample_factor', 50))


def _make_template_match_tracker(**kwargs):
    from trackers.template_match import TemplateMatchTracker
    # Defaults mirror the manually-selected target ROI (~64px) and the
    # reference config's search_margin=70, rather than an arbitrary smaller
    # crop that throws away most of the visible checkerboard target.
    return TemplateMatchTracker(
        template_size=kwargs.get('template_size', 64),
        search_margin=kwargs.get('search_margin', 70),
    )


def _make_icgn_tracker(**kwargs):
    from trackers.icgn import ICGNTracker
    return ICGNTracker(
        template_size=kwargs.get('template_size', 64),
        interp_order=kwargs.get('interp_order', 3),
    )


def _make_gabor_phase_tracker(**kwargs):
    from trackers.phase_based import GaborPhaseTracker
    return GaborPhaseTracker(template_size=kwargs.get('template_size', 64))


def _make_corner_klt_tracker(**kwargs):
    from trackers.corner_klt import CornerKLTTracker
    return CornerKLTTracker(template_size=kwargs.get('template_size', 64))


TRACKER_REGISTRY = {
    'dft': _make_dft_tracker,
    'template_match': _make_template_match_tracker,
    'icgn': _make_icgn_tracker,
    'gabor_phase': _make_gabor_phase_tracker,
    'corner_klt': _make_corner_klt_tracker,
}


# --------------------------------------------------------------------------
# Compensators: name -> factory(**kwargs) -> compensator instance
# --------------------------------------------------------------------------

def _make_none_compensator(**kwargs):
    from motion.differential_ref import NoCompensator
    return NoCompensator()


def _make_differential_compensator(**kwargs):
    from motion.differential_ref import DifferentialReferenceCompensator
    return DifferentialReferenceCompensator()


# affine_model -> motion_model kwarg understood by AffineFlowCompensator.
# 'similarity' (4-DOF: translation+rotation+uniform scale, cv2.estimateAffinePartial2D)
#   is what templatematch_affine_config uses, and is what this project
#   called 'affine' before this flag was split out - kept as the default.
# 'full' (6-DOF affine: cv2.estimateAffine2D, adds independent x/y scale + shear)
#   is a new option for comparison, not used by the template-match + affine-flow configuration.
# 'homography' (8-DOF projective, cv2.findHomography) - tested, underperformed
#   with only ~2 small/clustered background ROIs (see README).
AFFINE_MODEL_TO_MOTION_MODEL = {
    'similarity': 'affine',
    'full': 'full_affine',
    'homography': 'homography',
}


def _make_affine_flow_compensator(**kwargs):
    from motion.affine_flow import AffineFlowCompensator
    affine_model = kwargs.get('affine_model', 'similarity')
    motion_model = AFFINE_MODEL_TO_MOTION_MODEL[affine_model]
    return AffineFlowCompensator(
        motion_model=motion_model,
        transform_smoothing=kwargs.get('transform_smoothing', 'none'),
        smoothing_alpha=kwargs.get('smoothing_alpha', 0.3),
    )


COMPENSATOR_REGISTRY = {
    'none': _make_none_compensator,
    'differential': _make_differential_compensator,
    'affine_flow': _make_affine_flow_compensator,
}

# affine_flow needs raw background pixel data (patches or full frames - it
# detects/tracks its own corner features), not point trajectories - it goes
# through a different pipeline method and isn't currently supported by
# --from-cache.
COMPENSATORS_NEEDING_RAW_PATCHES = {'affine_flow'}


# --------------------------------------------------------------------------
# Filters: name -> function(signal, fs, cutoff_hz=None, band_hz=None) -> filtered signal
# --------------------------------------------------------------------------

def _filter_none(signal, fs, **kwargs):
    return signal


def _filter_highpass(signal, fs, cutoff_hz=1.0, **kwargs):
    from dsp.alignment import zero_phase_filter
    return zero_phase_filter(signal, fs=fs, cutoff_hz=cutoff_hz, btype='high')


def _filter_bandpass(signal, fs, band_hz=(8.0, 16.0), **kwargs):
    from dsp.alignment import zero_phase_filter
    return zero_phase_filter(signal, fs=fs, cutoff_hz=tuple(band_hz), btype='band')


def _filter_bandpass_wiener(signal, fs, band_hz=(0.2, 20.0), wiener_alpha=16.0, **kwargs):
    """Bandpass, then graded spectral (Wiener) shrinkage - see
    dsp/wiener_denoise.py. Unlike a narrower hard bandpass this keeps the full
    0.2-20Hz band (so it does not create the asymmetric-filter artifact) and
    instead attenuates only the bins that are noise-dominated, with the noise
    floor estimated from the vision signal itself (no LDS needed)."""
    from dsp.alignment import zero_phase_filter
    from dsp.wiener_denoise import wiener_denoise
    y = zero_phase_filter(signal, fs=fs, cutoff_hz=tuple(band_hz), btype='band')
    return wiener_denoise(y, fs=fs, alpha=wiener_alpha)


FILTER_REGISTRY = {
    'none': _filter_none,
    'highpass': _filter_highpass,
    'bandpass': _filter_bandpass,
    'bandpass_wiener': _filter_bandpass_wiener,
}


# --------------------------------------------------------------------------
# ROI methods: name -> generate(rois_path, video_path, **kwargs)
# Only invoked by run_experiment.py when the target ROI file doesn't already
# exist - existing files are just loaded as-is (this is what keeps ROI
# selection decoupled from tracking; see pipeline.py's module docstring).
# --------------------------------------------------------------------------

def _roi_method_manual(rois_path, video_path, **kwargs):
    raise RuntimeError(
        f"ROI file {rois_path} doesn't exist and --roi-method manual can't run "
        f"unattended (it needs a GUI). Run it yourself first:\n"
        f"    python data/roi_select.py \"{video_path}\" --out \"{rois_path}\""
    )


def _roi_method_auto(rois_path, video_path, **kwargs):
    import subprocess
    import sys
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'bootstrap_rois_auto.py')
    subprocess.run([sys.executable, script, video_path, '--out', rois_path], check=True)


def _roi_method_import(rois_path, video_path, **kwargs):
    roi_source = kwargs.get('roi_source')
    if not roi_source:
        raise ValueError("--roi-method import requires --roi-source <manual_rois.json>")
    import subprocess
    import sys
    script = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                           'data', 'import_manual_rois.py')
    cmd = [sys.executable, script, roi_source, '--out', rois_path, '--video', video_path]
    subprocess.run(cmd, check=True)


ROI_METHOD_REGISTRY = {
    'manual': _roi_method_manual,
    'auto': _roi_method_auto,
    'import': _roi_method_import,
}


# --------------------------------------------------------------------------
# Recipes: name -> dict of CLI flag overrides, for one-command reproduction
# of known leaderboard results. `python run_experiment.py --recipe NAME`
# applies these as new argparse *defaults* - any flag also passed explicitly
# on the command line still takes priority over the recipe.
# --------------------------------------------------------------------------

# Recipes leave video/lds unset and let run_experiment.py fill them in from
# paths.py (env var or configs/paths.json). Baking machine-specific absolute
# paths in here is what previously made the repo unrunnable elsewhere.
_DEFAULT_VIDEO = None
_DEFAULT_LDS = None

RECIPES = {
    'icgn_differential_bp9-14': {
        'tracker': 'icgn', 'compensator': 'differential',
        'filter': 'bandpass', 'filter_band_hz': [9.0, 14.0],
        'video': _DEFAULT_VIDEO, 'lds': _DEFAULT_LDS,
        'run_name': 'icgn_differential_bp9-14',
    },
    'templatematch_differential_bp9-14': {
        'tracker': 'template_match', 'compensator': 'differential',
        'filter': 'bandpass', 'filter_band_hz': [9.0, 14.0],
        'video': _DEFAULT_VIDEO, 'lds': _DEFAULT_LDS,
        'run_name': 'templatematch_differential_bp9-14',
    },
    'icgn_affineflow_fullframe_bp9-14': {
        'tracker': 'icgn', 'compensator': 'affine_flow', 'affine_model': 'similarity',
        'filter': 'bandpass', 'filter_band_hz': [9.0, 14.0],
        'video': _DEFAULT_VIDEO, 'lds': _DEFAULT_LDS,
        'run_name': 'icgn_affineflow_fullframe_bp9-14',
    },
}
