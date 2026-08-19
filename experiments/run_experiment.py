"""
Experiment harness: run one (tracker, compensator, filter) combo end-to-end
against real data, save its outputs, and append a row to leaderboard.csv so
multiple methods accumulate into one comparable table over time.

Usage:
    python experiments/run_experiment.py --tracker dft --compensator differential \
        --video "<path>/Video.MP4" --lds "<path>/LDS data.xlsx" --rois configs/rois.json

Or replay a known-good result in one command:
    python experiments/run_experiment.py --recipe icgn_differential_bp9-14

Every tracker/compensator/filter/ROI-method choice comes from the registries
in experiments/registries.py - add a new method by adding one file + one
registry entry there; this file's CLI parsing never needs to change.

Every run saves a self-contained config.json (resolved args, git commit +
dirty status, package versions, RNG seed) alongside metrics.json, so any
leaderboard row can be replayed later without the operator remembering what
flags produced it.
"""
import argparse
import csv
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from paths import video_path, lds_path
from data.lds import load_lds_prepared, load_lds_raw
from dsp.alignment import align_signals_native_lds
from pipeline import ROITracker
from eval.metrics import MetricsReport
from experiments import repro
from experiments.registries import (
    TRACKER_REGISTRY, COMPENSATOR_REGISTRY, COMPENSATORS_NEEDING_RAW_PATCHES,
    FILTER_REGISTRY, ROI_METHOD_REGISTRY, RECIPES,
)


def rebuild_leaderboard(out_root):
    """
    Regenerate leaderboard.csv from every experiments/results/*/metrics.json,
    sorted by RMSE ascending. Rebuilding from these self-contained per-run
    files - rather than incrementally appending CSV rows - avoids silent
    column misalignment if the metrics schema changes between runs (a real
    bug hit earlier: 'filter' was added to the metrics dict after the CSV
    header had already been written by an earlier run, silently shifting
    every later row by one column - see README Bug #9).
    """
    rows = []
    for metrics_path in sorted(glob.glob(os.path.join(out_root, '*', 'metrics.json'))):
        with open(metrics_path) as f:
            rows.append(json.load(f))
    rows.sort(key=lambda r: r.get('rmse_mm', float('inf')))

    fieldnames = []
    for r in rows:
        for k in r:
            if k not in fieldnames:
                fieldnames.append(k)

    leaderboard_path = os.path.normpath(os.path.join(out_root, os.pardir, 'leaderboard.csv'))
    with open(leaderboard_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    return leaderboard_path


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--recipe', default=None, choices=list(RECIPES.keys()),
                         help='Shortcut for a known-good flag combination (see registries.py). '
                              'Sets defaults for the flags below - anything also passed '
                              'explicitly on the command line still overrides the recipe.')
    parser.add_argument('--tracker', choices=list(TRACKER_REGISTRY.keys()))
    parser.add_argument('--compensator', default='differential', choices=list(COMPENSATOR_REGISTRY.keys()))
    parser.add_argument('--affine-model', default='similarity', choices=['similarity', 'full', 'homography'],
                         dest='affine_model',
                         help='affine_flow compensator only. similarity = 4-DOF (what templatematch_affine_config uses); full = 6-DOF affine (comparison only); '
                              'homography = 8-DOF projective (tested, underperformed here - see README).')
    parser.add_argument('--transform-smoothing', default='none', choices=['none', 'ema', 'kalman'],
                         dest='transform_smoothing',
                         help='affine_flow compensator only. Temporal smoothing of the per-frame '
                              'RANSAC-fit background transform (each frame is currently fit '
                              'independently). "ema" = exponential moving average over the decomposed '
                              'translation/rotation/scale params (see --smoothing-alpha); "kalman" = '
                              'constant-velocity Kalman+RTS smoother over the same params, weighted '
                              'per-frame by RANSAC inlier count/LK error. Both are risky if the camera '
                              'itself picks up real ~11.7Hz motion - always check phase_lag_deg_11.7hz '
                              'AND amplitude_ratio_11.7hz before vs. after (see README).')
    parser.add_argument('--smoothing-alpha', type=float, default=0.3, dest='smoothing_alpha',
                         help='--transform-smoothing ema only: weight on the new (raw) sample each '
                              'frame - 1.0 = no smoothing, smaller = heavier smoothing.')
    parser.add_argument('--video')
    parser.add_argument('--lds')
    parser.add_argument('--rois', default=None, help='Default: configs/rois.json next to this script')
    parser.add_argument('--roi-method', default=None, choices=list(ROI_METHOD_REGISTRY.keys()), dest='roi_method',
                         help='Only used if --rois does not already exist: which method to generate it '
                              'with. Existing ROI files are always just loaded as-is (ROI selection stays '
                              'decoupled from tracking - see pipeline.py).')
    parser.add_argument('--roi-source', default=None, dest='roi_source',
                         help='Source JSON for --roi-method import')
    parser.add_argument('--max-frames', type=int, default=None, dest='max_frames')
    parser.add_argument('--run-name', default=None, dest='run_name')
    parser.add_argument('--out-dir', default=None, dest='out_dir')
    parser.add_argument('--filter', default='none', choices=list(FILTER_REGISTRY.keys()),
                         help="Zero-phase post-filter applied to the vision displacement only "
                              "(LDS is already narrowband in this dataset - see README - so any "
                              "low-frequency vision content is compensation/tracking drift, not real "
                              "motion). Default 'none' shows raw tracker output.")
    parser.add_argument('--symmetric-lds-filter', action='store_true', dest='symmetric_lds_filter',
                         help='Also apply --filter to the LDS reference before comparison (matches the '
                              'template-match + affine-flow configuration\'s methodology). Default off, so historical '
                              'leaderboard rows stay comparable to new ones.')
    parser.add_argument('--filter-cutoff-hz', type=float, default=1.0, dest='filter_cutoff_hz',
                         help='highpass cutoff')
    parser.add_argument('--filter-band-hz', type=float, nargs=2, default=[8.0, 16.0], dest='filter_band_hz',
                         help='bandpass [low, high]')
    parser.add_argument('--wiener-alpha', type=float, default=16.0, dest='wiener_alpha',
                         help='--filter bandpass_wiener only: noise-floor over-subtraction factor. '
                              '0 disables denoising. Measured optimum on this dataset is ~16 '
                              '(selected on the first 30s, validated on the held-out last 30s: '
                              '-17%% RMSE). See dsp/wiener_denoise.py.')
    parser.add_argument('--from-cache', default=None, dest='from_cache',
                         help='Path to a previously saved trajectories.npz. Skips tracking entirely '
                              'and re-runs compensation/projection/filter/alignment from the cached '
                              'raw pixel trajectories - use this to sweep --compensator/--filter in '
                              'seconds instead of re-tracking the whole video each time.')
    parser.add_argument('--seed', type=int, default=42,
                         help='RNG seed for cv2 RANSAC-based estimators (affine/homography fitting), '
                              'so results are exactly reproducible run to run.')
    parser.add_argument('--lds-alignment', default='decimated', choices=['decimated', 'native'],
                         dest='lds_alignment',
                         help='"decimated" (default): LDS is anti-alias-filtered and decimated to 50Hz '
                              'once, then time-aligned against vision at that rate - unchanged behavior. '
                              '"native": matches the template-match + affine-flow configuration, which never decimates '
                              'LDS at all - the fine alignment stage instead interpolates the native '
                              '10kHz signal directly at each candidate lag (align_signals_native_lds), '
                              'so the LDS values used are never passed through the decimation filter. '
                              'See README "direct LDS interpolation" - real but modest improvement, '
                              'and only when the decimated search would have landed on a worse lag.')
    return parser


def resolve_args(argv=None):
    """Two-pass parse: find --recipe first, apply it as new defaults, then
    parse everything for real so explicit CLI flags still win over the recipe."""
    parser = build_parser()
    prelim, _ = parser.parse_known_args(argv)
    if prelim.recipe:
        parser.set_defaults(**RECIPES[prelim.recipe])
    args = parser.parse_args(argv)

    # Fall back to the configured data locations (paths.py: environment
    # variables or configs/paths.json) when --video/--lds are not given. This
    # is the normal path - the recipes deliberately no longer carry absolute
    # paths, since baking one machine's layout into the repo is what made it
    # unrunnable anywhere else.
    for name, resolver in (('video', video_path), ('lds', lds_path)):
        if getattr(args, name) is None:
            try:
                setattr(args, name, resolver())
            except FileNotFoundError as exc:
                parser.error(
                    f"--{name} was not given and could not be resolved.\n{exc}")

    if args.tracker is None:
        parser.error("missing required argument: --tracker (or use --recipe)")
    return args


def ensure_rois(rois_path, video_path, roi_method, roi_source):
    if os.path.exists(rois_path):
        return
    if roi_method is None:
        raise FileNotFoundError(
            f"{rois_path} does not exist and no --roi-method was given to generate it. "
            f"Either create it (see data/roi_select.py / data/bootstrap_rois_auto.py / "
            f"data/import_manual_rois.py) or pass --roi-method."
        )
    print(f"[ROI] {rois_path} not found - generating via --roi-method {roi_method}...")
    ROI_METHOD_REGISTRY[roi_method](rois_path, video_path, roi_source=roi_source)


def main():
    args = resolve_args()
    repro.set_deterministic_seed(args.seed)

    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rois_path = args.rois or os.path.join(project_root, 'configs', 'rois.json')
    out_root = args.out_dir or os.path.join(project_root, 'experiments', 'results')
    run_name = args.run_name or f"{args.tracker}_{args.compensator}"
    run_dir = os.path.join(out_root, run_name)
    os.makedirs(run_dir, exist_ok=True)

    ensure_rois(rois_path, args.video, args.roi_method, args.roi_source)

    print(f"[LDS] Loading and decimating reference signal...")
    lds_50hz, _, meta_lds = load_lds_prepared(args.lds, fs_target=50, cutoff_hz=25)
    baseline_rmse = float(np.std(lds_50hz))
    print(f"    {len(lds_50hz)} samples @ 50 Hz, zero-predictor RMSE = {baseline_rmse:.4f} mm")

    use_native_lds = args.lds_alignment == 'native'
    if use_native_lds:
        print(f"[LDS] --lds-alignment native: skipping decimation for the fine alignment stage "
              f"(align_signals_native_lds interpolates the native 10kHz signal directly instead)")
        lds_native, fs_lds_native = load_lds_raw(args.lds)

    roi_tracker = ROITracker(video_path=args.video, rois_json_path=rois_path)
    compensator = COMPENSATOR_REGISTRY[args.compensator](
        affine_model=args.affine_model,
        transform_smoothing=args.transform_smoothing,
        smoothing_alpha=args.smoothing_alpha,
    )
    uses_raw_patches = args.compensator in COMPENSATORS_NEEDING_RAW_PATCHES

    # When using native-LDS alignment, pipeline.py's own align_signals() call
    # is skipped (lds_ref=None below) so it returns raw, mean-subtracted-only
    # displacement instead - alignment happens explicitly afterward, against
    # the native (undecimated) LDS, instead of the pre-decimated array.
    lds_for_pipeline = None if use_native_lds else lds_50hz

    t0 = time.time()
    if uses_raw_patches:
        if args.from_cache:
            raise ValueError(f"--compensator {args.compensator} needs raw background data "
                              f"and doesn't support --from-cache yet.")
        tracker = TRACKER_REGISTRY[args.tracker]()
        results = roi_tracker.run_full_pipeline_affine_flow(
            tracker=tracker, affine_compensator=compensator, lds_ref=lds_for_pipeline, max_frames=args.max_frames
        )
    elif args.from_cache:
        print(f"[CACHE] Reusing trajectories from {args.from_cache} (no re-tracking)")
        cached = np.load(args.from_cache)
        tracking_dict = {
            'target_trajectory_px': cached['target_trajectory_px'],
            'bg_trajectories_px': list(cached['bg_trajectories_px']),
            'target_confidence': cached['target_confidence'],
            'tracker_meta': {'tracker': f"{args.tracker} (from cache)"},
            'video_meta': {},
        }
        results = roi_tracker.run_from_trajectories(tracking_dict, compensator, lds_ref=lds_for_pipeline)
    else:
        tracker = TRACKER_REGISTRY[args.tracker]()
        results = roi_tracker.run_full_pipeline(
            tracker=tracker, compensator=compensator, lds_ref=lds_for_pipeline, max_frames=args.max_frames
        )
    elapsed = time.time() - t0

    disp_mm = results['displacement_mm']
    lds_ref = results['lds_reference']

    if use_native_lds:
        # disp_mm here is raw (mean-subtracted only, per pipeline.py's
        # lds_ref=None path) - undo that mean-subtraction isn't needed since
        # align_signals_native_lds mean-subtracts internally too; just align.
        disp_mm, lds_ref, lag_s_native, _, _ = align_signals_native_lds(
            disp_mm, lds_50hz, lds_native, fs_lds_native, fs=50, max_lag_s=1.0
        )
        results['lag_s'] = lag_s_native

    if lds_ref is None or len(lds_ref) != len(disp_mm):
        print("[ERROR] Could not align with LDS reference; aborting metrics/save.")
        return

    disp_mm_prefilter = disp_mm.copy()
    lds_ref_prefilter = lds_ref.copy()

    # Post-filter, applied to the vision signal always, and (if
    # --symmetric-lds-filter) to LDS as well - see that flag's help text.
    filter_kwargs = {'cutoff_hz': args.filter_cutoff_hz, 'band_hz': args.filter_band_hz,
                      'wiener_alpha': args.wiener_alpha}
    disp_mm = FILTER_REGISTRY[args.filter](disp_mm, fs=50, **filter_kwargs)
    if args.symmetric_lds_filter and args.filter != 'none':
        lds_ref = FILTER_REGISTRY[args.filter](lds_ref, fs=50, **filter_kwargs)

    # --- Asymmetric-filter artifact floor ---------------------------------
    # Filtering the vision signal but NOT the LDS reference means every bit of
    # LDS energy outside the passband is, by construction, unreachable: the
    # vision signal has exactly zero content there, so it enters the RMSE as a
    # fixed penalty that no tracker or compensator can ever reduce. This is
    # not a subtlety - for the long-used `--filter bandpass --filter-band-hz 9
    # 14` default it is 0.129mm, i.e. ~87% of a 0.149mm "result", and it
    # silently masked real differences between compensators for most of this
    # project's history (see README "the metric convention trap"). Quantify
    # and surface it rather than letting the headline number stand alone.
    metric_floor_mm = None
    if args.filter != 'none' and not args.symmetric_lds_filter:
        lds_out_of_band = lds_ref - FILTER_REGISTRY[args.filter](lds_ref, fs=50, **filter_kwargs)
        metric_floor_mm = float(np.sqrt(np.mean(lds_out_of_band ** 2)))

    report = MetricsReport(disp_mm, lds_ref, fs=50)
    print(f"\n{report}")
    if metric_floor_mm is not None:
        true_err = float(np.sqrt(max(report.rmse_mm ** 2 - metric_floor_mm ** 2, 0.0)))
        print(f"\n[METRIC WARNING] --filter {args.filter} was applied to the vision signal but not to "
              f"LDS (no --symmetric-lds-filter).\n"
              f"    {metric_floor_mm:.4f} mm of this run's {report.rmse_mm:.4f} mm RMSE is out-of-band LDS "
              f"energy that no method can reach.\n"
              f"    In-band (i.e. actually attributable) error is ~{true_err:.4f} mm. For a like-for-like "
              f"comparison against the\n"
              f"    template-match + affine-flow configuration, use --filter bandpass --filter-band-hz 0.2 20 "
              f"--symmetric-lds-filter.")

    # --- Save submission-format CSV: time_s, displacement_mm @ 50 Hz ---
    csv_path = os.path.join(run_dir, 'displacement.csv')
    time_s = np.arange(len(disp_mm)) / 50.0
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time_s', 'displacement_mm'])
        for t, d in zip(time_s, disp_mm):
            writer.writerow([f"{t:.4f}", f"{d:.6f}"])

    # --- Always save the aligned, pre-filter (displacement_mm, lds_reference)
    # pair - tiny arrays either way, and reusable to test any --filter later
    # without re-tracking. Without this, reloading displacement.csv (which is
    # POST-filter) and comparing it against a freshly-decimated LDS array is
    # silently wrong on two counts: it's already filtered, and it's already
    # lag-shifted/edge-trimmed by align_signals so it no longer lines up
    # sample-for-sample with a plain lds_50hz[:len(disp_mm)] slice. (Bit us
    # once - see README Reference Reproduction section.) ---
    np.savez(
        os.path.join(run_dir, 'aligned_signals.npz'),
        displacement_mm=disp_mm_prefilter,
        lds_reference=lds_ref_prefilter,
    )

    # --- Cache raw trajectories so post-processing tweaks don't require re-tracking ---
    # (affine_flow's tracking_dict holds raw bg patches/frames, not point
    # trajectories - those are large and cheap to regenerate, so skipped here.)
    # --- Transform-smoothing debug CSV: per-frame raw vs smoothed background
    # transform params (+ the inlier/error diagnostics that drove Kalman's
    # per-frame measurement weighting), so smoothing can be checked by hand
    # (e.g. plotting) for lag/attenuation at the VIV resonance instead of
    # just trusted from the aggregate metrics below. ---
    if args.compensator == 'affine_flow' and args.transform_smoothing != 'none':
        debug_arrays = compensator.get_transform_debug_arrays()
        if debug_arrays is not None:
            debug_csv_path = os.path.join(run_dir, 'transform_smoothing_debug.csv')
            debug_fieldnames = list(debug_arrays.keys())
            n_rows = len(debug_arrays['frame'])
            with open(debug_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(debug_fieldnames)
                for i in range(n_rows):
                    writer.writerow([debug_arrays[k][i] for k in debug_fieldnames])
            print(f"[DEBUG] Wrote per-frame raw-vs-smoothed transform params to {debug_csv_path}")

    tracking_dict = results['tracking_dict']
    if not uses_raw_patches:
        npz_path = os.path.join(run_dir, 'trajectories.npz')
        np.savez(
            npz_path,
            target_trajectory_px=tracking_dict['target_trajectory_px'],
            bg_trajectories_px=np.array(tracking_dict['bg_trajectories_px']),
            target_confidence=tracking_dict['target_confidence'],
            displacement_mm=disp_mm,
            lds_reference=lds_ref,
        )

    # --- Metrics JSON ---
    metrics = {
        'run_name': run_name,
        'tracker': args.tracker,
        'compensator': args.compensator,
        'affine_model': args.affine_model if args.compensator == 'affine_flow' else '',
        'transform_smoothing': args.transform_smoothing if args.compensator == 'affine_flow' else '',
        'smoothing_alpha': args.smoothing_alpha if (args.compensator == 'affine_flow'
                                                      and args.transform_smoothing == 'ema') else '',
        'lds_alignment': args.lds_alignment,
        'wiener_alpha': args.wiener_alpha if args.filter == 'bandpass_wiener' else '',
        'filter': args.filter,
        # Out-of-band LDS energy included in rmse_mm purely because --filter was
        # applied asymmetrically; blank when the comparison is symmetric (or
        # unfiltered) and the RMSE is therefore fully attributable.
        'metric_artifact_floor_mm': metric_floor_mm if metric_floor_mm is not None else '',
        'symmetric_lds_filter': bool(args.symmetric_lds_filter),
        'rmse_mm': report.rmse_mm,
        'normalized_rmse': report.normalized_rmse,
        'mae_mm': report.mae,
        'phase_lag_deg_11.7hz': report.phase_lag_deg,
        'coherence_11.7hz': report.coherence_117hz,
        'amplitude_ratio_11.7hz': report.amplitude_ratio_117hz,
        'baseline_rmse_mm': baseline_rmse,
        'improvement_pct': (1 - report.rmse_mm / baseline_rmse) * 100,
        'n_frames': int(tracking_dict['target_trajectory_px'].shape[0]),
        'lag_s': results['lag_s'],
        'mm_per_px': results['mm_per_px'],
        'elapsed_s': elapsed,
        'seed': args.seed,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
    }
    with open(os.path.join(run_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)

    # --- Self-contained repro config: resolved args + git state + package
    # versions, so this exact row can be replayed without remembering flags. ---
    config = repro.build_run_config(args, repo_dir=project_root)
    with open(os.path.join(run_dir, 'config.json'), 'w') as f:
        json.dump(config, f, indent=2)
    if config['git']['dirty']:
        print("[WARN] Working tree has uncommitted changes - config.json records the base "
              "commit, but exact code state isn't fully captured until you commit.")

    leaderboard_path = rebuild_leaderboard(out_root)

    print("\n" + "=" * 70)
    print(f"SAVED: {run_dir}")
    print(f"RMSE = {report.rmse_mm:.4f} mm  |  baseline (zero) = {baseline_rmse:.4f} mm  "
          f"|  improvement = {metrics['improvement_pct']:.1f}%")
    print(f"Elapsed: {elapsed:.1f}s ({elapsed / metrics['n_frames']:.3f} s/frame)")
    print(f"Leaderboard: {leaderboard_path}")
    print("=" * 70)


if __name__ == '__main__':
    main()
