"""
Reproduce every result in the report, end to end, from the raw video.

This is the single entry point a reader should use to verify the project. It
runs the full ablation grid, the diagnostic experiments, the regression check
and the figure/report generation, in dependency order, and prints a pass/fail
comparison against the values published in docs/report.html.

    python experiments/reproduce_all.py                 # everything (~45 min)
    python experiments/reproduce_all.py --stage ablation
    python experiments/reproduce_all.py --list
    python experiments/reproduce_all.py --dry-run

Prerequisites: the input data paths must be configured - see paths.py
(environment variables, or configs/paths.json copied from
configs/paths.example.json). `python paths.py` prints what is currently set.

Every stage is idempotent: results land in experiments/results/<run_name>/ and
re-running overwrites that directory. The leaderboard is always regenerated
from the per-run metrics.json files, never appended to.
"""
import argparse
import json
import os
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import video_path, lds_path, templatematch_affine_script, repo_root

ROOT = repo_root()
PY = sys.executable
RES = os.path.join(ROOT, 'experiments', 'results')

# Reference values published in docs/report.html. reproduce_all checks against
# these so a regression is reported as a number, not just a crash.
EXPECTED = {
    'ABL_icgn_none': 1.7004, 'ABL_corner_klt_none': 1.6987, 'ABL_template_match_none': 1.7025,
    'icgn_differential_bp0.2-20_symlds_SIGNFIX': 0.1482,
    'ABL_corner_klt_differential': 0.3953, 'ABL_template_match_differential': 0.1502,
    'icgn_affineflow_bp0.2-20_symlds_SIGNFIX': 0.0711,
    'postaudit_cornerklt_affineflow': 0.0707,
    'templatematch_affineflow_bp0.2-20_symlds_SIGNFIX': 0.0773,
    'postaudit_icgn_affineflow_full': 0.1931,
    'postaudit_icgn_affineflow_homography': 0.1280,
    'postaudit_gabor_affineflow': 0.1432,
    'postaudit_icgn_affineflow_ema0.5': 0.2089,
    'postaudit_icgn_affineflow_kalman': 0.1792,
    'FINAL_corner_klt_affineflow_wiener16': 0.0577,
    'FINAL_icgn_affineflow_wiener16': 0.0583,
    'templatematch_affine_config': 0.07039,
}

# The evaluation convention used for every comparable number in the report:
# both signals band-limited to 0.2-20 Hz. Mixing conventions in one table is
# exactly what hid a 0.129 mm artifact for most of the project (report §4).
SYM = ['--filter', 'bandpass', '--filter-band-hz', '0.2', '20', '--symmetric-lds-filter']


def _exp(run_name, *extra):
    """One run_experiment.py invocation, with data paths filled in from paths.py."""
    return [PY, os.path.join('experiments', 'run_experiment.py'),
            '--video', video_path(), '--lds', lds_path(),
            '--run-name', run_name, *extra]


def _script(name, *extra):
    return [PY, os.path.join('experiments', name), *extra]


# ---------------------------------------------------------------------------
# Stages. Each is (name, description, [commands]).
# ---------------------------------------------------------------------------
def build_stages():
    aff = ['--compensator', 'affine_flow', '--affine-model', 'similarity']
    stages = []

    stages.append(('ablation', 'Tracker x compensator grid (report Fig 7, Table §3)', [
        # no compensation - establishes what the camera motion alone costs
        _exp('ABL_icgn_none', '--tracker', 'icgn', '--compensator', 'none', *SYM),
        _exp('ABL_corner_klt_none', '--tracker', 'corner_klt', '--compensator', 'none', *SYM),
        _exp('ABL_template_match_none', '--tracker', 'template_match', '--compensator', 'none', *SYM),
        # translation-only (2 background points)
        _exp('icgn_differential_bp0.2-20_symlds_SIGNFIX', '--tracker', 'icgn',
             '--compensator', 'differential', *SYM),
        _exp('ABL_corner_klt_differential', '--tracker', 'corner_klt', '--compensator', 'differential', *SYM),
        _exp('ABL_template_match_differential', '--tracker', 'template_match',
             '--compensator', 'differential', *SYM),
        # full-frame RANSAC similarity - the working configuration
        _exp('icgn_affineflow_bp0.2-20_symlds_SIGNFIX', '--tracker', 'icgn', *aff, *SYM),
        _exp('postaudit_cornerklt_affineflow', '--tracker', 'corner_klt', *aff, *SYM),
        _exp('templatematch_affineflow_bp0.2-20_symlds_SIGNFIX', '--tracker', 'template_match', *aff, *SYM),
    ]))

    stages.append(('variants', 'Affine DOF, extra trackers, transform smoothing (report §7)', [
        _exp('postaudit_icgn_affineflow_full', '--tracker', 'icgn',
             '--compensator', 'affine_flow', '--affine-model', 'full', *SYM),
        _exp('postaudit_icgn_affineflow_homography', '--tracker', 'icgn',
             '--compensator', 'affine_flow', '--affine-model', 'homography', *SYM),
        _exp('postaudit_gabor_affineflow', '--tracker', 'gabor_phase', *aff, *SYM),
        _exp('postaudit_icgn_affineflow_ema0.5', '--tracker', 'icgn', *aff,
             '--transform-smoothing', 'ema', '--smoothing-alpha', '0.5', *SYM),
        _exp('postaudit_icgn_affineflow_ema0.3', '--tracker', 'icgn', *aff,
             '--transform-smoothing', 'ema', '--smoothing-alpha', '0.3', *SYM),
        _exp('postaudit_icgn_affineflow_ema0.1', '--tracker', 'icgn', *aff,
             '--transform-smoothing', 'ema', '--smoothing-alpha', '0.1', *SYM),
        _exp('postaudit_icgn_affineflow_kalman', '--tracker', 'icgn', *aff,
             '--transform-smoothing', 'kalman', *SYM),
    ]))

    stages.append(('final', 'Best configuration: + spectral (Wiener) denoising (report §6)', [
        _exp('FINAL_corner_klt_affineflow_wiener16', '--tracker', 'corner_klt', *aff,
             '--filter', 'bandpass_wiener', '--filter-band-hz', '0.2', '20',
             '--wiener-alpha', '16', '--symmetric-lds-filter'),
        _exp('FINAL_icgn_affineflow_wiener16', '--tracker', 'icgn', *aff,
             '--filter', 'bandpass_wiener', '--filter-band-hz', '0.2', '20',
             '--wiener-alpha', '16', '--symmetric-lds-filter'),
    ]))

    stages.append(('regression', "frozen-config exact-reproduction check (report §8)",
                   [_script('run_templatematch_affine_config.py')]))

    stages.append(('diagnostics', 'Noise budget and component ablation (report §5, §8)', [
        _script('noise_budget.py'),
        _script('ablation_component_swap.py'),
        _script('phase_lag_isolation.py'),
    ]))

    stages.append(('report', 'Figures, then the HTML/Word report (report all figures)', [
        _script('make_report_figures.py'),
        _script('build_report.py'),
    ]))
    return stages


def run(cmds, dry_run=False):
    for cmd in cmds:
        pretty = ' '.join(f'"{c}"' if ' ' in str(c) else str(c) for c in cmd)
        print(f'\n  $ {pretty}\n', flush=True)
        if dry_run:
            continue
        t0 = time.time()
        r = subprocess.run(cmd, cwd=ROOT)
        if r.returncode != 0:
            print(f'  [FAIL] exit {r.returncode} after {time.time()-t0:.0f}s')
            return False
        print(f'  [ok] {time.time()-t0:.0f}s')
    return True


def verify():
    """Compare every reproduced RMSE against the published value."""
    print('\n' + '=' * 78)
    print('VERIFICATION against values published in docs/report.html')
    print('=' * 78)
    print(f'{"run":52s} {"published":>10} {"actual":>10}  status')
    worst = 0.0
    missing = 0
    for name, want in sorted(EXPECTED.items(), key=lambda kv: kv[1]):
        p = os.path.join(RES, name, 'metrics.json')
        if not os.path.exists(p):
            print(f'{name:52s} {want:10.4f} {"-":>10}  NOT RUN')
            missing += 1
            continue
        got = json.load(open(p)).get('rmse_mm', float('nan'))
        rel = abs(got - want) / max(want, 1e-9)
        worst = max(worst, rel)
        flag = 'ok' if rel < 0.02 else ('DRIFT' if rel < 0.10 else 'MISMATCH')
        print(f'{name:52s} {want:10.4f} {got:10.4f}  {flag} ({rel*100:.1f}%)')
    print('-' * 78)
    print(f'worst relative deviation: {worst*100:.2f}%   runs not present: {missing}')
    print('Tolerance note: RANSAC is seeded (--seed, default 42) so runs are '
          'reproducible;\n  small deviations across OpenCV/SciPy versions are expected, '
          'large ones are not.')
    return worst < 0.10 and missing == 0


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--stage', action='append', default=None,
                    help='run only this stage (repeatable). Default: all, in order.')
    ap.add_argument('--list', action='store_true', help='list stages and exit')
    ap.add_argument('--dry-run', action='store_true', help='print commands without running')
    ap.add_argument('--skip-verify', action='store_true')
    args = ap.parse_args()

    stages = build_stages()
    if args.list:
        print('stages (in dependency order):')
        for n, d, c in stages:
            print(f'  {n:12s} {len(c):2d} command(s)  {d}')
        return 0

    # fail early and clearly if the data paths are not configured
    if not args.dry_run:
        try:
            video_path(), lds_path()
        except FileNotFoundError as e:
            print(f'[ERROR] {e}')
            return 2

    selected = [s for s in stages if args.stage is None or s[0] in args.stage]
    if not selected:
        print(f'no stage matched {args.stage}; use --list')
        return 2

    t0 = time.time()
    for name, desc, cmds in selected:
        print('\n' + '=' * 78)
        print(f'STAGE {name} — {desc}')
        print('=' * 78)
        if not run(cmds, dry_run=args.dry_run):
            print(f'\nstage {name} failed; stopping.')
            return 1
    print(f'\nall selected stages finished in {(time.time()-t0)/60:.1f} min')

    if not args.dry_run and not args.skip_verify:
        return 0 if verify() else 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
