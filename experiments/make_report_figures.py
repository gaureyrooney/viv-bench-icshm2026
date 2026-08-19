"""
Generate every figure used in the final report (docs/report.html).

All figures are produced from saved run artefacts (aligned_signals.npz,
metrics.json) plus one video frame - no re-tracking - so the report can be
regenerated cheaply and stays consistent with the leaderboard.
"""
import glob
import json
import os
import sys

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sps

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from paths import video_path, lds_path
from data.lds import load_lds_raw
from dsp.alignment import zero_phase_filter
from dsp.wiener_denoise import wiener_denoise, estimate_noise_psd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RES = os.path.join(ROOT, 'experiments', 'results')
OUT = os.path.join(ROOT, 'docs', 'figures')
TIFF_OUT = os.path.join(OUT, 'tiff')      # archival 300 dpi copies
VIDEO = video_path()
LDS = lds_path()
BEST = 'postaudit_cornerklt_affineflow'

FG, BG, GRID = '#1a1a1a', '#ffffff', '#d8d8d8'
C_VIS, C_LDS, C_ERR, C_ACC = '#0b6fb8', '#c8102e', '#8a8a8a', '#1b7f4b'
plt.rcParams.update({'figure.facecolor': BG, 'axes.facecolor': BG, 'savefig.facecolor': BG,
                     'axes.edgecolor': FG, 'axes.labelcolor': FG, 'text.color': FG,
                     'xtick.color': FG, 'ytick.color': FG, 'font.size': 9,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6,
                     'axes.spines.top': False, 'axes.spines.right': False})

bp = lambda x, b=(0.2, 20.): zero_phase_filter(x, fs=50, cutoff_hz=b, btype='band')


def save(fig, name, tiff_dpi=300):
    """Write each figure twice.

    - docs/figures/<name>          web-optimised PNG/JPEG at 150 dpi, embedded
                                   in the HTML/Word report
    - docs/figures/tiff/<stem>.tif lossless LZW-compressed TIFF at 300 dpi,
                                   the archival copy for reuse in papers,
                                   posters and journal submissions (most
                                   publishers require TIFF at >=300 dpi)

    Both come from the same figure object, so they can never disagree.
    """
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(TIFF_OUT, exist_ok=True)
    fig.tight_layout()

    p = os.path.join(OUT, name)
    fig.savefig(p, dpi=150)

    stem = os.path.splitext(name)[0]
    p_tiff = os.path.join(TIFF_OUT, stem + '.tif')
    # pil_kwargs routes LZW compression through Pillow's TIFF writer; without
    # it a 300 dpi RGBA TIFF of these figures runs to tens of megabytes.
    fig.savefig(p_tiff, dpi=tiff_dpi, format='tiff',
                pil_kwargs={'compression': 'tiff_lzw'})

    plt.close(fig)
    print(f'  wrote {os.path.relpath(p, ROOT)}  +  {os.path.relpath(p_tiff, ROOT)} ({tiff_dpi} dpi)')


def load(run):
    d = np.load(os.path.join(RES, run, 'aligned_signals.npz'))
    return bp(d['displacement_mm']), bp(d['lds_reference'])


# ---------------------------------------------------------------- fig 1: scene
def fig_scene():
    cap = cv2.VideoCapture(VIDEO)
    ok, f = cap.read()
    cap.release()
    if not ok:
        return
    rois = json.load(open(os.path.join(ROOT, 'configs', 'rois.json')))
    img = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.imshow(img)
    for i, (x, y, w, h) in enumerate(rois['bg_boxes']):
        ax.add_patch(plt.Rectangle((x, y), w, h, ec=C_ACC, fc='none', lw=2))
        ax.annotate(f'background ROI {i+1}\n({int(w)}x{int(h)} px, {"8" if i==0 else "9"} features)',
                    (x + w, y), xytext=(x + w + 260, y - 190), color=C_ACC, fontsize=8,
                    arrowprops=dict(arrowstyle='->', color=C_ACC, lw=1.2))
    tx, ty = rois['target_point']
    ax.add_patch(plt.Circle((tx, ty), 60, ec=C_LDS, fc='none', lw=2))
    ax.annotate('checkerboard target\non the stay cable', (tx, ty), xytext=(tx - 900, ty - 330),
                color=C_LDS, fontsize=8, arrowprops=dict(arrowstyle='->', color=C_LDS, lw=1.2))
    u = np.array(rois['cable_axis']['viv_direction'])
    ax.arrow(tx, ty, u[0] * 320, u[1] * 320, color=C_LDS, width=8, head_width=45)
    ax.annotate('VIV measurement\ndirection', (tx + u[0] * 320, ty + u[1] * 320),
                xytext=(tx + 300, ty - 480), color=C_LDS, fontsize=8)
    ax.set_axis_off()
    ax.set_title('Fig 1 — Measurement scene (frame 0, 3840x2160). Target and the two background '
                 'reference ROIs.', fontsize=9, loc='left')
    save(fig, 'fig1_scene.png')


# --------------------------------------------- fig 2: time series + zoom
def fig_timeseries():
    v, l = load(BEST)
    t = np.arange(len(v)) / 50.
    fig, ax = plt.subplots(2, 1, figsize=(9, 5.2), gridspec_kw={'height_ratios': [1, 1]})
    ax[0].plot(t, l, color=C_LDS, lw=0.8, label='LDS reference')
    ax[0].plot(t, v, color=C_VIS, lw=0.8, alpha=0.85, label='vision (corner_klt + affine_flow)')
    ax[0].set_xlim(0, t[-1]); ax[0].set_ylabel('displacement (mm)')
    ax[0].legend(loc='upper right', frameon=False, ncol=2)
    ax[0].set_title('Fig 2 — Full 60 s record and a 1.5 s detail. RMSE = 0.0707 mm against a '
                    '0.376 mm RMS signal.', fontsize=9, loc='left')
    m = (t >= 30) & (t <= 31.5)
    ax[1].plot(t[m], l[m], color=C_LDS, lw=1.6, marker='o', ms=3, label='LDS')
    ax[1].plot(t[m], v[m], color=C_VIS, lw=1.6, marker='s', ms=3, label='vision')
    ax[1].set_xlabel('time (s)'); ax[1].set_ylabel('displacement (mm)')
    ax[1].legend(loc='upper right', frameon=False, ncol=2)
    save(fig, 'fig2_timeseries.png')


# ------------------------------------------------- fig 3: metric artifact
def fig_metric_artifact():
    v, l = load(BEST)
    lraw = np.load(os.path.join(RES, BEST, 'aligned_signals.npz'))['lds_reference']
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.6))
    band = (9., 14.)
    lb = zero_phase_filter(lraw, fs=50, cutoff_hz=band, btype='band')
    out = lraw - lb
    floor = np.sqrt(np.mean(out ** 2))
    inb = np.sqrt(np.mean((zero_phase_filter(np.load(os.path.join(RES, BEST, 'aligned_signals.npz'))['displacement_mm'],
                                              fs=50, cutoff_hz=band, btype='band') - lb) ** 2))
    tot = np.sqrt(floor ** 2 + inb ** 2)
    ax[0].bar(['reported\nRMSE', 'metric\nartifact', 'true in-band\nerror'], [tot, floor, inb],
              color=[C_ERR, C_LDS, C_ACC])
    for i, val in enumerate([tot, floor, inb]):
        ax[0].text(i, val + 0.003, f'{val:.4f}', ha='center', fontsize=8)
    ax[0].set_ylabel('mm'); ax[0].set_title('(a) Asymmetric 9-14 Hz filtering:\n87% of the number was '
                                            'unreachable', fontsize=9, loc='left')
    f, P = sps.welch(lraw, fs=50, nperseg=1024)
    ax[1].semilogy(f, P, color=C_LDS, lw=1.2)
    ax[1].axvspan(*band, color=C_ACC, alpha=0.15, label='vision kept here')
    ax[1].axvspan(0.2, band[0], color=C_LDS, alpha=0.12)
    ax[1].axvspan(band[1], 25, color=C_LDS, alpha=0.12, label='LDS energy discarded\nfrom vision only')
    ax[1].set_xlim(0, 25); ax[1].set_xlabel('frequency (Hz)'); ax[1].set_ylabel('LDS PSD (mm$^2$/Hz)')
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title('(b) 12.2% of LDS energy sits outside 9-14 Hz', fontsize=9, loc='left')
    save(fig, 'fig3_metric_artifact.png')


# --------------------------------------------------- fig 4: polarity bug
def fig_polarity():
    from scipy import ndimage
    d = np.load(os.path.join(RES, 'icgn_differential_bp9-14_v2', 'trajectories.npz'))
    lds_raw, fsn = load_lds_raw(LDS)
    sys.path.insert(0, ROOT)
    from data.lds import load_lds_prepared
    from pipeline import ROITracker
    from experiments.registries import COMPENSATOR_REGISTRY
    rt = ROITracker(video_path=VIDEO, rois_json_path=os.path.join(ROOT, 'configs', 'rois.json'))
    td = {'target_trajectory_px': d['target_trajectory_px'],
          'bg_trajectories_px': list(d['bg_trajectories_px']),
          'target_confidence': d['target_confidence'], 'tracker_meta': {}, 'video_meta': {}}
    disp = rt.project_to_viv_direction(
        rt.apply_compensation(td, COMPENSATOR_REGISTRY['differential'](affine_model='similarity'))['trajectory_px']
    ) * rt.mm_per_px
    lds, _, _ = load_lds_prepared(LDS)
    n = min(len(disp), len(lds)); disp, lds = disp[:n], lds[:n]
    lags = np.arange(0.10, 0.32, 0.002)
    fig, ax = plt.subplots(figsize=(9, 3.4))
    for sgn, col, lab in [(1, C_ERR, 'polarity +1 (what the old search assumed)'),
                          (-1, C_ACC, 'polarity -1 (correct)')]:
        e = []
        for lg in lags:
            sh = ndimage.shift(disp, shift=-lg * 50, order=3, mode='nearest') * sgn
            s = slice(20, n - 20)
            e.append(np.sqrt(np.mean((bp(sh[s]) - bp(lds[s])) ** 2)))
        ax.plot(lags, e, color=col, lw=1.6, label=lab)
    ax.axvline(0.240, color=C_ERR, ls=':', lw=1.2)
    ax.axvline(0.200, color=C_ACC, ls=':', lw=1.2)
    ax.annotate('old search locked here\n(0.240 s, +1)', (0.240, 0.26), xytext=(0.252, 0.30),
                color=C_ERR, fontsize=8, arrowprops=dict(arrowstyle='->', color=C_ERR))
    ax.annotate('true optimum\n(0.200 s, -1)', (0.200, 0.146), xytext=(0.140, 0.09),
                color=C_ACC, fontsize=8, arrowprops=dict(arrowstyle='->', color=C_ACC))
    ax.set_xlabel('assumed lag (s)'); ax.set_ylabel('RMSE 0.2-20 Hz (mm)')
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Fig 4 — The polarity bug. A sign-blind search finds a decoy optimum half a VIV '
                 'period (43 ms) away.', fontsize=9, loc='left')
    save(fig, 'fig4_polarity.png')


# ------------------------------------------------ fig 5: noise budget/bands
def fig_bands_budget():
    v, l = load(BEST)
    r = v - l
    bands = [(0.2, 1), (1, 2), (2, 5), (5, 8), (8, 10), (10, 11), (11, 12.5), (12.5, 14), (14, 20)]
    lab = [f'{a}-{b}' for a, b in bands]
    rr = [zero_phase_filter(r, fs=50, cutoff_hz=b, btype='band').std() for b in bands]
    ll = [zero_phase_filter(l, fs=50, cutoff_hz=b, btype='band').std() for b in bands]
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    x = np.arange(len(bands))
    ax[0].bar(x - 0.2, ll, 0.4, color=C_LDS, label='LDS signal')
    ax[0].bar(x + 0.2, rr, 0.4, color=C_ERR, label='residual error')
    ax[0].set_yscale('log'); ax[0].set_xticks(x); ax[0].set_xticklabels(lab, rotation=45, fontsize=7)
    ax[0].set_xlabel('band (Hz)'); ax[0].set_ylabel('RMS (mm)'); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title('(a) Error is broadband; signal is not', fontsize=9, loc='left')
    src = ['observed\nresidual', 'compensation\nnoise', 'tracker\nnoise', 'LDS sensor\nnoise', 'systematic\n(unexplained)']
    val = [0.0707, 0.0187, 0.0095, 0.00002, 0.0682]
    ax[1].bar(src, val, color=[C_ERR, C_VIS, C_VIS, C_ACC, '#b07d2b'])
    for i, vv in enumerate(val):
        ax[1].text(i, vv + 0.002, f'{vv:.4f}', ha='center', fontsize=7)
    ax[1].set_ylabel('mm'); ax[1].tick_params(axis='x', labelsize=7)
    ax[1].set_title('(b) Noise budget: 93% of variance is not stage noise', fontsize=9, loc='left')
    save(fig, 'fig5_bands_budget.png')


# ---------------------------------------------------- fig 6: wiener result
def fig_wiener():
    v, l = load(BEST)
    n = len(v); h = n // 2
    Pnn = estimate_noise_psd(v[:h], fs=50)
    alphas = [0, 1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 128]
    tr, te = [], []
    for a in alphas:
        vt = wiener_denoise(v[:h], fs=50, alpha=a, noise_psd=Pnn) if a else v[:h]
        vv = wiener_denoise(v[h:], fs=50, alpha=a, noise_psd=Pnn) if a else v[h:]
        tr.append(np.sqrt(np.mean((vt - l[:h]) ** 2)))
        te.append(np.sqrt(np.mean((vv - l[h:]) ** 2)))
    fig, ax = plt.subplots(1, 2, figsize=(9.5, 3.6))
    ax[0].plot(alphas, tr, 'o-', color=C_VIS, ms=4, label='train (first 30 s)')
    ax[0].plot(alphas, te, 's-', color=C_ACC, ms=4, label='held-out (last 30 s)')
    ax[0].axvline(16, color=FG, ls=':', lw=1)
    ax[0].annotate('interior optimum\nselected on train', (16, min(te)), xytext=(30, max(te) * 0.95),
                   fontsize=8, arrowprops=dict(arrowstyle='->', color=FG))
    ax[0].set_xscale('symlog'); ax[0].set_xlabel(r'over-subtraction factor $\alpha$')
    ax[0].set_ylabel('RMSE (mm)'); ax[0].legend(frameon=False, fontsize=8)
    ax[0].set_title(r'(a) An interior optimum exists — so this is denoising,''\n''not degeneration to a narrow band',
                    fontsize=9, loc='left')
    vd = wiener_denoise(v, fs=50, alpha=16, noise_psd=Pnn)
    f, Pv = sps.welch(v, fs=50, nperseg=1024)
    _, Pd = sps.welch(vd, fs=50, nperseg=1024)
    _, Pl = sps.welch(l, fs=50, nperseg=1024)
    ax[1].semilogy(f, Pv, color=C_ERR, lw=1, label='vision, raw')
    ax[1].semilogy(f, Pd, color=C_VIS, lw=1.2, label=r'vision, Wiener $\alpha$=16')
    ax[1].semilogy(f, Pl, color=C_LDS, lw=1, ls='--', label='LDS')
    ax[1].set_xlim(0, 25); ax[1].set_xlabel('frequency (Hz)'); ax[1].set_ylabel('PSD (mm$^2$/Hz)')
    ax[1].legend(frameon=False, fontsize=8)
    ax[1].set_title('(b) Noise-dominated bins are shrunk;\nthe 11.7 Hz resonance is preserved', fontsize=9, loc='left')
    save(fig, 'fig6_wiener.png')


# ------------------------------------------------------- fig 7: ablation
def fig_ablation():
    rows = []
    for p in glob.glob(os.path.join(RES, '*', 'metrics.json')):
        m = json.load(open(p))
        rows.append(m)
    def get(name):
        for m in rows:
            if m.get('run_name') == name:
                return m.get('rmse_mm')
        return None
    grid = {
        'none': {'icgn': get('ABL_icgn_none'), 'corner_klt': get('ABL_corner_klt_none'),
                 'template_match': get('ABL_template_match_none')},
        'differential': {'icgn': get('icgn_differential_bp0.2-20_symlds_SIGNFIX'),
                         'corner_klt': get('ABL_corner_klt_differential'),
                         'template_match': get('ABL_template_match_differential')},
        'affine_flow': {'icgn': get('icgn_affineflow_bp0.2-20_symlds_SIGNFIX'),
                        'corner_klt': get('postaudit_cornerklt_affineflow'),
                        'template_match': get('templatematch_affineflow_bp0.2-20_symlds_SIGNFIX')},
    }
    trackers = ['icgn', 'corner_klt', 'template_match']
    comps = ['none', 'differential', 'affine_flow']
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    w = 0.25
    for i, c in enumerate(comps):
        vals = [grid[c][t] if grid[c][t] else np.nan for t in trackers]
        b = ax.bar(np.arange(len(trackers)) + (i - 1) * w, vals, w, label=c)
        for xx, vv in zip(np.arange(len(trackers)) + (i - 1) * w, vals):
            if np.isfinite(vv):
                ax.text(xx, vv * 1.04, f'{vv:.3f}', ha='center', fontsize=7)
    ax.axhline(0.07039, color=C_LDS, ls='--', lw=1.2, label='reference impl. 0.0704')
    ax.set_yscale('log'); ax.set_xticks(range(len(trackers))); ax.set_xticklabels(trackers)
    ax.set_ylabel('RMSE (mm), sym 0.2-20 Hz'); ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.set_title('Fig 7 — Tracker x compensator ablation. Compensation dominates; tracker choice is '
                 'second order.', fontsize=9, loc='left')
    save(fig, 'fig7_ablation.png')


if __name__ == '__main__':
    print('generating figures...')
    for fn in [fig_scene, fig_timeseries, fig_metric_artifact, fig_polarity,
               fig_bands_budget, fig_wiener, fig_ablation]:
        try:
            fn()
        except Exception as e:
            print(f'  [skip] {fn.__name__}: {type(e).__name__}: {e}')
    print('done')
