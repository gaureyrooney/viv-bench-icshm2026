# Investigation log

Chronological record of every experiment run on this dataset, including the
negative results and the two measurement bugs that invalidated a large part of
the earlier work. Kept verbatim rather than tidied, because the sequence in
which the errors were found is itself the most useful part.

For the finished write-up see [report.html](report.html) (or `report.docx` /
`report.md`). For usage see the [README](../README.md).

---

## Literature-informed experiments

Brief review of 2024-2026 UAV/vision structural-displacement literature suggested
three specific things to try; each was implemented and tested, not just discussed:

1. **Homography vs. affine ego-motion model.** Several papers prefer full
   projective (homography) over affine for UAV compensation, since it also
   captures perspective effects from small attitude/depth changes. Added
   `motion_model='homography'` to `AffineFlowCompensator`
   (`cv2.findHomography` + RANSAC), validated against a synthetic known-camera-
   motion test (0.06px residual error) before trusting it on real data.
   **Result: worse** (0.279mm vs. 0.181mm affine). With only ~35 feature points
   clustered in 2 small background ROIs, the extra 2 projective DOF are
   underconstrained and amplify noise rather than capture real perspective
   effects - would likely need more, and more spatially spread, background
   features to pay off (e.g. the 3rd, larger `bg_roi` from the manual
   selection that's currently unused).
2. **VMD (Variational Mode Decomposition) residual cleanup**, per a two-stage
   UAV correction approach (geometric compensation, then adaptive decomposition
   of the residual, keeping only modes near the known structural frequency)
   (`dsp/vmd_filter.py`, using `vmdpy`). **Result: substantially worse**
   (~0.42mm vs. 0.181mm bandpass), stable across K=3-8 and alpha=200-10000.
   Reconstructing from a subset of VMD modes necessarily discards information;
   here it seems to discard more real signal than noise. Not pursued further.
3. **Cubic (vs. bilinear) interpolation in IC-GN's warp sampling**, to reduce
   DIC-literature-documented "peak-locking" bias (sub-pixel estimates biased
   toward integer pixels, worse for high-spatial-frequency content like a
   checkerboard). Confirmed a 13x precision gain on synthetic shifts
   (0.011px -> 0.0008px mean error) - but **no meaningful change on real data**
   (0.1805mm vs. 0.1808mm). Confirms the tracker's own sub-pixel precision was
   already not the bottleneck; real-world error comes from elsewhere.

Net effect of this round: none of the three closed the gap directly, but ruling
them out redirected the search - trying `differential` instead of `affine_flow`
with the now-precise IC-GN tracker is what actually produced the new best result
(0.150mm, see above).

## Additional trackers (method survey)

Two more independent trackers, chosen to cover mechanisms not yet represented
(see Results table above for numbers):

- **`GaborPhaseTracker`** (`trackers/phase_based.py`) - genuine phase-based
  tracking (the technique the competition's own reference [1] falls under),
  replacing `trackers/classical_methods.py`'s `PhaseBasedTracker`, which built
  real Gabor filters but never used them (fell back to an ad hoc frame-
  difference/gradient-ratio hack with a hardcoded scale constant - left in
  place but superseded). Measures the checkerboard's 2 dominant spatial
  frequencies from the frame-0 template via 2D FFT, builds complex Gabor
  quadrature filters tuned to each, and reads displacement directly off the
  unwrapped phase trajectory - no correlation-peak search, so it's structurally
  immune to the peak-locking bias literature attributes to every other tracker
  here. Validated on synthetic shifts (correct sign, sub-0.02px error for
  realistic small shifts, growing for larger shifts as expected of a small-
  motion linearization) before real data.
- **`CornerKLTTracker`** (`trackers/corner_klt.py`) - several sub-pixel
  corners per ROI, tracked independently via pyramidal LK optical flow
  anchored to frame 0, averaged. From the original project plan, not
  previously implemented. Validated on synthetic shifts (0.004px mean error).

Both perform comparably to IC-GN/template-match (within the same 0.150-0.158mm
band with `differential` compensation) - reinforcing the "tracker choice
doesn't explain the gap" finding rather than beating it.

## Reference reproduction

`experiments/run_templatematch_affine_config.py` loads a teammate's reference
implementation (`templatematch_affine_config/icshm2026_project1_templatematch_affine_config(1).py`, claims 0.070mm
RMSE / 0.983 correlation) via `importlib` - its functions completely
unmodified - and drives them with the ROIs from its already-saved
`templatematch_affine_config_json.json` instead of its script's interactive `cv2.selectROI`
step. Everything else (template tracking, affine ego-motion compensation,
filtering, LDS comparison) runs exactly as she wrote it, against our own copy
of the video/LDS files.

**Result: exact reproduction, 0.07039mm RMSE / 0.98290 correlation, matching
her saved result to 5 decimal places.** Confirms the number is real and
genuinely reproducible from scratch, not an artifact of one lucky run or an
evaluation-methodology difference. `configs/rois.json` was then updated to her
exact values (previously we'd imported a teammate-adjacent but not identical
manual ROI set).

**Diagnosing why its `affine_flow`-equivalent outperforms ours**, now that we
can run its exact code side-by-side with ours on identical frames:

1. *Hypothesis: its background ROIs are tighter than our fixed 100x100 patch,
   picking up fewer/cleaner features.* Her config reports 17 total detected
   features vs. our ~35 with the default 100px patch. Tested: cropped our
   background patches to its exact bounding boxes (67x67, 67x70) via
   `load_roi_patches`'s new `sizes=` parameter. **Result: no real improvement**
   (0.356mm vs ~0.348mm with the same 0.2-20Hz filter) - ruled out.
2. *Actual finding: the optical flow pyramid itself is structurally weaker on
   small patches.* Her `estimate_camera_affine` runs `cv2.calcOpticalFlowPyrLK`
   on the **full 3840x2160 frame** (a real 4-level pyramid: 3840->1920->960
   ->480->240px) and gets ~100% RANSAC inlier rate (16-17 of 17 points, every
   frame checked). `AffineFlowCompensator` runs the same call with identical
   parameters (`winSize=31`, `maxLevel=4`, `ransacReprojThreshold=3.0`) but on
   small (67-100px) cropped patches - a 4-level pyramid there bottoms out
   around 4x4px, essentially degenerate for coarse-to-fine matching. Measured
   inlier rate on matching data: ~13.6/~15 (still high, but a real gap, and
   inlier *count* likely understates how much noisier the accepted points are
   given a worse pyramid). This is an architectural consequence of our
   streaming-crop design (built to avoid decoding the whole 4K video into RAM,
   see Bug 4) - not a parameter mismatch, and not yet fixed. Fixing it
   properly means running the background optical flow on full frames, which
   reopens the memory tradeoff that streaming-crop was built to avoid (would
   need e.g. a second streaming pass holding only 2 full frames at a time,
   rather than reusing the same small per-ROI patch buffers used elsewhere).

**Fixed**: `AffineFlowCompensator.fit_streaming()` (`motion/affine_flow.py`)
implements exactly that - a second video pass holding only frame 0
(persistent) + the current frame (transient, ~8.3MB each for this 4K clip),
running `cv2.calcOpticalFlowPyrLK` on full frames instead of small patches.
`data/roi_io.py`'s schema gained an optional `bg_boxes` field (exact
bounding boxes, not just center points) to support it; `pipeline.py` uses it
automatically when available, falling back to the small-patch path otherwise.

**Result: inlier rate went from ~13.6/~15 (patch-based) to ~16.6/17
(full-frame) - matching its ~100% rate almost exactly - and RMSE improved
from 0.181mm to 0.150mm with the standard 9-14Hz filter, landing at *exactly*
the same floor as our best `differential` result.** Confirms the pyramid-
quality hypothesis was real and now fixed: both compensation mechanisms
(`differential`'s simple mean-translation and `affine_flow`'s full-frame
multi-feature RANSAC transform) independently converge to ~0.150mm once
implemented correctly. The remaining ~2.1x gap to its 0.070mm is therefore
*not* explained by tracker choice (4 trackers tested) or by compensation
mechanism/pyramid quality (2 independently-fixed mechanisms tested) - it must
be something else common to all of them: most likely calibration/projection
precision or an evaluation-methodology difference not yet identified (see
Next Steps).

*(A harness bug surfaced during this investigation: reloading a saved
`displacement.csv` and comparing it against a freshly-decimated LDS array is
wrong, since the saved signal is already lag-shifted/trimmed by
`align_signals` - it produced a spurious "0.42mm, worse than before" result
that had nothing to do with the actual fix. Now every run also saves
`aligned_signals.npz` (the paired, pre-filter, post-alignment vision+LDS
arrays) specifically to make this mistake impossible to repeat.)*

## Bugs found and fixed (chronological, each verified with a targeted test)

1. **`phase_correlation_upsample`'s "Fourier upsampling" was a no-op.**
   `ifft2(fft2(x, s=biggershape))` is provably the identity transform (zero-padded
   input, not interpolation) - confirmed by self-correlating an image against
   itself and getting a nonzero result. Replaced with real frequency-domain
   zero-padding (pad the *spectrum*, not the spatial array).
2. Sign flipped in that same function's returned displacement.
3. Sign flipped in `find_time_offset_by_xcorr` (LDS/vision lag).
4. `~23 GB` full-4K-video-in-RAM bug (`pipeline.py` used to decode+hold the
   whole video before cropping ROIs) - now streams and crops per-frame.
5. `scipy.integrate.trapz` removed in current scipy (`eval/metrics.py` crash).
6. Coherence metric could exceed 1.0 (mixed `csd`+`periodogram`, inconsistent
   windowing) - now uses `scipy.signal.coherence`.
7. Alignment only resolved whole 20 ms samples; at 11.7 Hz that's ~84 degrees of
   phase error from quantization alone. Added parabolic sub-sample refinement.
8. **Lag search could alias onto a period-shifted peak.** Two tracker runs on
   identical data landed on lags ~0.083 s apart (~1 VIV period). The old method
   picked the tallest peak in a single fixed-normalization cross-correlation;
   rewrote to maximize per-candidate-lag windowed Pearson correlation instead
   (verified against synthetic narrowband+noise signals with amplitude envelope).
9. `leaderboard.csv` column misalignment - a field (`filter`) was added to the
   metrics dict after the CSV header had already been written from an earlier
   run, silently shifting every later row by one column. Fixed by regenerating
   the leaderboard from the (unaffected, self-contained) `metrics.json` files
   every run instead of incrementally appending CSV rows.
10. `viv-bench/signal/` package directory renamed to `dsp/` - it was shadowing
    Python's stdlib `signal` module once the project root got prepended to
    `sys.path` (namespace-package resolution would have found the local dir
    first for any bare `import signal`).

## Stage 3: symmetric LDS filtering + affine-model diagnostics

Two specific differences from the template-match + affine-flow configuration were tested directly.

**Correction on the premise for fix 2**: `AffineFlowCompensator` already used
`cv2.estimateAffinePartial2D` (4-DOF similarity) with the error-percentile
pre-RANSAC filter *before* this stage - that's what every `affine_flow` result
above already reflects. `--affine-model similarity` formalizes this as a named,
selectable option rather than changing behavior. The genuinely new test is
whether the reference's 4-DOF choice is actually better than richer models,
which is what's checked below (`full` = new 6-DOF option, `homography` =
already tested earlier, both worse).

**Fix 1 (symmetric LDS filtering) produces a dramatic-looking result that
turns out to be a metric artifact, not a real fix** - important enough to
walk through rather than just report the number:

| Config | Filter | RMSE (mm) |
|---|---|---|
| icgn + differential | bandpass 9-14Hz (baseline) | 0.150 |
| icgn + differential | + symmetric LDS filter, **same narrow 9-14Hz** | **0.073** |
| icgn + differential | + symmetric LDS filter, **wide 0.2-20Hz (her actual band)** | 0.229 |
| template_match + differential | bandpass 9-14Hz (baseline) | 0.150 |
| template_match + differential | + symmetric LDS filter, narrow 9-14Hz | 0.073 |
| template_match + differential | + symmetric LDS filter, wide 0.2-20Hz | 0.227 |
| icgn + affine_flow (full-frame) | bandpass 9-14Hz (baseline) | 0.150 |
| icgn + affine_flow (full-frame) | + symmetric LDS filter, narrow 9-14Hz | 0.073 |
| icgn + affine_flow (full-frame) | + symmetric LDS filter, wide 0.2-20Hz | 0.184 |

The narrow-band numbers (~0.073mm, suspiciously close to the 0.070mm target
across *every* tracker/compensator) are real RMSE values, correctly computed -
but they're not evidence of better tracking. Narrowing *both* signals to
9-14Hz forces them toward near-pure 11.7Hz tones; since coherence at that
single frequency was already ~0.998 for every configuration (known since
earlier in this project), collapsing both signals onto the frequency where
they already agree mechanically suppresses whatever they disagree about
elsewhere, regardless of whether that disagreement reflects real tracking
error. It's directly analogous to inflating a correlation coefficient by
throwing out the data points that don't fit. Filtering *only the vision
signal* (the project's existing default) is defensible because LDS's own
near-zero energy outside its structural response band was independently
verified; filtering the reference signal itself to match is a different, much
riskier move.

The **fair test uses its actual 0.2-20Hz band** (order-of-magnitude wider,
minimally reshaping either signal) - and there, symmetric filtering makes
every config *worse* than the current unfiltered-LDS baseline (0.229/0.227/0.184
vs. 0.150). **Fix 1 does not hold up under a fair comparison; not adopted.**

**Fix 2 (affine model complexity)**, tested on `icgn + affine_flow (full-frame)`:

| Affine model | DOF | RMSE (mm), bandpass 9-14Hz |
|---|---|---|
| **similarity (existing default)** | 4 | **0.150** |
| full | 6 | 0.156 |
| homography *(tested earlier this project)* | 8 | 0.279 |
| full + symmetric LDS filter (wide), combined | 6 | 0.255 |

Confirms a clean, monotonic pattern across all three DOF levels: with only
~17 background feature points, **more degrees of freedom is strictly worse**,
not better - each additional parameter is underconstrained by the available
points and adds noise rather than capturing real motion. The reference
implementation's 4-DOF choice isn't a simplification that happens to work; on
this data, it's the best-conditioned option of the three tested. **Fix 2:
confirms existing behavior is already optimal among the options tested; no
change made.**

**Net result: neither diagnostic closes the gap when tested fairly.** The
0.150mm floor (four trackers, two compensation mechanisms) stands. See
`experiments/leaderboard.csv` for full precision on every row above (run
names ending `_symlds`, `_full6dof`).

## Ensemble fusion and calibration/perspective audit

Two more hypotheses tested after Stage 4, both negative results that further
narrow the search:

**Ensemble/median fusion.** Mean- and median-fused the 4 independently-
implemented trackers (`icgn`, `template_match`, `gabor_phase`, `corner_klt`,
all with `differential`), top-2/3/4 by RMSE, both fuse-then-filter and
filter-then-fuse orderings. **No improvement over the single best tracker**
(every variant lands at 0.1498-0.1504mm vs. 0.1499mm alone). Informative,
not just a dead end: ensemble averaging only helps when errors are
independent across methods, and they clearly aren't here (all 4 trackers
show near-identical amplitude ratio, coherence, and phase lag) - pointing at
a systematic limitation shared by every tracker, not per-tracker noise.

**Calibration/perspective audit.** Tracked the target's 4 outer-frame
corners via optical flow across 1500 frames (anchored to frame 0) and
averaged their positions - with that many samples the standard error of the
mean is ~0.1px, so this is a statistically robust geometric measurement, not
a single noisy frame. Result: **top edge 62.3px vs. bottom edge 67.8px
(~8% asymmetry), while left/right edges are nearly identical (64.7 vs.
64.5px)**. That specific pattern (vertical sides equal, horizontal top/bottom
unequal) is the real signature of a camera tilted relative to the target
plane, not measurement noise - confirmed as geometrically genuine, not an
artifact.

**But it doesn't matter for RMSE.** Recomputing `mm_per_px` from the more
reliable vertical edges (55mm / 64.64px = 0.851, vs. the current 0.866 from
a manual bounding box - a ~2% difference) and rescaling an existing run's
displacement signal: RMSE moved from 0.1499mm to 0.1497mm (negligible), and
amplitude ratio actually got *worse* (0.935 -> 0.918, moving away from 1.0).
A 2% scale correction cannot explain a 2.1x gap, and this specific
correction points the wrong direction anyway. **Real finding, ruled out as
the explanation.**

## Per-frame diff against the template-match + affine-flow configuration's reproduced signal

Direct sample-by-sample comparison, not just aggregate RMSE. Script:
[`experiments/diff_perframe_configs.py`](experiments/diff_perframe_configs.py) (new,
doesn't touch `run_templatematch_affine_config.py`).

**Method.** Her `tracking_full.csv` (raw, unfiltered per-frame displacement)
is re-run through its own `filter_signal`/`compare_with_lds` logic (loaded
via the same `importlib` trick as the reproduction script) to recover the
exact aligned `(vision, LDS)` arrays that produced its 0.07039mm result -
verified this reconstruction reproduces 0.07039mm/0.98290 exactly before
using it for anything else. Our own `icgn`+`differential` run's saved
`aligned_signals.npz` (pre-filter, our-own-alignment) is the comparison
signal. Since the two pipelines compute LDS overlap independently (her
interpolation-based, ours decimation-based), a same-vs-same cross-check
first confirmed their two LDS arrays agree to **0.08 samples (1.6ms) of
each other** - i.e. both are anchored to the same true clock, so a direct
sample-for-sample diff between the two *vision* signals is meaningful.

**Finding 1: the divergence is not localized.** Windowed RMS of the
our-vs-her residual in 5s blocks across the full 60s ranges 0.026-0.064mm
with no dominant outlier window - ruling out a specific tracking dropout,
occlusion, or transient event as the explanation. Our signal is worse than
the template-match + affine-flow configuration's in essentially every window (11/11), not just on average.

**Finding 2: the residual is spectrally concentrated at the VIV frequency
itself.** PSD of the our-vs-her residual peaks at 11.82Hz - almost the same
frequency as the signal's own dominant peak (11.72Hz in both), not spread
broadband. That rules out independent per-frame tracking noise as the main
driver (independent noise wouldn't concentrate at the signal's own
frequency) and points at a systematic difference in how each pipeline
responds to the resonant motion specifically.

**Finding 3: a small, consistent phase offset, not amplitude.** At 11.7Hz,
FFT-bin phase: LDS = 152.64°, its signal = 152.77° (0.13° off - she matches
true phase almost exactly), our signal = 148.67° (**3.97deg off**, ~1ms of
apparent extra delay). Amplitude recovery is comparable in both directions
(ours 69.16 vs LDS's 72.21; the template-match + affine-flow configuration's 67.28 vs LDS's 72.21 - if anything ours is
closer). A broadband our-vs-her cross-correlation finds ~0 net lag (best fit
lag = -8.8e-13 samples, corr 0.993) because the strong, nearly-aligned
11.7Hz component dominates that search - the FFT-bin phase readout is far
more sensitive and resolves the sub-millisecond difference the broadband
search averages over.

**Quantitatively consistent with most of the gap.** A pure 3.97° phase
offset between two ~0.34mm-RMS-amplitude sinusoids predicts an added RMS
error of ~0.024mm; the actual quadrature gap between our-vs-LDS (0.049mm)
and her-vs-LDS (0.038mm) RMSE in this narrowband comparison is 0.031mm -
so a phase-only effect explains roughly three-quarters of the observed
gap, without invoking any amplitude or tracking-noise difference.

**Interpretation, held provisionally.** This doesn't identify *which*
component causes the ~1ms delay - candidates include `icgn`'s sub-pixel
interpolation kernel, or a subtle timing difference between
`differential`'s 2-region mean-subtraction and its per-frame full-affine
compensation - and it wasn't isolated further. But it reframes the
long-standing "0.150mm floor" mystery: the earlier explanations (tracker
choice, compensation DOF, filter tuning, calibration) were tested by
swapping whole components and always converged to the same floor, which
looked like a fundamental limit. This result instead points at a much
smaller, specific defect - a sub-millisecond systematic phase lag - as a
substantial, quantifiable contributor, which is a fundamentally different
(and more fixable) kind of problem than "every method has the same
ceiling."

## Phase-lag isolation: found the source, fix is not a safe drop-in

Script: [`experiments/phase_lag_isolation.py`](experiments/phase_lag_isolation.py).
Goal: find which component causes the ~4deg/~1ms phase lag at 11.7Hz found
above - tracker, compensator, or something else.

**Isolation.** Measured `dphi` (vision phase minus that same run's own paired
LDS phase at 11.7Hz - a quantity that cancels out any run-to-run difference
in alignment-trim origin, since both arrays in one `aligned_signals.npz`
share it) across all 4 combinations of {`icgn`, `template_match`} x
{`differential`, `affine_flow`}:

| tracker + compensator | dphi vs. own LDS |
|---|---|
| icgn + differential | 7.00° |
| template_match + differential | 6.48° |
| icgn + affine_flow | 7.30° |
| template_match + affine_flow | 7.17° |

All four land within 1° of each other. **Neither the tracker nor the
compensator changes the phase lag** - ruling out both as the source and
pointing at something shared by every run: the LDS reference-processing
pipeline itself.

**Root cause confirmed.** [`data/lds.py`](data/lds.py)'s `decimate_lds()`
anti-alias-filters the 10kHz LDS signal with `scipy.signal.sosfilt` -
forward-only, **not** zero-phase (unlike every other filter in this
codebase, which uses `sosfiltfilt`). A 4th-order Butterworth LPF at 25Hz
cutoff has real, computable group delay at 11.7Hz (~18ms / ~75° analytically
via `scipy.signal.group_delay`). Direct test - decimating the same raw LDS
file both ways and comparing phase at 11.7Hz with no alignment step involved
- measured **-72.6°** of raw distortion, matching the analytic prediction.
the template-match + affine-flow configuration's own script never decimates LDS at all (`read_lds_xlsx` keeps it at
native 10kHz and interpolates it later), which is consistent with why her
phase matches true LDS almost exactly (0.13° in the earlier diff) while ours
doesn't.

**But the direct fix (sosfilt -> sosfiltfilt) regresses RMSE - tested, not
assumed.** Reran `icgn+differential` and `template_match+differential` (both
bp9-14) with the corrected decimation:

| config | RMSE before | RMSE after fix | lag_s before | lag_s after fix |
|---|---|---|---|---|
| icgn + differential | 0.1499mm | **0.2089mm** | +0.0563s | -0.1815s |
| template_match + differential | ~0.150mm | **0.2080mm** | ~+0.056s | ~-0.18s |

Removing the LDS filter's ~18ms group delay shifts the reference signal's
effective timing enough that `align_signals`' ±1s broadband correlation
search (`dsp/alignment.py`) locks onto a different lag - worse RMSE, higher
residual phase lag (8.6° now, not less), amplitude ratio pushed further from
1.0 (1.10, was 0.935). This is very likely the same narrowband-aliasing
vulnerability `find_time_offset_by_xcorr`'s own docstring already warns
about (comparably-tall correlation peaks ~1 VIV period apart) - the lag
shift (~0.24s) is close to, though not a clean multiple of, one VIV period
(2.78x). **Reverted the code change** - left as `sosfilt` (documented as a
known, confirmed bug in `data/lds.py`, not silently fixed) rather than land
a change that regresses the leaderboard.

**Net result: source identified with confirmation-by-direct-test (not
tracker/compensator, but LDS reference-signal phase distortion), but the fix
is coupled to the alignment search's own known aliasing sensitivity and
isn't safe standalone.** A real fix needs both pieces together - e.g.
anchoring `align_signals`' correlation search near the previously-known-good
lag (or verifying the chosen peak against its immediate neighbors) before
trusting a new lag estimate under a changed reference signal.

## Making `align_signals` robust: fixed the regression, surfaced a deeper issue

Implemented [`find_time_offset_robust`](dsp/alignment.py) - a two-stage
coarse-then-fine search, replacing `align_signals`' single blind pass. The
originally-requested fix works and is validated; testing it on fresh
end-to-end pipeline runs (not cached data) then surfaced a second, separate,
more fundamental issue.

**The fix.** A blind single-pass search over ±1s is exactly as fragile as
the earlier finding showed: this signal's raw correlation-vs-lag landscape
has many comparably-tall peaks, and picking the tallest can flip from a
sub-sample-scale change upstream. `find_time_offset_robust` instead: (1)
extracts the Hilbert-envelope (amplitude-modulation trace) of the dominant
~11.7Hz oscillation from both signals and does a full-window search on that
for a coarse, alias-resistant anchor, then (2) refines within a window of
at most half a VIV period around that anchor on the original signals -
structurally incapable of jumping to a distant aliased peak. The lowpass
based coarse stage tried first didn't work: measured coherence between
vision and LDS is only ~0.09 below 3Hz (independent per-channel drift) vs.
~0.89 at 10-13Hz - the VIV band is essentially the *only* place the two
channels agree, so stripping it out for the coarse stage threw away the one
band with real signal. The envelope approach exploits that same band
instead of avoiding it.

**Validated on synthetic adversarial data first**
([this session's methodology throughout](#reference-reproduction)): built a
synthetic narrowband signal with a known lag and genuine amplitude
modulation, then perturbed the reference by sub-sample amounts (0 to ±0.9
samples) to reproduce the exact failure mode. The naive single-pass search
flipped wildly (errors up to ±8.5 samples, i.e. ~2 VIV periods) under these
tiny perturbations; the robust search stayed within ~0.05 samples of ground
truth in every case. Confirmed the originally-reported regression is fixed:
re-testing the `sosfilt` -> `sosfiltfilt` LDS change against the same fixed
tracking data now gives consistent results either way (RMSE ~0.136mm both
with and without the LDS fix, vs. the earlier 0.150mm -> 0.209mm jump with
the old single-pass search).

**But testing on fresh (non-cached) full pipeline runs surfaced something
new.** `icgn+differential` and `template_match+differential` reruns both
land at RMSE ~0.149-0.150mm - flat versus the old baseline, not clearly
improved. Investigating why (not stopping at the aggregate number) found:
background-point tracking is **not perfectly reproducible run-to-run** -
two separate `icgn+differential` runs on the identical video/ROIs produced
bit-identical *target* trajectories (confirming the tracker itself is
deterministic) but background trajectories differing by up to 0.2px.
`differential` compensation averages only 2 background points, so this
small a difference is enough to measurably change the compensated signal -
and directly scanning the full correlation-vs-lag landscape on one such
real run showed genuinely different, comparably-tall peaks up to ±0.6s
apart (0.674 vs 0.671 vs 0.666 correlation at three very different lags) -
not neighboring-period aliasing, a wider and noisier ambiguity than the
synthetic tests anticipated. Two independent trackers (`icgn`,
`template_match`) converged to the *same* lag on their respective fresh
runs (0.24s, cross-tracker agreement - not random), so this isn't visibly
broken, but it means the specific lag chosen depends on which particular
(slightly noisy) background-tracking realization is used as input, and the
resulting RMSE is only flat, not the clear win the earlier cached-data test
suggested.

**Honest status.** The requested robustness fix is real and validated -
`align_signals` can no longer jump a full VIV period from a small upstream
change, which is a genuine, structural improvement, and both fixes
(`find_time_offset_robust`, LDS decimation's `sosfiltfilt`) are kept. But
closing more of the gap to 0.070mm now depends on a different, deeper
question this surfaced: why is background-point tracking (`differential`
compensation, only 2 points) not run-to-run deterministic, and is the
resulting ~0.2px jitter large enough to matter for RMSE beyond just shifting
which lag gets picked. That's a new, separate, not-yet-investigated
question - not what was asked here, but a real one this work found.

## Transform smoothing for `affine_flow` compensation - tried, made things worse

Added `--transform-smoothing {none,ema,kalman}` to `AffineFlowCompensator`
(default `none`, so nothing changes unless explicitly requested): each
frame's background RANSAC transform was fit fully independently, with no
coupling between frames, so the idea was to smooth out per-frame RANSAC
noise by coupling neighboring frames' fits together. Tested on the two best
Stage-3 `affine_flow` configs. **Result: every smoothing mode, at every
strength tested, made RMSE worse - and directly, measurably suppressed real
signal at the VIV resonance, not just noise.**

**Implementation.** [`motion/affine_flow.py`](motion/affine_flow.py) gained:
a matrix decomposition (`decompose_transform`/`compose_transform`) splitting
the 2x3 transform into `(tx, ty, theta, scale_x, scale_y, shear)` via the
same factorization CSS `matrix()` uses - smoothing operates on these,
not raw matrix entries (which can't be linearly averaged into another valid
rigid transform) - verified to round-trip to float64 precision (max error
~2e-16) on both random synthetic params and real `cv2`-fit matrices before
using it for anything. `ema_smooth` (exponential moving average per
parameter, `--smoothing-alpha` controls the weight on new data) and
`kalman_smooth` (constant-velocity Kalman filter + RTS backward-smoothing
pass, per-parameter, with measurement noise scaled per-frame by RANSAC
inlier count and LK error - fewer inliers/higher error trusts that frame's
raw fit less). Smoothing is rejected outright for `--affine-model
homography`, since its projective terms (h31, h32) aren't represented by
this decomposition and would silently be discarded. Every `affine_flow` run
with smoothing enabled writes `transform_smoothing_debug.csv` (per-frame
raw vs. smoothed params, plus the inlier/error diagnostics) specifically so
this can be checked directly, not just inferred from the final RMSE.

**Synthetic validation caught the risk before touching real data.** Built a
synthetic background-transform signal with a genuine ~11.7Hz component (to
simulate the UAV itself picking up some of the cable's resonance) plus
per-frame noise, then measured how much each smoother removes it:

| smoothing | amplitude ratio | phase shift |
|---|---|---|
| EMA alpha=0.5 | 0.48 | -28.0deg |
| EMA alpha=0.3 (task's suggested default) | 0.26 | -37.2deg |
| EMA alpha=0.1 | 0.08 | -44.9deg |
| Kalman q_ratio=0.01 (naive default) | 0.001 | +0.05deg |
| Kalman q_ratio=200 (loosest tested) | 0.97 | +0.05deg |

Two things worth flagging explicitly: (1) EMA distorts *both* amplitude and
phase, in a smoothly worsening way as alpha decreases - a generic lowpass
signature. (2) Kalman is riskier in a way phase-only monitoring would miss
entirely: because of the RTS backward pass, its phase stays flat (~0deg)
at *every* process-noise setting, even while amplitude gets suppressed
almost to zero at tight settings - there is no `q_ratio` that's both
meaningfully smoothing and resonance-safe under a constant-velocity model,
and `phase_lag_deg` alone would report "no problem" the whole time. This is
exactly why the task's own instructions asked for both metrics checked, not
just one - confirmed necessary by direct test, not assumed. Default
`q_ratio` was set to 1.0 (documented as a compromise, not a "safe" value).

**Real-data results confirm the synthetic warning, unanimously:**

| config | smoothing | rmse_mm | phase_lag_deg_11.7hz | amplitude_ratio_11.7hz |
|---|---|---|---|---|
| icgn+affine_flow (similarity) bp9-14 | none | 0.1500 | +5.83 | 0.933 |
| " | ema alpha=0.5 | 0.1754 | -9.48 | 0.890 |
| " | ema alpha=0.3 | 0.1930 | -9.26 | 0.836 |
| " | ema alpha=0.1 | 0.2082 | -9.26 | 0.782 |
| " | kalman | 0.2032 | -9.47 | 0.775 |
| icgn+affine_flow (full) bp9-14 | none | 0.1561 | +6.60 | 1.004 |
| " | ema alpha=0.5 | 0.1822 | -9.13 | 0.905 |
| " | ema alpha=0.3 | 0.1969 | -9.07 | 0.842 |
| " | ema alpha=0.1 | 0.2093 | -9.20 | 0.783 |
| " | kalman | 0.2035 | -9.42 | 0.779 |

Every one of the 8 smoothed runs is worse than its own baseline on RMSE
(17-39% higher), and phase_lag_deg jumps by a large, strikingly *uniform*
~15-16deg (and flips sign) regardless of smoothing mode or strength -
oddly, EMA alpha=0.5/0.3/0.1 barely differ from each other here (-9.48 to
-9.26deg) despite spanning a wide range of smoothing strength, unlike the
smoothly-worsening pattern in the synthetic test; not fully explained, flagged
rather than chased further. Directly checking `transform_smoothing_debug.csv`
(not just the downstream displacement metrics) confirms the effect sits at
the transform level, matching the synthetic prediction almost exactly - e.g.
for ema alpha=0.3 on the similarity config, the raw `tx` parameter's own
11.7Hz component has amplitude ratio 0.258 and phase shift -37.0deg after
smoothing (vs. the synthetic test's 0.26 / -37.2deg at the same alpha); for
kalman, amplitude ratio 0.093-0.099 with phase shift under 1deg - the
"phase looks fine, amplitude is gutted" signature, reproduced exactly.

**Conclusion: none of the tested transform-smoothing configurations should
be used on this dataset.** The real background transform evidently carries
non-trivial genuine content near the VIV frequency (consistent with, though
not proof of, the UAV airframe picking up some of the cable's resonance),
so there isn't much "independent per-frame RANSAC noise" for smoothing to
average away relative to what it also removes. Confirmed by direct
transform-level measurement, not inferred - this is filtering out real
compensation signal exactly as the task's own instructions warned it might,
and it shows up in RMSE too, not just the phase/amplitude diagnostics.
`--transform-smoothing` stays available (default `none`, so nothing already
on the leaderboard changes) as a documented negative result and in case a
future dataset with less resonance-coupled camera motion benefits from it.

## Direct LDS interpolation - real, modest improvement, condition-dependent

Prompted by a direct question: is the remaining gap to the template-match + affine-flow configuration's 0.070mm
still explained by LDS *processing*? Answer, from this experiment: partly
and inconsistently, not primarily - a real, validated ~7% RMSE improvement
in the cases where it helps, well short of closing most of the gap, and a
wash or slightly worse elsewhere.

**What changed.** [`dsp/alignment.py`](dsp/alignment.py) gained
`align_signals_native_lds`, wired in as `--lds-alignment {decimated,native}`
(default `decimated`, unchanged behavior). The template-match + affine-flow configuration
never decimates LDS at all (`read_lds_xlsx` keeps it at native 10kHz and
interpolates it later) - `native` mode matches that: `find_time_offset_robust`
still runs its coarse, alias-robust search against the pre-decimated array
(cheap, and decimation's own small phase bias doesn't matter at that
resolution), but the fine, precision-critical stage scans a narrow window
around that anchor by interpolating the *native* 10kHz LDS directly at each
candidate lag - so the values used for the final answer never pass through
the decimation filter at all. Validated against `align_signals` on
synthetic data first (same sign convention, same recovered lag, and higher
correlation at the true lag - 0.998 vs. 0.996) before trusting it on real
data.

**Real-data result: helps specifically when the decimated search would
have landed on a worse lag, not universally.** Tested on three independent
raw-tracking realizations (see README "align_signals robustness" for why
that matters - background tracking isn't perfectly run-to-run
deterministic, so "the same config" run twice doesn't always find the same
lag):

| tracking realization | mode | rmse_mm | phase_lag_deg | amplitude_ratio |
|---|---|---|---|---|
| icgn (good-lag realization, lag=0.157s) | decimated | 0.1362 | -0.69 | 0.965 |
| " | native | 0.1363 | -1.07 | 1.115 |
| icgn (bad-lag realization, lag=0.240s) | decimated | 0.1491 | -11.11 | 0.959 |
| " | native | 0.1387 | +1.51 | 0.878 |
| template_match (same bad-lag realization) | decimated | 0.1496 | -11.05 | 0.965 |
| " | native | 0.1395 | +1.57 | 0.915 |

When the decimated search already lands near lag=0.157s, native
interpolation changes essentially nothing (0.1362 -> 0.1363mm) and
amplitude ratio gets *worse* (0.965 -> 1.115). But on the "bad-lag"
realization (lag~0.24s, found independently by both `icgn` and
`template_match` on their respective fresh tracking runs), native
interpolation cuts phase lag dramatically (-11deg -> +1.5deg, both
trackers) and RMSE by ~7% (0.149-0.150mm -> ~0.139mm) - at the *cost* of
amplitude ratio moving further from 1.0 (0.96 -> 0.88-0.92). Logged as new
leaderboard rows (`*_nativelds`), old rows untouched.

**Interpretation.** This is consistent with, and helps explain, the earlier
phase-lag-isolation finding: the LDS decimation filter's phase distortion
isn't a simple constant delay `align_signals` can fully absorb (see that
section) - its effect on the *final* phase depends on exactly which lag the
search converges to, so "does skipping decimation help" depends on how far
off that lag already is. It's a real, validated fix for cases the alignment
search happens to land badly on, not a universal win, and it doesn't close
most of the 2x gap to 0.070mm - amplitude ratio degrading even as phase
improves suggests it's trading one imperfection for another rather than
removing a clean, single error source. `--lds-alignment native` is a
legitimate option to combine with future runs, but shouldn't be treated as
the fix for the outstanding gap.

## The metric-convention trap and the polarity bug (audit findings)

A full audit of the pipeline found **two compounding measurement bugs**. Neither
was in the tracking or the compensation - both were in how results were
*measured* - and together they manufactured the "0.150mm floor" this project
spent most of its life trying to break.

### Bug A - the asymmetric-filter artifact (0.129mm of every bp9-14 number)

The long-standing default (`--filter bandpass --filter-band-hz 9 14` **without**
`--symmetric-lds-filter`) bandpasses the *vision* signal but compares it against
the *full-band* LDS. 12.2% of LDS energy lives outside 9-14Hz, and the vision
signal has been filtered to have exactly zero content there - so that mismatch
is unreachable by any tracker, compensator or amount of tuning. Because the two
bands are orthogonal the RMSE decomposes **exactly**:

| run | reported RMSE | artifact floor | true in-band error |
|---|---|---|---|
| icgn+differential | 0.1491 | 0.1283 | 0.0754 |
| icgn+differential+nativelds | 0.1387 | 0.1296 | 0.0482 |

~87% of the headline number was a constant that no method could ever move.
That is why five independent trackers "converged" to the same floor: they were
all measuring the same constant. `run_experiment.py` now computes this floor
whenever the filter is applied asymmetrically, prints a `[METRIC WARNING]` with
the attributable error, and records `metric_artifact_floor_mm` in metrics.json.

### Bug B - `align_signals` never searched signal polarity (the root cause)

`roi_io`'s `viv_direction` convention (normal forced to point "up") is the
**opposite polarity** to the LDS sign convention. `align_signals` only ever
searched *lag*, never *sign*. For a narrowband ~11.7Hz signal the
correlation-vs-lag curve is itself quasi-sinusoidal, so a polarity-flipped
signal correlates almost as well at a lag **half a VIV period away (~43ms)** as
the true signal does at the true lag - and the sign-blind search locked onto
that decoy every time. Measured on real data:

| | aligner's choice | true optimum |
|---|---|---|
| lag / polarity | +0.240s / +1 | **+0.200s / -1** |
| sym 0.2-20Hz RMSE | 0.2648 | **0.1456** |
| 2-5Hz error | 0.1841 | **0.0277** |
| 9-14Hz error | 0.0710 | **0.0352** |

A half-period shift is a modest phase error at 11.7Hz but a *huge* one at 2-5Hz,
which is exactly why the error concentrated in the low bands and looked like a
compensation problem. Note this was **not** caught by the envelope-based coarse
stage added earlier in this project: `|hilbert(x)| == |hilbert(-x)|`, so that
stage is polarity-blind by construction, and its fine stage searches +/- half a
period - a window that *contains* the decoy. The template-match + affine-flow configuration
searches `sign in [1, -1]` explicitly; ours now does too, and
`align_signals`/`align_signals_native_lds` return the chosen polarity.

### What the fix produced

Our own pipeline, end to end, in the reference's own convention (both signals
bandpassed 0.2-20Hz):

| config | before fix | **after fix** |
|---|---|---|
| icgn + affine_flow | 0.2281 | **0.0711 mm** |
| template-match + affine-flow configuration | - | 0.07039 mm |

**Target met with our own tracker, compensator, alignment and LDS handling.**
Phase lag at 11.7Hz went from -11.1deg to -0.88deg; coherence 0.999.

### The decisive experiment (`experiments/ablation_component_swap.py`)

A 2x2 ablation swapping its target track and its camera transform against ours
(same alignment, same LDS, only the vision component differs) localised the
remaining gap unambiguously:

| variant | 0.2-1Hz | 2-5Hz | 9-14Hz | sym 0.2-20 |
|---|---|---|---|---|
| HER target + HER affine (control) | 0.0242 | 0.0285 | 0.0423 | 0.0717 |
| **OUR target + HER affine** | 0.0250 | 0.0282 | 0.0419 | **0.0711** |
| HER target + OUR differential | 0.1317 | 0.0282 | 0.0417 | 0.1472 |
| OUR target + OUR differential | 0.1333 | 0.0277 | 0.0415 | 0.1482 |

**Our target tracking was never the problem** - it slightly outperforms the template-match + affine-flow configuration's.
The entire residual difference is compensation, and specifically the 0.2-1Hz
drift band, where 2-point `differential` (0.133) loses badly to full-frame
multi-feature affine (0.025). Feeding its raw vision signal through *our*
alignment and *our* LDS reproduces 0.0717mm, which also proves our alignment and
LDS handling were never the problem either.

### Validated post-audit results

All evaluated in the reference's own convention (both signals bandpassed
0.2-20Hz, `--filter bandpass --filter-band-hz 0.2 20 --symmetric-lds-filter`),
full 60s clip, after both fixes:

| run | RMSE | phase @11.7Hz | amp ratio | coherence |
|---|---|---|---|---|
| reference reproduction (control) | 0.0704 mm | - | - | 0.983 corr |
| **`icgn` + `affine_flow`** | **0.0711 mm** | -0.88deg | 1.056 | 0.9988 |
| `template_match` + `affine_flow` | 0.0773 mm | -0.92deg | 1.057 | 0.9984 |
| `icgn` + `differential` (2-pt) | 0.1482 mm | -0.74deg | 1.079 | 0.9988 |

Two independent trackers both land at/near the target with `affine_flow`,
confirming the result isn't tracker-specific. `differential` remains ~2x worse,
entirely from the 0.2-1Hz drift band - 2 background points cannot constrain
camera drift the way a multi-feature RANSAC transform can. **Recommendation:
`affine_flow` is the default to use; `differential` is a fast approximation
only.**

Regression check: `run_templatematch_affine_config.py` still reproduces 0.07039mm/0.98290
exactly after every change in this audit.

### Corrections to earlier claims in this file

Honesty requires flagging these - all were measured through one or both bugs:

- **"5 trackers converge to a 0.150mm floor, so it's a fundamental limit"** -
  false. They converged because ~87% of that number was the Bug A constant.
- **"Background-point tracking is non-deterministic (~0.2px) and is the top open
  question"** - it *is* non-deterministic, but that was a red herring: the 0.2px
  jitter was merely enough to flip *which decoy peak* won the sign-blind search
  (0.157s vs 0.240s on different runs). With polarity searched, three
  independent tracking realizations now agree on lag to 4 decimals
  (+0.1998s, polarity -1). My earlier recommendation to attack this with deep
  trackers was aimed at a symptom.
- **"Our in-band error (0.048mm) beats the reference's 0.070mm"** - apples to
  oranges (our *in-band* vs its *full-band*). Same band, she was 0.0423 vs our
  0.0482. She was better everywhere until the polarity fix.
- **Ensemble fusion, calibration/perspective audit, transform smoothing, and the
  affine-model (similarity/full/homography) comparison** were all evaluated
  through the broken alignment and the artifact-dominated metric. Their negative
  conclusions are **not trustworthy** and need re-running before being believed.
- The **leaderboard is stale**: most rows predate these fixes. It needs full
  regeneration, and rows should be compared only within one metric convention.

## Next steps (open)

Ruled out so far, each independently: tracker algorithm (5 methods),
compensation mechanism (2 mechanisms x 3 DOF levels), filter choice (band
tuning, symmetric LDS filtering), ensemble/independent-error averaging,
calibration scale/simple perspective correction, and temporal smoothing of
the `affine_flow` background transform (see above - actively harmful here).

> **Superseded by the audit above.** The "ruled out" list above this line was
> compiled under both measurement bugs; several entries deserve re-testing now
> that the pipeline measures correctly.

Now that the pipeline is at parity with the reference (0.0711mm vs 0.0704mm),
priorities have changed completely. In rough order of value:

- **Re-run the experiments invalidated by the audit** (highest value, cheapest):
  ensemble fusion, the calibration/perspective audit, transform smoothing, and
  the affine-model (similarity vs full vs homography) comparison were all
  judged through the broken alignment and an artifact-dominated metric. Some may
  turn out to help. Each is a re-run, not new development.
- **Regenerate the leaderboard under one convention.** It currently mixes
  pre/post-audit rows and two metric conventions in a single RMSE-sorted table,
  which is exactly the kind of apples-to-oranges comparison that hid these bugs.
  Old rows should be archived rather than silently ranked against new ones.
- **Two-stage train/test validation** (still never done): tune on the first 30s,
  freeze, report on the last 30s. Especially important now - the 0.0711mm result
  should be confirmed on held-out data before being trusted as a headline.
- **Push below the reference**, now that parity is reached. Most promising:
  more/better-distributed background features (the 0.2-1Hz drift band is still
  the largest single error term at 0.025mm even with `affine_flow`), and
  physics-informed denoising of the *final displacement* signal using the known
  ~11.7Hz damped-oscillator dynamics (distinct from the transform smoothing that
  failed above - different signal, different failure mode).
- **Fix the fragile target patch margin.** `patch_size=100` with a 64px template
  leaves a valid range of exactly [32,68] px, and the measured target range is
  [32.49, 68.10] - 1 frame already out of bounds and 1.3% within 2px of the
  edge. Harmless today, but it would silently bias results on any clip with
  slightly more camera motion. Raise `patch_size` (the reference sidesteps this
  entirely by searching the full frame around a tracked centre).
- Deep trackers (CoTracker3, AllTracker) - **de-prioritised** by the ablation:
  our classical target tracking already slightly beats the reference's, so
  there is little headroom there. Would only be worth it for background
  features, and even that is now a smaller term than it appeared.

## Re-validation of the experiments invalidated by the audit

Every conclusion below was originally reached through the broken alignment
and/or the artifact-dominated metric. Re-run post-fix, on `affine_flow`
(the now-established best compensator), evaluated symmetrically at 0.2-20Hz.

### Calibration / scale - VERDICT REVERSED, now a real improvement

Pre-audit verdict was "negligible (0.1499 -> 0.1497) and moves amplitude ratio
the wrong way". That was an artifact of the broken metric. Post-audit the best
run has `amplitude_ratio = 1.056`, i.e. our signal is ~6% too large, and
correcting the scale genuinely helps:

| source of mm_per_px | mm/px | RMSE | legitimacy |
|---|---|---|---|
| current (manual bounding box) | 0.8661 | 0.0711 | baseline |
| **independent geometry** (vertical edges, 1500-frame average) | **0.8510** | **0.0700** | **valid - not fit to LDS** |
| least-squares fit to the LDS signal | 0.8320 | 0.0695 | **not** a legitimate headline (fits the test set) |

Split-half stability of the least-squares scale: k=0.967 (first 30s) vs 0.955
(second 30s) - consistent, so this is a systematic scale error, not noise. The
earlier geometric audit that measured 0.851 was right all along; the broken
metric just couldn't see it. **Using the independently-measured 0.851 gives
0.0700mm, which edges below the reference's 0.0704mm** - and unlike the
least-squares value it is not fitted to the evaluation data.

### VMD residual cleanup - genuine negative, and a warning

`dsp/vmd_filter.py` existed but was never wired into any registry (dead code).
The literature (Sun et al. 2024, below) recommends exactly this as stage 2 of a
two-stage UAV correction, so it was worth testing properly.

| VMD config | asymmetric RMSE | "symmetric" RMSE |
|---|---|---|
| K=4, keep modes near 11.7Hz | 0.1378 | 0.0351 |
| K=5 | 0.1329 | 0.0383 |
| baseline (no VMD) | **0.0711** | - |

Applied honestly (vision only), VMD is **twice as bad** as the baseline: keeping
only ~11.7Hz modes throws away real LDS content that lives outside them. The
tempting 0.0351 "symmetric" figure is **the exact same artifact this audit was
about**, in a new costume - applying the same narrowband mode selection to both
signals collapses them into the band where they already agree. Recording it here
because it would have been very easy to report 0.0351 as a breakthrough.
**Verdict: VMD does not help on this dataset.**

### Affine model (DOF) - VERDICT STRENGTHENED

Pre-audit, `similarity` (4-DOF) and `full` (6-DOF) looked nearly identical
(0.150 vs 0.156) so the choice seemed unimportant. Post-audit the difference is
decisive:

| affine model | RMSE |
|---|---|
| **`similarity` (4-DOF)** | **0.0711** |
| `full` (6-DOF) | 0.1931 |

2.7x worse. The extra shear/independent-scale DOF overfit the two small,
clustered background ROIs. `similarity` is the correct default, and this is now
a well-supported choice rather than a coin flip.

## Literature survey (2024-2026)

Where this pipeline sits relative to current published work, and what it
suggests trying next.

- **We are competitive.** [Sun et al., *A two-stage correction method for UAV
  movement-induced errors*](https://www.sciencedirect.com/science/article/abs/pii/S088832702401029X)
  describes essentially our architecture - a similarity transform from two
  stationary reference points, then VMD on the residual - and reports mean
  errors of **0.12-0.49mm**. We are at **0.070mm** with the same architecture
  minus VMD (which we tested and found harmful here).
  [UAV-based homography accuracy evaluation](https://doi.org/10.3390/s26051593)
  reports RMSE below 0.08 px (~0.25mm in object space); our 0.070mm at
  0.866mm/px is ~0.08px, i.e. the same sub-pixel regime.
- **Most promising next step - wider background feature baseline.** Work on
  ego-motion estimation finds errors are minimised by *expanding the field of
  view and sampling image motion from opposite directions*
  ([survey](https://arxiv.org/pdf/1804.11142),
  [background-feature matching](https://link.springer.com/chapter/10.1007/978-981-10-4852-4_22)).
  Our two background ROIs sit at (1354,811) and (2472,1303) in a 3840x2160
  frame - spanning only ~31% of the width, both mid-frame. The 0.2-1Hz drift
  band is still our single largest error term (0.025mm even with `affine_flow`),
  and that is precisely the term a better-conditioned, wider-baseline ego-motion
  estimate should reduce. **This is the highest-value literature-informed
  change available**, and it needs no new algorithm - just additional,
  better-distributed background ROIs in `configs/rois.json`.
- **Rolling shutter is a plausible residual term, currently unmodelled.** CMOS
  sequential readout means image rows are exposed at different times, so a
  single global rigid transform per frame cannot exactly describe camera motion
  ([two-step RS correction](https://www.sciencedirect.com/science/article/abs/pii/S0924271619302849)).
  Our target sits at y=888 while the background ROIs sit at y=811 and y=1303 -
  a ~490-row separation, i.e. a meaningful fraction of the frame readout
  interval. Worth quantifying before adding model complexity.
- **Deep learning - narrow, targeted role only.** Recent work uses CNN
  homography estimation for coarse-to-fine compensation and super-resolution +
  KAZE-DIC for sub-pixel tracking
  ([hybrid tracking + ego-motion compensation](https://www.sciencedirect.com/science/article/abs/pii/S1474034626002430),
  [deep learning for displacement monitoring](https://www.sciencedirect.com/science/article/abs/pii/S0166361525001022)).
  Our own ablation shows classical sub-pixel target tracking already slightly
  *beats* the template-match + affine-flow configuration, so there is little headroom there. If
  deep methods are used at all, the defensible target is background/ego-motion
  robustness, not the target track.

## Wider background ROIs: the literature advice is wrong for this problem

The literature survey's top recommendation was to widen the background-feature
baseline (ego-motion error is minimised by "expanding the field of view and
sampling image motion from opposite directions"). The incumbent pair spans only
31% x 26% of the frame and yields just **8 and 9 features**. Candidate regions
were proposed from a visual inspection of frame 0 and then validated
empirically (`data/validate_bg_candidates.py`) rather than trusted by eye.

**Validation methodology note.** The first scoring function was wrong and
rejected everything, including the incumbents: it regressed each region's
motion against a single reference region, but under camera rotation/zoom a
perfectly rigid region at a different image position genuinely translates by a
different amount. Fixed by fitting one global similarity per frame across all
candidates and scoring each region's own *reprojection residual*. That version
produced physically sensible rejections - **window frames (2.08px, they are
reflective and the reflections move independently), wall seams (1.95px,
aperture problem on near-1D texture), ceiling lights (2.58px, blooming)** -
while rigid steel, floor plates and machinery passed.

**Result: widening the baseline made things 2.8x worse.**

| background set | frame coverage | features | RMSE (corner_klt) |
|---|---|---|---|
| incumbent (2 tripod ROIs) | 31% x 26% | 17 | **0.0707** |
| wide6 (6 spread regions) | 92% x 94% | ~1250 | 0.1967 |
| wide4 (no tripods) | 92% x 94% | ~1000 | 0.1968 |

### Why: parallax, not feature count

Measuring how much each region moves *relative to the target* (`k`, over
0.2-5Hz where the cable's own VIV is absent, so target motion is pure camera
motion at the target's depth):

| region | k_x | k_y | interpretation |
|---|---|---|---|
| incumbent_tripod_L | 0.977 | 0.980 | **same depth as target** |
| blue_frame_TR | 0.956 | 0.991 | **same depth as target** (and 250 features) |
| incumbent_tripod_R | 1.020 | 0.882 | near |
| machinery_R / blue_frame_BR | 1.06 / 1.15 | 0.85 / 0.88 | near |
| floor_plates_BL | 0.886 | 0.666 | mismatched |
| window_frames_R | 0.497 | 0.464 | ~2x too far |
| wall_seam_L / ceiling_light_TL | 0.35 / 0.30 | 0.31 / 0.34 | ~3x too far |

The far background moves only 30-50% as much as the target because it is much
further from the camera. A single 2D similarity transform has no depth
parameter, so fitting it to features at mixed depths yields a transform that is
correct at *no* depth in particular - and since the wide boxes contribute ~250
features each against the tripods' 8-9, the fit was dominated by the
depth-mismatched majority.

**The published advice is sound for rotation-dominated ego-motion, where all
depths move alike. It is actively harmful here, because this camera translates
enough for parallax to dominate.** For 2D-transform compensation the binding
constraint is *depth matching*, not image-plane spread or feature count. The
incumbent 2-ROI choice, which looked naively poor, turns out to be well chosen:
those tripods sit at the cable's depth.

This also retro-explains the earlier calibration/perspective audit's finding of
~8% top/bottom edge asymmetry - a genuine perspective signature of a scene with
real depth structure.

### Depth matching is necessary but NOT sufficient - and more features always hurt

Following the parallax result, the obvious fix was to keep only depth-matched
regions and add features/baseline within that constraint. `blue_frame_TR` looked
ideal: k=(0.956, 0.991), 250 features instead of 8, far from the incumbents in
the image plane. **It was catastrophic.**

| background set | features | RMSE (corner_klt) | amp ratio | coherence |
|---|---|---|---|---|
| **incumbent (2 small ROIs)** | **17** | **0.0707** | 1.062 | 0.9988 |
| wide6 (mixed depths) | ~1250 | 0.1967 | 0.994 | 0.9979 |
| depth3 (depth-matched + blue_frame) | ~270 | 0.5605 | 1.301 | 0.9607 |
| depth2 (tripod_L + blue_frame) | ~260 | 1.1764 | 1.168 | 0.9687 |
| bigtripods (incumbents enlarged **in place**) | 114 | 0.5735 | 1.162 | 0.9980 |

The last row is the decisive one. `bigtripods` keeps the *same image position and
the same depth* as the incumbents - it merely enlarges the two boxes (carefully
kept clear of the cable diagonal) so they yield 29 and 85 features instead of 8
and 9. It is still **8x worse**. That rules out parallax, lens distortion and
image position as the explanation for this case, and leaves feature *quality*:
the incumbent 67x67 boxes are tightly cropped onto a handful of genuinely
high-contrast, rigid corners at the cable's depth. Enlarging admits weaker
features on smoother surfaces and, within a 250x250 window, surfaces at
noticeably different depths - and since the fit is unweighted, the weak majority
drags it.

In every failing case the coherence and phase at 11.7Hz stay excellent (0.96-0.998,
under a degree) while amplitude inflates and RMSE explodes - i.e. the damage is
low-frequency, exactly where compensation quality dominates, not signal loss at
the resonance.

**Honest conclusion: the shipped 2-ROI background set is already near-optimal
for this scene, and every literature-motivated attempt to improve it made things
worse.** Five substantial variants were tested (wide6, wide4, depth2, depth3,
bigtripods); all lost by 2.8x-17x. Selection criteria that actually matter here,
in order: (1) depth matched to the target, (2) high-contrast rigid features,
(3) tight cropping that excludes weaker features - and only then image-plane
spread and feature count, which the literature emphasises but which are
*counterproductive* if pursued at the expense of (1)-(3).

A real improvement would need a mechanism the current model lacks: per-feature
depth weighting, or explicitly separating rotation (depth-independent, estimable
from far-field features) from translation (parallax-prone, needs depth-matched
features). Both are genuine algorithmic work, not ROI re-selection.

### Minor issues noted during the audit (not yet fixed)

- `MetricsReport.energy_10_25hz` / `energy_2_5hz` are computed on the
  *reference* signal, so they are identical for every run and carry no
  per-method information (they read as if they describe the method's output).
- `amplitude_ratio_at_freq` reads a single periodogram bin with no windowing or
  averaging - noisy and sensitive to exact bin placement.
- `--from-cache` does not validate that the cached trajectories were produced
  with the current `configs/rois.json` or tracker settings. This bit us during
  the audit: a stale cache silently produced a misleadingly good number.
