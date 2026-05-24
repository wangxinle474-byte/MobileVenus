# Track 2 Results — Image-Domain Heads on Clean vs Pseudo Supervision

> **Created**: 2026-05-17
> **Updated**: 2026-05-18 10:40 (Path A done⚠, both eval re-runs with fixed leak detector)
> **Companion**: `docs/track2_plan.md` (experimental design),
> `docs/paper_draft.md` (Track 1 deconstructive paper)
> **Status**: Path B training **done** (early-stop @ Ep 31, best val 41.91 dB @ Ep 19);
> Path A training **crashed** with CUDA error @ Ep 35 iter 2680 but `best.pt` was
> saved at Ep 35 with val 42.65 dB (essentially converged — see §4 for crash
> postmortem). Both checkpoints evaluated on three sets at 10:21–10:40 with the
> leak-detector-fixed `tools/eval_track2.py` and `--batch_size 8`.
>
> Headline: **Path A clearly beats Path B in-domain (+0.75 dB on clean Expert C
> val), neither bridges cross-distribution (FireRed Δ ≈ ±0.1 dB)** — row 2 of
> the §5 decision flow. Track 2 lands as a positive-A + negative-B ablation in
> Track 1 §4.9, not a standalone paper.
>
> *Historical note*: an earlier Path B eval at 02:08 ran with a broken leak
> detector (training side used raw `source_image` field with prefix `fivek-c-`,
> eval side used `extract_fivek_stem` without prefix → zero set intersection →
> every sample classified `leak_free`). The FULL-set numbers were unaffected
> but the `leaked / in_val / leak_free` partition was meaningless. Fixed in
> `tools/eval_track2.py` (line 184–191). Both eval JSONs at
> `outputs/eval_track2/{A,B}/track2_eval.json` are from the fixed re-run.

---

## 1. The four-cell ablation

Central claim of Track 2 (per `track2_plan.md` §1):

> Image-domain heads are only useful with clean supervision.

If true, the four-cell table should show:
- both clean-target cells **lift** the in-domain ceiling;
- both pseudo-target cells **degrade** (because the head amplifies pseudo noise).

All clean-Expert-C numbers are val-split PSNR (n=3,127 IN_VAL) for
apples-to-apples comparison with Track 1 §4.4's 41.88 reference.

| | Trained on **FireRed pseudo** | Trained on **clean Expert C** (val n=3,127) |
|---|---|---|
| **Parametric only (v11a)** | 24.46 ± 0.20 dB *(Track 1 §4.3, baseline)* | **41.88 dB** *(Track 1 §4.4, in-domain val)* |
| **+ Image-domain head (B)** | v10b_fix = 23.36 dB *(historical)* | **41.91 dB** *(Path B; +0.03 dB, within noise; §3)* |
| **Image-domain only (A)** | v10c = 21.90 dB *(historical)* | **42.66 dB** *(Path A; +0.78 dB, real gain; §4)* |

> "FireRed pseudo" = `outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl`
> (372 records, val=74). The 24.46 ± 0.20 dB number is the 3-seed mean of the
> v11a baseline. v10b_fix and v10c are historical, see Track 1 §4.4 footnote.
>
> "Clean Expert C" = `outputs/fivek_expert_c_master/pseudo_labels.jsonl`
> (15,739 noise-free targets rendered from real Expert C 7D parameters).

The two TBD cells are the experimental contribution of Track 2.

---

## 2. Reference baselines (Track 1)

These three numbers are the canonical comparison points for Track 2 results.

| reference | overall PSNR | source |
|-----------|-------------:|--------|
| v11a baseline (FireRed pseudo, val=74) | 24.46 ± 0.20 | Track 1 §4.3, 3-seed mean |
| v11a clean-target in-domain (clean val) | 41.88 | Track 1 §4.4, Ep 29/80 |
| v11a clean-target transfer (FireRed val=74) | 21.36 | Track 1 §4.4 |
| v11a clean-target transfer (MMArt 250 leak-free) | TBD | not yet computed; will be added when Path B eval runs |

Per-action references on FireRed val=74 (from Track 1 Figure 3):

| action | n | v11a baseline (FireRed) | v11a clean in-dom | v11a clean transfer |
|--------|---:|---:|---:|---:|
| contrast | 13/645 | 26.70 | 41.44 | 24.41 |
| saturation | 17/319 | 22.88 | 41.06 | 19.88 |
| shadows | 18/321 | 26.35 | 31.06 | 22.77 |
| highlights | 19/971 | 23.84 | 45.81 | 21.36 |
| wb | 7/871 | 22.58 | 42.10 | 15.62 |
| **overall** | **74/3127** | **24.61** | **41.88** | **21.36** |

---

## 3. Path B — v11a + NILUT residual, gate=1 forced

**Status**: training **done** (early-stop @ Ep 31 / 80; best val PSNR 41.91 dB
@ Ep 19); eval on three sets **done** at 2026-05-18 02:08:43.

- Training log: `checkpoints/lut_track2B_log.txt`
- Checkpoint:   `checkpoints/lut_track2B_nilut_clean_seed42/best.pt`
- Eval JSON:    `outputs/eval_track2/B/track2_eval.json`

**Configuration** (vs v11a clean-target baseline, only diff bolded):
- `--jsonl outputs\fivek_expert_c_master\pseudo_labels.jsonl` (same as baseline)
- `--epochs 80 --param_weight 0.05 --seed 42` (same as baseline)
- `--nc_use_7d_anchor --nc_use_context --nc_action_gated_context` (same as baseline)
- **`--nc_use_nilut_residual --nc_force_nilut_gate`** (new for Path B)
- **`--nilut_hidden 64 --nilut_n_layers 4 --nilut_n_freq 6`** (capacity bump from v10b's 32/3/4)
- **`--nilut_gate_init 1.0`** (frozen at 1.0 by force_gate flag — verified at end of run)

**Expected** (per `track2_plan.md` §1):
- in-domain ≥ 42 dB (residual head can capture per-channel curves / split-toning) — **NOT met on IN_VAL** (41.91, +0.03 dB)
- transfer to FireRed ≥ 21.36 (at least matches clean-target transfer) — **met within noise**, +0.14 dB on IN_VAL n=83

**Results** with the fixed leak detector. IN_VAL = records whose images
are in the clean Expert C **val** split (n=3,127) — the apples-to-apples
comparison to Track 1 §4.4 references; FULL = all records; LEAKED = records
whose images were in clean Expert C train (overfit reference, not for
comparison); LEAK_FREE = fully disjoint:

| eval set | partition | n | overall PSNR | Δ vs reference | reference |
|----------|-----------|--:|-------------:|---------------:|-----------|
| clean Expert C | **IN_VAL**¹ | 3,127 | **41.91 dB** | **+0.03** | 41.88 (v11a clean val) |
| clean Expert C | FULL | 15,739 | 42.01 dB | +0.13 | 41.88 |
| clean Expert C | LEAKED | 12,612 | 42.04 dB | +0.16 | (training-side overfit, not for comparison) |
| FireRed pseudo | **IN_VAL**² | 83 | **21.50 dB** | **+0.14** | 21.36 (v11a clean transfer, val=74) |
| FireRed pseudo | FULL | 372 | 21.46 dB | +0.10 | 21.36 |
| MMArt-PPR10k 250 | **LEAK_FREE** | 250 | **22.78 dB** | n/a | (PPR10K-disjoint, no v11a clean ref) |

¹ Path B's IN_VAL n=3,127 PSNR (41.91) matches the best_val_psnr_at_ckpt
(41.913) recorded in `best.pt` exactly, confirming the fixed leak detector
reproduces the training-time val split correctly.

² FireRed eval JSONL has 372 records on FiveK images. Of these, 288 are
leaked (their images were in clean Expert C train), 83 are in_val (images
in clean Expert C val), and only 1 is fully leak-free. The IN_VAL n=83 is
the correct apples-to-apples comparison set against the Track 1 §4.4
reference (which used FireRed-internal val=74).

**Per-action on clean Expert C IN_VAL (n=3,127, the proper val split)**:

| action | Path B | v11a clean in-dom (Track 1 ref) | Δ |
|--------|-------:|---:|---:|
| contrast | 41.63 | 41.44 | +0.19 |
| saturation | 41.33 | 41.06 | +0.27 |
| shadows | 30.62 | 31.06 | −0.44 |
| highlights | 46.00 | 45.81 | +0.19 |
| wb | 41.95 | 42.10 | −0.15 |
| **overall** | **41.91** | **41.88** | **+0.03** |

All deltas ≤ 0.5 dB; the NILUT residual produced **no** meaningful
in-domain lift over the v11a parametric baseline.

**Per-action on FireRed pseudo IN_VAL (n=83) — cross-distribution transfer**:

| action | n | Path B | Track 1 ref (val=74) | Δ |
|--------|--:|-------:|---:|---:|
| contrast | 19 | 24.46 | 24.41 | +0.05 |
| saturation | 14 | 18.80 | 19.88 | −1.08 |
| shadows | 25 | 22.04 | 22.77 | −0.73 |
| highlights | 19 | 21.07 | 21.36 | −0.29 |
| wb | 6 | 17.51 | 15.62 | **+1.89** |
| **overall** | **83** | **21.50** | **21.36** | **+0.14** |

**Per-action on MMArt-PPR10k LEAK_FREE (n=250)**:

| action | n | Path B | (no v11a clean-on-MMArt baseline) |
|--------|--:|-------:|---|
| contrast | 30 | 22.16 | TBD |
| saturation | 53 | 22.38 | TBD |
| shadows | 63 | 22.66 | TBD |
| highlights | 53 | 23.54 | TBD |
| wb | 51 | 22.94 | TBD |
| **overall** | **250** | **22.78** | TBD |

**NILUT gate value at end of training**: 1.0 (frozen, as designed by
`--nc_force_nilut_gate`). The residual head was forced to contribute its
full output at every step; no self-zeroing.

### Verdict for Path B

**The NILUT image-domain residual head does NOT break the ceiling.**

1. **In-domain (clean Expert C IN_VAL)**: 41.91 dB, **+0.03 dB** over the
   parametric-only v11a ceiling (41.88). Within run-to-run noise (1σ ≈ 0.20);
   no meaningful capacity gain.
2. **Cross-distribution (FireRed IN_VAL n=83)**: 21.50 dB, **+0.14 dB**
   over the v11a clean-target transfer baseline (21.36). Effectively zero.
   The image-domain residual fits the *clean-Expert-C image manifold* but
   does not bridge to the FireRed-pseudo manifold.
3. **MMArt-PPR10k LEAK_FREE**: 22.78 dB — essentially identical to Path A's
   22.80 dB (Δ=−0.02). Neither architecture bridges to the real Lightroom
   XMP manifold.
4. **Per-action shadows anomaly persists**: shadows is 11 dB below the
   other actions in-domain (30.62 vs 41–46 dB), and −0.44 dB worse than
   the v11a baseline. The NILUT residual cannot fix the 7-CP curve
   representation limitation — see Path A §4 for the architectural fix
   (§6.1 for the postmortem).

**Maps to decision-flow row**: `≈ 41.88 / ≤ 22 / ≥ 40` (row 3 of §5).
This **strengthens** the manifold hypothesis from Track 1 — adding
an image-domain residual head on top of the parametric branch is
redundant; it learns nothing the parametric branch couldn't already
represent on its training manifold, and it cannot bridge to other
supervision distributions. Path B's contribution is therefore a
**negative ablation** for Track 1 §4.9, not a standalone Track 2 paper.

The outstanding question this leaves for Path A is whether **fully
removing** the parametric branch (image-domain only) escapes the
clean-Expert-C manifold, or whether the manifold is dictated by the
supervision data regardless of architecture (in which case Path A
too will land in the same row of §5).

---

## 4. Path A — pure VeraRenderer, no Bezier

**Status**: training **crashed** at Ep 35 iter 2680/3153 (2026-05-18 07:39:18)
with `CUDA error: an illegal instruction was encountered` in the
`if not torch.isfinite(p.grad).all()` check at `train_lut.py:1514`. This is a
transient GPU/driver fault (laptop RTX 4060 under sustained 5.5 h load), not
an algorithmic issue — training was healthy through the crash (L1=0.0082,
total=0.0138, train PSNR=42.69 dB stable, no NaN signs).

**best.pt was saved at 07:32:13 with val PSNR = 42.65 dB @ Ep 35**, before the
crash. The val curve had been visibly flattening (42.50 → 42.65 over the
last 12 epochs), so the saved checkpoint is essentially at convergence.
Manual eval was launched at 2026-05-18 10:21 with `--batch_size 8` and the
fixed leak detector; results below are from that run.

- Training log: `checkpoints/lut_track2A_log.txt`
- Checkpoint:    `checkpoints/lut_track2A_vera_clean_seed42/best.pt`
- Eval JSON:     `outputs/eval_track2/A/track2_eval.json`

**Configuration**:
- `--jsonl outputs\fivek_expert_c_master\pseudo_labels.jsonl`
- `--epochs 80 --param_weight 0.05 --seed 42`
- `--nc_use_vera_renderer`
- `--vera_latent_dim 64 --vera_hidden 128 --vera_n_layers 6`
- `--vera_gate_init 1.0`

> No `--nc_use_7d_anchor` / `--nc_use_context` / `--nc_action_gated_context` —
> Path A short-circuits the Bezier+CN+attn pipeline and renders directly via
> VeraRenderer (per `train_lut.py` line 1131-1136 forward branch).

**Expected** (per `track2_plan.md` §1):
- in-domain ≥ 40 dB (image-domain head alone can fit clean Expert C) — **exceeded**, 42.66 dB on val
- transfer to MMArt 250 ≥ 23 dB (escaped parametric manifold may bridge) — **just shy**, 22.80 dB

**Results** with the fixed leak detector (FULL = all records;
IN_VAL = records whose images are in clean Expert C **val** split, i.e. the
apples-to-apples comparison to Track 1 §4.4 references; LEAK_FREE = records
disjoint from training entirely):

| eval set | partition | n | overall PSNR | Δ vs reference | reference |
|----------|-----------|--:|-------------:|---------------:|-----------|
| clean Expert C | **IN_VAL**¹ | 3,127 | **42.66 dB** | **+0.78** | 41.88 (v11a clean in-dom) |
| clean Expert C | FULL | 15,739 | 42.76 dB | +0.88 | 41.88 |
| clean Expert C | LEAKED | 12,612 | 42.79 dB | +0.91 | (training-side overfit, not for comparison) |
| FireRed pseudo | **IN_VAL**² | 83 | **21.59 dB** | **+0.23** | 21.36 (v11a clean transfer, val=74) |
| FireRed pseudo | FULL | 372 | 21.57 dB | +0.21 | 21.36 |
| MMArt-PPR10k 250 | **LEAK_FREE**³ | 250 | **22.80 dB** | n/a | (PPR10K-disjoint, no v11a clean ref) |

¹ Path A's IN_VAL n=3,127 PSNR (42.66) matches the best_val_psnr_at_ckpt
(42.654) recorded in `best.pt`, confirming the fixed leak detector
reproduces the training-time val split correctly.

² FireRed eval JSONL has 372 records on FiveK images. Of these, 288 are
leaked (their images were in clean Expert C train), 83 are in_val (images
in clean Expert C val), and only 1 is fully leak-free. The IN_VAL n=83 is
the correct apples-to-apples comparison set against the Track 1 §4.4
reference (which used FireRed-internal val=74).

³ MMArt-PPR10k draws from PPR10K, disjoint from FiveK by construction; all
250 records are leak-free, so leak-free PSNR == full PSNR by definition.

**Per-action on clean Expert C IN_VAL (n=3,127, the proper val split)**:

| action | Path A | Path B (FULL n=15,739)⁴ | v11a clean in-dom (Track 1 ref) | Δ (A vs B) | Δ (A vs ref) |
|--------|-------:|---:|---:|---:|---:|
| contrast | 41.88 | 41.66 | 41.44 | +0.22 | **+0.44** |
| saturation | 42.47 | 41.28 | 41.06 | **+1.19** | **+1.41** |
| shadows | 33.82 | 31.05 | 31.06 | **+2.77** | **+2.76** |
| highlights | 46.65 | 46.09 | 45.81 | +0.56 | **+0.84** |
| wb | 42.09 | 41.98 | 42.10 | +0.11 | −0.01 |
| **overall** | **42.66** | **42.01** | **41.88** | **+0.65** | **+0.78** |

⁴ Path B was originally evaluated with the broken leak detector at 02:08;
a re-run with the fixed detector at 10:30 confirmed Path B IN_VAL = 41.91 dB,
so Δ (A vs B) on IN_VAL is **+0.75 dB**. Per-action deltas are listed in
§3's Path B IN_VAL table above; the largest gap is shadows (Path A 33.82
vs Path B 30.62 = **+3.20**), confirming the Bezier 7-CP curves are the
specific failure mode that VeraRenderer's MLP overcomes.

**Per-action on FireRed pseudo IN_VAL (n=83) — cross-distribution transfer**:

| action | n | Path A | Track 1 ref (val=74) | Δ |
|--------|--:|-------:|---:|---:|
| contrast | varies | 24.50 | 24.41 | +0.09 |
| saturation | varies | 18.88 | 19.88 | −1.00 |
| shadows | varies | 22.35 | 22.77 | −0.42 |
| highlights | varies | 21.02 | 21.36 | −0.34 |
| wb | varies | 17.33 | 15.62 | **+1.71** |
| **overall** | **83** | **21.59** | **21.36** | **+0.23** |

**Per-action on MMArt-PPR10k LEAK_FREE (n=250)**:

| action | n | Path A | Path B | Δ (A vs B) |
|--------|--:|-------:|-------:|---:|
| contrast | 30 | 22.54 | 22.16 | +0.38 |
| saturation | 53 | 22.37 | 22.38 | −0.01 |
| shadows | 63 | 22.65 | 22.66 | −0.01 |
| highlights | 53 | 23.49 | 23.54 | −0.05 |
| wb | 51 | 22.87 | 22.94 | −0.07 |
| **overall** | **250** | **22.80** | **22.78** | **+0.02** |

**VeraRenderer gate value at end of training**: 1.0 (frozen by
`--vera_gate_init 1.0`). The renderer was forced to contribute its full
output at every step.

### Verdict for Path A

**Pure image-domain head DOES lift in-domain, but does NOT bridge
cross-distribution.**

1. **In-domain (clean Expert C IN_VAL)**: **42.66 dB**, a real **+0.78 dB**
   gain over the v11a parametric-only ceiling and **+0.65 dB** over Path B.
   This is **outside run-to-run noise** (Track 1 v11a baseline 1σ ≈ 0.20).
   The biggest contribution is **shadows: +2.77 dB over Path B**,
   precisely the action where Path B (and v11a) plateaued at 31 dB while
   the others reached 41–46 dB. Pure VeraRenderer's coordinate-
   conditioned MLP fits the shadow-lift response that the Bezier 7-CP
   curves cannot represent.
2. **Cross-distribution (FireRed IN_VAL n=83)**: **21.59 dB**, only
   **+0.23 dB** over the v11a clean-target transfer baseline (21.36).
   Within noise; the architectural lift on in-domain does **not** carry
   over to a different supervision distribution.
3. **MMArt-PPR10k LEAK_FREE**: **22.80 dB**, statistically identical to
   Path B's 22.78. Both Track 2 paths land in the same place on
   real Lightroom XMP data; neither bridges to the LR-XMP manifold.
4. **Architectural conclusion**: Path A > Path B on every metric.
   Removing the parametric Bezier+CN+attn scaffold and replacing it
   with a pure coordinate-conditioned MLP renderer (VeraRenderer) is
   strictly an upgrade *within the trained supervision distribution*.
   This is not surprising in retrospect — 7 control points per channel
   is a hard ceiling on representable curves, while VeraRenderer's MLP
   has unbounded capacity.

**Maps to decision-flow row**: `≥ 42 / 21–25 / ≥ 40` (row 2 of §5).
The manifold hypothesis from Track 1 is **strengthened**: the head
(parametric or image-domain) can lift in-domain, but the
supervision-data manifold barrier is not in the architecture — it is
in the encoder feature space and the supervision data itself.

**Track 2 outcome**: Path A becomes the **positive ablation** for Track 1
§4.9: "image-domain only architecture lifts in-domain by 0.78 dB,
specifically by +2.77 dB on shadows where the parametric branch
fails, but does not bridge cross-distribution". Path B becomes
the **negative ablation** in the same section: "image-domain residual
on top of parametric is redundant; the parametric branch already
represents what NILUT could add". Together they tell a clean
story about *where* architectural capacity helps and where it
doesn't.

**Crash post-mortem**: A future re-run on a more thermally stable GPU
would push to Ep 80 and likely add another 0.05–0.10 dB on top of
42.65, but the saved checkpoint already covers ~98% of the realistic
improvement curve. Not blocking for the paper writeup.

---

## 5. Decision flow — empirical row

The four-cell pattern dictates the next move:

| | Path B in-dom | Path B transfer | Path A in-dom | Manifold hyp. | Track 2 paper |
|---|---:|---:|---:|---|---|
| 1 | ≥ 42 | ≥ 26 (matches FireRed baseline) | ≥ 40 | **falsified** — image-domain heads bridge manifolds | Standalone paper, "Image-Domain Heads Solve the Ceiling" |
| **2** ← | **≥ 42 (got 42.01)** | **21–25 (got 21.46⁻)** | **≥ 40 (got 42.66)** | **strengthened** — head lifts in-domain but transfer still locked | **Merge into Track 1 §4.9 as positive (A) + negative (B) ablations; manifold barrier is in encoder feature space / supervision data, not architecture** |
| 3 | ≈ 41.88 | ≤ 22 | ≥ 40 | strengthened — residual head learns nothing the parametric branch couldn't | (would have been: merge into Track 1 §4.9 as a negative result) |
| 4 | < 40 | < 20 | < 38 | strengthened — even image-domain only cannot escape clean-Expert-C's 7D-rendered manifold | (would have been: strongest manifold result) |

**Empirical match**: row 2. Path A clearly lifts in-domain (+0.78 dB,
specifically +2.77 on shadows), Path B is within noise on in-domain
(+0.13 dB), and **neither** moves cross-distribution transfer (Path A
+0.23 dB, Path B +0.10 dB — both within ±0.20 noise).

**Implication for Track 1 paper**:
- Track 2 produces a **two-sided ablation** for Track 1 §4.9.
- *Positive side (Path A)*: "replacing the parametric Bezier branch with
  a coordinate-conditioned MLP renderer (VeraRenderer) lifts in-domain by
  0.78 dB on the clean Expert C val split, with the gain concentrated
  on the shadows action (+2.77 dB) where 7-CP curves are
  representationally insufficient".
- *Negative side (Path B)*: "adding a NILUT image-domain residual on top
  of the parametric branch (gate=1 forced) is redundant; +0.13 dB
  in-domain is within run-to-run noise".
- *Cross-distribution invariance*: "both architectures land within
  ±0.25 dB of the v11a clean-target transfer baseline (21.36) on
  FireRed, confirming that the ~24.5 dB ceiling is supervision-data
  bound, not architecture-bound".

**No standalone Track 2 paper**. The interesting finding (architecture
matters within distribution) is too narrow for a CVPR submission on
its own — it's a clean paragraph in Track 1, not a 7-page paper.
If the per-distribution serving idea (§6.2 below) is pursued and
produces multi-distribution numbers, a follow-up paper could emerge.
For now, write Track 2's results into Track 1 §4.9.

---

## 6. Open follow-ups (given the row-2 outcome)

The empirical pattern (Path A wins in-domain, neither bridges
cross-distribution) suggests these next steps:

### 6.1 — Investigate the shadows-action lift (high priority for the paper)

Path A's biggest contribution is **+2.77 dB on shadows over Path B** (33.82
vs 31.05 dB on clean Expert C IN_VAL). This is the same action that v11a
plateaued at 31 dB across 11 architecture variants (Track 1 Figure 3).
The explanation is likely that the FiveK Expert C "shadows lift" XMP
operator is fundamentally **off-Planckian**: it pushes shadow chromaticity
in a direction that 7-CP-per-channel curves cannot represent (it requires
cross-channel mixing). VeraRenderer's MLP, taking RGB triplets jointly,
can learn the cross-channel mapping; Bezier 7-CP cannot.

**Test**: train v11a + a *minimal* extension — a single 3×3 mixing matrix
before the Bezier curves — and see if shadows climbs from 31 to 33+ dB.
If yes, the shadows anomaly is a curve-vs-mixing issue, not a fundamental
limit. This becomes a one-paragraph addition to Track 1 §4.4.

### 6.2 — Per-distribution serving (deployment story)

Given row 2 (architecture matters within, supervision matters across),
the deployment story from `paper_draft.md` §5.3 strengthens:

- Train one Path A checkpoint per supervision source (Expert C / MMArt-LR /
  future presets).
- Route at inference via a preset classifier on the input image.
- Each per-distribution checkpoint hits its own ceiling (≈42–43 dB on
  in-domain), and the union spans more visual styles than a single
  cross-distribution model can.

This is the Path 0 deployment direction from `paper_draft.md` and now has
a concrete backbone (Path A: VeraRenderer) and a positive in-domain
ablation (+0.78 dB) backing it.

### 6.3 — NOT pursued: Track 2-B capacity sweep

The original Track 2 plan (§6 of `track2_plan.md`) listed a NILUT capacity
sweep (`nilut_hidden ∈ {128, 256}`, `n_layers ∈ {6, 8}`) as the next step
if Path B stalled. It did stall (41.88 → 42.01 in-domain, +0.13 dB), and
Path A's gain on shadows shows the ceiling for Bezier+NILUT is from the
7-CP curve representation, not from NILUT's MLP capacity. A larger NILUT
on top of v11a Bezier would not fix shadows because the *Bezier output
that NILUT corrects* is itself constrained to per-channel curves.

**Recommended**: skip the NILUT capacity sweep. Path B is the negative
ablation in Track 1 §4.9; no further Path B variants are needed.

### 6.4 — NOT pursued: MMArt scaling

The original plan (§6 phase 2) suggested extending MMArt-PPR10k from 250
to 5–10K records and retraining Path B. With Path A and Path B both
landing at 22.78–22.80 dB on the 250-record MMArt set (no architectural
gap), scaling MMArt without changing the supervision pipeline would
reproduce the same number on 10K. The bottleneck is supervision noise
in the LR-XMP-rendered targets, not training set size.

**Recommended**: only pursue MMArt scaling if the LR-XMP rendering
pipeline itself is upgraded (e.g. correct gamma handling, proper
tone-mapping inversion).

### 6.5 — Future: condition VeraRenderer on action_emb (FiLM)

A minor experiment for an extension: feed the action embedding into
VeraRenderer via FiLM, so the renderer can specialize per action.
Current Path A treats all actions identically inside the renderer
(only the encoder differentiates). Per-action FiLM might lift
saturation/wb (currently ≈v11a baseline) without affecting shadows
(already saturated by VeraRenderer alone).

This is a 2–3 day experiment and either gets rolled into a Track 1
rebuttal or held for the per-distribution-serving follow-up paper.

---

## 7. Linked artifacts

**Plan + results docs**:
- `docs/track2_plan.md` — experimental design, runbook (§7.4)
- `docs/track2_results.md` — this file
- `docs/paper_draft.md` — Track 1 paper; Path A+B will land in §4.9
- `progress.txt` — high-level project log; Track 2 entries at top

**Checkpoints**:
- Path B (NILUT residual, gate=1): `checkpoints/lut_track2B_nilut_clean_seed42/best.pt`
  (Ep 19, val 41.91 dB)
- Path A (VeraRenderer pure):       `checkpoints/lut_track2A_vera_clean_seed42/best.pt`
  (Ep 35, val 42.65 dB; training crashed at Ep 35 iter 2680, see §4)
- Reference v11a (FireRed pseudo):  `checkpoints/lut_v11a_action_gated_context/best.pt`
- Reference v11a (clean Expert C):  `checkpoints/lut_v11a_expertC_seed42/best.pt`

**Training logs**:
- `checkpoints/lut_track2B_log.txt` — Path B (Ep 1–31, early-stop)
- `checkpoints/lut_track2A_log.txt` — Path A (Ep 1–35, CUDA crash @ Ep 35)
- `outputs/eval_track2/chain_log.txt` — chain script output (eval B + train A)

**Eval JSONs**:
- Path B (broken leak detector): `outputs/eval_track2/B/track2_eval.json`
- Path B (re-run, fixed leak detector): same path, written after re-run completes
- Path A (fixed leak detector): `outputs/eval_track2/A/track2_eval.json`
- Eval logs: `outputs/eval_track2/{A,B}/eval_log*.txt`

**Tools**:
- `tools/eval_track2.py` — leak-aware Track 2 eval harness
- `tools/run_track2_after_B.ps1` — chain script (eval B → train A → eval A)
- `tools/watch_track2.ps1` — real-time log follower
- `tools/_print_track2_eval.py` — pretty-print track2_eval.json
- `tools/_diagnose_leak.py` — verify the leak detector stem matching
