# Track 2 Plan — Constructive Follow-up to the 24.5 dB Ceiling Paper

> **Created**: 2026-05-17
> **Companion**: `docs/paper_draft.md` (Track 1 deconstructive paper)
> **Goal**: Test whether image-domain residual learning (Path B) and full
> image-domain INR rendering (Path A) can break the parametric ceiling
> identified in Track 1.

---

## 0. The non-obvious twist

The two architectures we plan to test (NILUT residual head, VeraRetouch-style
per-pixel renderer) **already exist** in the codebase as `models/nilut.py`
and `models/vera_renderer.py`. They were tested in v10b and v10c with
**negative results**:

| variant | head | gate | trained on | val PSNR | note |
|---------|------|-----:|-----------|---------:|------|
| v10b (Plan C) | NILUT residual | 0.0 (locked) | FireRed pseudo + Qwen WB | 24.47 | gate stayed at 0 → identity |
| v10b_fix | NILUT residual | 1.0 (forced) | FireRed pseudo + Qwen WB | **23.36** | −1 dB negative |
| v10c | VeraRenderer (replaces Bezier) | 1.0 (default) | FireRed pseudo + Qwen WB | **21.90** | catastrophic |

**Why they failed**: both were trained on FireRed-style pseudo-labels.
Track 1's §4.4–§4.8 *already proved* that pseudo-label distribution is
the bottleneck. By forcing a learnable image-domain head to fit those
broken targets, v10b_fix and v10c amplified the supervision noise rather
than escaping it.

**Track 2's actual contribution is therefore not the architecture, but
the supervision-data switch.** Re-train the *same* architectures on
clean Expert C targets (the 15,739 noise-free set from §4.4) and ask
whether the ceiling moves.

This insight makes Track 2 cheaper: ~2-3 weeks instead of 6-8.

---

## 1. Two settings, one architecture pool

### Path B: v11a + NILUT residual, trained on clean Expert C

```
Image (B,3,256,256)
  │
  ├─► MobileViT-S encoder ──► feat (B, 384)
  │       │
  │       ├─► action_emb fusion ──► fused (B, 416)
  │       │       │
  │       │       ├─► BCPE ──► Bezier control points
  │       │       └─► context head ──► action-gated context map
  │       │
  │       └─► 7D ISP anchor ──► coarse_diff_isp render ──► coarse
  │
  ├─► Color naming ──► CN probability maps
  │
  └─► Bezier blend (BCPE × CN) ──► refined
                                       │
                                       ▼
              ┌──────────────────────────────────┐
              │  PATH B: NILUT residual head     │
              │   gate=1 (forced, not learnable) │
              │   trained on clean Expert C      │
              └──────────────────────────────────┘
                                       │
                                       ▼
                                    output
```

**Hyper-parameter difference vs v10b**:

- `nilut_gate_init=1.0` and **gate is registered as `requires_grad=False`**
  (forces the residual head to always contribute; no self-zeroing).
- Capacity bump: `nilut_hidden=64`, `nilut_n_layers=4`, `nilut_n_freq=6`
  (≈12K params instead of v10b's ~3K).
- Training data: **15,739 clean Expert C targets** (from §4.4), not
  FireRed pseudo. Same train/val split as v11a (seed=42, image-stem level).
- 7D anchor weight raised to 0.10 (vs v11a's 0.05) so the parametric branch
  remains a strong prior.

**Hypothesis**: With clean targets, the residual head can learn the
manifold offset between *what 7D ISP can render* and *what Expert C
actually produced* (e.g., per-channel curves, hue rotation, split-toning),
pushing the in-domain ceiling above 41.88 dB. Cross-distribution transfer
to FireRed pseudo val=74 may also recover some of the −19.5 dB drop.

### Path A: pure VeraRenderer, trained on clean Expert C

```
Image (B,3,256,256)
  │
  ├─► MobileViT-S encoder ──► feat (B, 384)
  │       │
  │       └─► action_emb fusion ──► fused (B, 416)
  │
  └─► VeraRenderer(img, fused) ──► output
       (no BCPE, no Bezier, no diff_isp)
```

**Hyper-parameter difference vs v10c**:

- Capacity bump: `latent_dim=64`, `hidden=128`, `n_layers=6`
  (≈80K params instead of v10c's ~25K).
- `gate_init=1.0` (was already default; we keep it).
- Training data: **15,739 clean Expert C targets**, not FireRed pseudo.
- *No 7D anchor* (we want to test the upper bound of image-domain only).

**Hypothesis**: Without the parametric prior, can a pure image-domain
INR fit the clean Expert C targets *better than* v11a + NILUT residual?
If yes (Path A > Path B in-domain), the parametric branch is a real
constraint. If no (Path A ≤ Path B), interpretability is free.

### Path A+B comparison cells (the four-cell ablation)

| | Train on **FireRed pseudo** (broken) | Train on **clean Expert C** (Track 2) |
|---|---|---|
| **Parametric only (v11a)** | v11a baseline = 24.46 ± 0.20 dB (Track 1) | Track 1 §4.4 clean = 41.88 dB in-domain |
| **+ Image-domain head** | v10b_fix = 23.36 dB *(negative)* | **B-clean** (this plan, expected: > 42 dB?) |
| **Image-domain only** | v10c = 21.90 dB *(catastrophic)* | **A-clean** (this plan, expected: 40-43 dB?) |

The four-cell pattern is the central scientific claim of Track 2:
*image-domain heads are only useful with clean supervision*.

---

## 2. Datasets

| Dataset | Records | Use | Status |
|---------|--------:|-----|--------|
| Clean Expert C | **15,739** | Path A & B training | exists (`outputs/fivek_expert_c_master/pseudo_labels.jsonl`) |
| Clean Expert C val | ~3,000 | Path A & B in-domain val | derived from above by seed=42 stem split |
| FireRed pseudo val=74 | 74 | cross-distribution transfer | exists (canonical Track 1 val) |
| MMArt-PPR10k 250 | 250 | cross-source transfer | exists (`outputs/mmart_pseudo_labels/v1_250/`) |

**No new data collection needed** for Phase 1. Phase 2 may extend MMArt
to 5-10K if Phase 1 results are promising.

---

## 3. Evaluation protocol (matches Track 1)

For each trained checkpoint, compute three numbers:

1. **In-domain PSNR** — clean Expert C val, all 5 actions.
   *Comparison*: v11a on this val ≈ 41.88 dB (§4.4).
2. **Transfer to FireRed pseudo val=74** — same protocol as Track 1
   §4.7 / §4.8.
   *Comparison*: v11a baseline = 24.46 dB; clean-target transfer = 21.36 dB.
3. **Transfer to MMArt-PPR10k 250** — leak-aware (PPR10k disjoint from
   FiveK by construction).
   *Comparison*: v11a head-to-head = 20.79 dB.

A successful Path B is one where in-domain ≥ 42 dB AND transfer ≥ 21.36 dB
(at least matches clean-target transfer; ideally exceeds it).
A successful Path A is one where in-domain ≥ 40 dB (image-domain alone
can recover the ceiling) AND transfer ≥ 21 dB.

---

## 4. Schedule (3 weeks, ≈4 training runs)

### Week 1 — Path B (low risk)

| day | task | output |
|-----|------|--------|
| 1 | Edit `NamedCurvesPredictor` to register `nilut.gate` with `requires_grad=False` when a new `--nc_force_nilut_gate` flag is set. Bump capacity (`--nilut_hidden 64 --nilut_n_layers 4 --nilut_n_freq 6`). | code change |
| 2 | Add `--clean_target_jsonl` flag for direct training on `outputs/fivek_expert_c_master/pseudo_labels.jsonl` (already supported via `--extra_jsonl`; just verify). | code verify |
| 3-4 | Train Path B (60 epochs, batch=4, AdamW). | `checkpoints/lut_track2B_nilut_clean_seed42/best.pt` |
| 5 | Eval Path B on (clean val, FireRed val=74, MMArt 250). | `outputs/eval_track2/B_results.json` |

### Week 2 — Path A (medium risk)

| day | task | output |
|-----|------|--------|
| 6 | Bump VeraRenderer capacity (`--vera_latent_dim 64 --vera_hidden 128 --vera_n_layers 6`). Verify training pipeline runs with `--nc_use_vera_renderer` and clean targets. | code change |
| 7-8 | Train Path A (60 epochs). | `checkpoints/lut_track2A_vera_clean_seed42/best.pt` |
| 9 | Eval Path A on three eval sets. | `outputs/eval_track2/A_results.json` |
| 10 | Aggregate four-cell results table. | `docs/track2_results.md` |

### Week 3 — Decision + write-up

| day | task | output |
|-----|------|--------|
| 11-12 | Failure-mode analysis (which actions improve, which don't). Per-action breakdown. | `outputs/eval_track2/per_action.csv` |
| 13 | Decide: (i) merge Track 2 into Track 1 paper as new §4.9, OR (ii) write Track 2 as standalone follow-up paper. | decision doc |
| 14-15 | If (ii): start standalone draft. If (i): integrate findings into existing draft. | updated `paper_draft.md` |

---

## 5. Falsification criteria

We falsify the manifold hypothesis (the central claim of Track 1 §5.2)
if **either** of the following holds:

- **Path B in-domain ≥ 43 dB AND Path B transfer to FireRed val=74 ≥ 26 dB**
  → image-domain residuals do bridge manifolds when trained on clean targets.
- **Path A in-domain ≥ 42 dB AND Path A transfer to MMArt 250 ≥ 23 dB**
  → pure image-domain INR escapes the parametric manifold entirely.

If both Path A and Path B in-domain are >42 dB but transfer numbers stay
near v11a/clean-target baselines, the manifold hypothesis is *strengthened*:
even removing the 7D ISP bottleneck doesn't recover transfer, suggesting
the manifold barrier is in the *encoder feature space* itself, not in
the rendering head. This would be a valuable negative result.

---

## 6. Risks and mitigations

| risk | mitigation |
|------|-----------|
| Path B in-domain fails to exceed 41.88 (the v11a clean-target ceiling) → NILUT residual is redundant given the 7D anchor. | Test alternate residual designs: per-pixel residual via VeraRenderer wired in residual mode (gate=1, output added to v11a fused). Also test removing 7D anchor for Path B (treat it as Path A with structural prior). |
| Path A blows up training (catastrophic like v10c). | Add SSIM+L1 loss curriculum: 10 epochs L1-only warmup, then add SSIM. Initialize VeraRenderer to identity (zero-init final layer is already done). |
| Clean Expert C train data lacks WB diversity (all are Planckian since extracted via 1D temperature). | This is acknowledged in Track 1 §4.6 (boundary-biased temperature distribution). For Path A and B WB action specifically, augment with synthetic off-Planckian samples via 2D chromaticity perturbation [R1]. |
| The 15,739 clean records have all 5 actions per image with the same Expert C target — Path A might over-fit to the trivial action collapse. | Include a strong action conditioning loss: per-action L1 weighted by action_onehot, plus a contrastive loss between same-image-different-action samples. |

---

## 7. Implementation checklist (week 1 detail)

### 7.1 Code edits to `training/firered_baseline/train_lut.py`

- [ ] Add CLI flag `--nc_force_nilut_gate` (default off) → in
      `NamedCurvesPredictor.__init__`, after `self.nilut = NILUT(...)`,
      conditionally do `self.nilut.gate.requires_grad_(False)`.
- [ ] Verify `--clean_target_jsonl` works as `--extra_jsonl` substitute.
- [ ] Add CLI flag `--vera_latent_dim`, `--vera_hidden`, `--vera_n_layers`
      to forward to `VeraRenderer(...)` constructor.

### 7.2 New evaluation tool

- [ ] `tools/eval_track2.py` — given a checkpoint, runs eval on three sets
      (clean val, FireRed val=74, MMArt 250) and emits a single JSON file
      compatible with the existing `outputs/eval_leak_aware/` schema.

### 7.3 Training commands (planned)

> **Note**: We use `--jsonl <clean>` (which **overrides** the default
> `outputs/inverse_fit_pilot/fivek_500_master/pseudo_labels.jsonl`)
> rather than `--extra_jsonl <clean>` (which would *append* clean to
> the default FireRed pseudo set, recreating exactly the joint-mixing
> failure mode of Track 1 §4.7). Clean-target-only is the central
> design choice of Track 2.

> Verified config from `checkpoints/lut_v11a_expertC_seed42/best.pt`
> (Track 1 §4.4 = 41.88 dB at Ep 29 / 80): `--jsonl=clean`, `--epochs=80`,
> `--param_weight=0.05`, all `nc_use_context`/`nc_action_gated_context`/
> `nc_use_7d_anchor` set. Path B keeps **all of these unchanged** and adds
> only `--nc_use_nilut_residual --nc_force_nilut_gate` plus capacity flags.
> This is the strict A/B comparison.

```powershell
# Path B: v11a Expert C + NILUT residual (gate=1 forced) — clean A/B vs v11a
python -u training\firered_baseline\train_lut.py `
  --jsonl outputs\fivek_expert_c_master\pseudo_labels.jsonl `
  --named_curves --nc_n_colors 3 --nc_n_control_points 7 `
  --nc_use_7d_anchor --nc_use_context --nc_action_gated_context `
  --nc_use_nilut_residual --nc_force_nilut_gate `
  --nilut_hidden 64 --nilut_n_layers 4 --nilut_n_freq 6 `
  --nilut_gate_init 1.0 `
  --param_weight 0.05 --dropout 0.5 `
  --epochs 80 --seed 42 `
  --out_dir checkpoints\lut_track2B_nilut_clean_seed42 `
  2>&1 | Tee-Object -FilePath checkpoints\lut_track2B_log.txt

python tools\eval_track2.py `
  --ckpt checkpoints\lut_track2B_nilut_clean_seed42\best.pt `
  --out_dir outputs\eval_track2\B

# Path A: pure VeraRenderer (no Bezier/CN/diff_isp) — image-domain only
python -u training\firered_baseline\train_lut.py `
  --jsonl outputs\fivek_expert_c_master\pseudo_labels.jsonl `
  --named_curves --nc_n_colors 3 --nc_n_control_points 7 `
  --nc_use_vera_renderer `
  --vera_latent_dim 64 --vera_hidden 128 --vera_n_layers 6 `
  --vera_gate_init 1.0 `
  --param_weight 0.05 --dropout 0.5 `
  --epochs 80 --seed 42 `
  --out_dir checkpoints\lut_track2A_vera_clean_seed42 `
  2>&1 | Tee-Object -FilePath checkpoints\lut_track2A_log.txt

python tools\eval_track2.py `
  --ckpt checkpoints\lut_track2A_vera_clean_seed42\best.pt `
  --out_dir outputs\eval_track2\A
```

### 7.4 Runbook — resume after wall-clock break

> Copy-pasteable checklist for resuming Track 2 work after stepping away.
> Designed so a future Cascade session (or the user alone) can pick up
> without re-deriving any context. Replace `PathBCkpt` etc. variables
> with the actual paths if running by hand.

**Step 1 — verify Path B finished cleanly**

```powershell
# Should print val_psnr (target ≥ 35 dB) and a recent epoch number.
python -c "import torch; ck=torch.load('checkpoints/lut_track2B_nilut_clean_seed42/best.pt', map_location='cpu', weights_only=False); print('val_psnr=', ck['val_psnr'], 'epoch=', ck['epoch'])"

# Tail the training log to confirm '[DONE]' line is present.
Get-Content checkpoints\lut_track2B_log.txt -Tail 20
```

If `val_psnr < 30` or no `[DONE]` line, training did NOT finish — check
log for traceback and re-run from §7.3 Path B command (the `--out_dir`
will resume if checkpoint exists).

**Step 2 — eval Path B on three sets**

```powershell
python tools\eval_track2.py `
  --ckpt checkpoints\lut_track2B_nilut_clean_seed42\best.pt `
  --out_dir outputs\eval_track2\B `
  2>&1 | Tee-Object -FilePath outputs\eval_track2\B\eval_log.txt
```

Expected ~5 min on RTX 4060 (15,739 + 372 + 250 records, batch=1).

**Step 3 — record Path B numbers in `track2_results.md`**

Fill in the three TBDs in §3 of `docs/track2_results.md` from
`outputs/eval_track2/B/track2_eval.json`. Specifically:
- `eval_sets.clean_expertC.subsets.full.overall` → in-domain row
- `eval_sets.firered_pseudo.subsets.leak_free.overall` → FireRed row
- `eval_sets.mmart_real_lr.subsets.leak_free.overall` → MMArt row

Plus the per-action breakdown from `firered_pseudo.subsets.leak_free.per_action`.

**Step 4 — sanity check NILUT gate at end of training**

```powershell
python -c "import torch; ck=torch.load('checkpoints/lut_track2B_nilut_clean_seed42/best.pt', map_location='cpu', weights_only=False); print('nilut.gate=', float(ck['model_state_dict']['nilut.gate']))"
```

Must print `1.0` (force-gate worked). If it drifted, file a bug.

**Step 5 — launch Path A training**

```powershell
python -u training\firered_baseline\train_lut.py `
  --jsonl outputs\fivek_expert_c_master\pseudo_labels.jsonl `
  --named_curves --nc_n_colors 3 --nc_n_control_points 7 `
  --nc_use_vera_renderer `
  --vera_latent_dim 64 --vera_hidden 128 --vera_n_layers 6 `
  --vera_gate_init 1.0 `
  --param_weight 0.05 --dropout 0.5 `
  --epochs 80 --seed 42 `
  --out_dir checkpoints\lut_track2A_vera_clean_seed42 `
  *>&1 | Tee-Object -FilePath checkpoints\lut_track2A_log.txt
```

ETA: ~12-15 hours. (Path A may be slightly faster than B because no LUT
basis branch, but VeraRenderer's per-pixel MLP is the dominant cost.)

> ⚠ Path A does **not** set `--nc_use_7d_anchor`/`--nc_use_context`/
> `--nc_action_gated_context`. The forward branch in `train_lut.py:1131`
> short-circuits to VeraRenderer when `use_vera_renderer` is true, so
> those flags are dead weight that would only confuse param_head training.

**Step 6 — eval Path A**

```powershell
python tools\eval_track2.py `
  --ckpt checkpoints\lut_track2A_vera_clean_seed42\best.pt `
  --out_dir outputs\eval_track2\A `
  2>&1 | Tee-Object -FilePath outputs\eval_track2\A\eval_log.txt
```

**Step 7 — record Path A numbers in `track2_results.md`**

Same as Step 3 but fills §4 Path A table.

**Step 8 — verify VeraRenderer gate**

```powershell
python -c "import torch; ck=torch.load('checkpoints/lut_track2A_vera_clean_seed42/best.pt', map_location='cpu', weights_only=False); print('vera.gate=', float(ck['model_state_dict']['vera.gate']))"
```

Should be `1.0` (Path A's gate is NOT force-frozen; if it drifted to 0,
the VeraRenderer learned to disable itself — important diagnostic).

**Step 9 — fill four-cell decision table in `track2_results.md` §5**

Pick the row in the decision table that matches your numbers. The four
outcomes have pre-written verdicts already.

**Step 10 — choose paper outcome**

| If §5 picks row 1 | Track 2 = standalone paper. Open `docs/track2_paper_outline.md` (create when needed) and start writing. |
| If §5 picks row 2-3 | Merge into Track 1. Add §4.9 to `docs/paper_draft.md` titled "Image-domain heads on clean supervision (Track 2 ablation)". |
| If §5 picks row 4 | Merge into Track 1 §4.9 as the strongest manifold result; this is the negative-but-conclusive outcome. |

---

## 8. Open questions

1. Should Path B's NILUT residual condition on the action label?
   (Current `nilut.py` is action-agnostic.) If yes, prepend a small
   FiLM layer projecting `action_emb → (gain, bias)` for the NILUT MLP.
2. Should Path A's VeraRenderer keep coordinate encoding `(x, y)`?
   This is mostly useful for spatially-varying edits; for global retouching
   it may add noise. Test both with and without coord_enc.
3. Should we add MMArt-PPR10k as **augmentation** to clean Expert C
   training (not as replacement, like §4.8 did) to fight per-action
   distributional sparsity? This would be Phase 2 if Phase 1 is positive.

---

## 9. Linked artifacts

- Track 1 paper draft: `docs/paper_draft.md`
- Track 1 outline: `docs/paper_outline.md`
- v11a base architecture: `training/firered_baseline/train_lut.py:777`
  (class `NamedCurvesPredictor`)
- NILUT module: `models/nilut.py` (already implemented)
- VeraRenderer module: `models/vera_renderer.py` (already implemented)
- Clean targets dataset: `outputs/fivek_expert_c_master/pseudo_labels.jsonl`
  (15,739 records)
- v10b/v10c historical results: `progress.txt:99-129, 195-205`
- v11a baseline checkpoint: `checkpoints/lut_v11a_action_gated_context/best.pt`
