# LoRA v1 Eval Report

- **Base model**: `/root/autodl-tmp/models/Qwen3-VL-4B-Instruct`
- **Adapter**: `/root/autodl-tmp/checkpoints/intelligence_camera/lora_v1`
- **Val set**: `/root/autodl-tmp/datasets/ArtEdit-Bench/sharegpt/ArtEdit_LoRA_val.json`
- **Samples**: 75
- **Runtime**: 476.2s (6.3s/sample)

## 1. Format compliance

| Metric | Value |
|---|---|
| Pred parsed_ok | **75/75 (100.0%)** |
| GT parsed_ok | 75/75 (100.0%) |
| CN parsed_ok | 35/35 (100.0%) |
| EN parsed_ok | 40/40 (100.0%) |

## 2. Key usage frequency (pred vs GT)

| Key | Pred count | GT count | Pred/GT ratio |
|---|---:|---:|---:|
| `contrast` | 75 | 75 | 1.00 |
| `shadows` | 74 | 67 | 1.10 |
| `vibrance` | 70 | 64 | 1.09 |
| `dehaze` | 65 | 48 | 1.35 |
| `exposure` | 65 | 65 | 1.00 |
| `blacks` | 61 | 63 | 0.97 |
| `saturation` | 52 | 41 | 1.27 |
| `clarity` | 30 | 38 | 0.79 |
| `tint` | 12 | 14 | 0.86 |
| `temp` | 9 | 16 | 0.56 |
| `texture` | 4 | 9 | 0.44 |
| `highlights` | 2 | 4 | 0.50 |
| `whites` | 1 | 6 | 0.17 |

## 3. Per-key value statistics (pred)

| Key | n | mean | std | min | max |
|---|---:|---:|---:|---:|---:|
| `contrast` | 75 | 13.20 | 8.10 | -22 | 40 |
| `shadows` | 74 | 7.19 | 14.52 | -18 | 42 |
| `vibrance` | 70 | 12.36 | 4.75 | -10 | 20 |
| `dehaze` | 65 | 10.00 | 0.00 | 10 | 10 |
| `exposure` | 65 | 0.14 | 0.30 | -1 | 1 |
| `blacks` | 61 | -8.93 | 5.88 | -15 | 10 |
| `saturation` | 52 | 6.27 | 7.18 | -10 | 20 |
| `clarity` | 30 | 9.67 | 3.86 | -10 | 15 |
| `tint` | 12 | 10.00 | 0.00 | 10 | 10 |
| `temp` | 9 | 10.67 | 4.64 | 3 | 15 |
| `texture` | 4 | 10.00 | 0.00 | 10 | 10 |
| `highlights` | 2 | -25.00 | 5.00 | -30 | -20 |
| `whites` | 1 | 10.00 | 0.00 | 10 | 10 |

## 4. Pred vs GT MAE (paired keys only)

| Key | n_paired | MAE | Median |
|---|---:|---:|---:|
| `contrast` | 75 | 5.76 | 4.00 |
| `shadows` | 67 | 12.70 | 6.00 |
| `vibrance` | 62 | 3.79 | 2.00 |
| `dehaze` | 46 | 0.93 | 0.00 |
| `exposure` | 60 | 0.13 | 0.08 |
| `blacks` | 52 | 8.02 | 5.00 |
| `saturation` | 33 | 6.82 | 5.00 |
| `clarity` | 20 | 2.55 | 0.00 |
| `tint` | 3 | 2.67 | 3.00 |
| `temp` | 5 | 812.80 | 3.00 |
| `whites` | 1 | 110.00 | 110.00 |

## 5. Best 5 samples (lowest L1 to GT)

| sid | lang | L1 | pred | gt |
|---|---|---:|---|---|
| 184 | CN | 10.10 | `{"contrast": 20, "shadows": 10, "blacks": -10, "clarity": 10, "dehaze": 10, "vibrance": 10, "saturation": 10}` | `{"temp": 5, "exposure": -0.1, "contrast": 15, "shadows": 10, "blacks": -10, "clarity": 10, "dehaze": 10, "vibrance": 10, "saturation": 10}` |
| 175 | CN | 20.00 | `{"exposure": 0.12, "contrast": 10, "shadows": 10, "blacks": -10, "dehaze": 10, "vibrance": 10, "saturation": 10}` | `{"exposure": 0.12, "contrast": 10, "shadows": 10, "blacks": -10, "dehaze": 10, "vibrance": 10, "saturation": -10}` |
| 173 | EN | 20.00 | `{"contrast": 20, "blacks": -15, "vibrance": 15, "dehaze": 10}` | `{"contrast": 20, "blacks": -20, "vibrance": 20}` |
| 110 | CN | 20.00 | `{"clarity": 10, "dehaze": 10, "contrast": 10, "shadows": 10, "vibrance": 10, "saturation": 10}` | `{"contrast": 20, "shadows": 10, "blacks": -10, "clarity": 10, "dehaze": 10, "vibrance": 10, "saturation": 10}` |
| 93 | CN | 20.00 | `{"exposure": 0.12, "contrast": 10, "shadows": 10, "vibrance": 15, "dehaze": 10}` | `{"exposure": 0.12, "contrast": 10, "shadows": 15, "vibrance": 20, "dehaze": 10, "blacks": -10}` |

## 6. Worst 5 samples (highest L1 to GT)

| sid | lang | L1 | pred | gt |
|---|---|---:|---|---|
| 368 | EN | 4077.48 | `{"temp": 3, "exposure": -0.23, "contrast": 10, "shadows": -10, "blacks": -10, "clarity": 10, "dehaze": 10, "vibrance": 10, "saturation": 10}` | `{"temp": 4000, "tint": 30, "exposure": 0.25, "contrast": 10, "shadows": -5, "blacks": -5}` |
| 341 | CN | 235.05 | `{"exposure": 0.23, "contrast": 10, "highlights": -30, "shadows": 10, "whites": 10, "blacks": -10, "vibrance": 10, "dehaze": 10, "tint": 10}` | `{"exposure": 0.28, "contrast": -14, "tint": 13, "whites": -100, "blacks": 28}` |
| 222 | EN | 230.28 | `{"exposure": 0.12, "contrast": -22, "shadows": 28, "blacks": 10, "dehaze": 10, "vibrance": 10, "saturation": 10, "tint": 10}` | `{"temp": 100, "tint": 10, "exposure": 0.4, "contrast": 10, "clarity": -25, "dehaze": -10, "vibrance": 20, "saturation": 5}` |
| 307 | CN | 186.10 | `{"exposure": -0.23, "contrast": 10, "shadows": 20, "blacks": -10, "dehaze": 10, "vibrance": 10, "saturation": -5}` | `{"exposure": -0.13, "contrast": 7, "highlights": -53, "shadows": 35, "whites": 50, "blacks": -50}` |
| 163 | CN | 178.31 | `{"exposure": 0.52, "contrast": 10, "shadows": 20, "blacks": -10, "clarity": 10, "dehaze": 10, "vibrance": 20, "saturation": 10}` | `{"exposure": 0.83, "contrast": 14, "highlights": -100, "shadows": -32, "blacks": -10, "vibrance": 18, "dehaze": 10}` |