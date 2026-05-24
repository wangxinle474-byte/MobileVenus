"""Path X smoke test: verify implicit residual head wires correctly.

Tests:
  1. Construct NamedCurvesPredictor with use_implicit_head=True
  2. Load v11a backbone checkpoint (implicit_head should be reported missing)
  3. Verify freeze_backbone == only implicit_head trainable
  4. Forward pass shape + identity behavior at gate=0
  5. Gate sensitivity: refined-output should change with gate=0.5

Usage:
  python tools/smoke_test_pathX.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from training.firered_baseline.train_lut import NamedCurvesPredictor


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'[device] {device}')

    # Match v11a config: nc_n_colors=3, nc_n_control_points=7,
    # use_attention=False, use_context=True, action_gated_context=True,
    # use_7d_anchor=True, dropout=0.5
    print('\n=== Step 1: construct NamedCurvesPredictor + implicit_head ===')
    model = NamedCurvesPredictor(
        n_colors=3,
        n_control_points=7,
        use_attention=False,
        use_context=True,
        action_gated_context=True,
        use_7d_anchor=True,
        use_implicit_head=True,
        implicit_head_base_ch=32,
        implicit_head_gate_init=0.0,
        image_size=256,
        dropout=0.5,
    ).to(device)

    n_total = sum(p.numel() for p in model.parameters())
    n_implicit = sum(p.numel() for n, p in model.named_parameters()
                     if n.startswith('implicit_head'))
    n_backbone = n_total - n_implicit
    print(f'  total: {n_total/1e6:.2f}M, '
          f'backbone: {n_backbone/1e6:.2f}M, '
          f'implicit_head: {n_implicit/1e6:.2f}M')
    assert n_implicit > 1e5, 'implicit_head too small'

    # Step 2: load v11a backbone
    print('\n=== Step 2: load v11a backbone weights ===')
    ck_path = PROJECT_ROOT / 'checkpoints/lut_v11a_action_gated_context/best.pt'
    if not ck_path.exists():
        print(f'  [skip] backbone ckpt not found: {ck_path}')
    else:
        bk = torch.load(str(ck_path), map_location=device, weights_only=False)
        missing, unexpected = model.load_state_dict(
            bk['model_state_dict'], strict=False)
        non_head_missing = [m for m in missing
                            if not m.startswith('implicit_head')]
        print(f'  total missing: {len(missing)} '
              f'(non-head: {len(non_head_missing)})')
        print(f'  unexpected: {len(unexpected)}')
        assert len(non_head_missing) == 0, \
            f'non-head missing: {non_head_missing[:5]}'
        assert all(m.startswith('implicit_head') for m in missing), \
            f'unexpected missing: {missing[:5]}'
        print(f'  loaded val_psnr={bk.get("val_psnr", 0):.2f} '
              f'@ Ep{bk.get("epoch", "?")}')

    # Step 3: freeze logic
    print('\n=== Step 3: freeze backbone, only implicit_head trainable ===')
    for name, p in model.named_parameters():
        if name.startswith('implicit_head'):
            p.requires_grad_(True)
        else:
            p.requires_grad_(False)
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_frozen = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    print(f'  trainable: {n_trainable/1e6:.2f}M (target: '
          f'{n_implicit/1e6:.2f}M)')
    print(f'  frozen:    {n_frozen/1e6:.2f}M')
    assert n_trainable == n_implicit, \
        f'trainable mismatch: {n_trainable} vs {n_implicit}'

    # Step 4: forward pass + identity at gate=0
    print('\n=== Step 4: forward pass at gate=0 (identity) ===')
    model.eval()
    B = 2
    enc_in = torch.randn(B, 3, 256, 256).to(device)
    orig = torch.rand(B, 3, 256, 256).to(device)
    a_oh = torch.zeros(B, 5).to(device)
    a_oh[:, 0] = 1.0  # contrast

    with torch.no_grad():
        # Disable implicit head temporarily by zero gate (init value)
        gate_val = float(model.implicit_head.gate.item())
        print(f'  current gate = {gate_val}')
        refined_at_gate0, _, _, _ = model(enc_in, orig, a_oh)
        print(f'  refined shape: {tuple(refined_at_gate0.shape)}, '
              f'range [{refined_at_gate0.min():.3f}, '
              f'{refined_at_gate0.max():.3f}]')

        # Step 5: gate sensitivity — set gate=0.5, expect outputs change
        print('\n=== Step 5: gate sensitivity test (gate=0 vs gate=0.5) ===')
        # First disable implicit head and capture baseline
        model.use_implicit_head = False
        baseline, _, _, _ = model(enc_in, orig, a_oh)
        model.use_implicit_head = True

        # Now with implicit head + gate=0
        gate_orig = model.implicit_head.gate.data.clone()
        model.implicit_head.gate.data.fill_(0.0)
        out_g0, _, _, _ = model(enc_in, orig, a_oh)
        diff_g0 = (out_g0 - baseline).abs().mean().item()
        print(f'  gate=0.0  vs no-head baseline: mean abs diff = {diff_g0:.6f}'
              f' (should be ~0)')

        model.implicit_head.gate.data.fill_(0.5)
        out_g05, _, _, _ = model(enc_in, orig, a_oh)
        diff_g05 = (out_g05 - baseline).abs().mean().item()
        print(f'  gate=0.5  vs no-head baseline: mean abs diff = {diff_g05:.6f}'
              f' (should be > 0 only if residual nonzero)')
        # Note: after zero-init, residual is 0, so even gate=0.5 produces 0
        # change before training. We can verify by injecting noise into final
        # conv weights:
        w = model.implicit_head.final.weight
        b = model.implicit_head.final.bias
        torch.nn.init.normal_(w, mean=0.0, std=0.01)
        torch.nn.init.normal_(b, mean=0.0, std=0.01)
        out_g05_noisy, _, _, _ = model(enc_in, orig, a_oh)
        diff_noisy = (out_g05_noisy - baseline).abs().mean().item()
        print(f'  gate=0.5 + noisy final: mean abs diff = {diff_noisy:.6f}'
              f' (should be > 1e-3)')
        assert diff_noisy > 1e-3, \
            f'implicit head not contributing: {diff_noisy}'

        # Restore
        model.implicit_head.gate.data.copy_(gate_orig)

    # Step 6: gradient check with REAL init (gate=1.0, final conv zero-init).
    # This is the working cold-start config: zero-init final conv → residual=0
    # at start → output=refined (identity), but gradient w.r.t. final conv
    # weights = gate * sech²(0) * d1_features ≠ 0 so the head can learn.
    print('\n=== Step 6: gradient flow at real init (gate=1, final=0) ===')
    # Stay in eval mode: encoder uses Dropout(0.5) which would randomize
    # the backbone output and break the identity check. Gradient flow does
    # not require train mode.
    model.eval()
    torch.nn.init.zeros_(model.implicit_head.final.weight)
    torch.nn.init.zeros_(model.implicit_head.final.bias)
    model.implicit_head.gate.data.fill_(1.0)  # the new default
    # Zero existing grads from any earlier autograd
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    target = torch.rand(B, 3, 256, 256).to(device)
    refined, _, _, _ = model(enc_in, orig, a_oh)
    # Sanity: gate=1 + final=0 should still produce identity output
    diff_identity = (refined - baseline).abs().mean().item()
    print(f'  identity check (refined - no_head_baseline) = {diff_identity:.6e}'
          f' (should be ~0)')
    assert diff_identity < 1e-5, f'not identity at init: {diff_identity}'

    loss = (refined - target).abs().mean()
    loss.backward()
    n_with_grad = sum(1 for n, p in model.named_parameters()
                      if p.grad is not None and p.grad.abs().sum() > 0)
    n_implicit_with_grad_step1 = sum(
        1 for n, p in model.named_parameters()
        if n.startswith('implicit_head')
        and p.grad is not None and p.grad.abs().sum() > 0)
    n_implicit_total = sum(1 for n, _ in model.named_parameters()
                           if n.startswith('implicit_head'))
    print(f'  step1 trainable params with non-zero grad: {n_with_grad}')
    print(f'  step1 implicit_head non-zero grad: '
          f'{n_implicit_with_grad_step1}/{n_implicit_total} '
          f'(only final.{{w,b}} expected; upstream d_loss/d_d1 = '
          f'gate*sech²(0)*final.w = 0 because final.w=0 at init)')

    # final conv weights MUST have nonzero grad (gate=1, sech²(0)=1)
    fw_grad = model.implicit_head.final.weight.grad.abs().sum().item()
    fb_grad = model.implicit_head.final.bias.grad.abs().sum().item()
    print(f'  final conv weight grad sum: {fw_grad:.6e}')
    print(f'  final conv bias   grad sum: {fb_grad:.6e}')
    assert fw_grad > 1e-8, 'final conv weight not getting gradient'
    assert fb_grad > 1e-8, 'final conv bias not getting gradient'

    # After 1 manual SGD step on final, upstream layers should also get grad
    print('\n=== Step 7: after 1 step, upstream layers get gradient ===')
    with torch.no_grad():
        model.implicit_head.final.weight.add_(
            -1e-2 * model.implicit_head.final.weight.grad)
        model.implicit_head.final.bias.add_(
            -1e-2 * model.implicit_head.final.bias.grad)
    for p in model.parameters():
        if p.grad is not None:
            p.grad.zero_()
    refined2, _, _, _ = model(enc_in, orig, a_oh)
    loss2 = (refined2 - target).abs().mean()
    loss2.backward()
    n_implicit_with_grad_step2 = sum(
        1 for n, p in model.named_parameters()
        if n.startswith('implicit_head')
        and p.grad is not None and p.grad.abs().sum() > 0)
    print(f'  step2 implicit_head non-zero grad: '
          f'{n_implicit_with_grad_step2}/{n_implicit_total} '
          f'(should be most/all)')
    assert n_implicit_with_grad_step2 >= 30, \
        f'upstream gradient flow blocked: {n_implicit_with_grad_step2}'
    gate_g = model.implicit_head.gate.grad.item()
    print(f'  step2 gate grad: {gate_g:.6e} (should be != 0 now)')
    assert abs(gate_g) > 1e-8, 'gate not learning'

    print('\n[OK] All Path X smoke tests passed')


if __name__ == '__main__':
    main()
