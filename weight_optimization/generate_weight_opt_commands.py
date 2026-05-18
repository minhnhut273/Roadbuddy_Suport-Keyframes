#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Utility script to print ready-to-copy PowerShell commands for weight optimization runs.

Usage examples:
    python generate_weight_opt_commands.py --mode lr_root
    python generate_weight_opt_commands.py --mode lr_inside
    python generate_weight_opt_commands.py --mode hardneg_root
    python generate_weight_opt_commands.py --mode hardneg_inside
"""

from __future__ import annotations

import argparse
from textwrap import dedent


def lr_root() -> str:
    return dedent(r'''
$log = "weight_optimization\outputs\lr_sweep_sign_policy.txt"
$lrs = @(0.001,0.003,0.005,0.01)

"=== START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Encoding utf8

foreach ($lr in $lrs) {
  $tag = $lr.ToString().Replace('.','')
  $outdir = "weight_optimization/outputs/weights_sign_lr_${tag}"

  "=== SIGN_POLICY | LR=$lr | START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8

  python weight_optimization/optimize/ranking_optimizer.py `
    --data_file weight_optimization/outputs/metrics/training_data.json `
    --baseline_config configs/selector.yaml `
    --output_dir $outdir `
    --epochs 50 `
    --learning_rate $lr `
    --margin 0.1 `
    --pairs_per_support 32 `
    --after_penalty 1.0 `
    --hard_negative_ratio 0.5 `
    --policy sign_policy 2>&1 | Out-File -FilePath $log -Append -Encoding utf8

  python weight_optimization/eval/evaluate_weights.py `
    --weights_file "$outdir\optimized_weights_sign_policy.yaml" `
    --baseline_config configs/selector.yaml `
    --data_file weight_optimization/outputs/metrics/training_data.json `
    --output_file "weight_optimization/outputs/metrics/eval_sign_lr_${tag}.json" `
    --after_penalty 1.0 `
    --policy sign_policy 2>&1 | Out-File -FilePath $log -Append -Encoding utf8

  "=== SIGN_POLICY | LR=$lr | END $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8
  "" | Out-File -FilePath $log -Append -Encoding utf8
}

"=== END $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8
Write-Host "Done. Log saved to $log"
''').strip() + "\n"


def lr_inside() -> str:
    return dedent(r'''
$log = "outputs\lr_sweep_sign_policy.txt"
$lrs = @(0.001,0.003,0.005,0.01)

"=== START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Encoding utf8

foreach ($lr in $lrs) {
  $tag = $lr.ToString().Replace('.','')
  $outdir = "outputs/weights_sign_lr_${tag}"

  "=== SIGN_POLICY | LR=$lr | START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8

  python optimize/ranking_optimizer.py `
    --data_file outputs/metrics/training_data.json `
    --baseline_config ..\configs\selector.yaml `
    --output_dir $outdir `
    --epochs 50 `
    --learning_rate $lr `
    --margin 0.1 `
    --pairs_per_support 32 `
    --after_penalty 1.0 `
    --hard_negative_ratio 0.5 `
    --policy sign_policy 2>&1 | Out-File -FilePath $log -Append -Encoding utf8

  python eval/evaluate_weights.py `
    --weights_file "$outdir\optimized_weights_sign_policy.yaml" `
    --baseline_config ..\configs\selector.yaml `
    --data_file outputs/metrics/training_data.json `
    --output_file "outputs/metrics/eval_sign_lr_${tag}.json" `
    --after_penalty 1.0 `
    --policy sign_policy 2>&1 | Out-File -FilePath $log -Append -Encoding utf8

  "=== SIGN_POLICY | LR=$lr | END $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8
  "" | Out-File -FilePath $log -Append -Encoding utf8
}

"=== END $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8
Write-Host "Done. Log saved to $log"
''').strip() + "\n"


def hardneg_root() -> str:
    return dedent(r'''
$log = "weight_optimization\outputs\all_hardneg_policy_sweep.txt"
$policies = @("sign_policy","lane_policy","coverage_policy")
$ratios = @(0.25,0.5,0.75)

"=== START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Encoding utf8

foreach ($p in $policies) {
  foreach ($r in $ratios) {
    $tag = $r.ToString().Replace('.','')
    $outdir = "weight_optimization/outputs/weights_${p}_r${tag}"

    "=== POLICY=$p | HARD_NEG_RATIO=$r | START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8

    python weight_optimization/optimize/ranking_optimizer.py `
      --data_file weight_optimization/outputs/metrics/training_data.json `
      --baseline_config configs/selector.yaml `
      --output_dir $outdir `
      --epochs 50 `
      --pairs_per_support 32 `
      --after_penalty 1.0 `
      --hard_negative_ratio $r `
      --policy $p 2>&1 | Out-File -FilePath $log -Append -Encoding utf8

    "=== EVAL POLICY=$p | HARD_NEG_RATIO=$r ===" | Out-File -FilePath $log -Append -Encoding utf8

    python weight_optimization/eval/evaluate_weights.py `
      --weights_file "$outdir\optimized_weights_${p}.yaml" `
      --baseline_config configs/selector.yaml `
      --data_file weight_optimization/outputs/metrics/training_data.json `
      --output_file "weight_optimization/outputs/metrics/eval_${p}_r${tag}.json" `
      --after_penalty 1.0 `
      --policy $p 2>&1 | Out-File -FilePath $log -Append -Encoding utf8

    "=== POLICY=$p | HARD_NEG_RATIO=$r | END $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8
    "" | Out-File -FilePath $log -Append -Encoding utf8
  }
}

"=== END $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8
Write-Host "Done. Log saved to $log"
''').strip() + "\n"


def hardneg_inside() -> str:
    return dedent(r'''
$log = "outputs\all_hardneg_policy_sweep.txt"
$policies = @("sign_policy","lane_policy","coverage_policy")
$ratios = @(0.25,0.5,0.75)

"=== START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Encoding utf8

foreach ($p in $policies) {
  foreach ($r in $ratios) {
    $tag = $r.ToString().Replace('.','')
    $outdir = "outputs/weights_${p}_r${tag}"

    "=== POLICY=$p | HARD_NEG_RATIO=$r | START $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8

    python optimize/ranking_optimizer.py `
      --data_file outputs/metrics/training_data.json `
      --baseline_config ..\configs\selector.yaml `
      --output_dir $outdir `
      --epochs 50 `
      --pairs_per_support 32 `
      --after_penalty 1.0 `
      --hard_negative_ratio $r `
      --policy $p 2>&1 | Out-File -FilePath $log -Append -Encoding utf8

    "=== EVAL POLICY=$p | HARD_NEG_RATIO=$r ===" | Out-File -FilePath $log -Append -Encoding utf8

    python eval/evaluate_weights.py `
      --weights_file "$outdir\optimized_weights_${p}.yaml" `
      --baseline_config ..\configs\selector.yaml `
      --data_file outputs/metrics/training_data.json `
      --output_file "outputs/metrics/eval_${p}_r${tag}.json" `
      --after_penalty 1.0 `
      --policy $p 2>&1 | Out-File -FilePath $log -Append -Encoding utf8

    "=== POLICY=$p | HARD_NEG_RATIO=$r | END $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8
    "" | Out-File -FilePath $log -Append -Encoding utf8
  }
}

"=== END $(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') ===" | Out-File -FilePath $log -Append -Encoding utf8
Write-Host "Done. Log saved to $log"
''').strip() + "\n"


MODES = {
    "lr_root": lr_root,
    "lr_inside": lr_inside,
    "hardneg_root": hardneg_root,
    "hardneg_inside": hardneg_inside,
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=sorted(MODES.keys()), required=True)
    args = parser.parse_args()
    print(MODES[args.mode](), end="")


if __name__ == "__main__":
    main()
