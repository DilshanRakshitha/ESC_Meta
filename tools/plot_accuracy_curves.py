#!/usr/bin/env python3
"""
Plot training and validation ("testing") accuracy curves from training logs and optional CV bars.

Sources parsed:
- logs_<model>.txt produced by run_batch_models.sh (lines like: "Epoch 10/50: Train=88.12% Val=85.47%")
- <model>.json for per-fold CV accuracies (optional)

Usage examples:
  python tools/plot_accuracy_curves.py --models rapid_kan_lite ickan
  python tools/plot_accuracy_curves.py --models rapid_kan_lite --also-folds
  python tools/plot_accuracy_curves.py --models rapid_kan ickan_deep --output-dir plots
"""
import argparse
import os
import re
import json
from typing import List, Tuple
import matplotlib.pyplot as plt

EPOCH_LINE_RE = re.compile(r"Epoch\s+(\d+)/(\d+):\s*Train=([0-9.]+)%\s*Val=([0-9.]+)%")
FOLD_PREFIX_RE = re.compile(r"Fold\s+(\d+)")


def parse_log_file(path: str, only_fold: int | None = None, auto_first_fold: bool = False) -> Tuple[List[float], List[float]]:
    train, val = [], []
    if not os.path.exists(path):
        return train, val
    with open(path, 'r', errors='ignore') as f:
        current_fold = None
        selected_fold = only_fold  # will be set if auto_first_fold and fold markers encountered
        for line in f:
            # Detect fold header lines if present
            fold_match = FOLD_PREFIX_RE.search(line)
            if fold_match:
                current_fold = int(fold_match.group(1))
                if auto_first_fold and selected_fold is None:
                    selected_fold = current_fold
            m = EPOCH_LINE_RE.search(line)
            if m:
                # Determine effective fold filter
                effective_fold = selected_fold if selected_fold is not None else only_fold
                if effective_fold is not None and current_fold is not None and current_fold != effective_fold:
                    continue
                # epoch_idx = int(m.group(1))  # not used, order is by occurrence
                # total_epochs = int(m.group(2))
                train_acc = float(m.group(3))
                val_acc = float(m.group(4))
                train.append(train_acc)
                val.append(val_acc)
    return train, val


def plot_curves(model: str, train: List[float], val: List[float], out_dir: str) -> str:
    if not train and not val:
        return ""
    os.makedirs(out_dir, exist_ok=True)
    epochs = list(range(1, max(len(train), len(val)) + 1))
    plt.figure(figsize=(8, 5))
    if train:
        plt.plot(range(1, len(train)+1), train, label='Train Acc', color='#1f77b4')
    if val:
        plt.plot(range(1, len(val)+1), val, label='Validation Acc', color='#ff7f0e')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.title(f'ICKAN deep — Accuracy vs Epochs')
    plt.grid(True, alpha=0.3)
    plt.legend()
    out_path = os.path.join(out_dir, f'ICKAN_deep_accuracy_curve.png')
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close()
    return out_path


def plot_cv_bars(model: str, out_dir: str) -> str:
    json_path = f'{model}.json'
    if not os.path.exists(json_path):
        return ""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        accs = data.get('cv', {}).get('individual_accuracies', [])
        if not accs:
            return ""
        os.makedirs(out_dir, exist_ok=True)
        plt.figure(figsize=(6, 4))
        plt.bar([f'Fold {i+1}' for i in range(len(accs))], accs, color='#2ca02c')
        plt.ylabel('Accuracy (%)')
        plt.title(f'{model} — Fold Accuracies')
        for i, v in enumerate(accs):
            plt.text(i, v + 0.5, f'{v:.1f}%', ha='center', va='bottom', fontsize=8)
        out_path = os.path.join(out_dir, f'{model}_fold_accuracies.png')
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
        return out_path
    except Exception:
        return ""


def main():
    parser = argparse.ArgumentParser(description='Plot training/validation accuracy curves from logs and optional CV bars')
    parser.add_argument('--models', nargs='+', required=True, help='Model names (e.g., rapid_kan_lite ickan)')
    parser.add_argument('--logs-pattern', default='logs_{model}.txt', help='Pattern to find logs per model')
    parser.add_argument('--output-dir', default='plots', help='Directory to save plots')
    parser.add_argument('--also-folds', action='store_true', help='Also plot per-fold accuracy bars from <model>.json')
    parser.add_argument('--fold', type=int, default=None, help='Plot only this fold (number). Overrides default first-fold behavior.')
    parser.add_argument('--all-folds', action='store_true', help='Plot epochs from all folds concatenated (legacy behavior)')
    args = parser.parse_args()

    produced = []
    for model in args.models:
        log_path = args.logs_pattern.format(model=model)
        auto_first = (args.fold is None and not args.all_folds)
        train, val = parse_log_file(log_path, only_fold=args.fold, auto_first_fold=auto_first)
        if args.all_folds:
            print(f'[INFO] Using all folds for model {model}')
        elif args.fold is not None:
            print(f'[INFO] Using specified fold {args.fold} for model {model}')
        elif auto_first:
            print(f'[INFO] Auto-selected first fold encountered in log for model {model}')
        curve_path = plot_curves(model, train, val, args.output_dir)
        if curve_path:
            print(f'[OK] Saved: {curve_path}')
            produced.append(curve_path)
        else:
            print(f'[WARN] No epoch curves parsed for {model} (log not found or format mismatch): {log_path}')
        if args.also_folds:
            bar_path = plot_cv_bars(model, args.output_dir)
            if bar_path:
                print(f'[OK] Saved: {bar_path}')
                produced.append(bar_path)
            else:
                print(f'[WARN] No fold accuracies found for {model}.json')

    if not produced:
        print('No plots produced. Ensure logs_<model>.txt exist and contain epoch lines from training.')


if __name__ == '__main__':
    main()
