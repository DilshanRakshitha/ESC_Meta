import os
import time
import json
import numpy as np
import torch
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader, TensorDataset


def _compute_latency(model: torch.nn.Module, device: torch.device, val_items, batch_size: int = 32,
                     warmup_batches: int = 3, timed_batches: int = 20) -> Dict[str, Any]:
    """Measure batch and per-sample latency plus throughput on real validation items."""
    feats, labels = [], []
    for feat, lab in val_items:
        t = torch.tensor(feat, dtype=torch.float32)
        if t.ndim == 3 and t.shape[-1] in (1, 3) and t.shape[0] not in (1, 3):
            t = t.permute(2, 0, 1)
        feats.append(t)
        labels.append(lab)
    if not feats:
        return {}
    X = torch.stack(feats)
    y = torch.tensor(labels, dtype=torch.long)
    ds = TensorDataset(X, y)
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False, pin_memory=(device.type == 'cuda'))

    model.eval()
    timings = []
    with torch.no_grad():
        it = iter(loader)
        for _ in range(warmup_batches):
            try:
                xb, _ = next(it)
            except StopIteration:
                break
            xb = xb.to(device)
            _ = model(xb)
        it = iter(loader)
        count = 0
        while count < timed_batches:
            try:
                xb, _ = next(it)
            except StopIteration:
                break
            xb = xb.to(device)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(xb)
            if device.type == 'cuda':
                torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            timings.append((dt, xb.size(0)))
            count += 1

    if not timings:
        return {}
    batch_times = [t for t, _ in timings]
    sample_times = [t / n for t, n in timings]

    def _stats(arr):
        return {
            'mean_ms': float(np.mean(arr) * 1000),
            'p50_ms': float(np.percentile(arr, 50) * 1000),
            'p90_ms': float(np.percentile(arr, 90) * 1000),
            'p99_ms': float(np.percentile(arr, 99) * 1000),
            'min_ms': float(np.min(arr) * 1000),
            'max_ms': float(np.max(arr) * 1000),
        }

    return {
        'device': str(device),
        'batches_timed': len(batch_times),
        'batch_latency': _stats(batch_times),
        'per_sample_latency': _stats(sample_times),
        'throughput_samples_per_sec': float((batch_size / np.mean(batch_times))) if batch_times else None,
    }


def export_model_and_generate_reports(model_name: str,
                                      model_factory,
                                      device: torch.device,
                                      cv_results: Dict[str, Any],
                                      folds_data_cache: Dict[int, Any],
                                      best_fold_id: Optional[int],
                                      standardized_model_path: Optional[str] = None,
                                      batch_size: int = 32,
                                      measure_latency: bool = True) -> Dict[str, Any]:
    """Export best checkpoint, measure latency, and write text+JSON reports.

    Returns dict with paths and metrics.
    """
    # Resolve standardized path
    standardized_model_path = standardized_model_path or f"{model_name}.pth"

    # Pick checkpoint path
    best_ckpt_path = None
    if best_fold_id is not None:
        candidate = f"best_model_fold_{best_fold_id}.pth"
        if os.path.exists(candidate):
            best_ckpt_path = candidate
    if best_ckpt_path is None:
        fold_candidates = [fn for fn in os.listdir('.') if fn.startswith('best_model_fold_') and fn.endswith('.pth')]
        if fold_candidates:
            fold_candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            best_ckpt_path = fold_candidates[0]

    model_for_export = model_factory().to(device)
    if best_ckpt_path:
        try:
            state = torch.load(best_ckpt_path, map_location=device)
            state_dict = state.get('model_state_dict', state)
            model_for_export.load_state_dict(state_dict)
        except Exception as e:
            print(f"Warning: failed to load checkpoint '{best_ckpt_path}': {e}")

    torch.save(model_for_export.state_dict(), standardized_model_path)
    param_count = sum(p.numel() for p in model_for_export.parameters())
    file_size_bytes = os.path.getsize(standardized_model_path) if os.path.exists(standardized_model_path) else 0
    file_size_mb = file_size_bytes / (1024 * 1024)

    # Latency fold selection
    if isinstance(folds_data_cache, dict) and best_fold_id in folds_data_cache:
        val_items = folds_data_cache[best_fold_id]
    else:
        val_items = list(folds_data_cache.values())[0]
    latency_stats = _compute_latency(model_for_export, device, val_items, batch_size=batch_size) if measure_latency else {}

    # Reports
    text_path = f"{model_name}.txt"
    json_path = f"{model_name}.json"
    try:
        with open(text_path, 'w') as rf:
            rf.write(f"Model: {model_name}\n")
            rf.write(f"Device: {device}\n")
            rf.write(f"Parameters: {param_count:,}\n")
            rf.write(f"Saved Model Size: {file_size_mb:.2f} MB\n")
            rf.write(f"CV Mean Accuracy: {cv_results['mean_accuracy']:.2f}%\n")
            rf.write(f"CV Std Accuracy: {cv_results['std_accuracy']:.2f}%\n")
            if latency_stats:
                rf.write("\nInference Latency (real inputs)\n")
                rf.write(f"  Batches timed: {latency_stats.get('batches_timed',0)}\n")
                bl = latency_stats.get('batch_latency', {})
                ps = latency_stats.get('per_sample_latency', {})
                rf.write(f"  Batch mean: {bl.get('mean_ms','n/a'):.2f} ms | p50: {bl.get('p50_ms','n/a'):.2f} ms | p90: {bl.get('p90_ms','n/a'):.2f} ms | p99: {bl.get('p99_ms','n/a'):.2f} ms\n")
                rf.write(f"  Per-sample mean: {ps.get('mean_ms','n/a'):.2f} ms | p50: {ps.get('p50_ms','n/a'):.2f} ms | p90: {ps.get('p90_ms','n/a'):.2f} ms | p99: {ps.get('p99_ms','n/a'):.2f} ms\n")
                rf.write(f"  Throughput: {latency_stats.get('throughput_samples_per_sec','n/a'):.2f} samples/s\n")
        print(f"Report saved to {text_path}")
        try:
            json_payload = {
                'model': model_name,
                'device': str(device),
                'parameters': int(param_count),
                'saved_model_path': standardized_model_path,
                'saved_model_size_mb': float(file_size_mb),
                'cv': {
                    'mean_accuracy': float(cv_results['mean_accuracy']),
                    'std_accuracy': float(cv_results['std_accuracy']),
                    'individual_accuracies': [float(x) for x in cv_results.get('individual_accuracies', [])],
                    'best_fold_acc': float(cv_results.get('best_fold_acc', 0.0)),
                    'worst_fold_acc': float(cv_results.get('worst_fold_acc', 0.0)),
                },
                'latency': latency_stats or None,
                'best_checkpoint': best_ckpt_path,
            }
            with open(json_path, 'w') as jf:
                json.dump(json_payload, jf, indent=2)
            print(f"JSON report saved to {json_path}")
        except Exception as je:
            print(f"Warning: failed to write JSON report '{json_path}': {je}")
    except Exception as e:
        print(f"Warning: failed to write text report '{text_path}': {e}")

    return {
        'text_report': text_path,
        'json_report': json_path,
        'model_path': standardized_model_path,
        'latency': latency_stats,
        'parameters': param_count,
        'file_size_mb': file_size_mb,
        'best_checkpoint_source': best_ckpt_path,
    }


def measure_latency_from_checkpoint(model_name: str,
                                    model_factory,
                                    device: torch.device,
                                    folds_data_cache: Dict[int, Any],
                                    best_fold_id: Optional[int] = None,
                                    batch_size: int = 32,
                                    warmup_batches: int = 3,
                                    timed_batches: int = 20) -> Dict[str, Any]:
    """Load <model_name>.pth and append latency metrics to existing reports."""
    model_path = f"{model_name}.pth"
    model = model_factory().to(device)
    if os.path.exists(model_path):
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
    else:
        raise FileNotFoundError(f"Saved model not found: {model_path}")

    # Select validation items
    if isinstance(folds_data_cache, dict):
        if best_fold_id is None and folds_data_cache:
            best_fold_id = next(iter(folds_data_cache.keys()))
        val_items = folds_data_cache[best_fold_id]
    else:
        raise ValueError("folds_data_cache must be a dict of fold_id -> items")

    # Measure latency
    latency = _compute_latency(model, device, val_items, batch_size=batch_size,
                               warmup_batches=warmup_batches, timed_batches=timed_batches)

    # Update reports if they exist
    text_path = f"{model_name}.txt"
    json_path = f"{model_name}.json"
    if os.path.exists(text_path):
        with open(text_path, 'a') as rf:
            rf.write("\n[Post-hoc Latency]\n")
            rf.write(f"  Batches timed: {latency.get('batches_timed',0)}\n")
            bl = latency.get('batch_latency', {})
            ps = latency.get('per_sample_latency', {})
            rf.write(f"  Batch mean: {bl.get('mean_ms','n/a'):.2f} ms | p50: {bl.get('p50_ms','n/a'):.2f} ms | p90: {bl.get('p90_ms','n/a'):.2f} ms | p99: {bl.get('p99_ms','n/a'):.2f} ms\n")
            rf.write(f"  Per-sample mean: {ps.get('mean_ms','n/a'):.2f} ms | p50: {ps.get('p50_ms','n/a'):.2f} ms | p90: {ps.get('p90_ms','n/a'):.2f} ms | p99: {ps.get('p99_ms','n/a'):.2f} ms\n")
            rf.write(f"  Throughput: {latency.get('throughput_samples_per_sec','n/a'):.2f} samples/s\n")
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r') as jf:
                data = json.load(jf)
        except Exception:
            data = {}
        data['latency'] = latency
        with open(json_path, 'w') as jf:
            json.dump(data, jf, indent=2)

    return latency


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Measure latency from saved checkpoint")
    parser.add_argument('--model', required=True, help='Model name (basename of saved .pth/.json/.txt)')
    parser.add_argument('--device', default='cpu', choices=['cpu','cuda'], help='Device to use')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--timed', type=int, default=20)
    parser.add_argument('--fold', type=int, default=None, help='Validation fold to use for latency')
    args = parser.parse_args()

    dev = torch.device('cuda' if (args.device=='cuda' and torch.cuda.is_available()) else 'cpu')
    print(f"Using device: {dev}")
    # Deferred imports to avoid circulars in CLI mode
    from main import create_model_factory, load_fsc22_folds
    fold_info = load_fsc22_folds(args.model)
    model_factory = create_model_factory(args.model, fold_info['num_classes'])
    latency = measure_latency_from_checkpoint(
        model_name=args.model,
        model_factory=model_factory,
        device=dev,
        folds_data_cache=fold_info['folds_data'],
        best_fold_id=args.fold,
        batch_size=args.batch_size,
        warmup_batches=args.warmup,
        timed_batches=args.timed,
    )
    print("Latency:", json.dumps(latency, indent=2))
