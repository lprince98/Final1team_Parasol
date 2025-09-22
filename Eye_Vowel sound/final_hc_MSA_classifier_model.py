# -*- coding: utf-8 -*-
"""
Pipeline runner that:
  1) Reads a JSON config
  2) Makes a subject-wise train/test split (fixed list or stratified ratio)
  3) Builds a temporary train-only view of the dataset (hardlink/copy)
  4) Imports and calls `run_training` from train_multibranch_voice_pd.py
  5) Evaluates Test ACC/F1/AUC using the saved fold checkpoints (probability ensemble)
  6) Exports an averaged ensemble checkpoint and optional TorchScript

Usage examples
--------------
# subject-wise random 20% test split
python pipeline_train_test_export.py \
  --config configs/hc_vs_msa_strong.json \
  --test_ratio 0.2 --device cuda --export_torchscript

# fixed subject list for test (one subject ID per line: e.g., HC12a1, MSA07b3)
python pipeline_train_test_export.py \
  --config configs/hc_vs_msa_strong.json \
  --test_subjects_file configs/test_subjects.txt --device cuda

Notes
-----
- This script **does not modify** your original training script. It imports `run_training` and
  points it to a temporary directory that contains only the TRAIN/VAL subset to avoid leakage.
- Subject ID is derived as the filename stem before the first underscore (e.g., "HC10a1_clean.wav" -> subject "HC10a1").
  Adjust `extract_subject_id` if your naming differs.
- Requires scikit-learn for stratified subject split.
"""

from __future__ import annotations
import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import re

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report, confusion_matrix

import torch
import torch.nn as nn

# --- import training module ---
# Assumes train_multibranch_voice_pd.py is in the same folder or PYTHONPATH
try:
    import train_multibranch_voice_pd as trainmod
except Exception as e:
    print(f"[ERROR] cannot import train_multibranch_voice_pd: {e}\nMake sure this script runs from the directory containing train_multibranch_voice_pd.py")
    raise


# ---------- helpers ----------
LABEL_ALLOWED = ("HC", "PD", "MSA", "PSP", "APD")


def extract_label_from_name(path: Path) -> Optional[str]:
    name_u = path.name.upper()
    m = re.match(r"([A-Z]+)", name_u)
    if not m:
        return None
    return m.group(1)


def extract_subject_id(path: Path) -> str:
    # e.g., HC10a1_clean.wav -> HC10a1
    stem = path.stem
    if "_" in stem:
        return stem.split("_")[0].upper()
    return stem.upper()


def scan_files(data_root: str, pattern: str, include: Optional[List[str]] = None,
               rename_map: Optional[Dict[str, str]] = None) -> Tuple[List[str], List[str]]:
    include_set = set(s.upper() for s in include) if include else None
    rmap = {k.upper(): v.upper() for k, v in (rename_map or {}).items()}
    files = sorted(Path(data_root).rglob(pattern))
    paths, labels = [], []
    for p in files:
        lab0 = extract_label_from_name(p)
        if not lab0:
            continue
        if include_set is not None and lab0 not in include_set:
            continue
        lab = rmap.get(lab0, lab0)
        if lab not in LABEL_ALLOWED:
            continue
        paths.append(str(p))
        labels.append(lab)
    if not paths:
        raise SystemExit(f"[ERROR] No files matched under '{data_root}' with pattern '{pattern}'.")
    return paths, labels


def subject_stratified_split(files: List[str], labels: List[str], test_ratio: float,
                             fixed_subjects: Optional[List[str]] = None,
                             seed: int = 42) -> Tuple[List[str], List[str]]:
    subjects = np.array([extract_subject_id(Path(fp)) for fp in files])
    labels_arr = np.array(labels)
    if fixed_subjects:
        fixed = set(s.upper() for s in fixed_subjects)
        mask_test = np.array([s in fixed for s in subjects])
        te_idx = np.where(mask_test)[0]
        tr_idx = np.where(~mask_test)[0]
    elif test_ratio and test_ratio > 0:
        # pick unique subjects and stratify on their label
        subj_unique = np.array(sorted(set(subjects.tolist())))
        # label per subject = majority label among its files
        subj2lab: Dict[str, str] = {}
        for s in subj_unique:
            labs = labels_arr[subjects == s]
            vals, counts = np.unique(labs, return_counts=True)
            subj2lab[s] = vals[np.argmax(counts)]
        subj_labels = np.array([subj2lab[s] for s in subj_unique])
        sss = StratifiedShuffleSplit(n_splits=1, test_size=test_ratio, random_state=seed)
        idx = np.arange(len(subj_unique))
        tr_s, te_s = next(sss.split(idx, subj_labels))
        test_subj = set(subj_unique[te_s])
        mask_test = np.array([s in test_subj for s in subjects])
        te_idx = np.where(mask_test)[0]
        tr_idx = np.where(~mask_test)[0]
    else:
        # no split requested -> all train
        te_idx = np.array([], dtype=int)
        tr_idx = np.arange(len(files))
    tr_files = [files[i] for i in tr_idx]
    te_files = [files[i] for i in te_idx]
    return tr_files, te_files


def make_train_view(train_files: List[str], view_dir: Path) -> None:
    view_dir.mkdir(parents=True, exist_ok=True)
    # flatten into the view dir; cache still keyed by audio hash so it's fine
    for src in train_files:
        src_p = Path(src)
        dst = view_dir / src_p.name
        if dst.exists():
            continue
        try:
            os.link(src_p, dst)  # hardlink (same volume)
        except Exception:
            shutil.copy2(src_p, dst)


def average_fold_state_dicts(ckpt_paths: List[Path]) -> Dict[str, torch.Tensor]:
    sd_avg = None
    n = 0
    for p in ckpt_paths:
        ck = torch.load(p, map_location='cpu')
        sd = ck['model']
        if sd_avg is None:
            sd_avg = {k: v.clone().float() for k, v in sd.items()}
        else:
            for k, v in sd.items():
                sd_avg[k] += v.float()
        n += 1
    if sd_avg is None or n == 0:
        raise RuntimeError("No checkpoints to average")
    for k in sd_avg:
        sd_avg[k] /= float(n)
    return sd_avg


def evaluate_test(out_dir: Path, data_root: str, pattern: str, include: Optional[List[str]],
                  device: str = 'cpu', batch: int = 16) -> Tuple[float, float, float]:
    # Load class mapping from a checkpoint to keep order consistent
    ck0 = None
    for k in range(1, 100):  # search model_fold1..99
        p = out_dir / f"model_fold{k}.pt"
        if p.is_file():
            ck0 = torch.load(p, map_location='cpu')
            break
    if ck0 is None:
        raise SystemExit(f"[ERROR] no fold ckpt found under {out_dir}")

    label2idx = ck0['label2idx']
    classes = [kv[0] for kv in sorted(label2idx.items(), key=lambda kv: kv[1])]

    files, labels = scan_files(data_root, pattern, include=include)
    y = np.array([label2idx[lab] for lab in labels], dtype=np.int64)

    # build dataset for these files
    ds = trainmod.VoiceDataset(files, y.tolist(), trainmod.CFG.sample_rate,
                               cache_dir=str(out_dir / 'feature_cache_test'),
                               cache_rebuild=False, use_cache=True, aug_wave=False)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=False,
                                     num_workers=min(12, os.cpu_count() or 4),
                                     pin_memory=device.startswith('cuda'),
                                     persistent_workers=True, prefetch_factor=4)

    # ensemble probabilities across available folds
    probs_ens = None
    y_true_ref = None
    used = 0
    crit = nn.CrossEntropyLoss()
    for k in range(1, 100):
        p = out_dir / f"model_fold{k}.pt"
        if not p.is_file():
            continue
        ck = torch.load(p, map_location=device)
        model = trainmod.MultiBranchNet(tab_dim=ck['tab_dim'], num_classes=len(classes)).to(device)
        model.load_state_dict(ck['model'])
        model.eval()
        _, _, _, y_true, _, y_prob = trainmod.evaluate(model, dl, crit, device, use_tqdm=False)
        if probs_ens is None:
            probs_ens = y_prob
            y_true_ref = y_true
        else:
            assert (y_true_ref == y_true).all(), "Test order mismatch"
            probs_ens += y_prob
        used += 1
    if used == 0:
        raise SystemExit("[ERROR] no fold checkpoints found")

    probs_ens /= used
    y_pred = probs_ens.argmax(1)
    acc = accuracy_score(y_true_ref, y_pred)
    f1 = f1_score(y_true_ref, y_pred, average='macro')
    try:
        if len(classes) == 2 and len(np.unique(y_true_ref)) == 2:
            auc = roc_auc_score(y_true_ref, probs_ens[:, 1])
        else:
            auc = roc_auc_score(y_true_ref, probs_ens, multi_class='ovr', average='macro')
    except Exception:
        auc = float('nan')

    rpt = classification_report(y_true_ref, y_pred, target_names=classes, digits=4, zero_division=0)
    cm = confusion_matrix(y_true_ref, y_pred)
    with open(out_dir / 'test_report.txt', 'w', encoding='utf-8') as f:
        f.write(rpt + "\n")
        f.write(str(cm) + "\n")
        f.write(f"ACC={acc:.6f} F1_macro={f1:.6f} AUC_macro={auc:.6f}\n")
        f.write(f"files_test={len(files)}\n")
    np.save(out_dir / 'test_y_true.npy', y_true_ref)
    np.save(out_dir / 'test_y_pred.npy', y_pred)
    np.save(out_dir / 'test_y_prob.npy', probs_ens)
    print(f"[TEST] ACC {acc:.3f} | F1 {f1:.3f} | AUC {auc:.3f} | files={len(files)} (folds used={used})")
    return acc, f1, auc


def export_ensemble(out_dir: Path, export_torchscript: bool = False, device: str = 'cpu') -> Path:
    # collect fold ckpts
    ckpts = sorted([p for p in out_dir.glob('model_fold*.pt') if p.is_file()])
    if not ckpts:
        raise SystemExit(f"[ERROR] no model_fold*.pt in {out_dir}")
    ck0 = torch.load(ckpts[0], map_location='cpu')
    label2idx = ck0['label2idx']
    tab_dim = ck0['tab_dim']

    sd_avg = average_fold_state_dicts(ckpts)

    # build model and save averaged state dict
    model = trainmod.MultiBranchNet(tab_dim=tab_dim, num_classes=len(label2idx)).to('cpu')
    model.load_state_dict(sd_avg)
    bundle = {
        'model': model.state_dict(),
        'label2idx': label2idx,
        'cfg': trainmod.CFG.__dict__,
        'tab_dim': tab_dim,
        'folds_averaged': len(ckpts),
    }
    out_ckpt = out_dir / 'model_ensemble_avg.pt'
    torch.save(bundle, out_ckpt)
    print(f"[EXPORT] Averaged ensemble saved -> {out_ckpt}")

    if export_torchscript:
        model.eval()
        # dummy inputs for tracing
        n_mels = trainmod.CFG.n_mels
        T_cnn = trainmod.CFG.cnn_frames
        T_rnn = trainmod.CFG.rnn_frames
        mfcc_dim = trainmod.CFG.n_mfcc * 3
        ex_mel = torch.zeros(1, 1, n_mels, T_cnn)
        ex_mfcc = torch.zeros(1, T_rnn, mfcc_dim)
        ex_tab = torch.zeros(1, tab_dim)
        ts = torch.jit.trace(model, (ex_mel, ex_mfcc, ex_tab))
        out_ts = out_dir / 'model_ensemble_avg.ts'
        ts.save(str(out_ts))
        print(f"[EXPORT] TorchScript saved -> {out_ts}")
        return out_ts
    return out_ckpt


# ---------- main ----------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', type=str, required=True, help='training JSON config (same schema as original)')
    ap.add_argument('--test_ratio', type=float, default=0.2, help='subject-wise test ratio; ignored if --test_subjects_file given')
    ap.add_argument('--test_subjects_file', type=str, default=None, help='file with one subject id per line')
    ap.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--export_torchscript', action='store_true')
    ap.add_argument('--skip_train', action='store_true', help='only evaluate/export using existing checkpoints')
    args = ap.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        cfg = json.load(f)

    data_root = cfg.get('data_root', './DATA')
    out_dir = Path(cfg.get('out', './runs/mbvpd'))
    pattern = cfg.get('pattern', '*_clean*.wav')
    include = cfg.get('include', None)
    rename_map = cfg.get('rename_map', None)

    # 1) scan & split subjects
    files_all, labels_all = scan_files(data_root, pattern, include=include, rename_map=rename_map)

    fixed_subjects = None
    if args.test_subjects_file and os.path.isfile(args.test_subjects_file):
        with open(args.test_subjects_file, 'r', encoding='utf-8') as f:
            fixed_subjects = [ln.strip() for ln in f if ln.strip()]
        print(f"[SPLIT] using fixed subject list ({len(fixed_subjects)} entries) for TEST")

    train_files, test_files = subject_stratified_split(files_all, labels_all, test_ratio=args.test_ratio,
                                                       fixed_subjects=fixed_subjects, seed=42)

    print(f"[SPLIT] train/val files = {len(train_files)} | test files = {len(test_files)}")
    if len(test_files) == 0:
        print('[WARN] test set is empty. Set --test_ratio or --test_subjects_file for holdout evaluation.')

    # 2) build train-only view
    split_root = out_dir / 'split_train_view'
    if split_root.exists():
        shutil.rmtree(split_root)
    make_train_view(train_files, split_root)

    # 3) run training on train-only view
    if not args.skip_train:
        cfg_train = dict(cfg)  # shallow copy
        cfg_train['data_root'] = str(split_root)
        # isolate caches/output for this run
        cfg_train['out'] = str(out_dir)
        if cfg.get('cache_dir'):
            cfg_train['cache_dir'] = str(out_dir / 'feature_cache_train')

        # coerce CLI-like kwargs for run_training
        kw = {
            'data_root': cfg_train['data_root'],
            'out_dir': cfg_train['out'],
            'epochs': int(cfg_train.get('epochs', 30)),
            'batch_size': int(cfg_train.get('batch', 16)),
            'folds': int(cfg_train.get('folds', 5)),
            'lr': float(cfg_train.get('lr', 3e-4)),
            'device': args.device,
            'pattern': cfg_train.get('pattern', '*_clean*.wav'),
            'use_tqdm': True,
            'amp': bool(cfg_train.get('amp', False)),
            # cache
            'cache_dir': cfg_train.get('cache_dir', './feature_cache'),
            'cache_rebuild': bool(cfg_train.get('cache_rebuild', False)),
            'no_cache': bool(cfg_train.get('no_cache', False)),
            # aug
            'aug_spec': bool(cfg_train.get('aug_spec', False)),
            'spec_p': float(cfg_train.get('spec_p', 0.7)),
            'spec_freq_masks': int(cfg_train.get('spec_freq_masks', 2)),
            'spec_time_masks': int(cfg_train.get('spec_time_masks', 2)),
            'spec_F': int(cfg_train.get('spec_F', 12)),
            'spec_T_pct': float(cfg_train.get('spec_T_pct', 0.10)),
            'aug_seq': bool(cfg_train.get('aug_seq', False)),
            'seq_p': float(cfg_train.get('seq_p', 0.7)),
            'seq_time_masks': int(cfg_train.get('seq_time_masks', 1)),
            'seq_T_pct': float(cfg_train.get('seq_T_pct', 0.05)),
            'aug_wave': bool(cfg_train.get('aug_wave', False)),
            'wave_p': float(cfg_train.get('wave_p', 0.3)),
            'wave_gain_db': float(cfg_train.get('wave_gain_db', 3.0)),
            'wave_pitch_semitones': float(cfg_train.get('wave_pitch_semitones', 0.5)),
            # optim
            'dropout': float(cfg_train.get('dropout', 0.45)),
            'label_smoothing': float(cfg_train.get('label_smoothing', 0.05)),
            'weight_decay': float(cfg_train.get('weight_decay', 2e-4)),
            'clip_grad': None,
            # sampler & sched
            'sampler_type': cfg_train.get('sampler', 'weighted'),
            'sched': cfg_train.get('sched', 'cosine'),
            'warmup_epochs': int(cfg_train.get('warmup_epochs', 3)),
            # filter/grouping
            'include': cfg_train.get('include', None),
            'rename_map': cfg_train.get('rename_map', None),
        }
        print("[TRAIN] kwargs ->", {k: v for k, v in kw.items() if k not in ('data_root','out_dir')})
        trainmod.set_seed(trainmod.CFG.seed)
        # ensure device checked
        _ = trainmod.check_cuda_and_log(args.device)
        trainmod.run_training(**kw)
    else:
        print('[TRAIN] skipped by --skip_train (using existing checkpoints)')

    # 4) evaluate test on original files (not in train view)
    if len(test_files) > 0:
        # Write a tiny manifest dir with only test files so we can use the evaluator uniformly
        test_view = out_dir / 'split_test_view'
        if test_view.exists():
            shutil.rmtree(test_view)
        make_train_view(test_files, test_view)
        evaluate_test(out_dir, str(test_view), pattern, include=include, device=args.device, batch=int(cfg.get('batch', 16)))
    else:
        print('[TEST] skipped (empty test set)')

    # 5) export ensemble (and optionally TorchScript)
    export_ensemble(out_dir, export_torchscript=args.export_torchscript, device=args.device)


if __name__ == '__main__':
    main()
