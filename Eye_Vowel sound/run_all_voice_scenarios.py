# -*- coding: utf-8 -*-
"""
Quick runner: 여러 시나리오를 '빠른 비교' 용도로 학습/요약
 - 정확히 '*_clean.wav'만 사용 (이전의 '*_clean*.wav'로 인한 과집계 방지)
 - 베이스파일명(파일명만) 중복 제거 옵션 제공
 - 캐시 100% 활용 (시나리오별 분리 + Prewarm)
 - 증강 OFF (spec/seq/wave)
 - 입력 프레임 축소(CFG.cnn_frames=128, CFG.rnn_frames=160)
 - Fold=2, Epoch=10 (조절 가능)
 - tqdm 출력 최소화
 - run_training()이 None을 반환해도 out_dir의 txt/json로 복구
 - 이진 분류 시 AUC(bin)도 요약

출력:
  ./runs/<scenario_slug>/
    ├─ model_fold*.pt, report_fold*.txt, summary.json(=fold별 리포트)
  ./runs/scenario_summary.json / .csv
"""

import os, re, json, csv, shutil, statistics, math
from typing import List, Tuple, Optional
from collections import Counter

# BLAS 쓰레드 고정(안정화)
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from train_multibranch_voice_pd import (
    run_training, set_seed, check_cuda_and_log, CFG,
    auto_guess_data_root, list_files_by_name, VoiceDataset
)
from torch.utils.data import DataLoader

import numpy as np
from sklearn.metrics import roc_auc_score

# ====== 경로/상수 ======
BASE_OUT      = "./runs"
BASE_CACHE    = "./feature_cache"
SCENARIO_DATA = "./DATA/_scenarios"

# 정확히 "_clean.wav"만 매칭
FILE_PATTERN = "*_clean.wav"

# 동일 파일명이 여러 폴더에 중복돼 있으면 1개만 사용
DEDUP_BY_BASENAME = True

# ====== Quick 모드 파라미터 ======
QUICK = True
Q_EPOCHS = 10           # 30 -> 10
Q_FOLDS  = 2            # 5  -> 2
Q_BATCH  = 32           # 16 -> 32 (OOM 시 자동 하향)
Q_WARMUP = 1            # 3  -> 1
Q_USE_TQDM = False      # 콘솔 출력 최소화

# 입력 프레임 축소(속도↑/메모리↓, 성능은 보통 소폭 하락)
CFG.cnn_frames = 128    # 256 -> 128
CFG.rnn_frames = 160    # 300 -> 160

# ====== 시나리오 정의 ======
SCENARIOS = {
    # 3-class
    "HC_PD_MSA": {"include": ["HC","PD","MSA"]},

    # 2-class (원래)
    "HC_vs_MSA": {"include": ["HC","MSA"]},
    "HC_vs_PD":  {"include": ["HC","PD"]},
    "HC_vs_PSP": {"include": ["HC","PSP"]},

    # 2-class (그룹핑) : 라벨 리네임으로 APD 묶음
    "HC_vs_APD(MSA, PSP)": {
        "include": ["HC","MSA","PSP"],
        "rename_map": {"MSA":"APD", "PSP":"APD"}
    },
    "PD_vs_APD(MSA, PSP)": {
        "include": ["PD","MSA","PSP"],
        "rename_map": {"MSA":"APD", "PSP":"APD"}
    },
}

COMMON = dict(
    epochs=Q_EPOCHS, batch_size=Q_BATCH, folds=Q_FOLDS, lr=3e-4,
    # 증강 OFF (캐시 100% 활용)
    aug_spec=False, aug_seq=False, aug_wave=False,
    # 캐시: 시나리오별 분리, 재빌드 끔(핵심!)
    cache_dir=None, cache_rebuild=False, no_cache=False,
    dropout=0.45, label_smoothing=0.05, weight_decay=2e-4,
    sampler_type='weighted', sched='cosine', warmup_epochs=Q_WARMUP,
)

# ====== 유틸 ======
def _slug(name: str) -> str:
    s = name.lower().strip()
    s = re.sub(r"\s+", "_", s)
    # 기존: s = re.sub(r"[^a-z0-9_.()\-\]+", "_", s)   # <- \] 때문에 문자클래스가 안 닫힘
    s = re.sub(r"[^a-z0-9_.()\-]+", "_", s)            # (),._- 만 허용
    return s

def _safe_copy(src: str, dst_dir: str) -> str:
    os.makedirs(dst_dir, exist_ok=True)
    base = os.path.basename(src)
    name, ext = os.path.splitext(base)
    dst = os.path.join(dst_dir, base)
    k = 1
    while os.path.exists(dst):
        dst = os.path.join(dst_dir, f"{name}__{k}{ext}")
        k += 1
    shutil.copy2(src, dst)
    return dst

def _dedupe_by_basename(paths: List[str], labs: List[str]) -> Tuple[List[str], List[str]]:
    """동일 베이스파일명 중복을 제거(첫 번째만 유지)."""
    seen = set()
    keep_p, keep_l, dups = [], [], []
    for p, l in zip(paths, labs):
        base = os.path.basename(p).lower()
        if base in seen:
            dups.append(p)
            continue
        seen.add(base)
        keep_p.append(p); keep_l.append(l)
    if dups:
        print(f"[WARN] duplicate basenames ignored: {len(dups)} (showing up to 5)")
        for x in dups[:5]:
            print("   ", x)
    return keep_p, keep_l

def _collect_files(data_root: str, include: Optional[List[str]]=None) -> Tuple[List[str], List[str]]:
    """list_files_by_name를 사용해 *_clean.wav만 수집하고, 필요시 라벨 필터/중복 제거."""
    paths, labs = list_files_by_name(data_root, FILE_PATTERN)
    if DEDUP_BY_BASENAME:
        paths, labs = _dedupe_by_basename(paths, labs)
    if include:
        inc = set(x.upper() for x in include)
        filt = [(p,l) for p,l in zip(paths,labs) if l.upper() in inc]
        paths = [p for p,_ in filt]; labs = [l for _,l in filt]
    return paths, labs

def _apply_rename(labs: List[str], rename_map: Optional[dict]) -> List[str]:
    if not rename_map: return [l.upper() for l in labs]
    rmap = {k.upper(): v.upper() for k, v in rename_map.items()}
    return [rmap.get(l.upper(), l.upper()) for l in labs]

def _build_scenario_dataset(data_root: str, include: List[str], scenario_name: str) -> Tuple[str, int]:
    """시나리오별 원본 파일만 복사해서 작은 데이터 루트를 구성."""
    scen_dir = os.path.join(SCENARIO_DATA, _slug(scenario_name))
    if os.path.isdir(scen_dir):
        shutil.rmtree(scen_dir)
    os.makedirs(scen_dir, exist_ok=True)

    paths, _ = _collect_files(data_root, include=include)
    for p in paths:
        _safe_copy(p, scen_dir)
    return scen_dir, len(paths)

def _print_label_distributions(data_root: str):
    """정확 매칭 + 중복 제거 후 라벨 분포와, 시나리오별 그룹핑 분포를 함께 출력."""
    _, labs_all = _collect_files(data_root, include=None)
    print("[CHECK] all-label distribution (raw):", Counter(labs_all))
    for name, cfg in SCENARIOS.items():
        inc  = cfg.get("include")
        rmap = cfg.get("rename_map")
        _, labs_raw = _collect_files(data_root, include=inc)
        labs_grp = _apply_rename(labs_raw, rmap)
        print(f"[CHECK] {name:20s} raw:{Counter(labs_raw)} => grouped:{Counter(labs_grp)}")

def _load_reports_from_json(out_dir: str) -> Optional[List[dict]]:
    for fn in ("scenario_fold_reports.json","summary.json","cv_reports.json"):
        p = os.path.join(out_dir, fn)
        if os.path.isfile(p):
            try:
                data = json.load(open(p,"r",encoding="utf-8"))
                if isinstance(data, list) and data: return data
            except Exception:
                pass
    return None

def _parse_reports_from_txt(out_dir: str) -> Optional[List[dict]]:
    # (Fallback 필요 시 구현) report_fold*.txt에서 파싱 — 현재는 생략
    return None

def _coerce_reports(reports_obj, out_dir: str) -> Optional[List[dict]]:
    if isinstance(reports_obj, list) and reports_obj:
        return reports_obj
    js = _load_reports_from_json(out_dir)
    if js:
        return js
    txt = _parse_reports_from_txt(out_dir)
    if txt:
        try:
            json.dump(txt, open(os.path.join(out_dir,"scenario_fold_reports.json"),"w",encoding="utf-8"),
                      indent=2, ensure_ascii=False)
        except Exception:
            pass
        return txt
    return None

# ====== Binary AUC 계산 로직 (리포트 보정용, 보통은 트레이너에서 제공) ======
def _try_get(d: dict, keys: List[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    for parent in ("val","valid","metrics","extra","eval","oof","fold"):
        sub = d.get(parent)
        if isinstance(sub, dict):
            for k in keys:
                if k in sub and sub[k] is not None:
                    return sub[k]
    return None

def _sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0/(1.0+np.exp(-x))

def _softmax(z):
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / np.sum(e, axis=1, keepdims=True)

def _compute_binary_auc_from_report(rep: dict) -> Optional[float]:
    y_true = _try_get(rep, ["y_true","val_targets","targets","val_y","labels_true","labels","target"])
    if y_true is None:
        return None
    y_true = np.asarray(y_true).astype(int).ravel()
    uniq = np.unique(y_true)
    if uniq.size != 2:
        return None
    if set(uniq) != {0,1}:
        mapping = {uniq[0]:0, uniq[1]:1}
        y_true = np.vectorize(mapping.get)(y_true)

    score = _try_get(rep, ["y_score","y_scores","val_scores","val_score",
                           "y_prob","y_probs","val_probs","probs","prob"])
    if score is None:
        logits = _try_get(rep, ["y_logits","val_logits","logits"])
        if logits is not None:
            logits = np.asarray(logits)
            if logits.ndim == 1:
                score = _sigmoid(logits)
            elif logits.ndim == 2 and logits.shape[1] >= 2:
                score = _softmax(logits)[:, 1]
    if score is None:
        return None

    score = np.asarray(score)
    if score.ndim == 2 and score.shape[1] >= 2:
        score = score[:, 1]
    score = score.ravel()

    n = min(len(y_true), len(score))
    if n == 0:
        return None
    y_true = y_true[:n]; score = score[:n]

    try:
        return float(roc_auc_score(y_true, score))
    except Exception:
        return None

def prewarm_cache(all_files: List[str], cache_dir: str, sample_rate: int):
    """특징 캐시를 미리 생성(각 파일 1회만 계산)."""
    if not all_files: return
    labels = [0]*len(all_files)
    ds = VoiceDataset(all_files, labels, sample_rate,
                      cache_dir=cache_dir, cache_rebuild=False,
                      use_cache=True, aug_wave=False)
    dl = DataLoader(ds, batch_size=64, shuffle=False,
                    num_workers=min(8, os.cpu_count() or 4),
                    pin_memory=False, persistent_workers=False, prefetch_factor=4)
    for _ in dl:
        pass

def _try_run_training(kwargs, batch_candidates=(64, 48, 32, 24, 16)):
    """CUDA OOM 시 자동으로 배치 크기를 낮춰 재시도."""
    last_err = None
    for bs in batch_candidates:
        run_kwargs = dict(kwargs)
        run_kwargs["batch_size"] = bs
        try:
            print(f"[INFO] run_training with batch_size={bs}")
            return run_training(**run_kwargs)
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda error" in msg:
                print(f"[WARN] OOM with batch={bs} -> retry with smaller batch")
                last_err = e
                continue
            raise
    if last_err:
        raise last_err

# ====== 메인 플로우 ======
def main():
    # 1) 데이터 루트
    data_root = auto_guess_data_root() or next((d for d in ["./DATA/records","./DATA"] if os.path.isdir(d)), None)
    if not data_root:
        raise SystemExit("[ERROR] *_clean.wav 데이터 경로를 찾지 못했습니다. ./DATA/records 또는 ./DATA 를 준비하세요.")
    print(f"[CHECK] data_root = {data_root}")

    # 2) 디바이스/AMP
    set_seed(CFG.seed)
    device = check_cuda_and_log('cuda')
    use_amp = device.startswith('cuda')

    # 3) 라벨 분포 체크 (정확 매칭 + 중복 제거, 리네임 전/후)
    _print_label_distributions(data_root)

    # 4) 실행 베이스 경로
    os.makedirs(BASE_OUT, exist_ok=True)
    os.makedirs(BASE_CACHE, exist_ok=True)
    os.makedirs(SCENARIO_DATA, exist_ok=True)

    summary_rows = []

    for name, cfg in SCENARIOS.items():
        out_dir = os.path.join(BASE_OUT, _slug(name))
        os.makedirs(out_dir, exist_ok=True)

        # 시나리오 데이터 구성(원본 라벨 기준 include로 필터링해서 복사)
        scen_root, n_files = _build_scenario_dataset(data_root, cfg["include"], name)
        if n_files == 0:
            print(f"[ERROR] '{name}' 시나리오 파일 0개")
            summary_rows.append({"scenario":name,"classes":",".join(cfg["include"]),
                                 "acc_mean":None,"acc_std":None,"f1_mean":None,"f1_std":None,
                                 "auc_macro_mean":None,"auc_binary_mean":None})
            continue
        print(f"\n=== Scenario: {name} | include={cfg['include']} ===")
        print(f"[INFO] scenario_data_root={scen_root} (files={n_files})")

        # 캐시 분리 + 사전 Prewarm (핵심)
        scen_cache = os.path.join(BASE_CACHE, _slug(name))
        os.makedirs(scen_cache, exist_ok=True)
        files, _ = list_files_by_name(scen_root, FILE_PATTERN)
        if DEDUP_BY_BASENAME:
            files, _ = _dedupe_by_basename(files, [0]*len(files))
        print(f"[INFO] prewarm cache: {len(files)} files -> {scen_cache}")
        prewarm_cache(files, scen_cache, CFG.sample_rate)

        # 학습 실행(빠른 설정)
        base_kwargs = {
            "data_root": scen_root,
            "out_dir": out_dir,
            "device": device,
            "amp": use_amp,
            **{**COMMON, "cache_dir": scen_cache},
            "use_tqdm": Q_USE_TQDM,
            "pattern": FILE_PATTERN,           # 중요: 훈련도 *_clean.wav만 사용
            "include": cfg.get("include"),
            "rename_map": {k.upper(): v.upper() for k,v in cfg.get("rename_map", {}).items()} if cfg.get("rename_map") else None,
        }

        try:
            raw_reports = _try_run_training(base_kwargs)
        except Exception as e:
            print(f"[WARN] run_training 예외 — out_dir 로그 파싱 시도: {e}")
            raw_reports = None

        # 리포트 수집
        reports = _coerce_reports(raw_reports, out_dir)
        if not reports:
            # 디버깅 로그: 폴더 내용/파일 유무/JSON 파싱
            try:
                print(f"[DEBUG] listdir({out_dir}):", os.listdir(out_dir))
            except Exception as e:
                print(f"[DEBUG] listdir 실패: {e}")
            for cand in ("scenario_fold_reports.json","summary.json","cv_reports.json"):
                p = os.path.join(out_dir, cand)
                print(f"[DEBUG] exists({cand}) = {os.path.isfile(p)}")
                if os.path.isfile(p):
                    try:
                        js = json.load(open(p,"r",encoding="utf-8"))
                        print(f"[DEBUG] {cand} loaded, type={type(js)}, len={len(js) if isinstance(js,list) else 'n/a'}")
                    except Exception as e:
                        print(f"[DEBUG] {cand} JSON 로드 실패: {e}")

            print(f"[WARN] {name}: 리포트 수집 실패 (out_dir={out_dir})")
            summary_rows.append({"scenario":name,"classes":",".join(cfg["include"]),
                                 "acc_mean":None,"acc_std":None,"f1_mean":None,"f1_std":None,
                                 "auc_macro_mean":None,"auc_binary_mean":None})
            continue

        # 시나리오가 실제로 이진인지 (리네임 반영)
        _, labs_raw = _collect_files(data_root, include=cfg.get("include"))
        labs_grp = _apply_rename(labs_raw, cfg.get("rename_map"))
        is_binary = len(set(labs_grp)) == 2

        # 요약 통계
        accs = [r.get('val_acc') for r in reports if r.get('val_acc') is not None]
        f1s  = [r.get('val_f1')  for r in reports if r.get('val_f1')  is not None]
        aucs_macro = [r.get('val_auc_macro_ovr') for r in reports if r.get('val_auc_macro_ovr') is not None]

        aucs_bin = []
        if is_binary:
            for r in reports:
                v = r.get('val_auc_binary') or r.get('val_auc') or None
                if v is None or (isinstance(v, float) and math.isnan(v)):
                    v = _compute_binary_auc_from_report(r)
                if v is not None and not (isinstance(v, float) and math.isnan(v)):
                    aucs_bin.append(float(v))

        def _m(v): return round(statistics.mean(v),4) if v else None
        def _s(v): return round(statistics.pstdev(v),4) if v and len(v)>1 else (0.0 if len(v)==1 else None)

        summary_rows.append({
            "scenario": name,
            "classes": ",".join(cfg["include"]) + (f" | rename:{cfg['rename_map']}" if cfg.get("rename_map") else ""),
            "acc_mean": _m(accs), "acc_std": _s(accs),
            "f1_mean":  _m(f1s),  "f1_std":  _s(f1s),
            "auc_macro_mean": _m(aucs_macro),
            "auc_binary_mean": _m(aucs_bin) if is_binary else None,
        })

        # fold별 리포트 JSON 보존
        try:
            json_path = os.path.join(out_dir, "scenario_fold_reports.json")
            if not os.path.isfile(json_path):
                json.dump(reports, open(json_path,"w",encoding="utf-8"), indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[WARN] {name}: scenario_fold_reports.json 저장 실패: {e}")

    # 5) 요약 저장
    os.makedirs(BASE_OUT, exist_ok=True)
    json.dump(summary_rows, open(os.path.join(BASE_OUT,"scenario_summary.json"),"w",encoding="utf-8"),
              indent=2, ensure_ascii=False)
    with open(os.path.join(BASE_OUT, "scenario_summary.csv"), "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["scenario","classes","acc_mean","acc_std","f1_mean","f1_std","auc_macro_mean","auc_binary_mean"])
        for r in summary_rows:
            w.writerow([r["scenario"], r["classes"],
                        "" if r["acc_mean"] is None else r["acc_mean"],
                        "" if r["acc_std"]  is None else r["acc_std"],
                        "" if r["f1_mean"] is None else r["f1_mean"],
                        "" if r["f1_std"]  is None else r["f1_std"],
                        "" if r["auc_macro_mean"] is None else r["auc_macro_mean"],
                        "" if r["auc_binary_mean"] is None else r["auc_binary_mean"]])

    # 6) 콘솔 요약
    print("\n=== Summary (Quick mode: mean±std over folds) ===")
    for r in summary_rows:
        acc = f"{r['acc_mean']}±{r['acc_std']}" if r['acc_mean'] is not None else "N/A"
        f1  = f"{r['f1_mean']}±{r['f1_std']}"   if r['f1_mean']  is not None else "N/A"
        parts = [f"ACC={acc}", f"F1={f1}"]
        if r.get("auc_binary_mean") is not None:
            parts.append(f"AUC(bin)={r['auc_binary_mean']}")
        if r.get("auc_macro_mean") is not None:
            parts.append(f"AUC(macro)={r['auc_macro_mean']}")
        print(f"{r['scenario']:22s} | " + " | ".join(parts))

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
