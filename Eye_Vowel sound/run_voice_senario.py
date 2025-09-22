# -*- coding: utf-8 -*-
"""
F5 한 번으로 3개 시나리오 순차 실행:
  1) HC vs PD vs MSA (3-class)
  2) HC vs MSA (binary)
  3) HC vs PD  (binary)

출력:
  - 각 시나리오 폴더: ./runs/<scenario_name_lower>/
      ├─ model_fold*.pt, report_fold*.txt, scenario_fold_reports.json
  - 통합 요약: ./runs/scenario_summary.json, ./runs/scenario_summary.csv
  - 콘솔: mean±std 요약 테이블
"""
import os, json, csv, statistics
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from train_multibranch_voice_pd import (
    run_training, set_seed, check_cuda_and_log, CFG, auto_guess_data_root
)

BASE_OUT = "./runs"

SCENARIOS = {
    "HC_PD_MSA": {"include": ["HC","PD","MSA"]},
    "HC_vs_MSA": {"include": ["HC","MSA"]},
    "HC_vs_PD":  {"include": ["HC","PD"]},
}

COMMON = dict(
    epochs=30, batch_size=16, folds=5, lr=3e-4,
    aug_spec=True, aug_seq=True, aug_wave=True,
    cache_dir="./feature_cache", cache_rebuild=False, no_cache=False,
    dropout=0.45, label_smoothing=0.05, weight_decay=2e-4,
    sampler_type='weighted', sched='cosine', warmup_epochs=3,
)

def main():
    # 1) 데이터 루트 자동 탐지 (*_clean*.wav가 있는 폴더)
    data_root = auto_guess_data_root()
    if not data_root:
        for c in ["./DATA/wav", "./DATA"]:
            if os.path.isdir(c):
                data_root = c
                break
    if not data_root:
        raise SystemExit("[ERROR] *_clean*.wav 데이터 경로를 찾지 못했습니다. ./DATA/wav 를 준비하세요.")

    # 2) 디바이스/AMP
    set_seed(CFG.seed)
    device = check_cuda_and_log('cuda')
    use_amp = device.startswith('cuda')

    # 3) 시나리오 순차 실행
    os.makedirs(BASE_OUT, exist_ok=True)
    summary_rows = []

    for name, cfg in SCENARIOS.items():
        out_dir = os.path.join(BASE_OUT, name.lower())
        os.makedirs(out_dir, exist_ok=True)
        print(f"\n=== Scenario: {name} | include={cfg['include']} ===")

        reports = run_training(
            data_root=data_root,
            out_dir=out_dir,
            include=cfg["include"],
            merge=None,             # 병합 안 함(삼진/이진 모두 그대로)
            device=device,
            amp=use_amp,
            **COMMON
        )
        accs = [r['val_acc'] for r in reports]
        f1s  = [r['val_f1']  for r in reports]
        aucs = [r.get('val_auc_macro_ovr', float('nan')) for r in reports]
        auc_mean = (statistics.mean([a for a in aucs if a == a]) if any(a == a for a in aucs) else None)

        summary_rows.append({
            "scenario": name,
            "classes": ",".join(cfg["include"]),
            "acc_mean": round(statistics.mean(accs), 4),
            "acc_std":  round(statistics.pstdev(accs), 4),
            "f1_mean":  round(statistics.mean(f1s), 4),
            "f1_std":   round(statistics.pstdev(f1s), 4),
            "auc_macro_mean": (round(auc_mean, 4) if auc_mean is not None else None),
        })

        with open(os.path.join(out_dir, "scenario_fold_reports.json"), "w", encoding="utf-8") as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)

    # 4) 전체 요약 저장 (JSON + CSV)
    with open(os.path.join(BASE_OUT, "scenario_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary_rows, f, indent=2, ensure_ascii=False)

    csv_path = os.path.join(BASE_OUT, "scenario_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["scenario","classes","acc_mean","acc_std","f1_mean","f1_std","auc_macro_mean"])
        for r in summary_rows:
            writer.writerow([r["scenario"], r["classes"], r["acc_mean"], r["acc_std"], r["f1_mean"], r["f1_std"], r["auc_macro_mean"]])

    # 5) 콘솔 요약 출력
    print("\n=== Summary (mean±std over folds) ===")
    for r in summary_rows:
        auc_part = f", AUC(macro)={r['auc_macro_mean']}" if r['auc_macro_mean'] is not None else ""
        print(f"{r['scenario']:10s} | ACC={r['acc_mean']}±{r['acc_std']} | F1={r['f1_mean']}±{r['f1_std']}{auc_part}")

if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()
    main()
