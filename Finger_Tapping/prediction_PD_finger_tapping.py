
import os, time, argparse, datetime as dt
import numpy as np
import pandas as pd
import cv2 as cv

# ---------------- feature_extraction 가져오기 + 전역 심볼 핫픽스 ----------------
from feature_extraction import get_final_features, distance
import feature_extraction as fe  # 모듈 객체

# iqr 부재 핫픽스 (NaN-safe)
if not hasattr(fe, "iqr"):
    def iqr(a):
        import numpy as _np
        a = _np.asarray(a, dtype=float)
        if a.size == 0:
            return float("nan")
        q75 = _np.nanpercentile(a, 75)
        q25 = _np.nanpercentile(a, 25)
        return float(q75 - q25)
    fe.iqr = iqr

# scikit-learn 심볼 주입 (LinearRegression, r2_score, PolynomialFeatures)
try:
    from sklearn.linear_model import LinearRegression as _LR
    from sklearn.metrics import r2_score as _r2
    try:
        from sklearn.preprocessing import PolynomialFeatures as _PF
    except Exception:
        _PF = None
except Exception:
    _LR = _r2 = _PF = None

if not hasattr(fe, "LinearRegression"):
    if _LR is None:
        raise SystemExit(
            "scikit-learn이 필요합니다. 활성 venv에서\n"
            "  pip install scikit-learn\n"
            "설치 후 다시 실행해 주세요."
        )
    fe.LinearRegression = _LR
if not hasattr(fe, "r2_score") and _r2 is not None:
    fe.r2_score = _r2
if _PF is not None and not hasattr(fe, "PolynomialFeatures"):
    fe.PolynomialFeatures = _PF
# ---------------------------------------------------------------------------

# MediaPipe
import mediapipe as mp
Hands = mp.solutions.hands.Hands
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS
draw = mp.solutions.drawing_utils
from mediapipe.framework.formats import landmark_pb2

# Pillow(한글)
try:
    from PIL import ImageFont, ImageDraw, Image
except Exception:
    ImageFont = ImageDraw = Image = None

# ----------------- 유틸 -----------------
def ensure_dir(p): os.makedirs(p, exist_ok=True); return p
def now_tag(): return dt.datetime.now().strftime("%Y%m%d_%H%M%S")

def angle_deg(ax, ay, bx, by):
    dot = ax*bx + ay*by
    na = np.hypot(ax, ay); nb = np.hypot(bx, by)
    if na == 0 or nb == 0: return -1.0
    c = np.clip(dot/(na*nb), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))

def write_landmarks_csv(outdir, tag, hand_label, frames):
    rows = []
    for i, lm in enumerate(frames):
        row = {"frame": i, "hand": hand_label.upper()}
        if lm is None:
            for j in range(21):
                row[f"x_{j}"]=np.nan; row[f"y_{j}"]=np.nan; row[f"z_{j}"]=np.nan
        else:
            for j, l in enumerate(lm):
                row[f"x_{j}"]=l.x; row[f"y_{j}"]=l.y; row[f"z_{j}"]=l.z
        rows.append(row)
    pd.DataFrame(rows).to_csv(os.path.join(outdir, f"landmarks_{tag}_{hand_label.upper()}.csv"), index=False)

def collect_timeseries(session_landmarks, frame_size, duration_s):
    w, h = frame_size
    per_hand = {}

    for hand, frames in session_landmarks.items():
        D_raw, W_raw = [], []

        for lm in frames:
            if lm is None:
                D_raw.append(np.nan)  # 👈 NaN 기록
                W_raw.append((np.nan, np.nan))
                continue

            w_lm, cmc, th, idx = lm[0], lm[1], lm[4], lm[8]
            wx, wy = int(w_lm.x * w), int(w_lm.y * h)
            tx, ty = int(th.x * w), int(th.y * h)
            ix, iy = int(idx.x * w), int(idx.y * h)

            ang = angle_deg(tx - wx, ty - wy, ix - wx, iy - wy)
            D_raw.append(ang)

            cmx, cmy = int(cmc.x * w), int(cmc.y * h)
            wnorm = distance(wx, wy, cmx, cmy) or np.nan
            W_raw.append((wx / wnorm, wy / wnorm) if not np.isnan(wnorm) else (np.nan, np.nan))

        # === NaN 보간 처리 ===
        df_tmp = pd.DataFrame({
            "D_raw": D_raw,
            "W_raw_x": [w[0] for w in W_raw],
            "W_raw_y": [w[1] for w in W_raw]
        })
        df_tmp = df_tmp.interpolate(limit_direction="both")  # 앞뒤 값으로 보간

        # 다시 numpy로 변환
        D_raw = df_tmp["D_raw"].to_numpy()
        W_raw = list(zip(df_tmp["W_raw_x"], df_tmp["W_raw_y"]))

        per_hand[hand] = {
            "D_raw": D_raw,
            "W_raw": W_raw,
            "num_frames": len(frames),
            "duration": float(duration_s)
        }

    return per_hand

# -------------- 한글 텍스트 --------------
def _auto_korean_font():
    cands = [
        r"C:\Windows\Fonts\malgun.ttf", r"C:\Windows\Fonts\malgunbd.ttf",
        r"C:\Windows\Fonts\NanumGothic.ttf", r"C:\Windows\Fonts\gulim.ttc",
        "/System/Library/Fonts/AppleSDGothicNeo.ttc",
        "/usr/share/fonts/truetype/nanum/NanumGothic.ttf",
    ]
    for p in cands:
        if os.path.exists(p): return p
    return None

def put_kr(img_bgr, text, org, font_path=None, font_size=26,
           color=(255,255,255), stroke_color=(0,0,0), stroke_width=2):
    if ImageFont is None:
        cv.putText(img_bgr, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
        return img_bgr
    if not font_path or not os.path.exists(font_path):
        font_path = _auto_korean_font()
    try:
        font = ImageFont.truetype(font_path, font_size) if font_path else None
    except Exception:
        font = None
    if font is None:
        cv.putText(img_bgr, text, org, cv.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv.LINE_AA)
        return img_bgr

    def bgr2rgb(c): return (int(c[2]), int(c[1]), int(c[0]))
    img_rgb = cv.cvtColor(img_bgr, cv.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb); draw_pil = ImageDraw.Draw(pil_img)
    if stroke_width>0:
        draw_pil.text(org, text, font=font, fill=bgr2rgb(stroke_color),
                      stroke_width=stroke_width, stroke_fill=bgr2rgb(stroke_color))
    draw_pil.text(org, text, font=font, fill=bgr2rgb(color),
                  stroke_width=stroke_width, stroke_fill=bgr2rgb(stroke_color))
    return cv.cvtColor(np.array(pil_img), cv.COLOR_RGB2BGR)

def draw_panel(img, x, y, w, h, alpha=0.35, color=(0,0,0)):
    overlay = img.copy()
    cv.rectangle(overlay, (x, y), (x+w, y+h), color, thickness=-1)
    return cv.addWeighted(overlay, alpha, img, 1-alpha, 0)

# -------------- 미러 표시용 --------------
def draw_landmarks_mirrored(frame_disp, lm_list, connections):
    mirrored = landmark_pb2.NormalizedLandmarkList(
        landmark=[landmark_pb2.NormalizedLandmark(x=1.0 - l.x, y=l.y, z=l.z) for l in lm_list]
    )
    draw.draw_landmarks(frame_disp, mirrored, connections)

# ----------------- 메인 -----------------
def main():
    ap = argparse.ArgumentParser(description="(Lite) Webcam → MediaPipe → Final features CSV (Mirror+KR)")
    ap.add_argument("--artifact_path", type=str,
                default="./pd_hc_binary/best_pipeline_recall_ExtraTrees.joblib",
                help="저장된 파이프라인(.joblib) 또는 번들(dict) 경로")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="./outputs")
    ap.add_argument("--schema_csv", type=str, default="./severity_dataset_dropped_correlated_columns_modf.csv")
    ap.add_argument("--prep_seconds", type=int, default=5)
    ap.add_argument("--taps", type=int, default=10)
    ap.add_argument("--max_seconds", type=int, default=180)
    ap.add_argument("--tap_on_ratio", type=float, default=0.035)  # 접촉 임계
    ap.add_argument("--tap_off_ratio", type=float, default=0.06)  # 해제 임계(히스테리시스)
    ap.add_argument("--font", type=str, default="")
    ap.add_argument("--swap_labels_for_display", action="store_true", default=True,
                    help="거울 화면에서 라벨/카운트를 사용자 시점으로 스왑해 표시(기본 ON)")
    args = ap.parse_args()

    outdir = ensure_dir(args.outdir)
    schema_cols = list(pd.read_csv(args.schema_csv, nrows=1).columns)
    tag = now_tag()
    features_fname = f"features_{tag}.csv"  # ← 최종 피처 파일명
    font_path = args.font if args.font else _auto_korean_font()

    cap = cv.VideoCapture(args.camera)
    if not cap.isOpened(): raise RuntimeError(f"웹캠을 열 수 없습니다. --camera {args.camera}")

    hands = Hands(static_image_mode=False, max_num_hands=2, 
                  model_complexity=1, # 0 → 빠름(부정확), 1 → 균형, 2 → 더 정확
                  min_detection_confidence=0.6, min_tracking_confidence=0.8)  # 추적 안정성 (여기 올리는 게 효과 큼)

    # 1) 준비(거울 미리보기)
    prep_start = time.time()
    while True:
        ok, frame = cap.read()
        if not ok: cap.release(); cv.destroyAllWindows(); raise RuntimeError("카메라 프레임 수신 실패")
        h, w = frame.shape[:2]
        frame_disp = cv.flip(frame.copy(), 1)

        remain = args.prep_seconds - (time.time() - prep_start)
        if remain < 0: break

        frame_disp = put_kr(frame_disp, "준비 중 (거울 화면)", (10, 40), font_path, 28, (0,255,255))
        frame_disp = put_kr(frame_disp, f"시작까지: {int(np.ceil(remain))}초", (10, 80), font_path, 26, (0,255,255))
        frame_disp = put_kr(frame_disp, f"목표: 양손 각 {args.taps}회", (10, h-40), font_path, 26, (255,255,255))
        cv.imshow("Hand Capture (Mirror)", frame_disp)
        if cv.waitKey(1) & 0xFF == ord('q'):
            cap.release(); cv.destroyAllWindows(); return

    # 2) 측정(양손 모두 목표 도달 시 종료)
    session_landmarks = {"Left": [], "Right": []}   # 실제 손 기준으로 저장
    last_release, count = {"Left": True, "Right": True}, {"Left": 0, "Right": 0}
    min_side, meas_start = None, time.time()

    while True:
        ok, frame = cap.read()
        if not ok: break
        h, w = frame.shape[:2]
        if min_side is None: min_side = min(w, h)
        on_th, off_th = min_side*args.tap_on_ratio, min_side*args.tap_off_ratio

        # 탐지: 원본 프레임(실제 손 기준)
        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        res = hands.process(rgb)

        current = {"Left": None, "Right": None}
        frame_disp = cv.flip(frame.copy(), 1)  # 표시용 거울 프레임

        if res.multi_hand_landmarks and res.multi_handedness:
            for i, hd in enumerate(res.multi_handedness):
                label = hd.classification[0].label  # 'Left'|'Right' (실제 손)
                score = hd.classification[0].score
                if score < 0.5: continue
                lm = res.multi_hand_landmarks[i].landmark
                current[label] = lm

                # (a) 거울 프레임에 랜드마크 그리기
                draw_landmarks_mirrored(frame_disp, lm, HAND_CONNECTIONS)

                # (b) 라벨 표기: 거울 화면 기준 스왑 표기(사용자 시점)
                if args.swap_labels_for_display:
                    disp_label = "오른손" if label == "Left" else "왼손"
                else:
                    disp_label = "왼손" if label == "Left" else "오른손"
                posx = 10 if (label == 'Left') else w - 210
                score_pct = int(round(score * 100))
                frame_disp = put_kr(frame_disp, f"{disp_label}({score_pct}%)", (posx, 28), font_path, 24, (0,255,0))

                # (c) 탭 감지: 원본 좌표계로 계산(실제 손 기준 카운트)
                tx, ty = int(lm[4].x*w), int(lm[4].y*h)
                ix, iy = int(lm[8].x*w), int(lm[8].y*h)
                dpx = float(np.hypot(tx-ix, ty-iy))
                touching = dpx < on_th
                if last_release[label] and touching:
                    count[label] += 1
                    last_release[label] = False
                if dpx > off_th:
                    last_release[label] = True

                # (d) 노란 점/선: 거울 프레임 좌표로 변환 후 그리기
                mtx, mix = (w - 1 - tx), (w - 1 - ix)
                cv.circle(frame_disp, (mtx, ty), 4, (0,255,255), -1)
                cv.circle(frame_disp, (mix, iy), 4, (0,255,255), -1)
                cv.line(frame_disp, (mtx, ty), (mix, iy), (0,255,255), 2)

        session_landmarks["Left"].append(current["Left"])
        session_landmarks["Right"].append(current["Right"])

        # 하단 패널(가려지지 않게 반투명 배경)
        panel_h = 78
        frame_disp = draw_panel(frame_disp, 6, h - panel_h - 6, w - 12, panel_h, alpha=0.30, color=(0,0,0))

        # 실제 카운트(종료 판단용)
        elapsed = time.time() - meas_start
        lc_true = min(count["Left"],  args.taps)
        rc_true = min(count["Right"], args.taps)

        # 표시용 카운트(거울 화면 시점으로 스왑)
        if args.swap_labels_for_display:
            lc_disp, rc_disp = rc_true, lc_true
        else:
            lc_disp, rc_disp = lc_true, rc_true

        # 텍스트 표시
        frame_disp = put_kr(frame_disp, f"시간: {elapsed:4.1f}초", (12, h - panel_h - 6 + 26),
                            font_path, 26, (255,255,255))
        frame_disp = put_kr(frame_disp, f"왼손 {lc_disp}/{args.taps}  |  오른손 {rc_disp}/{args.taps}",
                            (12, h - panel_h - 6 + 56), font_path, 26, (255,255,255))
        cv.imshow("Hand Capture (Mirror)", frame_disp)

        # 종료: 실제 양손 모두 목표 도달
        if (lc_true >= args.taps) and (rc_true >= args.taps): break
        if elapsed >= args.max_seconds: break
        if cv.waitKey(1) & 0xFF == ord('q'): break

    cap.release(); cv.destroyAllWindows()

    frame_size = (int(w), int(h)); total_duration = time.time() - meas_start
    write_landmarks_csv(outdir, tag, "LEFT",  session_landmarks["Left"])
    write_landmarks_csv(outdir, tag, "RIGHT", session_landmarks["Right"])

    data_by_hand = collect_timeseries(session_landmarks, frame_size, total_duration)
    rows = []
    for label in ["Left", "Right"]:
        try:
            feats = get_final_features(data_by_hand[label]); feats["hand"] = label.lower()
        except Exception as e:
            print(f"[WARN] {label} 피처 계산 실패: {e}"); feats = {"hand": label.lower()}
        rows.append(feats)

    df = pd.DataFrame(rows)
    df["filename"] = features_fname
    for c in schema_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[schema_cols]

    out_path = os.path.join(outdir, features_fname)
    df.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")

  

    ########################
    # === 모델 예측 추가 ===
    ########################
    import traceback
    from joblib import load
    from sklearn.pipeline import Pipeline
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder

    pipe_path = args.artifact_path
    print(f"[INFO] Loading artifact: {pipe_path}")
    bundle = load(pipe_path)
    print(f"[INFO] Loaded object type: {type(bundle)}")

    # --------- 유틸 ---------
    def find_column_transformer(pipe: Pipeline):
        if isinstance(pipe, Pipeline):
            for name, step in pipe.steps:
                if isinstance(step, ColumnTransformer):
                    return step
        return None

    def align_raw_to_expected(df_raw, expected_cols, num_cols, cat_cols):
        # 누락 컬럼 채우기 (수치=0.0, 범주='unknown')
        for c in expected_cols:
            if c not in df_raw.columns:
                if c in num_cols:
                    df_raw[c] = 0.0
                else:
                    df_raw[c] = "unknown"
        df_infer = df_raw[expected_cols].copy()

        # NaN/dtype 보정
        for c in num_cols:
            if c in df_infer.columns:
                df_infer[c] = pd.to_numeric(df_infer[c], errors="coerce").fillna(0.0)
        for c in cat_cols:
            if c in df_infer.columns:
                df_infer[c] = df_infer[c].astype("object").fillna("unknown")
        return df_infer

    def build_prefixed_frame_from_raw(df_raw, expected_prefixed_cols):
        """전처리 없는 모델만 저장된 경우: num__/cat__ 스키마를 원본에서 생성"""
        out = pd.DataFrame(index=df_raw.index)
        df_norm = df_raw.copy()
        if "hand" in df_norm.columns:
            df_norm["hand"] = df_norm["hand"].astype(str).str.lower()

        for col in expected_prefixed_cols:
            if col.startswith("num__"):
                base = col.split("num__", 1)[1]
                out[col] = pd.to_numeric(df_norm.get(base, 0.0), errors="coerce").fillna(0.0)
            elif col.startswith("cat__"):
                base = col.split("cat__", 1)[1]
                if "_" in base:
                    feat, level = base.split("_", 1)
                    val = df_norm.get(feat, "unknown")
                    val = val.astype(str).fillna("unknown")
                    out[col] = (val.str.lower() == level.lower()).astype(int)
                else:
                    out[col] = (df_norm.get(base, pd.Series(["unknown"]*len(df_norm))).astype(str) != "unknown").astype(int)
            else:
                out[col] = 0.0

        out = out.reindex(columns=expected_prefixed_cols, fill_value=0.0)
        return out

    # --------- 번들 해체: Pipeline / Dict / Model 대응 ---------
    try:
        pipeline = None
        model = None
        pre = None
        selected_prefixed = None  # num__/cat__ 형태
        raw_features = None       # 원본 컬럼 리스트

        if isinstance(bundle, Pipeline):
            pipeline = bundle
        elif isinstance(bundle, dict):
            # 가장 우선: 완성 파이프라인이 담겨있다면 그걸 사용
            if "pipeline" in bundle and isinstance(bundle["pipeline"], Pipeline):
                pipeline = bundle["pipeline"]
            # 아니면 model과 preprocessor로 파이프라인 재구성
            if pipeline is None and ("model" in bundle):
                model = bundle["model"]
                pre = bundle.get("preprocessor", None)
                if pre is not None:
                    pipeline = Pipeline([("preprocessor", pre), ("classifier", model)])
            # 피처 이름 힌트
            selected_prefixed = bundle.get("selected_features") or bundle.get("final_selected_features")
            raw_features = bundle.get("raw_features")
        else:
            # Estimator 단독
            model = bundle

        # --- 입력 원본 로드 ---
        df_raw_all = pd.read_csv(out_path)  # 방금 저장한 features_*.csv
        # hand/gender 등 범주 표준화
        if "hand" in df_raw_all.columns:
            df_raw_all["hand"] = df_raw_all["hand"].astype(str).str.lower()

        # ---------- 분기 1: 파이프라인이 있는 경우 ----------
        if pipeline is not None:
            print("[INFO] Using Pipeline for inference.")
            pre_ct = find_column_transformer(pipeline)
            if pre_ct is None:
                # 드물게 전처리 없는 파이프라인이면 원본 전체를 그대로 사용
                df_infer = df_raw_all.copy()
            else:
                expected_cols = list(pre_ct.feature_names_in_)
                # 수치/범주 컬럼 구분
                num_cols, cat_cols = [], []
                for name, trans, cols in pre_ct.transformers_:
                    if name == "num":
                        num_cols = list(cols)
                    elif name == "cat":
                        cat_cols = list(cols)

                # raw_features 힌트가 있으면 그 집합을 우선 맞추고, 나머지 expected를 보정
                if raw_features is not None:
                    for c in raw_features:
                        if c not in df_raw_all.columns:
                            df_raw_all[c] = np.nan
                df_infer = align_raw_to_expected(df_raw_all, expected_cols, num_cols, cat_cols)

            # 예측
            y_pred = pipeline.predict(df_infer)
            y_prob = (
                pipeline.predict_proba(df_infer)[:, 1]
                if hasattr(pipeline, "predict_proba") else
                np.full(len(df_infer), np.nan)
            )

        # ---------- 분기 2: 모델만 있고, selected_features(접두사) 제공 ----------
        elif (model is not None) and (selected_prefixed is not None):
            print("[INFO] Using bare model + prefixed features (selected_features) for inference.")
            df_pref = build_prefixed_frame_from_raw(df_raw_all, selected_prefixed)
            y_pred = model.predict(df_pref)
            y_prob = (
                model.predict_proba(df_pref)[:, 1]
                if hasattr(model, "predict_proba") else
                np.full(len(df_pref), np.nan)
            )

        # ---------- 분기 3: 모델만 있고, feature_names_in_ 제공 ----------
        elif (model is not None) and hasattr(model, "feature_names_in_"):
            print("[INFO] Using bare model + feature_names_in_ for inference.")
            expected_prefixed = list(model.feature_names_in_)
            df_pref = build_prefixed_frame_from_raw(df_raw_all, expected_prefixed)
            y_pred = model.predict(df_pref)
            y_prob = (
                model.predict_proba(df_pref)[:, 1]
                if hasattr(model, "predict_proba") else
                np.full(len(df_pref), np.nan)
            )

        else:
            raise RuntimeError(
                "아티팩트를 해석할 수 없습니다. (pipeline/model/selected_features/feature_names_in_ 중 하나가 필요)"
            )

        # --- 결과 저장/출력 ---
        df_out = df_raw_all.copy()
        df_out["prediction"] = y_pred
        df_out["probability"] = y_prob
        for _, row in df_out.iterrows():
            label = "환자(PD)" if row["prediction"] == 1 else "정상(HC)"
            print(f"[RESULT] {row.get('hand','?')}: {label} (위험도={row['probability']:.2f})")

        pred_path = out_path.replace("features_", "predictions_")
        df_out.to_csv(pred_path, index=False)
        print(f"[INFO] Saved predictions to {pred_path}")

    except Exception as e:
        print("[ERROR] Inference failed.")
        traceback.print_exc()
        print("=== DEBUG HINTS ===")
        print(f"- Artifact path           : {pipe_path}")
        print(f"- Is Pipeline?            : {isinstance(bundle, Pipeline)}")
        if isinstance(bundle, dict):
            print(f"- Dict keys               : {list(bundle.keys())}")


if __name__ == "__main__":
    main()




