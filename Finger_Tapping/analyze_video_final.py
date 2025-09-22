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

def noisyor(probs: np.ndarray) -> float:
    """Noisy-OR 결합 확률 계산"""
    p = np.clip(np.asarray(probs, dtype=float), 0.0, 1.0)
    return float(1.0 - np.prod(1.0 - p))

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
                D_raw.append(np.nan)
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

        df_tmp = pd.DataFrame({
            "D_raw": D_raw,
            "W_raw_x": [w[0] for w in W_raw],
            "W_raw_y": [w[1] for w in W_raw]
        })
        df_tmp = df_tmp.interpolate(limit_direction="both")

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
    ap = argparse.ArgumentParser(description="(Lite) Webcam → Record → Analyze Video → Final features CSV (Mirror+KR)")

    # ==== 인자 정의 ====
    ap.add_argument("--artifact_path", type=str,
                    default="./pd_hc_binary/best_pipeline_auc_AdaBoost.joblib",
                    help="저장된 파이프라인(.joblib) 또는 번들(dict) 경로")
    ap.add_argument("--camera", type=int, default=0)
    ap.add_argument("--outdir", type=str, default="./outputs")
    ap.add_argument("--schema_csv", type=str, default="./severity_dataset_dropped_correlated_columns_modf.csv")
    ap.add_argument("--prep_seconds", type=int, default=5)
    ap.add_argument("--taps", type=int, default=10)
    ap.add_argument("--max_seconds", type=int, default=180)
    ap.add_argument("--tap_on_ratio", type=float, default=0.035)
    ap.add_argument("--tap_off_ratio", type=float, default=0.06)
    ap.add_argument("--font", type=str, default="")
    ap.add_argument("--swap_labels_for_display", action="store_true", default=True)
    ap.add_argument("--thresh", type=float, default=0.50, help="기본 PD 판정 임계값")
    ap.add_argument("--delta", type=float, default=0.05, help="옐로존 폭")
    ap.add_argument("--video_path", type=str, default="", help="분석할 영상 파일 경로")
    ap.add_argument("--strategy", type=str, choices=["recall", "youden", "accuracy", "default"],
                    default="default", help="threshold 기준 선택")
    args = ap.parse_args()

    # ==== 출력 폴더 및 파일명 ====
    outdir = ensure_dir(args.outdir)
    schema_cols = list(pd.read_csv(args.schema_csv, nrows=1).columns)
    tag = now_tag()
    features_fname = f"features_{tag}.csv"
    font_path = args.font if args.font else _auto_korean_font()

    # ==== 1. 녹화 or 기존 비디오 사용 ====
    recorded_video_path = args.video_path
    if not recorded_video_path:
        recorded_video_path = os.path.join(outdir, f"recording_{tag}.mp4")

        cap = cv.VideoCapture(args.camera)
        if not cap.isOpened():
            raise RuntimeError(f"웹캠을 열 수 없습니다. --camera {args.camera}")

        w = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv.CAP_PROP_FPS)) or 20
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        video_writer = cv.VideoWriter(recorded_video_path, fourcc, fps, (w, h))
        print(f"[INFO] 녹화를 시작합니다: {recorded_video_path}")

        hands_for_preview = Hands(static_image_mode=False, max_num_hands=2,
                                  model_complexity=1, min_detection_confidence=0.6,
                                  min_tracking_confidence=0.8)

        prep_start = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("카메라 프레임 수신 실패")
            video_writer.write(frame)
            remain = args.prep_seconds - (time.time() - prep_start)
            if remain < 0:
                break
            frame_disp = cv.flip(frame.copy(), 1)
            frame_disp = put_kr(frame_disp, "준비 중 (거울 화면)", (10, 40), font_path, 28, (0,255,255))
            frame_disp = put_kr(frame_disp, f"시작까지: {int(np.ceil(remain))}초", (10, 80), font_path, 26, (0,255,255))
            cv.imshow("Hand Capture (Mirror) - Recording...", frame_disp)
            if cv.waitKey(1) & 0xFF == ord('q'):
                cap.release(); video_writer.release(); cv.destroyAllWindows(); return

        last_release, count = {"Left": True, "Right": True}, {"Left": 0, "Right": 0}
        min_side, meas_start = None, time.time()

        # [FIX 1] 측정 루프에 화면 표시(imshow) 및 시각화 로직 추가
        while True:
            ok, frame = cap.read()
            if not ok: break
            video_writer.write(frame)
            h, w = frame.shape[:2]
            if min_side is None: min_side = min(w, h)
            on_th, off_th = min_side*args.tap_on_ratio, min_side*args.tap_off_ratio

            frame_disp = cv.flip(frame.copy(), 1)
            rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            res = hands_for_preview.process(rgb)

            if res.multi_hand_landmarks and res.multi_handedness:
                for i, hd in enumerate(res.multi_handedness):
                    label = hd.classification[0].label
                    score = hd.classification[0].score
                    if score < 0.5: continue
                    lm = res.multi_hand_landmarks[i].landmark
                    
                    draw_landmarks_mirrored(frame_disp, lm, HAND_CONNECTIONS)

                    tx, ty = int(lm[4].x*w), int(lm[4].y*h)
                    ix, iy = int(lm[8].x*w), int(lm[8].y*h)
                    dpx = float(np.hypot(tx-ix, ty-iy))
                    touching = dpx < on_th
                    if last_release[label] and touching:
                        count[label] += 1
                        last_release[label] = False
                    if dpx > off_th:
                        last_release[label] = True

            panel_h = 78
            frame_disp = draw_panel(frame_disp, 6, h - panel_h - 6, w - 12, panel_h, alpha=0.30, color=(0,0,0))
            elapsed = time.time() - meas_start
            lc_true = min(count["Left"],  args.taps)
            rc_true = min(count["Right"], args.taps)
            
            if args.swap_labels_for_display: lc_disp, rc_disp = rc_true, lc_true
            else: lc_disp, rc_disp = lc_true, rc_true

            frame_disp = put_kr(frame_disp, f"시간: {elapsed:4.1f}초", (12, h - panel_h - 6 + 26), font_path, 26, (255,255,255))
            frame_disp = put_kr(frame_disp, f"왼손 {lc_disp}/{args.taps}  |  오른손 {rc_disp}/{args.taps}", (12, h - panel_h - 6 + 56), font_path, 26, (255,255,255))
            
            cv.imshow("Hand Capture (Mirror) - Recording...", frame_disp)

            if (count["Left"] >= args.taps) and (count["Right"] >= args.taps): break
            if elapsed >= args.max_seconds: break
            if cv.waitKey(1) & 0xFF == ord('q'): break

        cap.release(); video_writer.release(); cv.destroyAllWindows()
        hands_for_preview.close()
        print("[INFO] 녹화가 완료되었습니다.")
        total_duration = time.time() - meas_start
    else:
        cap_tmp = cv.VideoCapture(recorded_video_path)
        fps = cap_tmp.get(cv.CAP_PROP_FPS)
        num_frames = cap_tmp.get(cv.CAP_PROP_FRAME_COUNT)
        total_duration = num_frames / fps if fps > 0 else 0
        cap_tmp.release()

    # [FIX 2] 랜드마크 추출 및 특징 계산 로직 복원
    print(f"[INFO] 녹화된 영상 분석을 시작합니다: {recorded_video_path}")
    cap_analysis = cv.VideoCapture(recorded_video_path)
    if not cap_analysis.isOpened():
        raise RuntimeError(f"녹화된 비디오 파일을 열 수 없습니다: {recorded_video_path}")

    analysis_w = int(cap_analysis.get(cv.CAP_PROP_FRAME_WIDTH))
    analysis_h = int(cap_analysis.get(cv.CAP_PROP_FRAME_HEIGHT))
    num_video_frames = int(cap_analysis.get(cv.CAP_PROP_FRAME_COUNT))
    frame_size = (analysis_w, analysis_h)
    
    hands_for_analysis = Hands(static_image_mode=True, max_num_hands=2,
                               model_complexity=1, min_detection_confidence=0.5)

    session_landmarks = {"Left": [], "Right": []}
    frame_count = 0
    while True:
        ok, frame = cap_analysis.read()
        if not ok: break
        
        frame_count += 1
        print(f"\r[INFO] 영상 분석 중... {frame_count}/{num_video_frames}", end="")

        rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        res = hands_for_analysis.process(rgb)
        
        current = {"Left": None, "Right": None}
        if res.multi_hand_landmarks and res.multi_handedness:
            for i, hd in enumerate(res.multi_handedness):
                label = hd.classification[0].label
                current[label] = res.multi_hand_landmarks[i].landmark
        
        session_landmarks["Left"].append(current["Left"])
        session_landmarks["Right"].append(current["Right"])
        
    cap_analysis.release()
    hands_for_analysis.close()
    print("\n[INFO] 영상 분석 완료.")

    data_by_hand = collect_timeseries(session_landmarks, frame_size, total_duration)
    rows = []
    for label in ["Left", "Right"]:
        try:
            feats = get_final_features(data_by_hand[label])
            feats["hand"] = label.lower()
        except Exception as e:
            print(f"[WARN] {label} 피처 계산 실패: {e}")
            feats = {"hand": label.lower()}
        rows.append(feats)

    df_raw_all = pd.DataFrame(rows)
    df_raw_all["filename"] = features_fname
    for c in schema_cols:
        if c not in df_raw_all.columns:
            df_raw_all[c] = np.nan
    df_infer = df_raw_all[schema_cols].copy()
    
    out_path = os.path.join(outdir, features_fname)
    df_infer.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")


    # ==== 3. 모델 불러오기 및 예측 ====
    from joblib import load
    from sklearn.pipeline import Pipeline
    
    bundle = load(args.artifact_path)
    
    pipeline = None
    if isinstance(bundle, Pipeline):
        pipeline = bundle
    elif isinstance(bundle, dict) and "pipeline" in bundle:
        pipeline = bundle["pipeline"]
    else:
        raise RuntimeError("아티팩트에서 파이프라인을 찾을 수 없습니다.")

    # ==== 4. threshold 전략 선택 ====
    person_thresh = None
    strategy_used = "manual"

    if isinstance(bundle, dict):
        default_strategy = bundle.get("threshold_strategy", "default")
        default_thresh = bundle.get("person_threshold", 0.5)

        if args.strategy == "default":
            person_thresh = default_thresh
            strategy_used = default_strategy
        else:
            metrics_all = bundle.get("metrics_all", {})
            if args.strategy in metrics_all:
                person_thresh = float(metrics_all[args.strategy].get("threshold", default_thresh))
                strategy_used = args.strategy
            else:
                person_thresh = default_thresh
                strategy_used = default_strategy
    else:
        person_thresh = 0.5

# ==== 사용자 코드 내에서 직접 조정값 입력 ====
    THRESH_OFFSET = 0.12   # 👉 여기서 보정값을 직접 입력 (+0.05면 threshold 0.05 올림, -0.02면 0.02 내림)
    person_thresh = person_thresh + THRESH_OFFSET

    print(f"[INFO] threshold_strategy='{strategy_used}', "
          f"artifact_base={default_thresh:.4f}, offset={THRESH_OFFSET:+.4f}, "
          f"final_person_threshold={person_thresh:.4f}")


    # ==== pipeline 예측 ====
    y_prob = pipeline.predict_proba(df_infer)[:, 1]
    df_out = df_raw_all.copy()
    df_out["probability"] = y_prob
    # 손 단위 예측은 최종 임계값을 따름
    df_out["prediction"] = (df_out["probability"] >= person_thresh).astype(int)

    # ==== 손 단위 결과 (Zone 기반) ====
    zones = {}
    for _, row in df_out.iterrows():
        prob = row['probability']
        hand = row['hand']

        if prob < (person_thresh - args.delta):
            zone = "green"
            label = "정상(HC)"
        elif prob < (person_thresh + args.delta):
            zone = "yellow"
            label = "주의/재검 권유"
        else:
            zone = "red"
            label = "환자(PD)"

        zones[hand] = zone
        print(f"[RESULT] {hand}: {label} (위험도={prob:.2f}, TH={person_thresh:.2f}, Zone={zone.upper()})")

    pred_path = out_path.replace("features_", "predictions_")
    df_out.to_csv(pred_path, index=False)
    print(f"[INFO] Saved predictions to {pred_path}")

    # ==== 사람 단위 최종 판정 (손 단위 Zone 종합) ====
    if all(z == "green" for z in zones.values()):
        final_pred = 0
        zone = "green"
        diagnosis = "정상(HC)"
    elif any(z == "red" for z in zones.values()):
        final_pred = 1
        zone = "red"
        diagnosis = "환자(PD)"
    else:
        final_pred = -1
        zone = "yellow"
        diagnosis = "주의/재검 권유"

    print("-" * 50)
    print(f"[PERSON-LEVEL PREDICTION]")
    print(f"  - 최종 진단: {diagnosis} (Zone: {zone.upper()})")
    print(f"  - 왼손 Zone: {zones.get('left')}, 오른손 Zone: {zones.get('right')}")
    print(f"  - 적용 전략: '{strategy_used}' (임계값: {person_thresh:.4f}, Δ={args.delta})")
    print("-" * 50)

    per_person = pd.DataFrame([{
        "filename": features_fname,
        "final_pred": final_pred,   # -1=주의, 0=정상, 1=PD
        "left_zone": zones.get("left"),
        "right_zone": zones.get("right"),
        "strategy": strategy_used,
        "threshold_used": person_thresh
    }])
    person_path = pred_path.replace("predictions_", "person_level_decision_")
    per_person.to_csv(person_path, index=False)
    print(f"[INFO] Saved person-level decision to {person_path}")


if __name__ == "__main__":
    main()