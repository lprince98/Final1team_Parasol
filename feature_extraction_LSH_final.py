
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
                D_raw.append(-1.0); W_raw.append((-1.0, -1.0)); continue
            w_lm, cmc, th, idx = lm[0], lm[1], lm[4], lm[8]
            wx, wy = int(w_lm.x*w), int(w_lm.y*h)
            tx, ty = int(th.x*w),   int(th.y*h)
            ix, iy = int(idx.x*w),  int(idx.y*h)
            ang = angle_deg(tx-wx, ty-wy, ix-wx, iy-wy)
            D_raw.append(ang)
            cmx, cmy = int(cmc.x*w), int(cmc.y*h)
            wnorm = distance(wx, wy, cmx, cmy) or 0.0
            W_raw.append((wx/wnorm, wy/wnorm) if wnorm else (-1.0, -1.0))
        per_hand[hand] = {"D_raw": np.asarray(D_raw, float), "W_raw": W_raw,
                          "num_frames": len(frames), "duration": float(duration_s)}
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

    hands = Hands(static_image_mode=False, max_num_hands=2, model_complexity=1,
                  min_detection_confidence=0.5, min_tracking_confidence=0.5)

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

    # 3) 저장(랜드마크 & 최종 피처) — 실제 손 기준
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
    df["filename"] = features_fname  # ← filename 컬럼에 최종 파일명 기록

    for c in schema_cols:
        if c not in df.columns: df[c] = np.nan
    df = df[schema_cols]

    out_path = os.path.join(outdir, features_fname)  # ← 파일명 변수 사용
    df.to_csv(out_path, index=False)
    print(f"[SAVED] {out_path}")

if __name__ == "__main__":
    main()
