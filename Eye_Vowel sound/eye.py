# -*- coding: utf-8 -*-
"""
Eye Tracking (Vertical) — UI & API
- MediaPipe FaceMesh로 좌/우 홍채 및 눈꺼풀 지표 추출
- 수직 응시 정규값: v_offset_norm = (iy - cy) / eye_height  (정면≈0, 위=+, 아래=-)
- UI 모드: '위/아래' 음성 큐, 왕복 5회 트라이얼, CSV 저장
- API 모드:
    • /api/analyze/eye-tracking (단일 이미지)
    • /api/analyze/video        (비디오 타임시리즈 + 요약 통계, vpp 포함)

Windows 11 / Python 3.10.9 기준
"""

import os, math, time, random, argparse, tempfile
from collections import deque
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import cv2 as cv
import mediapipe as mp

# UI용
import pyttsx3
import sounddevice as sd
import wave
from pathlib import Path

# API용
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import uvicorn


# ==============================
# 공통 설정
# ==============================
FPS_TARGET   = 30
BUF_SEC      = 8
CALIB_SECS   = 3.0
N_ROUNDS     = 5                 # 왕복 5회(UP→DOWN)
TRIAL_LAT_MAX= 0.700             # s
TRIAL_WIN    = 1.2               # s
PAUSE_BETWEEN= 0.9               # s
CUE_GAP_SEC  = 0.35              # s

# 사케이드 임계(상대 단위)
VEL_TH       = 0.9               # 상대 속도(/s)
AMP_TH       = 0.12              # 상대 진폭
PITCH_OK_DEG = 7.0               # |pitch| 초과 시 머리 고정 유도

# MediaPipe FaceMesh
mp_face = mp.solutions.face_mesh

# 홍채/눈꺼풀 인덱스 (refine_landmarks=True, 478포인트)
LEFT_IRIS  = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_TOP, LEFT_BOT, LEFT_L, LEFT_R     = 159, 145, 33, 133
RIGHT_TOP, RIGHT_BOT, RIGHT_L, RIGHT_R = 386, 374, 362, 263

# solvePnP용 3D 모델 포인트(mm) & 해당 2D 인덱스
FACE_3D_MODEL_POINTS = np.array([
    [0.0,   0.0,    0.0],    # nose tip
    [0.0, -330.0, -65.0],    # chin
    [-225.0, 170.0, -135.0], # left eye corner
    [225.0, 170.0, -135.0],  # right eye corner
    [-150.0, -150.0, -125.0],# left mouth
    [150.0, -150.0, -125.0], # right mouth
], dtype=np.float64)
FACE_2D_INDEXES = [1, 152, 33, 263, 61, 291]


# ==============================
# 유틸/코어 로직
# ==============================
def now_s() -> float:
    return time.perf_counter()

def moving_average(arr: np.ndarray, w: int = 3) -> np.ndarray:
    if len(arr) < w:
        return arr.copy()
    csum = np.cumsum(np.insert(arr, 0, 0))
    out = (csum[w:] - csum[:-w]) / float(w)
    head = np.repeat(out[0], w-1)
    return np.concatenate([head, out])

def _dist(p0, p1):
    return math.hypot(p0[0]-p1[0], p0[1]-p1[1])

def iris_center_from_landmarks(lms, idxs):
    xs = [lms[i].x for i in idxs]; ys = [lms[i].y for i in idxs]
    return (float(np.mean(xs)), float(np.mean(ys)))

def compute_vertical_gaze_ratio(lms) -> float:
    """
    수직 응시 비율(정규화, 정면≈0 / 위=양수 / 아래=음수)
    v_offset_norm = (iris_y - mid_y) / eye_height  (y축↓가 +라서 부호 반전)
    """
    Ltop, Lbot = lms[LEFT_TOP], lms[LEFT_BOT]
    Rtop, Rbot = lms[RIGHT_TOP], lms[RIGHT_BOT]
    Lspan = max(1e-6, _dist((Ltop.x, Ltop.y), (Lbot.x, Lbot.y)))
    Rspan = max(1e-6, _dist((Rtop.x, Rtop.y), (Rbot.x, Rbot.y)))
    Liris = iris_center_from_landmarks(lms, LEFT_IRIS)
    Riris = iris_center_from_landmarks(lms, RIGHT_IRIS)
    Lmid_y = (Ltop.y + Lbot.y) / 2.0
    Rmid_y = (Rtop.y + Rbot.y) / 2.0
    Lratio = - (Liris[1] - Lmid_y) / Lspan
    Rratio = - (Riris[1] - Rmid_y) / Rspan
    return (Lratio + Rratio) / 2.0

def eyelid_opening_pixels_and_points(lms, w, h):
    l_up = (int(lms[LEFT_TOP].x*w),  int(lms[LEFT_TOP].y*h))
    l_lo = (int(lms[LEFT_BOT].x*w),  int(lms[LEFT_BOT].y*h))
    r_up = (int(lms[RIGHT_TOP].x*w), int(lms[RIGHT_TOP].y*h))
    r_lo = (int(lms[RIGHT_BOT].x*w), int(lms[RIGHT_BOT].y*h))
    return _dist(l_up, l_lo), _dist(r_up, r_lo), l_up, l_lo, r_up, r_lo

def solve_head_pose(lms, w, h):
    pts_2d = np.array([(lms[i].x*w, lms[i].y*h) for i in FACE_2D_INDEXES], dtype=np.float64)
    focal = w
    cam_mtx = np.array([[focal, 0, w/2],[0, focal, h/2],[0,0,1]], dtype=np.float64)
    dist = np.zeros((4,1), dtype=np.float64)
    ok, rvec, tvec = cv.solvePnP(FACE_3D_MODEL_POINTS, pts_2d, cam_mtx, dist, flags=cv.SOLVEPNP_ITERATIVE)
    pitch = yaw = roll = 0.0
    if ok:
        rmat, _ = cv.Rodrigues(rvec)
        angles, *_ = cv.RQDecomp3x3(rmat)
        pitch, yaw, roll = float(angles[0]), float(angles[1]), float(angles[2])
    z_cm = float(tvec[2][0]/10.0) if ok else 0.0
    return ok, pitch, yaw, roll, z_cm


# ==============================
# 사케이드/버퍼 클래스 (UI)
# ==============================
@dataclass
class TrialResult:
    trial: int
    target: str            # "UP" | "DOWN"
    latency_ms: int        # -1: 실패
    peakVel_rel_per_s: float
    amplitude_rel: float
    success: bool
    tefr: float
    head_pitch_deg: float
    cue_time_hms: str

class GazeAnalyzer:
    def __init__(self):
        self.face = mp_face.FaceMesh(
            static_image_mode=False, max_num_faces=1, refine_landmarks=True,
            min_detection_confidence=0.5, min_tracking_confidence=0.5
        )
        self.buf_len = int(BUF_SEC * FPS_TARGET)
        self.gaze_buf = deque(maxlen=self.buf_len)
        self.time_buf = deque(maxlen=self.buf_len)
        self.frame_count = 0
        self.fail_count = 0
        self.neutral = 0.0
        self.calibrated = False
        self.last_pitch = 0.0
        self.trial_no = 0

    def tefr(self):
        return self.fail_count / max(1, self.frame_count)

    def current_velocity(self) -> float:
        if len(self.gaze_buf) < 3: return 0.0
        y = np.array(self.gaze_buf, dtype=np.float32)
        t = np.array(self.time_buf, dtype=np.float64)
        y_s = moving_average(y, 3)
        dy = y_s[-1] - y_s[-3]
        dt = t[-1] - t[-3]
        return float(dy/dt) if dt > 1e-6 else 0.0

    def process_frame(self, frame_bgr):
        self.frame_count += 1
        h, w = frame_bgr.shape[:2]
        rgb = cv.cvtColor(frame_bgr, cv.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        res = self.face.process(rgb)
        rgb.flags.writeable = True

        out = {"tracked": False, "gazeY": None, "velY": 0.0,
               "l_open_px": None, "r_open_px": None,
               "l_up_pt": None, "l_lo_pt": None, "r_up_pt": None, "r_lo_pt": None,
               "pitch": None, "yaw": None, "roll": None, "z_cm": None,
               "tefr": self.tefr(), "landmarks_for_draw": []}

        if not res.multi_face_landmarks:
            self.fail_count += 1
            return out

        lms = res.multi_face_landmarks[0].landmark
        gy_raw = compute_vertical_gaze_ratio(lms)
        gy = gy_raw - self.neutral if self.calibrated else gy_raw
        self.gaze_buf.append(gy); self.time_buf.append(now_s())

        vel = self.current_velocity()
        l_open, r_open, l_up, l_lo, r_up, r_lo = eyelid_opening_pixels_and_points(lms, w, h)
        hp_ok, pitch, yaw, roll, z_cm = solve_head_pose(lms, w, h)
        if hp_ok: self.last_pitch = pitch

        draw_pts = []
        for idx in LEFT_IRIS + RIGHT_IRIS + [LEFT_TOP, LEFT_BOT, RIGHT_TOP, RIGHT_BOT, LEFT_L, LEFT_R, RIGHT_L, RIGHT_R]:
            x, y = int(lms[idx].x*w), int(lms[idx].y*h); draw_pts.append((x, y))

        out.update(dict(tracked=True, gazeY=gy, velY=vel,
                        l_open_px=l_open, r_open_px=r_open,
                        l_up_pt=l_up, l_lo_pt=l_lo, r_up_pt=r_up, r_lo_pt=r_lo,
                        pitch=pitch if hp_ok else None, yaw=yaw if hp_ok else None,
                        roll=roll if hp_ok else None, z_cm=z_cm if hp_ok else None,
                        tefr=self.tefr(), landmarks_for_draw=draw_pts))
        return out

    def calibrate(self, seconds=CALIB_SECS):
        t_end = now_s() + seconds; samples = []
        while now_s() < t_end:
            time.sleep(0.033)
            if len(self.gaze_buf) > 0: samples.append(self.gaze_buf[-1])
        if len(samples) >= int(seconds*10):
            self.neutral = float(np.mean(samples)); self.calibrated = True
            return True, self.neutral
        return False, None

    def measure_saccade(self, direction: int, t0_perf: float, t0_hms: str) -> TrialResult:
        """direction: +1(UP)/-1(DOWN)"""
        self.trial_no += 1
        base = self.gaze_buf[-1] if len(self.gaze_buf) else 0.0
        t_end = t0_perf + TRIAL_WIN

        detected = False; t_start = None
        peak_vel = 0.0;   amp = 0.0

        while now_s() < t_end:
            time.sleep(0.01)
            if len(self.gaze_buf) < 2: continue
            gy = self.gaze_buf[-1]; vel = self.current_velocity()
            peak_vel = max(peak_vel, abs(vel))
            amp_candidate = abs(gy - base); amp = max(amp, amp_candidate)

            head_ok = abs(self.last_pitch) <= PITCH_OK_DEG
            if (not detected) and head_ok and ((direction*vel) > VEL_TH) and (amp_candidate > AMP_TH) and (now_s() >= t0_perf):
                detected = True; t_start = now_s()
            if detected and now_s() - t0_perf > 0.45: break

        lat_ms = -1; success = False
        if detected and (t_start is not None):
            lat_ms = int(round((t_start - t0_perf)*1000))
            success = (0 <= (t_start - t0_perf) <= TRIAL_LAT_MAX)

        return TrialResult(
            trial=self.trial_no, target="UP" if direction>0 else "DOWN",
            latency_ms=lat_ms, peakVel_rel_per_s=float(round(peak_vel,4)),
            amplitude_rel=float(round(amp,4)), success=bool(success),
            tefr=float(round(self.tefr(),4)), head_pitch_deg=float(round(self.last_pitch,2)),
            cue_time_hms=t0_hms
        )


# ==============================
# WAV 캐시(TTS) 유틸
# ==============================
def _read_wav_to_numpy(path: Path):
    with wave.open(str(path), 'rb') as wf:
        sr = wf.getframerate(); nchan = wf.getnchannels()
        sampwidth = wf.getsampwidth(); nframes = wf.getnframes()
        raw = wf.readframes(nframes)
    data = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if nchan == 2: data = data.reshape(-1,2).mean(axis=1)
    return data, sr

def build_cue_cache(tts_engine, cache_dir="cues"):
    Path(cache_dir).mkdir(parents=True, exist_ok=True)
    cache = {}
    for word, fname in [("위","up.wav"),("아래","down.wav")]:
        fpath = Path(cache_dir)/fname
        if not fpath.exists():
            tts_engine.save_to_file(word, str(fpath)); tts_engine.runAndWait()
        arr, sr = _read_wav_to_numpy(fpath); cache[word]=(arr, sr)
    return cache


# ==============================
# UI 모드 (웹캠 실험)
# ==============================
def run_ui(cam_index=0, log_frames=False, out_dir="results"):
    ga = GazeAnalyzer()

    tts = pyttsx3.init()
    try:
        for v in tts.getProperty('voices'):
            name=(v.name or "").lower(); lang=(getattr(v,"languages",[""])[0] or "").lower()
            if "ko" in name or "korean" in name or "ko" in lang:
                tts.setProperty('voice', v.id); break
    except Exception: pass
    cue_cache = build_cue_cache(tts, cache_dir="cues")

    def play_cue_and_mark(word: str):
        arr, sr = cue_cache[word]
        t0_perf = now_s(); t0_hms = datetime.now().strftime("%H:%M:%S")
        sd.play(arr, sr, blocking=False)
        return t0_perf, t0_hms

    os.makedirs(out_dir, exist_ok=True)
    frame_log = []

    WIN = "PSP Vertical Saccade — UI (C=Calib, T=Trials, Q=Quit)"
    cap = cv.VideoCapture(cam_index, cv.CAP_DSHOW if os.name=='nt' else 0)
    if not cap.isOpened(): raise RuntimeError("웹캠을 열 수 없습니다.")
    cap.set(cv.CAP_PROP_FRAME_WIDTH, 1280); cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720); cap.set(cv.CAP_PROP_FPS, FPS_TARGET)

    cue_text = "대기 (C=캘리브레이션, T=테스트, Q=종료)"
    results = []
    print("창 활성화 후: C=캘리브레이션, T=트라이얼 시작, Q=종료")

    while True:
        ok, frame = cap.read()
        if not ok: break
        frame = cv.flip(frame, 1)

        out = ga.process_frame(frame)

        # 프레임 로깅(옵션)
        if log_frames:
            time_hms = datetime.now().strftime("%H:%M:%S")
            frame_log.append({
                "time_hms": time_hms,
                "gazeY_norm": float(out["gazeY"]) if out["tracked"] else np.nan,
                "velY_rel_per_s": float(out["velY"]) if out["tracked"] else np.nan,
                "left_lid_px": float(out["l_open_px"]) if out["tracked"] else np.nan,
                "right_lid_px": float(out["r_open_px"]) if out["tracked"] else np.nan,
                "head_pitch_deg": float(out["pitch"]) if out["pitch"] is not None else np.nan,
                "head_yaw_deg": float(out["yaw"]) if out["yaw"] is not None else np.nan,
                "head_roll_deg": float(out["roll"]) if out["roll"] is not None else np.nan,
                "z_cm": float(out["z_cm"]) if out["z_cm"] is not None else np.nan,
                "tefr": float(out["tefr"]),
                "tracked": bool(out["tracked"])
            })

        # 그리기
        if out["tracked"]:
            for (x,y) in out["landmarks_for_draw"]:
                cv.circle(frame, (x,y), 2, (255,200,40), -1, cv.LINE_AA)
            if out["l_up_pt"] and out["l_lo_pt"]:
                cv.line(frame, out["l_up_pt"], out["l_lo_pt"], (0,255,0), 1, cv.LINE_AA)
            if out["r_up_pt"] and out["r_lo_pt"]:
                cv.line(frame, out["r_up_pt"], out["r_lo_pt"], (0,255,0), 1, cv.LINE_AA)

            cv.putText(frame, f"Tracking: OK  TEFR={out['tefr']*100:4.1f}%", (12,26),
                       cv.FONT_HERSHEY_SIMPLEX, 0.65, (50,220,120), 2)
            cv.putText(frame, f"GazeY(norm): {out['gazeY']:+.3f}   VelY(rel/s): {out['velY']:+.3f}", (12,50),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (240,240,240), 2)
            cv.putText(frame, f"Head Pitch:{(out['pitch'] if out['pitch'] is not None else 0):+0.1f}°",
                       (12,74), cv.FONT_HERSHEY_SIMPLEX, 0.7, (40,180,255), 2)
        else:
            cv.putText(frame, f"Tracking: FAIL  TEFR={ga.tefr()*100:4.1f}%", (12,26),
                       cv.FONT_HERSHEY_SIMPLEX, 0.65, (30,60,255), 2)

        cv.putText(frame, cue_text, (12,102), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255,210,40), 2)
        cv.imshow(WIN, frame)
        key = cv.waitKey(1) & 0xFF

        if key in (ord('q'), ord('Q')): break

        if key in (ord('c'), ord('C')):
            cue_text = "캘리브레이션(정면 3초)"
            ok_calib, neutral = ga.calibrate(CALIB_SECS)
            cue_text = "캘리 완료" if ok_calib else "캘리 실패(다시 시도)"
            if ok_calib: cue_text += f" (neutral={neutral:+.3f})"

        if key in (ord('t'), ord('T')):
            if not ga.calibrated:
                cue_text = "먼저 C로 캘리브레이션"; continue

            # 왕복 5회 시퀀스
            seq = []; 
            for _ in range(N_ROUNDS): seq.extend(["UP","DOWN"])  # 총 10회

            # 트라이얼
            for trg in seq:
                cue_text = "준비..."
                t_ready = now_s() + 0.25
                while now_s() < t_ready:
                    if cv.waitKey(1) & 0xFF in (ord('q'), ord('Q')): break
                    time.sleep(0.003)

                cue_kor = "위" if trg == "UP" else "아래"
                t0_perf = now_s(); t0_hms = datetime.now().strftime("%H:%M:%S")
                arr, sr = build_cue_cache(pyttsx3.init())["위" if trg=="UP" else "아래"]
                sd.play(arr, sr, blocking=False)

                direction = +1 if trg == "UP" else -1
                rec = ga.measure_saccade(direction, t0_perf, t0_hms)
                results.append(rec)

                cue_text = f"#{rec.trial} {rec.target} @{rec.cue_time_hms} lat={rec.latency_ms}ms v={rec.peakVel_rel_per_s:.2f} amp={rec.amplitude_rel:.2f} {'OK' if rec.success else 'NG'} pitch={rec.head_pitch_deg:+.1f}°"

                # 왕복 내 전환 간 간격
                t_gap = now_s() + CUE_GAP_SEC
                while now_s() < t_gap:
                    if cv.waitKey(1) & 0xFF in (ord('q'), ord('Q')): break
                    time.sleep(0.003)

            cue_text = "완료"
            if results:
                df = pd.DataFrame([asdict(r) for r in results])
                ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
                path = os.path.join(out_dir, f"psp_vertical_saccade_{ts_file}.csv")
                df.to_csv(path, index=False, encoding="utf-8-sig")
                cue_text = f"CSV 저장: {path}"

            t_rest = now_s() + PAUSE_BETWEEN
            while now_s() < t_rest:
                if cv.waitKey(1) & 0xFF in (ord('q'), ord('Q')): break
                time.sleep(0.01)

    cap.release(); cv.destroyAllWindows()

    # 프레임 로그 저장(옵션)
    if log_frames and frame_log:
        df_log = pd.DataFrame(frame_log)
        prefer = ["time_hms","gazeY_norm","velY_rel_per_s","left_lid_px","right_lid_px",
                  "head_pitch_deg","head_yaw_deg","head_roll_deg","z_cm","tefr","tracked"]
        cols = [c for c in prefer if c in df_log.columns] + [c for c in df_log.columns if c not in prefer]
        df_log = df_log.reindex(columns=cols)
        ts_file = datetime.now().strftime("%Y%m%d_%H%M%S")
        fpath = os.path.join(out_dir, f"frame_log_{ts_file}.csv")
        df_log.to_csv(fpath, index=False, encoding="utf-8-sig")
        print(f"[프레임 로그 저장] {fpath}")


# ==============================
# API 모드 (이미지/비디오)
# ==============================
def s_stats(x: List[float]) -> Dict[str, float]:
    a = np.array(x, dtype=np.float32)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": np.nan, "std": np.nan, "min": np.nan, "max": np.nan, "p50": np.nan}
    return {
        "mean": float(np.mean(a)), "std": float(np.std(a)),
        "min": float(np.min(a)),  "max": float(np.max(a)),
        "p50": float(np.percentile(a, 50)),
    }

def create_api_app():
    app = FastAPI(title="Eye Tracking API (Fusion)", description="이미지/비디오에서 눈 지표 추출", version="1.3.0")
    face_img = mp_face.FaceMesh(static_image_mode=True,  max_num_faces=1, refine_landmarks=True,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)
    face_vid = mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True,
                                min_detection_confidence=0.5, min_tracking_confidence=0.5)

    @app.get("/")
    async def root():
        return {"message": "Eye Tracking API (Fusion)", "version": "1.3.0"}

    @app.post("/api/analyze/eye-tracking")
    async def analyze(file: UploadFile = File(...)):
        try:
            contents = await file.read()
            nparr = np.frombuffer(contents, np.uint8)
            image = cv.imdecode(nparr, cv.IMREAD_COLOR)
            if image is None:
                raise HTTPException(status_code=400, detail="유효하지 않은 이미지 파일")

            h, w = image.shape[:2]
            rgb = cv.cvtColor(image, cv.COLOR_BGR2RGB)
            res = face_img.process(rgb)
            if not res.multi_face_landmarks:
                raise HTTPException(status_code=404, detail="얼굴을 찾을 수 없습니다")

            lms = res.multi_face_landmarks[0].landmark

            gaze_ratio = compute_vertical_gaze_ratio(lms)
            l_open, r_open, *_ = eyelid_opening_pixels_and_points(lms, w, h)
            hp_ok, pitch, yaw, roll, z_cm = solve_head_pose(lms, w, h)

            def pt(i):
                lm = lms[i]
                return {"x": int(lm.x*w), "y": int(lm.y*h)}

            left_iris_center  = iris_center_from_landmarks(lms, LEFT_IRIS)
            right_iris_center = iris_center_from_landmarks(lms, RIGHT_IRIS)

            resp = {
                "status": "success",
                "landmarks": {
                    "left_eye": {
                        "iris_center": {"x": int(left_iris_center[0]*w), "y": int(left_iris_center[1]*h)},
                        "iris_boundary": [pt(i) for i in LEFT_IRIS],
                    },
                    "right_eye": {
                        "iris_center": {"x": int(right_iris_center[0]*w), "y": int(right_iris_center[1]*h)},
                        "iris_boundary": [pt(i) for i in RIGHT_IRIS],
                    },
                    "image_size": {"width": w, "height": h}
                },
                "analysis": {
                    "gaze_ratio_norm": gaze_ratio,
                    "eyelid_opening_px": {"left": float(l_open), "right": float(r_open)},
                    "head_pose": {
                        "ok": hp_ok, "pitch_deg": pitch if hp_ok else None,
                        "yaw_deg": yaw if hp_ok else None, "roll_deg": roll if hp_ok else None,
                        "z_cm": z_cm if hp_ok else None
                    }
                }
            }
            return JSONResponse(content=resp)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

    @app.post("/api/analyze/video")
    async def analyze_video(file: UploadFile = File(...), downsample_hz: int = 10):
        """업로드된 비디오에서 타임시리즈(10Hz 기본)와 요약 통계(vpp 포함)를 반환"""
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp:
                tmp.write(await file.read())
                tmp_path = tmp.name

            cap = cv.VideoCapture(tmp_path)
            if not cap.isOpened():
                os.unlink(tmp_path); raise HTTPException(status_code=400, detail="비디오 열기 실패")

            fps = cap.get(cv.CAP_PROP_FPS)
            if not fps or fps <= 0 or fps > 120: fps = FPS_TARGET
            frame_interval = int(max(1, round(fps / max(1, downsample_hz))))

            ts, gazeY, velY, lidL, lidR, pitchL, yawL, rollL, zcmL = [], [], [], [], [], [], [], [], []

            # 버퍼(속도 계산용)
            gaze_buf: List[float] = []
            time_buf: List[float] = []
            def cur_vel() -> float:
                if len(gaze_buf) < 3: return 0.0
                y = np.array(gaze_buf, dtype=np.float32); t = np.array(time_buf, dtype=np.float64)
                y_s = moving_average(y, 3); dy = y_s[-1] - y_s[-3]; dt = t[-1] - t[-3]
                return float(dy/dt) if dt > 1e-6 else 0.0

            tracked = 0; total = 0; idx = 0; t0 = time.perf_counter()
            while True:
                ok, frame = cap.read()
                if not ok: break
                total += 1
                h, w = frame.shape[:2]
                rgb = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
                res = face_vid.process(rgb)

                if res.multi_face_landmarks:
                    lms = res.multi_face_landmarks[0].landmark
                    gy_raw = compute_vertical_gaze_ratio(lms)
                    gaze_buf.append(gy_raw); time_buf.append(time.perf_counter() - t0)
                    vel = cur_vel()
                    l_open, r_open, *_ = eyelid_opening_pixels_and_points(lms, w, h)
                    hp_ok, pitch, yaw, roll, z_cm = solve_head_pose(lms, w, h)

                    if idx % frame_interval == 0:
                        ts.append(idx / fps); gazeY.append(float(gy_raw)); velY.append(float(vel))
                        lidL.append(float(l_open)); lidR.append(float(r_open))
                        pitchL.append(float(pitch) if hp_ok else np.nan)
                        yawL.append(float(yaw) if hp_ok else np.nan)
                        rollL.append(float(roll) if hp_ok else np.nan)
                        zcmL.append(float(z_cm) if hp_ok else np.nan)

                    tracked += 1
                idx += 1

            cap.release(); os.unlink(tmp_path)

            # 요약 + vpp
            gaze_arr = np.array(gazeY, dtype=np.float32); gaze_arr = gaze_arr[np.isfinite(gaze_arr)]
            vpp = float(np.nanmax(gaze_arr) - np.nanmin(gaze_arr)) if gaze_arr.size else float("nan")
            saccade_hits = int(np.sum((np.abs(np.array(velY)) > VEL_TH) &
                                      (np.abs(np.array(gazeY) - (np.nanmedian(gazeY) if len(gazeY) else 0)) > AMP_TH)))

            resp = {
                "status": "success",
                "meta": {
                    "frames_total": total,
                    "fps_inferred": float(fps),
                    "tracked_ratio": float(tracked / max(1, total)),
                    "downsample_hz": int(downsample_hz),
                },
                "summary": {
                    "gazeY": s_stats(gazeY),
                    "velY": s_stats(velY),
                    "eyelid_left_px": s_stats(lidL),
                    "eyelid_right_px": s_stats(lidR),
                    "head_pitch_deg": s_stats(pitchL),
                    "head_yaw_deg": s_stats(yawL),
                    "head_roll_deg": s_stats(rollL),
                    "z_cm": s_stats(zcmL),
                    "saccade_event_count": saccade_hits,
                    "vpp": vpp,  # ★ 수직 peak-to-peak(정규화 단위)
                },
                "timeseries": {
                    "t_sec": ts, "gazeY": gazeY, "velY": velY,
                    "eyelid_left_px": lidL, "eyelid_right_px": lidR,
                    "head_pitch_deg": pitchL, "head_yaw_deg": yawL, "head_roll_deg": rollL, "z_cm": zcmL,
                }
            }
            return JSONResponse(content=resp)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"서버 오류: {str(e)}")

    return app


# ==============================
# 엔트리포인트
# ==============================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["ui", "api"], default="ui")
    parser.add_argument("--cam", type=int, default=0)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--log-frames", action="store_true", help="UI: 프레임별 지표 CSV 저장")
    parser.add_argument("--out-dir", type=str, default="results", help="CSV 저장 폴더")
    args = parser.parse_args()

    if args.mode == "ui":
        run_ui(cam_index=args.cam, log_frames=args.log_frames, out_dir=args.out_dir)
    else:
        app = create_api_app()
        uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
