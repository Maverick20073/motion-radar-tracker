"""
Single-target lock-on tracker + motion radar (webcam, OpenCV + NumPy) - v3.4

Install once in Command Prompt:
    pip install opencv-python numpy
(For a more stable tracker use:  pip uninstall opencv-python -y  then  pip install opencv-contrib-python numpy)

Run:
    python motion_radar_tracker.py

Controls (click the camera window first):
    Left-click on a MOVING object : LOCK on it
    L                             : lock on the biggest moving object
    U / right-click / UNLOCK btn  : unlock
    R                             : reset background (and unlock)
    N                             : radar ON / OFF
    T                             : hide / show ALL overlay (tracking & clicking stay active)
    F                             : fullscreen        B : black & white view        Q : quit
"""

import math
import threading
import time
import traceback
from collections import deque

import cv2
import numpy as np

clock = time.perf_counter

# ----------------------------------------------------------------------------- settings
CAMERA_INDEX = 0
MIRROR = True
MAX_WIDTH = 960
DET_SCALE = 0.5
TRACKER_SCALE = 1.0
GLOBAL_SCALE = 0.5             # lost-target search runs on a half-size frame (much faster)
GLOBAL_EVERY = 4               # search the whole frame every Nth frame while lost
DISPLAY_RATE = 18.0
DISPLAY_DEADBAND = 1.0
PROB_SMOOTH = 0.15
FRAGMENT_MIN = 250
MIN_BOX_AREA = 2000
MAX_BOX_FRAC = 0.30
MERGE_GAP = 20
CONFIRM_HITS = 3
COAST_OTHER = 0.5
MIN_SPEED = 40
PREDICT_SECONDS = 1.0
WARMUP_FRAMES = 40
ACCEL_STD = 500.0
MEAS_STD = 20.0
VISUAL_STD = 6.0
TRAIL_LEN = 30
COAST_DAMPING = 0.95
GRABCUT_ITERS = 2              # was 4; fewer passes = faster lock

# --- lock keeping
TRACKER_SIM_MIN = 0.15
REACQUIRE_SIM = 0.30
MOTION_REACQ_SIM = 0.12
TEMPLATE_MIN = 0.45
WEAK_FRAMES = 45
RECENT_LOSS = 3.0
OUT_OF_VIEW_AFTER = 1.2
GLOBAL_SIM = 0.35
GLOBAL_CONFIRM = 2
AUTO_RELOCK = True
SHAPE_THR = 80
CAND_THR = 90
REFINE_EVERY = 60
START_FULLSCREEN = True
BW_VIEW = True

# --- human filter (always on)
HUMAN_FILTER = True
SUPPRESS_BODY = True
SKIN_GROW = 21
SKIN_LOCK_MAX = 0.50
FACE_EVERY = 4
FACE_HOLD = 1.5

# --- overlay visibility
SHOW_HUD = True        # T: everything except tracking, reticle and unlock button
SHOW_RADAR = True      # N

DIRECTIONS = ["RIGHT", "UP-RIGHT", "UP", "UP-LEFT", "LEFT", "DOWN-LEFT", "DOWN", "DOWN-RIGHT"]
LABELS = DIRECTIONS + ["STILL"]

C_TRACK = (0, 255, 255)
C_LOCK = (0, 255, 0)
C_COAST = (0, 175, 255)
C_BAD = (70, 70, 235)
C_PRED = (70, 235, 255)
C_RADAR = (130, 255, 150)
C_TEXT = (245, 245, 245)
C_DARK = (22, 18, 16)
C_PANEL = (36, 28, 22)
C_DIM = (170, 170, 170)
C_BTN = (60, 60, 220)

RET_GREEN = (0, 255, 0)
RET_RED = (0, 0, 255)

HIST_RANGES = [0, 180, 0, 256, 0, 256]
FONT = cv2.FONT_HERSHEY_SIMPLEX
rng = np.random.default_rng()


# ----------------------------------------------------------------------------- threaded camera
class CameraStream:
    """Reads the webcam in a background thread so the main loop always gets the NEWEST frame."""

    def __init__(self, index):
        self.cap = self._open(index)
        self._lock = threading.Lock()
        self._frame = None
        self._count = 0
        self._last_ok = clock()
        self._running = False
        self._thread = None

    @staticmethod
    def _open(index):
        backend = getattr(cv2, "CAP_DSHOW", 0)
        cap = cv2.VideoCapture(index, backend) if backend else cv2.VideoCapture(index)
        if not cap.isOpened():
            cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            return cap
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
            cap.set(cv2.CAP_PROP_FPS, 30)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def start(self):
        if not self.cap.isOpened():
            return False
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        t0 = clock()
        while clock() - t0 < 5.0:
            with self._lock:
                if self._frame is not None:
                    return True
            time.sleep(0.01)
        return False

    def _loop(self):
        while self._running:
            ok, frame = self.cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._count += 1
                self._last_ok = clock()

    def read(self, last_count):
        with self._lock:
            if self._count == last_count or self._frame is None:
                return None, last_count
            return self._frame, self._count

    def dead(self):
        return clock() - self._last_ok > 3.0

    def release(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        self.cap.release()


# ----------------------------------------------------------------------------- human filter
def skin_mask(bgr):
    """255 where a pixel looks like human skin (YCrCb AND HSV rules)."""
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    m1 = cv2.inRange(ycc, (0, 133, 77), (255, 173, 127))
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m2 = cv2.inRange(hsv, (0, 25, 40), (25, 200, 255))
    m3 = cv2.inRange(hsv, (170, 25, 40), (180, 200, 255))
    return cv2.bitwise_and(m1, cv2.bitwise_or(m2, m3))


def skin_fraction(frame, box):
    H, W = frame.shape[:2]
    x, y, w, h = [int(round(v)) for v in box]
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return 0.0
    roi = frame[y1:y2, x1:x2]
    if max(roi.shape[:2]) > 120:
        f = 120.0 / max(roi.shape[:2])
        roi = cv2.resize(roi, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return cv2.countNonZero(skin_mask(roi)) / float(roi.shape[0] * roi.shape[1])


def looks_human(frame, box, base=0.0):
    """True if the box is mostly skin (a hand / arm / face)."""
    if not HUMAN_FILTER:
        return False
    return skin_fraction(frame, box) > max(SKIN_LOCK_MAX, base + 0.35)


def odd(n):
    n = max(3, int(round(n)))
    return n if n % 2 == 1 else n + 1


class HumanFilter:
    """Removes humans from the motion mask: skin pixels + face zone + body zone."""

    def __init__(self):
        self.cascade = None
        try:
            c = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            if not c.empty():
                self.cascade = c
        except Exception:
            self.cascade = None
        self.zones = []
        self.heads = []

    def apply(self, mask, small, frame_count, now):
        H, W = mask.shape[:2]
        sk = skin_mask(small)
        sk = cv2.morphologyEx(sk, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
        g = odd(SKIN_GROW * DET_SCALE)
        sk = cv2.dilate(sk, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (g, g)))
        mask[sk > 0] = 0

        if self.cascade is not None and frame_count % FACE_EVERY == 0:
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            faces = self.cascade.detectMultiScale(gray, 1.2, 4, minSize=(max(24, int(40 * DET_SCALE)),) * 2)
            if len(faces):
                self.zones, self.heads = [], []
                for (fx, fy, fw, fh) in faces:
                    head = (fx - 0.4 * fw, fy - 0.5 * fh, fx + 1.4 * fw, fy + 1.4 * fh, now)
                    self.zones.append(head)
                    self.heads.append(head)
                    if SUPPRESS_BODY:
                        self.zones.append((fx - 1.0 * fw, fy + fh, fx + 2.0 * fw, H, now))
        self.zones = [z for z in self.zones if now - z[4] < FACE_HOLD]
        self.heads = [z for z in self.heads if now - z[4] < FACE_HOLD]
        for (x1, y1, x2, y2, _t) in self.zones:
            mask[max(int(y1), 0):min(int(y2), H), max(int(x1), 0):min(int(x2), W)] = 0
        return mask


human = HumanFilter()


# ----------------------------------------------------------------------------- Kalman filter
def fq_matrices(dt):
    F = np.array([[1, 0, dt, 0],
                  [0, 1, 0, dt],
                  [0, 0, 1, 0],
                  [0, 0, 0, 1]], dtype=float)
    q = ACCEL_STD ** 2
    Q = q * np.array([[dt ** 4 / 4, 0, dt ** 3 / 2, 0],
                      [0, dt ** 4 / 4, 0, dt ** 3 / 2],
                      [dt ** 3 / 2, 0, dt ** 2, 0],
                      [0, dt ** 3 / 2, 0, dt ** 2]])
    return F, Q


class Kalman:
    H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)

    def __init__(self, x, y):
        self.x = np.array([x, y, 0.0, 0.0])
        self.P = np.diag([30.0 ** 2, 30.0 ** 2, 250.0 ** 2, 250.0 ** 2])

    def predict(self, dt, coasting=False):
        F, Q = fq_matrices(dt)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q
        if coasting:
            self.x[2:] *= COAST_DAMPING

    def update(self, z, std=MEAS_STD):
        R = np.eye(2) * std ** 2
        S = self.H @ self.P @ self.H.T + R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (np.asarray(z, float) - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def project(self, t):
        F, Q = fq_matrices(t)
        return F @ self.x, F @ self.P @ F.T + Q


class Track:
    def __init__(self, tid, cx, cy, bw, bh, now):
        self.id = tid
        self.kf = Kalman(cx, cy)
        self.bw, self.bh = float(bw), float(bh)
        self.hits = 1
        self.missed = 0
        self.last_seen = now
        self.trail = deque(maxlen=TRAIL_LEN)

    @property
    def x(self):
        return float(self.kf.x[0])

    @property
    def y(self):
        return float(self.kf.x[1])

    @property
    def vx(self):
        return float(self.kf.x[2])

    @property
    def vy(self):
        return float(self.kf.x[3])

    @property
    def confirmed(self):
        return self.hits >= CONFIRM_HITS

    @property
    def coasting(self):
        return self.missed > 0

    def box(self):
        return (self.x - self.bw / 2, self.y - self.bh / 2, self.x + self.bw / 2, self.y + self.bh / 2)


# ----------------------------------------------------------------------------- appearance helpers
def clamp_box(box, shape):
    H, W = shape[:2]
    x, y, w, h = [int(round(v)) for v in box]
    x, y = max(0, min(x, W - 2)), max(0, min(y, H - 2))
    w, h = max(16, min(w, W - x)), max(16, min(h, H - y))
    return x, y, w, h


def roi_hist(frame, box):
    H, W = frame.shape[:2]
    x, y, w, h = [int(v) for v in box]
    x1, y1, x2, y2 = max(x, 0), max(y, 0), min(x + w, W), min(y + h, H)
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    mx, my = int((x2 - x1) * 0.15), int((y2 - y1) * 0.15)
    x1, x2, y1, y2 = x1 + mx, x2 - mx, y1 + my, y2 - my
    if x2 - x1 < 4 or y2 - y1 < 4:
        return None
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0, 1, 2], None, [8, 4, 4], HIST_RANGES)
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    return hist


def similarity(frame, box, ref):
    cur = roi_hist(frame, box)
    if cur is None or ref is None:
        return 0.0
    return float(cv2.compareHist(ref, cur, cv2.HISTCMP_CORREL))


def fg_bg_hist(roi, fg):
    """Colour model rewarding colours of the OBJECT and ignoring colours also found in its background."""
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    bins = [12, 8, 8]
    f = cv2.calcHist([hsv], [0, 1, 2], fg, bins, HIST_RANGES)
    fs = float(f.sum())
    if fs < 1:
        return None
    b = cv2.calcHist([hsv], [0, 1, 2], cv2.bitwise_not(fg), bins, HIST_RANGES)
    bs = float(b.sum())
    f = f / fs
    b = b / bs if bs > 0 else np.zeros_like(b)
    score = f / (f + b + 1e-9)
    score[f < 0.002] = 0
    if score.max() < 0.05:
        plain = f.copy()
        cv2.normalize(plain, plain, 0, 255, cv2.NORM_MINMAX)
        return plain.astype(np.float32)
    return (score * 255).astype(np.float32)


def vivid(bgr):
    b, g, r = [int(round(c)) for c in bgr]
    h, s, v = [int(c) for c in cv2.cvtColor(np.uint8([[[b, g, r]]]), cv2.COLOR_BGR2HSV)[0, 0]]
    if s < 40:
        return (235, 235, 235) if v > 110 else (205, 205, 205)
    s, v = max(s, 120), max(v, 210)
    out = cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2BGR)[0, 0]
    return tuple(int(c) for c in out)


def refine_object(frame, box):
    """Cut the object out of its background with GrabCut (skin pixels are removed from the result)."""
    H, W = frame.shape[:2]
    x, y, w, h = box
    pad = int(0.15 * max(w, h))
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
    roi = frame[y0:y1, x0:x1]
    rh, rw = roi.shape[:2]
    if rw < 24 or rh < 24:
        return None
    scale = min(1.0, 320.0 / rw)
    small = roi if scale == 1.0 else cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sh, sw = small.shape[:2]
    rx, ry = max(int((x - x0) * scale), 1), max(int((y - y0) * scale), 1)
    rww = max(min(int(w * scale), sw - rx - 1), 4)
    rhh = max(min(int(h * scale), sh - ry - 1), 4)
    mask = np.zeros(small.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65), np.float64), np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(small, mask, (rx, ry, rww, rhh), bgd, fgd, GRABCUT_ITERS, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return None
    fg = (((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)) * 255
    if scale != 1.0:
        fg = cv2.resize(fg, (rw, rh), interpolation=cv2.INTER_LINEAR)
        _, fg = cv2.threshold(fg, 127, 255, cv2.THRESH_BINARY)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    cnts, _ = cv2.findContours(fg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    one = np.zeros_like(fg)
    cv2.drawContours(one, [c], -1, 255, -1)
    if cv2.countNonZero(one) < 0.12 * w * h:
        return None
    if HUMAN_FILTER:
        ns = cv2.bitwise_and(one, cv2.bitwise_not(skin_mask(roi)))
        if cv2.countNonZero(ns) >= 0.3 * cv2.countNonZero(one):
            cn, _ = cv2.findContours(ns, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if cn:
                c = max(cn, key=cv2.contourArea)
                one = np.zeros_like(fg)
                cv2.drawContours(one, [c], -1, 255, -1)
    bx, by, bw, bh = cv2.boundingRect(c)
    hist = fg_bg_hist(roi, one)
    if hist is None:
        return None
    color = cv2.mean(roi, mask=one)[:3]
    c = c + np.array([[x0, y0]], dtype=c.dtype)
    return dict(box=(x0 + bx, y0 + by, bw, bh), contour=c, color=vivid(color), shape_hist=hist)


def build_model(frame, box, from_motion):
    """Size, shape and colour of the thing you locked on."""
    x, y, w, h = clamp_box(box, frame.shape)
    ref = refine_object(frame, (x, y, w, h))
    if ref is not None and ref["box"][2] * ref["box"][3] >= 0.30 * w * h:
        return ref
    if from_motion:
        x, y, w, h = clamp_box((x + w * 0.12, y + h * 0.12, w * 0.76, h * 0.76), frame.shape)
    H, W = frame.shape[:2]
    pad = int(0.15 * max(w, h))
    rx0, ry0 = max(x - pad, 0), max(y - pad, 0)
    rx1, ry1 = min(x + w + pad, W), min(y + h + pad, H)
    roi = frame[ry0:ry1, rx0:rx1]
    ix, iy = int(w * 0.2), int(h * 0.2)
    fgm = np.zeros(roi.shape[:2], np.uint8)
    fgm[max(y + iy - ry0, 0):max(y + h - iy - ry0, 1), max(x + ix - rx0, 0):max(x + w - ix - rx0, 1)] = 255
    if HUMAN_FILTER:
        ns = cv2.bitwise_and(fgm, cv2.bitwise_not(skin_mask(roi)))
        if cv2.countNonZero(ns) >= 0.3 * cv2.countNonZero(fgm):
            fgm = ns
    hist = fg_bg_hist(roi, fgm)
    inner = frame[y + iy:y + h - iy, x + ix:x + w - ix]
    if inner.size == 0:
        inner = frame[y:y + h, x:x + w]
    if hist is None:
        hsv = cv2.cvtColor(inner, cv2.COLOR_BGR2HSV)
        hist = cv2.calcHist([hsv], [0, 1, 2], None, [12, 8, 8], HIST_RANGES)
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        hist = hist.astype(np.float32)
    return dict(box=(x, y, w, h), contour=None, color=vivid(cv2.mean(inner)[:3]), shape_hist=hist)


class ScaledTracker:
    def __init__(self, tracker, scale):
        self.tracker = tracker
        self.scale = scale

    def update(self, frame):
        small = frame if self.scale == 1.0 else cv2.resize(frame, None, fx=self.scale, fy=self.scale,
                                                            interpolation=cv2.INTER_AREA)
        ok, bb = self.tracker.update(small)
        if not ok:
            return False, None
        s = self.scale
        return True, (bb[0] / s, bb[1] / s, bb[2] / s, bb[3] / s)


def create_visual_tracker(frame, box):
    """Best appearance tracker available: CSRT > KCF > MIL."""
    box = tuple(float(v) for v in box)
    scale = TRACKER_SCALE if min(box[2], box[3]) * TRACKER_SCALE >= 24 else 1.0
    small = frame if scale == 1.0 else cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    sbox = tuple(max(2, int(round(v * scale))) for v in box)
    for name in ("TrackerCSRT_create", "TrackerKCF_create", "TrackerMIL_create"):
        for mod in (cv2, getattr(cv2, "legacy", None)):
            if mod is None or not hasattr(mod, name):
                continue
            try:
                tr = getattr(mod, name)()
                tr.init(small, sbox)
                return ScaledTracker(tr, scale), name.replace("Tracker", "").replace("_create", "")
            except Exception:
                continue
    return None, "TEMPLATE"


class LockedTarget(Track):
    """The one object you chose. Followed by appearance + Kalman + motion fallback."""

    def __init__(self, tid, frame, box, now, from_motion=False):
        model = build_model(frame, box, from_motion)
        x, y, w, h = clamp_box(model["box"], frame.shape)
        super().__init__(tid, x + w / 2, y + h / 2, w, h, now)
        self.hits = CONFIRM_HITS
        self.hist = roi_hist(frame, (x, y, w, h))
        self.hist_ref = None if self.hist is None else self.hist.copy()
        self.template = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).copy()
        self.tracker, self.tracker_name = create_visual_tracker(frame, (x, y, w, h))
        self.shape_hist = model["shape_hist"]
        self.contour = model["contour"]
        self.vis_box = [x, y, w, h] if self.contour is not None else None
        self.raw_box = None
        self._prev_vis = None
        self.base_area = float(w * h)
        self.base_skin = skin_fraction(frame, (x, y, w, h))
        self.lock_time = now
        self.sim = 1.0
        self.weak = 0
        self.fail = 0
        self.occluded = False
        self.cand_c = None
        self.cand_n = 0
        self.search_tick = 0
        self.disp = None
        self.kf.P = np.diag([8.0 ** 2, 8.0 ** 2, 120.0 ** 2, 120.0 ** 2])

    def reinit_tracker(self, frame, box):
        self.tracker, self.tracker_name = create_visual_tracker(frame, clamp_box(box, frame.shape))

    def out_of_view(self, now):
        return self.missed > 0 and (now - self.last_seen) > OUT_OF_VIEW_AFTER

    def draw_box(self):
        if self.vis_box is not None and self.contour is not None and not self.coasting:
            x, y, w, h = self.vis_box
            return (x, y, x + w, y + h)
        return self.box()

    def display_box(self, dt):
        t = np.array(self.draw_box(), dtype=float)
        if self.disp is None or float(np.max(np.abs(t - self.disp))) > 3.0 * max(self.bw, self.bh):
            self.disp = t
        else:
            rate = DISPLAY_RATE + math.hypot(self.vx, self.vy) / 40.0
            k = 1.0 - math.exp(-dt * rate)
            diff = t - self.disp
            diff[np.abs(diff) < DISPLAY_DEADBAND] = 0.0
            self.disp = self.disp + k * diff
        return tuple(float(v) for v in self.disp)


def update_shape(lt, frame, box):
    lt.contour, lt.vis_box, lt.raw_box = None, None, None
    if lt.shape_hist is None:
        return
    H, W = frame.shape[:2]
    x, y, w, h = [int(v) for v in box]
    pad = int(0.15 * max(w, h))
    x0, y0 = max(x - pad, 0), max(y - pad, 0)
    x1, y1 = min(x + w + pad, W), min(y + h + pad, H)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    bp = cv2.calcBackProject([hsv], [0, 1, 2], lt.shape_hist, HIST_RANGES, 1)
    bp = cv2.GaussianBlur(bp, (5, 5), 0)
    _, m = cv2.threshold(bp, SHAPE_THR, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return
    c = max(cnts, key=cv2.contourArea)
    area = cv2.contourArea(c)
    if area < 0.2 * w * h or area > 1.5 * w * h:
        return
    c = c + np.array([[x0, y0]], dtype=c.dtype)
    new = cv2.boundingRect(c)
    prev = lt._prev_vis
    lt.vis_box = list(new) if prev is None else [0.5 * o + 0.5 * n for o, n in zip(prev, new)]
    lt._prev_vis = list(lt.vis_box)
    lt.raw_box = new
    lt.contour = c


def template_search(frame, lt):
    th, tw = lt.template.shape[:2]
    H, W = frame.shape[:2]
    pad = int(1.2 * max(tw, th) + 60)
    cx, cy = int(lt.x), int(lt.y)
    x1, y1 = max(cx - pad - tw // 2, 0), max(cy - pad - th // 2, 0)
    x2, y2 = min(cx + pad + tw // 2, W), min(cy + pad + th // 2, H)
    if x2 - x1 <= tw or y2 - y1 <= th:
        return None
    gray = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2GRAY)
    res = cv2.matchTemplate(gray, lt.template, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    if score < TEMPLATE_MIN:
        return None
    return (x1 + loc[0], y1 + loc[1], tw, th)


def color_candidates(frame, lt, region=None, scale=1.0):
    if lt.shape_hist is None:
        return []
    H, W = frame.shape[:2]
    x0, y0, x1, y1 = (0, 0, W, H) if region is None else region
    if x1 - x0 < 16 or y1 - y0 < 16:
        return []
    roi = frame[y0:y1, x0:x1]
    if scale != 1.0:
        roi = cv2.resize(roi, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    bp = cv2.calcBackProject([hsv], [0, 1, 2], lt.shape_hist, HIST_RANGES, 1)
    bp = cv2.GaussianBlur(bp, (5, 5), 0)
    _, m = cv2.threshold(bp, CAND_THR, 255, cv2.THRESH_BINARY)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    ref = max(lt.base_area, 1.0)
    out = []
    for c in cnts:
        ratio = cv2.contourArea(c) / (scale * scale) / ref
        if ratio < 0.3 or ratio > 3.0:
            continue
        bx, by, bw, bh = cv2.boundingRect(c)
        out.append((ratio, (x0 + bx / scale, y0 + by / scale, bw / scale, bh / scale)))
    out.sort(key=lambda t: -t[0])
    return out[:6]


def best_match(frame, lt, cands, min_sim):
    best = None
    for _ratio, cand in cands:
        cb = clamp_box(cand, frame.shape)
        if looks_human(frame, cb, lt.base_skin):
            continue
        sim = max(similarity(frame, cb, lt.hist), similarity(frame, cb, lt.hist_ref))
        if sim >= min_sim and (best is None or sim > best[0]):
            best = (sim, cb)
    return best


def local_color_search(frame, lt, gate):
    H, W = frame.shape[:2]
    r = int(gate + max(lt.bw, lt.bh))
    region = (max(int(lt.x) - r, 0), max(int(lt.y) - r, 0), min(int(lt.x) + r, W), min(int(lt.y) + r, H))
    return best_match(frame, lt, color_candidates(frame, lt, region), REACQUIRE_SIM)


def global_search(frame, lt):
    # Runs on a downscaled frame; candidate boxes are returned in full-size coordinates.
    return best_match(frame, lt, color_candidates(frame, lt, None, GLOBAL_SCALE), GLOBAL_SIM)


def find_distractors(lt, dets):
    x1, y1, x2, y2 = lt.box()
    la = max((x2 - x1) * (y2 - y1), 1.0)
    out = []
    for (cx, cy, bw, bh) in dets:
        ix = min(x2, cx + bw / 2) - max(x1, cx - bw / 2)
        iy = min(y2, cy + bh / 2) - max(y1, cy - bh / 2)
        if ix > 0 and iy > 0 and ix * iy > 0.10 * la:
            if math.hypot(cx - lt.x, cy - lt.y) > 0.6 * max(lt.bw, lt.bh) or bw * bh > 2.0 * la:
                out.append((cx, cy, bw, bh))
    return out


def update_locked(lt, frame, dets, now, dt):
    """Move the lock forward one frame."""
    H, W = frame.shape[:2]
    lt.kf.predict(dt, coasting=lt.coasting)
    lt.occluded = lt.missed == 0 and bool(find_distractors(lt, dets))
    box, std = None, VISUAL_STD
    size = max(lt.bw, lt.bh)
    speed = math.hypot(lt.vx, lt.vy)
    recent = (now - lt.last_seen) < RECENT_LOSS

    # 1) appearance tracker (keeps following even when the object stops moving)
    if lt.tracker is not None:
        try:
            ok, bb = lt.tracker.update(frame)
        except Exception:
            ok, bb = False, None
        if ok:
            x, y, bw, bh = bb
            sane = (bw > 8 and bh > 8 and 0 < x + bw / 2 < W and 0 < y + bh / 2 < H
                    and 0.15 * lt.base_area <= bw * bh <= 6.0 * lt.base_area)
            if sane:
                if lt.occluded:
                    allowed = 25 + 0.5 * size + 2 * speed * dt
                else:
                    allowed = 60 + 1.5 * size + 0.5 * speed + 150 * min(now - lt.last_seen, 3.0)
                if not recent:
                    allowed = 0
                jump = math.hypot(x + bw / 2 - lt.x, y + bh / 2 - lt.y)
                sim = similarity(frame, (x, y, bw, bh), lt.hist)
                if jump <= allowed:
                    if sim >= TRACKER_SIM_MIN:
                        lt.weak = 0
                        box, lt.sim = (x, y, bw, bh), sim
                        std = VISUAL_STD * (1 + 6 * max(0.0, 0.7 - sim))
                    elif lt.weak < WEAK_FRAMES and not lt.occluded:
                        lt.weak += 1
                        box, lt.sim = (x, y, bw, bh), sim
                        std = VISUAL_STD * 6
            lt.fail = 0
        else:
            lt.fail += 1

    # 1b) the tracker must never slide onto a hand / arm / face
    if box is not None and looks_human(frame, clamp_box(box, frame.shape), lt.base_skin):
        box = None
        lt.sim = 0.0
        lt.weak = 0

    # 2) short loss: template search near the predicted position
    if box is None and recent:
        found = template_search(frame, lt)
        if found is not None and not looks_human(frame, clamp_box(found, frame.shape), lt.base_skin):
            box, std = found, VISUAL_STD * 2
            lt.sim = max(similarity(frame, box, lt.hist), 0.0)
            lt.reinit_tracker(frame, box)
            lt.weak = lt.fail = 0

    # 3) short loss: colour search nearby, then hop to the nearest MOVING blob of the right size
    if box is None and recent:
        gate = 140 + 100 * min(now - lt.last_seen, 3.0)
        res = local_color_search(frame, lt, gate)
        if res is not None:
            box, lt.sim, std = res[1], res[0], VISUAL_STD * 3
            lt.reinit_tracker(frame, box)
            lt.weak = lt.fail = 0
        elif dets:
            best = None
            for (cx, cy, bw, bh) in dets:
                d = math.hypot(cx - lt.x, cy - lt.y)
                if d >= gate:
                    continue
                ratio = (bw * bh) / max(lt.base_area, 1.0)
                if not 0.3 <= ratio <= 3.5:
                    continue
                cand = (cx - bw / 2, cy - bh / 2, bw, bh)
                if looks_human(frame, clamp_box(cand, frame.shape), lt.base_skin):
                    continue
                sim = similarity(frame, cand, lt.hist)
                if sim < MOTION_REACQ_SIM:
                    continue
                score = d / gate - sim
                if best is None or score < best[0]:
                    best = (score, sim, cand)
            if best is not None:
                box, lt.sim, std = best[2], best[1], MEAS_STD
                lt.reinit_tracker(frame, box)
                lt.weak = lt.fail = 0

    # 4) lost: search the frame every GLOBAL_EVERY frames; re-lock when the same-looking object is seen again
    if box is None and lt.missed >= 1:
        lt.search_tick += 1
        if lt.search_tick % GLOBAL_EVERY == 0:
            res = global_search(frame, lt)
            if res is not None:
                cx, cy = res[1][0] + res[1][2] / 2, res[1][1] + res[1][3] / 2
                gate = max(120.0, 2.0 * size)
                if lt.cand_c is not None and math.hypot(cx - lt.cand_c[0], cy - lt.cand_c[1]) < gate:
                    lt.cand_n += 1
                else:
                    lt.cand_n = 1
                lt.cand_c = (cx, cy)
                if lt.cand_n >= GLOBAL_CONFIRM:
                    box, lt.sim, std = res[1], res[0], VISUAL_STD * 3
                    lt.reinit_tracker(frame, box)
                    lt.weak = lt.fail = 0
                    lt.cand_n, lt.cand_c = 0, None
                    lt.kf = Kalman(cx, cy)
                    lt.kf.P = np.diag([8.0 ** 2, 8.0 ** 2, 120.0 ** 2, 120.0 ** 2])
                    lt.trail.clear()
                    lt.missed = 0
                    say("TARGET BACK IN RANGE - locked again", 3.0)
            else:
                lt.cand_n, lt.cand_c = 0, None

    # 5) last resort: restart the tracker on the predicted box if it still looks right
    if box is None and recent and lt.missed % 3 == 2:
        pb = (lt.x - lt.bw / 2, lt.y - lt.bh / 2, lt.bw, lt.bh)
        cpb = clamp_box(pb, frame.shape)
        if similarity(frame, cpb, lt.hist) >= 0.35 and not looks_human(frame, cpb, lt.base_skin):
            lt.reinit_tracker(frame, pb)

    if box is not None:
        bx, by, bw, bh = box
        if lt.occluded:
            std = max(std, VISUAL_STD * 3)
            lt.contour, lt.vis_box, lt.raw_box = None, None, None
        else:
            update_shape(lt, frame, box)
        mx, my, mw, mh, a = bx + bw / 2, by + bh / 2, bw, bh, 0.8
        if lt.raw_box is not None:
            rx, ry, rw, rh = lt.raw_box
            scx, scy = rx + rw / 2, ry + rh / 2
            if math.hypot(scx - mx, scy - my) <= 0.4 * max(bw, bh):
                mx, my, mw, mh, a = scx, scy, rw, rh, 0.7
                std = max(VISUAL_STD * 0.7, std * 0.7)
        lt.kf.update((mx, my), std)
        lt.bw = a * lt.bw + (1 - a) * mw
        lt.bh = a * lt.bh + (1 - a) * mh
        lt.hits += 1
        lt.missed = 0
        lt.last_seen = now
        if not lt.occluded and lt.weak == 0 and lt.sim > 0.7:
            if lt.hits % 5 == 0:
                new = roi_hist(frame, box)
                if new is not None and lt.hist is not None:
                    lt.hist = 0.9 * lt.hist + 0.1 * new
            if lt.hits % 15 == 0:
                x, y, w, h = clamp_box(box, frame.shape)
                lt.template = cv2.cvtColor(frame[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).copy()
            if lt.hits % REFINE_EVERY == 0 and lt.raw_box is not None and lt.sim > 0.8:
                rx, ry, rw, rh = lt.raw_box
                if rw * rh >= 0.5 * bw * bh:
                    lt.reinit_tracker(frame, (rx - 0.03 * rw, ry - 0.03 * rh, rw * 1.06, rh * 1.06))
    else:
        lt.missed += 1
        lt.contour, lt.vis_box, lt.raw_box = None, None, None
        lt._prev_vis = None

    lt.trail.append((lt.x, lt.y))


# ----------------------------------------------------------------------------- global state
tracks = {}
next_id = 1
lock_count = 0
locked = None
pending_click = None
pending_unlock = False
status_msg = ""
status_until = 0.0


def say(msg, seconds=2.5):
    global status_msg, status_until
    status_msg, status_until = msg, clock() + seconds


def lock_on(clean, box, now, from_motion=True):
    """Lock on ONE moving object. Refused while something is already locked or if it looks human."""
    global locked, lock_count
    if locked is not None:
        say("Already locked - press U (or click UNLOCK) before choosing another object", 3.5)
        return False
    x, y, w, h = clamp_box(box, clean.shape)
    if looks_human(clean, (x, y, w, h)):
        say("That looks like a hand / body part - not locked. Humans are ignored.", 3.5)
        return False
    lock_count += 1
    locked = LockedTarget(lock_count, clean, box, now, from_motion)
    tracks.clear()
    say(f"LOCKED on target #{lock_count}  ({locked.tracker_name} tracker)")
    return True


def unlock():
    global locked
    if locked is not None:
        locked = None
        say("Unlocked")


# ----------------------------------------------------------------------------- detection
def merge_boxes(boxes, gap):
    boxes = [list(b) for b in boxes]
    changed = True
    while changed:
        changed = False
        out = []
        while boxes:
            a = boxes.pop()
            i = 0
            while i < len(boxes):
                b = boxes[i]
                if (a[0] - gap < b[0] + b[2] and b[0] < a[0] + a[2] + gap and
                        a[1] - gap < b[1] + b[3] and b[1] < a[1] + a[3] + gap):
                    x1, y1 = min(a[0], b[0]), min(a[1], b[1])
                    x2 = max(a[0] + a[2], b[0] + b[2])
                    y2 = max(a[1] + a[3], b[1] + b[3])
                    a = [x1, y1, x2 - x1, y2 - y1]
                    boxes.pop(i)
                    changed = True
                    i = 0
                else:
                    i += 1
            out.append(a)
        boxes = out
    return [tuple(b) for b in boxes]


def find_detections(mask, s=1.0):
    """Clean the motion mask and return [(cx, cy, w, h), ...] in FULL-size pixels, one per object."""
    k_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(15 * s), odd(15 * s)))
    k_dil = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (odd(5 * s), odd(5 * s)))
    _, mask = cv2.threshold(mask, 200, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k_open)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k_close)
    mask = cv2.dilate(mask, k_dil, iterations=1)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    s2 = s * s
    boxes = [cv2.boundingRect(c) for c in contours if cv2.contourArea(c) >= FRAGMENT_MIN * s2]
    boxes = merge_boxes(boxes, MERGE_GAP * s)
    max_area = MAX_BOX_FRAC * mask.shape[0] * mask.shape[1]
    inv = 1.0 / s
    return [((x + w / 2) * inv, (y + h / 2) * inv, w * inv, h * inv)
            for x, y, w, h in boxes if MIN_BOX_AREA * s2 <= w * h <= max_area]


# ----------------------------------------------------------------------------- tracking of (clickable) moving objects
def update_tracks(dets, now, dt):
    global next_id

    for tr in tracks.values():
        tr.kf.predict(dt, coasting=tr.coasting)

    pairs = []
    for tid, tr in tracks.items():
        speed = math.hypot(tr.vx, tr.vy)
        gate = 70 + 0.6 * max(tr.bw, tr.bh) + 1.5 * speed * dt
        for di, (cx, cy, _w, _h) in enumerate(dets):
            d = math.hypot(cx - tr.x, cy - tr.y)
            if d < gate:
                pairs.append((d, tid, di))
    pairs.sort()

    used_t, used_d = set(), set()
    for _d, tid, di in pairs:
        if tid in used_t or di in used_d:
            continue
        cx, cy, bw, bh = dets[di]
        tr = tracks[tid]
        tr.kf.update((cx, cy))
        tr.bw = 0.7 * tr.bw + 0.3 * bw
        tr.bh = 0.7 * tr.bh + 0.3 * bh
        tr.hits += 1
        tr.missed = 0
        tr.last_seen = now
        used_t.add(tid)
        used_d.add(di)

    for tid, tr in list(tracks.items()):
        if tid not in used_t:
            tr.missed += 1
            if now - tr.last_seen > COAST_OTHER or (not tr.confirmed and tr.missed > 2):
                del tracks[tid]

    for di, (cx, cy, bw, bh) in enumerate(dets):
        if di not in used_d:
            tracks[next_id] = Track(next_id, cx, cy, bw, bh, now)
            next_id += 1

    ids = list(tracks)
    remove = set()
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            ia, ib = ids[i], ids[j]
            if ia in remove or ib in remove:
                continue
            a, b = tracks[ia], tracks[ib]
            ax1, ay1, ax2, ay2 = a.box()
            bx1, by1, bx2, by2 = b.box()
            if (bx1 <= a.x <= bx2 and by1 <= a.y <= by2) or (ax1 <= b.x <= ax2 and ay1 <= b.y <= ay2):
                remove.add(ia if a.hits < b.hits else ib)
    for tid in remove:
        del tracks[tid]

    for tr in tracks.values():
        if tr.confirmed:
            tr.trail.append((tr.x, tr.y))


def step(frame, sub, frame_count, now, dt):
    """One frame: detect motion (humans removed), then EITHER follow the lock OR list clickable objects."""
    global locked
    small = frame if DET_SCALE == 1.0 else cv2.resize(frame, None, fx=DET_SCALE, fy=DET_SCALE,
                                                      interpolation=cv2.INTER_AREA)
    blurred = cv2.GaussianBlur(small, (3, 3) if DET_SCALE < 1.0 else (5, 5), 0)
    mask = sub.apply(blurred, learningRate=0.003 if frame_count > WARMUP_FRAMES else -1)
    if frame_count <= WARMUP_FRAMES:
        return
    if HUMAN_FILTER:
        mask = human.apply(mask, blurred, frame_count, now)
    dets = find_detections(mask, DET_SCALE)

    if locked is not None:
        update_locked(locked, frame, dets, now, dt)
        tracks.clear()
        if not AUTO_RELOCK and locked.out_of_view(now):
            locked = None
            say("TARGET NOT LOCKED - out of range. Click a moving object to lock again", 4.0)
        return
    update_tracks(dets, now, dt)


def heading_probabilities(tr, n=300):
    mean = tr.kf.x[2:4]
    cov = tr.kf.P[2:4, 2:4]
    cov = (cov + cov.T) / 2 + np.eye(2) * 1e-3
    v = rng.multivariate_normal(mean, cov, n, check_valid="ignore")
    speed = np.hypot(v[:, 0], v[:, 1])
    ang = np.degrees(np.arctan2(-v[:, 1], v[:, 0])) % 360
    bins = ((ang + 22.5) // 45).astype(int) % 8
    bins[speed < MIN_SPEED] = 8
    return np.bincount(bins, minlength=9) / n


def heading_name(tr):
    speed = math.hypot(tr.vx, tr.vy)
    if speed < MIN_SPEED:
        return "STILL", speed
    ang = math.degrees(math.atan2(-tr.vy, tr.vx)) % 360
    return DIRECTIONS[int((ang + 22.5) // 45) % 8], speed


# ----------------------------------------------------------------------------- drawing
def blend_rect(frame, x1, y1, x2, y2, color, alpha):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), min(int(x2), W), min(int(y2), H)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    frame[y1:y2, x1:x2] = cv2.addWeighted(np.full_like(roi, color), alpha, roi, 1 - alpha, 0)


def overlay_draw(frame, rect, alpha, painter):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = rect
    x1, y1, x2, y2 = max(int(x1), 0), max(int(y1), 0), min(int(x2), W), min(int(y2), H)
    if x2 <= x1 or y2 <= y1:
        return
    roi = frame[y1:y2, x1:x2]
    layer = roi.copy()
    painter(layer, -x1, -y1)
    frame[y1:y2, x1:x2] = cv2.addWeighted(layer, alpha, roi, 1 - alpha, 0)


def chip(frame, text, org, fg=C_TEXT, bg=C_PANEL, scale=0.5, thick=1, alpha=0.78, pad=5):
    (tw, th), base = cv2.getTextSize(text, FONT, scale, thick)
    x, y = int(org[0]), int(org[1])
    blend_rect(frame, x - pad, y - th - pad, x + tw + pad, y + base + pad, bg, alpha)
    cv2.putText(frame, text, (x, y), FONT, scale, fg, thick, cv2.LINE_AA)
    return tw + 2 * pad


def brackets(frame, x1, y1, x2, y2, color, thick, length=26):
    for (px, py, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
        cv2.line(frame, (px, py), (px + dx * length, py), color, thick, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy * length), color, thick, cv2.LINE_AA)


def draw_other(frame, tr):
    x1, y1, x2, y2 = [int(v) for v in tr.box()]
    name, _speed = heading_name(tr)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_TRACK, 1, cv2.LINE_AA)
    L = max(6, min(14, (x2 - x1) // 4))
    brackets(frame, x1 - 2, y1 - 2, x2 + 2, y2 + 2, C_DARK, 4, L)
    brackets(frame, x1 - 2, y1 - 2, x2 + 2, y2 + 2, C_TRACK, 2, L)
    chip(frame, f"#{tr.id} {name}  click to lock", (x1, max(y1 - 8, 14)), C_DARK, C_TRACK, 0.38, 1, 0.9, 3)
    if math.hypot(tr.vx, tr.vy) >= MIN_SPEED:
        cv2.arrowedLine(frame, (int(tr.x), int(tr.y)), (int(tr.x + tr.vx * 0.4), int(tr.y + tr.vy * 0.4)),
                        C_TRACK, 1, cv2.LINE_AA, tipLength=0.3)


def draw_ignored_humans(frame):
    """Thin blue label on detected faces so you can see what is being ignored."""
    inv = 1.0 / DET_SCALE
    for (x1, y1, x2, y2, _t) in human.heads:
        p1, p2 = (int(x1 * inv), int(y1 * inv)), (int(x2 * inv), int(y2 * inv))
        cv2.rectangle(frame, p1, p2, (200, 120, 40), 1, cv2.LINE_AA)
        cv2.putText(frame, "HUMAN - ignored", (p1[0], max(p1[1] - 6, 12)), FONT, 0.4, (200, 120, 40), 1, cv2.LINE_AA)


def draw_reticle(frame, x, y, w, h, vx, vy, locked, anim=0.0, pulse=0.0, ring=0.0):
    """Target lock reticle: four corner brackets + a tilted square. Green when locked, red when lost."""
    color = RET_GREEN if locked else RET_RED
    H = frame.shape[0]

    x1, y1 = int(round(x)), int(round(y))
    x2, y2 = int(round(x + w)), int(round(y + h))
    bw_, bh_ = max(x2 - x1, 20), max(y2 - y1, 20)
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

    # Corner brackets, slightly outside the target box
    pad = 8
    L = max(12, int(min(bw_, bh_) * 0.22))
    corners = [
        (x1 - pad, y1 - pad, 1, 1),
        (x2 + pad, y1 - pad, -1, 1),
        (x1 - pad, y2 + pad, 1, -1),
        (x2 + pad, y2 + pad, -1, -1),
    ]
    for px, py, dx, dy in corners:
        cv2.line(frame, (px, py), (px + dx * L, py), color, 2, cv2.LINE_AA)
        cv2.line(frame, (px, py), (px, py + dy * L), color, 2, cv2.LINE_AA)

    # Tilted square at the center (rotated 30 degrees, like the reference image)
    half = max(8, int(min(bw_, bh_) * 0.18))
    radius = half * math.sqrt(2)
    base_angle = math.radians(30)
    pts = []
    for k in range(4):
        a = base_angle + k * math.pi / 2
        pts.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    poly = np.array(pts, dtype=np.float32).round().astype(np.int32).reshape((-1, 1, 2))
    cv2.polylines(frame, [poly], True, color, 2, cv2.LINE_AA)

    # Small center dot
    cv2.circle(frame, (cx, cy), 2, color, -1, cv2.LINE_AA)

    # Size label below the brackets
    label = f"{int(round(w))} x {int(round(h))} px"
    chip(frame, label, (x1 - pad, min(y2 + pad + 20, H - 22)), color, C_PANEL, 0.4, 1, 0.8, 3)


def draw_banner(frame, title, sub, color):
    h, w = frame.shape[:2]
    (tw, th), _ = cv2.getTextSize(title, FONT, 0.95, 2)
    (sw, sh), _ = cv2.getTextSize(sub, FONT, 0.5, 1)
    bw = max(tw, sw) + 56
    x1, x2 = w // 2 - bw // 2, w // 2 + bw // 2
    y1, y2 = 16, 16 + th + sh + 40
    blend_rect(frame, x1, y1, x2, y2, C_PANEL, 0.88)
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2, cv2.LINE_AA)
    brackets(frame, x1 - 4, y1 - 4, x2 + 4, y2 + 4, color, 3, 14)
    cv2.putText(frame, title, (w // 2 - tw // 2, y1 + th + 12), FONT, 0.95, color, 2, cv2.LINE_AA)
    cv2.putText(frame, sub, (w // 2 - sw // 2, y2 - 12), FONT, 0.5, C_TEXT, 1, cv2.LINE_AA)


def draw_locked(frame, tr, now, dt):
    if tr.out_of_view(now):
        draw_reticle(frame, tr.x - tr.bw / 2, tr.y - tr.bh / 2, tr.bw, tr.bh, 0.0, 0.0, locked=False)
        tr.disp = None
        return
    x1, y1, x2, y2 = tr.display_box(dt)
    draw_reticle(frame, x1, y1, max(x2 - x1, 1), max(y2 - y1, 1), tr.vx, tr.vy, locked=True)


def draw_radar(frame, now):
    h, w = frame.shape[:2]
    R = 92
    cx, cy = w - R - 18, h - R - 18
    blend_rect(frame, cx - R - 6, cy - R - 24, cx + R + 6, cy + R + 6, C_PANEL, 0.55)

    def paint_disc(layer, ox, oy):
        cv2.circle(layer, (cx + ox, cy + oy), R, (40, 60, 20), -1, cv2.LINE_AA)
    overlay_draw(frame, (cx - R - 2, cy - R - 2, cx + R + 3, cy + R + 3), 0.55, paint_disc)

    for r in (R // 3, 2 * R // 3, R):
        cv2.circle(frame, (cx, cy), r, (90, 150, 70), 1, cv2.LINE_AA)
    cv2.line(frame, (cx - R, cy), (cx + R, cy), (90, 150, 70), 1)
    cv2.line(frame, (cx, cy - R), (cx, cy + R), (90, 150, 70), 1)

    base = (now * 120) % 360
    for k in range(0, 45, 3):
        a = math.radians(base - k)
        f = 1 - k / 45
        col = tuple(int(c * f) for c in C_RADAR)
        cv2.line(frame, (cx, cy), (int(cx + R * math.cos(a)), int(cy - R * math.sin(a))), col, 1, cv2.LINE_AA)

    def to_radar(px, py):
        px = min(max(px, 0), w)
        py = min(max(py, 0), h)
        return (int(cx + (px / w - 0.5) * 2 * R * 0.9), int(cy + (py / h - 0.5) * 2 * R * 0.9))

    for tr in tracks.values():
        if tr.confirmed:
            cv2.circle(frame, to_radar(tr.x, tr.y), 3, C_TRACK, -1, cv2.LINE_AA)

    if locked is not None:
        gone = locked.out_of_view(now)
        color = C_BAD if gone else (C_COAST if locked.coasting else RET_GREEN)
        pts = list(locked.trail)
        for k, (px, py) in enumerate(pts[:-1]):
            f = (k + 1) / len(pts)
            cv2.circle(frame, to_radar(px, py), 1, tuple(int(c * f) for c in color), -1)
        blip = to_radar(locked.x, locked.y)
        pulse = 6 + int(3 * abs(math.sin(now * 4)))
        cv2.circle(frame, blip, pulse, color, 1, cv2.LINE_AA)
        if not gone:
            cv2.circle(frame, blip, 4, color, -1, cv2.LINE_AA)
            mean, _ = locked.kf.project(PREDICT_SECONDS)
            fut = to_radar(mean[0], mean[1])
            cv2.line(frame, blip, fut, C_PRED, 1, cv2.LINE_AA)
            cv2.circle(frame, fut, 4, C_PRED, 1, cv2.LINE_AA)
    cv2.putText(frame, "RADAR", (cx - R, cy - R - 8), FONT, 0.5, C_RADAR, 1, cv2.LINE_AA)


def draw_probability_panel(frame, probs):
    h, w = frame.shape[:2]
    x0, y0, pw = w - 240, 12, 228
    ph = 34 + 9 * 19
    blend_rect(frame, x0, y0, x0 + pw, y0 + ph, C_PANEL, 0.66)
    cv2.rectangle(frame, (x0, y0), (x0 + pw, y0 + ph), C_TRACK, 1, cv2.LINE_AA)
    cv2.putText(frame, "HEADING PROBABILITY", (x0 + 10, y0 + 20), FONT, 0.5, C_PRED, 1, cv2.LINE_AA)
    top = int(np.argmax(probs))
    for i, label in enumerate(LABELS):
        y = y0 + 34 + i * 19
        color = C_LOCK if i == top else C_TRACK
        cv2.putText(frame, label, (x0 + 10, y + 11), FONT, 0.42, C_TEXT, 1, cv2.LINE_AA)
        cv2.rectangle(frame, (x0 + 100, y), (x0 + 100 + 80, y + 12), (70, 60, 50), -1)
        cv2.rectangle(frame, (x0 + 100, y), (x0 + 100 + int(probs[i] * 80), y + 12), color, -1)
        cv2.putText(frame, f"{probs[i] * 100:.0f}%", (x0 + 188, y + 11), FONT, 0.42, C_TEXT, 1, cv2.LINE_AA)


def draw_corners(frame):
    h, w = frame.shape[:2]
    brackets(frame, 8, 8, w - 8, h - 8, (0, 170, 0), 2, 34)


def unlock_button_rect(h):
    return (16, h - 96, 150, h - 64)


def draw_unlock_button(frame):
    x1, y1, x2, y2 = unlock_button_rect(frame.shape[0])
    blend_rect(frame, x1, y1, x2, y2, C_BTN, 0.9)
    cv2.rectangle(frame, (x1, y1), (x2, y2), C_TEXT, 1, cv2.LINE_AA)
    cv2.putText(frame, "UNLOCK  (U)", (x1 + 12, y2 - 10), FONT, 0.55, C_TEXT, 2, cv2.LINE_AA)


# ----------------------------------------------------------------------------- interaction
def confirmed_tracks():
    return [t for t in tracks.values() if t.confirmed]


def biggest_track():
    cands = [t for t in confirmed_tracks() if not t.coasting]
    return max(cands, key=lambda t: t.bw * t.bh) if cands else None


def on_mouse(event, x, y, _flags, _param):
    global pending_click, pending_unlock
    if event == cv2.EVENT_LBUTTONDOWN:
        pending_click = (x, y)
    elif event == cv2.EVENT_RBUTTONDOWN:
        pending_unlock = True


def lock_from_point(clean, x, y, now):
    """Lock only on a MOVING object under / near the click. Still objects are not lockable."""
    if locked is not None:
        say("Already locked - press U (or click UNLOCK) before choosing another object", 3.5)
        return
    best, best_d = None, 120
    for tr in confirmed_tracks():
        x1, y1, x2, y2 = tr.box()
        d = 0 if (x1 <= x <= x2 and y1 <= y <= y2) else math.hypot(x - tr.x, y - tr.y)
        if d < best_d:
            best, best_d = tr.id, d
    if best is None:
        say("No MOVING object there - click on an object while it is moving", 3.5)
        return
    tr = tracks[best]
    x1, y1, x2, y2 = tr.box()
    if lock_on(clean, (x1, y1, x2 - x1, y2 - y1), now, True):
        tracks.pop(best, None)


def new_subtractor():
    return cv2.createBackgroundSubtractorMOG2(history=300, varThreshold=40, detectShadows=True)


def set_fullscreen(win, on):
    cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN,
                          cv2.WINDOW_FULLSCREEN if on else cv2.WINDOW_NORMAL)


def main():
    global locked, pending_click, pending_unlock, BW_VIEW
    global SHOW_HUD, SHOW_RADAR
    cam = CameraStream(CAMERA_INDEX)
    if not cam.start():
        print("Could not open camera. Close other apps using the webcam and try again.")
        cam.release()
        return

    win = "Motion Radar Tracker (Q to quit)"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    fullscreen = START_FULLSCREEN
    if fullscreen:
        set_fullscreen(win, True)
    else:
        cv2.resizeWindow(win, 1280, 720)
    cv2.setMouseCallback(win, on_mouse)

    sub = new_subtractor()
    frame_count = 0
    last = clock()
    last_cnt = 0
    fps = 0.0
    prob_s = None
    clean = None
    h = w = 0

    while True:
        key = cv2.waitKey(1) & 0xFF

        frame, last_cnt = cam.read(last_cnt)
        if frame is None and clean is None:
            if cam.dead():
                print("Camera returned no frame.")
                break
        elif frame is not None:
            if MIRROR:
                frame = cv2.flip(frame, 1)
            h, w = frame.shape[:2]
            if w > MAX_WIDTH:
                frame = cv2.resize(frame, (MAX_WIDTH, int(h * MAX_WIDTH / w)), interpolation=cv2.INTER_AREA)
                h, w = frame.shape[:2]
            now = clock()
            dt = min(max(now - last, 0.005), 0.2)
            last = now
            fps = 0.9 * fps + 0.1 / dt if fps else 1.0 / dt
            frame_count += 1

            clean = frame.copy()
            step(clean, sub, frame_count, now, dt)
            if BW_VIEW:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                frame[:] = cv2.cvtColor(cv2.convertScaleAbs(gray, alpha=1.15, beta=-10), cv2.COLOR_GRAY2BGR)

            if SHOW_HUD:
                draw_corners(frame)
                draw_ignored_humans(frame)
            if locked is None:
                for tr in confirmed_tracks():
                    draw_other(frame, tr)
            else:
                draw_locked(frame, locked, now, dt)
                draw_unlock_button(frame)

            if SHOW_HUD and SHOW_RADAR:
                draw_radar(frame, now)

            if frame_count <= WARMUP_FRAMES:
                if SHOW_HUD:
                    chip(frame, "CALIBRATING - keep the scene still", (16, 36), C_DARK, C_PRED, 0.7, 2, 0.9, 7)
            elif locked is None:
                prob_s = None
                if SHOW_HUD:
                    chip(frame, f"SCANNING   moving objects: {len(confirmed_tracks())}", (16, 34), C_TEXT, C_PANEL, 0.6, 1)
                    chip(frame, "Click a MOVING object to LOCK it  |  L: biggest moving object", (16, 64), C_DIM, C_PANEL, 0.5, 1)
            elif locked.out_of_view(now):
                draw_banner(frame, "TARGET NOT LOCKED",
                            "Out of range - searching. It locks again automatically when it returns.   U = unlock",
                            C_BAD)
            else:
                probs = heading_probabilities(locked)
                prob_s = probs if prob_s is None else (1 - PROB_SMOOTH) * prob_s + PROB_SMOOTH * probs
                name, speed = heading_name(locked)
                top = int(np.argmax(prob_s))
                col = C_COAST if locked.coasting else RET_GREEN
                if locked.coasting:
                    title = "SEARCHING FOR TARGET"
                elif locked.occluded:
                    title = "TARGET LOCKED - object passing"
                else:
                    title = "TARGET LOCKED"
                if SHOW_HUD:
                    chip(frame, title, (16, 34), C_DARK if col == C_LOCK else C_TEXT, col, 0.65, 2, 0.92, 7)
                    chip(frame, f"match {max(locked.sim, 0) * 100:.0f}%   tracker {locked.tracker_name}",
                         (16, 64), C_TEXT, C_PANEL, 0.5, 1)
                    chip(frame, f"heading {name}   speed {speed:.0f} px/s", (16, 90), C_TEXT, C_PANEL, 0.5, 1)
                    chip(frame, f"most likely next: {LABELS[top]} ({prob_s[top] * 100:.0f}%)", (16, 116), C_PRED, C_PANEL, 0.5, 1)
                    draw_probability_panel(frame, prob_s)

            if now < status_until:
                chip(frame, status_msg, (16, h - 46), C_DARK, C_PRED, 0.6, 2, 0.92, 6)
            if SHOW_HUD:
                chip(frame, "Click:lock  L:biggest  U/right-click:unlock  R:reset  N:radar  T:hide all  B:B&W  F:full  Q:quit",
                     (16, h - 16), C_DIM, C_PANEL, 0.45, 1)
                chip(frame, f"{fps:.0f} FPS   radar {'ON' if SHOW_RADAR else 'OFF'}",
                     (16, 150), C_DIM, C_PANEL, 0.45, 1)
            else:
                chip(frame, f"{fps:.0f} FPS  (T = show overlay)", (16, 150), C_DIM, C_PANEL, 0.45, 1)

            cv2.imshow(win, frame)

        if clean is None:
            continue

        if pending_unlock:
            pending_unlock = False
            unlock()
        if pending_click is not None:
            x, y = pending_click
            pending_click = None
            bx1, by1, bx2, by2 = unlock_button_rect(h)
            if locked is not None and bx1 <= x <= bx2 and by1 <= y <= by2:
                unlock()
            else:
                lock_from_point(clean, x, y, clock())

        if key == ord("q"):
            break
        elif key == ord("u"):
            if locked is None:
                say("Nothing is locked")
            else:
                unlock()
        elif key == ord("r"):
            sub = new_subtractor()
            tracks.clear()
            locked = None
            frame_count = 0
            say("Background reset")
        elif key == ord("f"):
            fullscreen = not fullscreen
            set_fullscreen(win, fullscreen)
        elif key == ord("b"):
            BW_VIEW = not BW_VIEW
            say("Black & white view ON" if BW_VIEW else "Colour view ON")
        elif key == ord("n"):
            SHOW_RADAR = not SHOW_RADAR
            say("Radar ON" if SHOW_RADAR else "Radar OFF", 1.5)
        elif key == ord("t"):
            SHOW_HUD = not SHOW_HUD
            say("Overlay ON" if SHOW_HUD else "Overlay HIDDEN - tracking and clicking still active", 2.0)
        elif key == ord("l"):
            if locked is not None:
                say("Already locked - press U (or click UNLOCK) first", 3.5)
            else:
                t = biggest_track()
                if t is not None:
                    x1, y1, x2, y2 = t.box()
                    if lock_on(clean, (x1, y1, x2 - x1, y2 - y1), clock(), True):
                        tracks.pop(t.id, None)
                else:
                    say("No moving object to lock - move an object in view first", 3.5)

    cam.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        print("\nThe program stopped because of an error:\n")
        traceback.print_exc()
        input("\nPress Enter to close this window...")
