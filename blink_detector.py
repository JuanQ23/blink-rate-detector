#!/usr/bin/env python3
"""
Blink Rate Detector — GUI
=========================
Detects blinks from your webcam in real time and reports:
  • Total blinks
  • Blinks per minute
  • Blinks per hour (projected from the whole session)
  • Blinks per hour (rolling, based on the last 5 minutes — more stable)
  • A live plot of blinks per minute over time (plus a large plot
    window and a per-minute CSV export)
  • 20-20-20 break notifications: every 20 minutes of session time
    a popup reminds you to look at something 20+ feet away for
    20 seconds, with a countdown, snooze and skip
  • Presence detection: if no face is seen for more than a few
    seconds (5s grace by default) you are counted as "away". Away
    periods are shaded red on the plot, minutes that are mostly
    away are marked unreliable, away time is excluded from the
    rate calculations, and the 20-20-20 timer pauses while away.
  • Activity categories: tag what you are doing (video games,
    watching a show, YouTube, your own custom labels…) and the
    session is split into activity segments. You get a live
    blink-rate breakdown per activity, activity bands on the
    plot, and activity columns in the CSV exports.
  • Optional auto-detection (Windows): the foreground window's
    program and title are checked every ~2 seconds and matched
    to an activity (YouTube/Netflix/etc. in a browser, known
    coding and call apps, fullscreen non-browser app = video
    games). Titles are read locally and never stored.

How it works:
  MediaPipe Face Mesh locates 6 landmarks around each eye.
  The Eye Aspect Ratio (EAR) drops sharply when you blink.
  When EAR stays below a threshold for a few frames and then rises
  again, that counts as one blink.

Requirements:
  pip install -r requirements.txt

Run:
  python blink_detector.py

Notes:
  - All processing is local. No video is recorded or sent anywhere.
  - Good, even lighting and looking roughly toward the camera works best.
  - Normal resting blink rate is roughly 900–1,200/hour (15–20/min),
    and it usually falls a lot during concentrated screen work.
"""

from __future__ import annotations

import csv
import math
import time
from collections import deque
from datetime import datetime
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover
    Image = ImageTk = None

try:
    import mediapipe as mp
except ImportError:  # pragma: no cover
    mp = None


# MediaPipe Face Mesh landmark indices for the eyes
# Order: p1 (outer corner), p2 (upper), p3 (upper-inner),
#        p4 (inner corner), p5 (lower-inner), p6 (lower)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

# Activity categories: pick one (or type your own) to tag what you are
# doing; the session splits into segments and blink rates are reported
# per activity. Each activity gets a (plot band, legend) color pair,
# assigned in the order activities first appear.
ACTIVITY_PRESETS = ["Video games", "Watching a show", "YouTube", "Reading",
                    "Coding / work", "Video call", "Browsing", "Other"]
ACTIVITY_PALETTE = [("#dceeff", "#1f77b4"), ("#ffe9d6", "#e67e22"),
                    ("#e6f4d7", "#2e9e4f"), ("#f3e2fd", "#8e44ad"),
                    ("#fdf3c4", "#b7950b"), ("#d9f2f0", "#148f77"),
                    ("#ffe0ec", "#c2185b"), ("#e8e8e8", "#666666")]
DEFAULT_ACTIVITY = "Uncategorized"

# --- Auto-detection of the foreground window (Windows only) -------------
# Every couple of seconds the app can look at which window is in front
# (process name + window title, read locally via the Windows API —
# nothing is stored or sent anywhere) and classify it into an activity.

_BROWSER_PROCS = {"chrome.exe", "msedge.exe", "firefox.exe", "brave.exe",
                  "opera.exe", "opera_gx.exe", "vivaldi.exe", "arc.exe",
                  "chromium.exe"}
_CODE_PROCS = {"code.exe", "cursor.exe", "pycharm.exe", "pycharm64.exe",
               "idea.exe", "idea64.exe", "webstorm.exe", "webstorm64.exe",
               "rider.exe", "rider64.exe", "devenv.exe", "notepad++.exe",
               "sublime_text.exe", "atom.exe"}
_CALL_PROCS = {"zoom.exe", "ms-teams.exe", "teams.exe", "webex.exe",
               "skype.exe"}
_VIDEO_PLAYER_PROCS = {"vlc.exe", "mpc-hc.exe", "mpc-hc64.exe",
                       "potplayer.exe", "potplayer64.exe", "wmplayer.exe",
                       "mpv.exe", "kmplayer.exe"}
_SHOW_TITLE_KEYWORDS = ["netflix", "hulu", "disney+", "disney plus", "hbo",
                        "max.com", "prime video", "amazon prime", "peacock",
                        "paramount+", "apple tv+", "crunchyroll", "twitch",
                        "plex"]


def foreground_window_info():
    """(process_name, window_title, is_fullscreen) for the front window.

    Windows only; returns None anywhere else or on any error. Everything
    is read locally and used immediately — titles are never stored.
    """
    import sys
    if sys.platform != "win32":
        return None
    try:
        import ctypes
        from ctypes import wintypes

        user32 = ctypes.windll.user32
        kernel32 = ctypes.windll.kernel32
        hwnd = user32.GetForegroundWindow()
        if not hwnd:
            return None

        length = user32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buf, length + 1)
        title = buf.value or ""

        pid = wintypes.DWORD()
        user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        process = ""
        if pid.value:
            handle = kernel32.OpenProcess(0x1000, False, pid.value)  # QUERY_LIMITED
            if handle:
                try:
                    path_buf = ctypes.create_unicode_buffer(1024)
                    size = wintypes.DWORD(1024)
                    if kernel32.QueryFullProcessImageNameW(
                            handle, 0, path_buf, ctypes.byref(size)):
                        process = path_buf.value.replace("\\", "/").rsplit("/", 1)[-1]
                finally:
                    kernel32.CloseHandle(handle)

        rect = wintypes.RECT()
        fullscreen = False
        if user32.GetWindowRect(hwnd, ctypes.byref(rect)):
            screen_w = user32.GetSystemMetrics(0)
            screen_h = user32.GetSystemMetrics(1)
            fullscreen = (rect.left <= 0 and rect.top <= 0
                          and rect.right >= screen_w and rect.bottom >= screen_h)
        return (process, title, fullscreen)
    except Exception:
        return None


def classify_activity(process: str, title: str, fullscreen: bool):
    """Map a foreground window to an activity name, or None to keep the
    current activity (unknown windows never force a guess)."""
    p = (process or "").lower()
    t = (title or "").lower()
    if not p and not t:
        return None
    if p in ("python.exe", "pythonw.exe"):
        return None  # the detector's own window — never switch for it
    if "youtube" in t:
        return "YouTube"
    if any(k in t for k in _SHOW_TITLE_KEYWORDS):
        return "Watching a show"
    if p in _CALL_PROCS or "zoom meeting" in t or "teams meeting" in t:
        return "Video call"
    if p in _CODE_PROCS:
        return "Coding / work"
    if p in _VIDEO_PLAYER_PROCS:
        return "Watching a show"
    if p in _BROWSER_PROCS:
        return "Browsing"
    if fullscreen and p not in ("explorer.exe",):
        return "Video games"
    return None



def euclidean(a, b) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def eye_aspect_ratio(landmarks, indices, img_w: int, img_h: int) -> float:
    pts = [(landmarks[i].x * img_w, landmarks[i].y * img_h) for i in indices]
    p1, p2, p3, p4, p5, p6 = pts
    vertical = euclidean(p2, p6) + euclidean(p3, p5)
    horizontal = 2.0 * euclidean(p1, p4)
    return vertical / horizontal if horizontal else 0.0


class BlinkDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Blink Rate Detector")
        self.geometry("1180x800")
        self.minsize(1020, 680)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # --- State ---
        self.cap = None
        self.face_mesh = None
        self.running = False
        self.after_id = None

        self.total_blinks = 0
        # Blink times are stored in *session-elapsed* seconds (the clock
        # that freezes while the camera is paused), so pauses and away
        # periods line up with the plot and the per-minute buckets.
        self.blink_timestamps: list[float] = []
        self.session_start: float | None = None
        self.elapsed_before_pause = 0.0

        # --- Presence / away state (session-elapsed seconds) ---
        self.presence_enabled_var = tk.BooleanVar(value=True)
        self.away_grace_var = tk.IntVar(value=5)  # seconds before "away"
        self.away_intervals: list[tuple[float, float]] = []  # (start, end)
        self.away_start: float | None = None   # open (confirmed) away interval
        self.no_face_since: float | None = None  # first frame with no face

        self.ear_below_frames = 0
        self.ear_history: deque[float] = deque(maxlen=90)

        # --- Activity segments (session-elapsed seconds) ---
        # [start, end_or_None, name]; the open segment runs to "now".
        self.activity_var = tk.StringVar(value=DEFAULT_ACTIVITY)
        self.activity_segments: list[list] = []
        self._activity_color_idx: dict[str, int] = {}
        self._breakdown_text_cache = ""
        self.auto_activity_var = tk.BooleanVar(value=False)
        self._auto_tick_count = 0

        # Settings (bound to sliders)
        self.threshold_var = tk.DoubleVar(value=0.21)
        self.min_frames_var = tk.IntVar(value=2)
        self.camera_var = tk.IntVar(value=0)

        # --- 20-20-20 break state ---
        # Rule: every 20 minutes, look at something 20+ feet away for 20 seconds.
        self.break_enabled_var = tk.BooleanVar(value=True)
        self.break_interval_var = tk.IntVar(value=20)   # minutes between breaks
        self.break_duration_var = tk.IntVar(value=20)   # seconds per break
        self.breaks_taken = 0
        self.breaks_skipped = 0
        # Session-elapsed seconds at which the next break is due.
        self.next_break_at = 20 * 60.0
        self.break_window = None          # tk.Toplevel while a break popup is open
        self.break_countdown_left = 0
        self.break_countdown_id = None

        # --- Plot window state ---
        self.plot_window = None           # tk.Toplevel while the big plot is open
        self.plot_canvas = None
        self.plot_info_var = None

        self._build_ui()
        self._update_stats_loop()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        root = ttk.Frame(self, padding=10)
        root.pack(fill="both", expand=True)
        root.columnconfigure(0, weight=3)
        root.columnconfigure(1, weight=2)
        root.rowconfigure(0, weight=1)

        # Left: video
        left = ttk.LabelFrame(root, text="Camera", padding=8)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 8))

        self.video_label = tk.Label(
            left, text="Camera is off\n\nPress  Start Camera  to begin",
            bg="#111111", fg="#dddddd", font=("TkDefaultFont", 13)
        )
        self.video_label.pack(fill="both", expand=True)

        self.status_var = tk.StringVar(value="Status: Idle  •  No face detected")
        ttk.Label(left, textvariable=self.status_var, font=("TkDefaultFont", 10)
                  ).pack(anchor="w", pady=(8, 0))

        # EAR meter
        ttk.Label(left, text="Eye openness (EAR) — blink when it dips below the red line:"
                  ).pack(anchor="w", pady=(8, 2))
        self.ear_canvas = tk.Canvas(left, height=70, bg="white", highlightthickness=1,
                                    highlightbackground="#cccccc")
        self.ear_canvas.pack(fill="x")

        # Right: stats + controls (scrollable — there are a lot of panels)
        right_outer = ttk.Frame(root)
        right_outer.grid(row=0, column=1, sticky="nsew")
        right_canvas = tk.Canvas(right_outer, highlightthickness=0, width=390)
        right_scroll = ttk.Scrollbar(right_outer, orient="vertical",
                                     command=right_canvas.yview)
        right_canvas.configure(yscrollcommand=right_scroll.set)
        right_scroll.pack(side="right", fill="y")
        right_canvas.pack(side="left", fill="both", expand=True)
        right = ttk.Frame(right_canvas)
        right_win = right_canvas.create_window((0, 0), window=right, anchor="nw")
        right.bind("<Configure>", lambda e: right_canvas.configure(
            scrollregion=right_canvas.bbox("all")))
        right_canvas.bind("<Configure>", lambda e: right_canvas.itemconfigure(
            right_win, width=e.width))
        right_canvas.bind_all("<MouseWheel>", lambda e: right_canvas.yview_scroll(
            int(-e.delta / 120), "units"))
        right.columnconfigure(0, weight=1)

        stats = ttk.LabelFrame(right, text="Your blink rate", padding=10)
        stats.pack(fill="x")
        stats.columnconfigure(1, weight=1)

        self.total_var = tk.StringVar(value="0")
        self.elapsed_var = tk.StringVar(value="00:00:00")
        self.bpm_var = tk.StringVar(value="—")
        self.bph_projected_var = tk.StringVar(value="—")
        self.bph_rolling_var = tk.StringVar(value="—")
        self.last_min_var = tk.StringVar(value="0")
        self.breaks_taken_var = tk.StringVar(value="0")
        self.next_break_var = tk.StringVar(value="starts with camera")
        self.away_var = tk.StringVar(value="00:00")

        rows = [
            ("Total blinks:", self.total_var, 16),
            ("Session time:", self.elapsed_var, 12),
            ("Time away (excluded):", self.away_var, 12),
            ("Blinks / minute:", self.bpm_var, 16),
            ("Blinks / hour (projected):", self.bph_projected_var, 16),
            ("Blinks / hour (last 5 min):", self.bph_rolling_var, 16),
            ("Blinks in last 60 seconds:", self.last_min_var, 12),
            ("20-20-20 breaks taken:", self.breaks_taken_var, 12),
            ("Next 20-20-20 break in:", self.next_break_var, 12),
        ]
        for i, (label, var, size) in enumerate(rows):
            ttk.Label(stats, text=label).grid(row=i, column=0, sticky="w", pady=3)
            ttk.Label(stats, textvariable=var,
                      font=("TkDefaultFont", size, "bold")).grid(row=i, column=1, sticky="e", pady=3)

        ttk.Label(stats,
                  text="All rates use time present only — away time is excluded.\n"
                       "Projected = whole-session average scaled to 1 hour.\n"
                       "Last 5 min = recent rate over present time in that window.",
                  foreground="#666666", wraplength=360, justify="left"
                  ).grid(row=len(rows), column=0, columnspan=2, sticky="w", pady=(6, 0))

        # Per-minute plot (line chart over time)
        chart_box = ttk.LabelFrame(right, text="Blinks per minute over time", padding=8)
        chart_box.pack(fill="x", pady=(10, 0))
        self.chart_canvas = tk.Canvas(chart_box, height=140, bg="white",
                                      highlightthickness=1, highlightbackground="#cccccc")
        self.chart_canvas.pack(fill="x")
        plot_btns = ttk.Frame(chart_box)
        plot_btns.pack(fill="x", pady=(6, 0))
        ttk.Button(plot_btns, text="Open full plot", command=self.open_plot_window).pack(
            side="left", fill="x", expand=True)
        ttk.Button(plot_btns, text="Save per-minute data (CSV)…",
                   command=self.save_per_minute_csv).pack(
            side="left", fill="x", expand=True, padx=(6, 0))

        # Activity categories
        activity_box = ttk.LabelFrame(right, text="What are you doing?", padding=10)
        activity_box.pack(fill="x", pady=(10, 0))
        self.activity_combo = ttk.Combobox(
            activity_box, textvariable=self.activity_var,
            values=ACTIVITY_PRESETS)
        self.activity_combo.pack(fill="x")
        self.activity_combo.bind("<<ComboboxSelected>>", self._on_activity_changed)
        self.activity_combo.bind("<Return>", self._on_activity_changed)
        self.activity_combo.bind("<FocusOut>", self._on_activity_changed)
        ttk.Checkbutton(activity_box, text="Auto-detect from the active window (Windows)",
                        variable=self.auto_activity_var).pack(anchor="w", pady=(6, 0))
        ttk.Label(activity_box,
                  text="Pick what you're doing — or type your own label and\n"
                       "press Enter. Switching starts a new segment, and the\n"
                       "breakdown below keeps a separate blink rate for each\n"
                       "activity. Away time is not counted against any of them.",
                  foreground="#666666", justify="left").pack(anchor="w", pady=(6, 0))

        breakdown_box = ttk.LabelFrame(right, text="Blink rate by activity", padding=8)
        breakdown_box.pack(fill="x", pady=(10, 0))
        self.breakdown_text = tk.Text(breakdown_box, height=6, wrap="word",
                                      relief="flat", bg="#f7f7f7",
                                      font=("TkDefaultFont", 10))
        self.breakdown_text.pack(fill="x")
        self.breakdown_text.insert("1.0", "Start the camera and pick what you're doing.")
        self.breakdown_text.config(state="disabled")

        # Controls
        controls = ttk.LabelFrame(right, text="Controls", padding=10)
        controls.pack(fill="x", pady=(10, 0))

        btn_row = ttk.Frame(controls)
        btn_row.pack(fill="x")
        self.start_btn = ttk.Button(btn_row, text="▶  Start Camera", command=self.start_camera)
        self.start_btn.pack(side="left", fill="x", expand=True)
        self.stop_btn = ttk.Button(btn_row, text="■  Stop", command=self.stop_camera, state="disabled")
        self.stop_btn.pack(side="left", fill="x", expand=True, padx=(6, 0))

        btn_row2 = ttk.Frame(controls)
        btn_row2.pack(fill="x", pady=(6, 0))
        ttk.Button(btn_row2, text="Reset stats", command=self.reset_stats).pack(
            side="left", fill="x", expand=True)
        ttk.Button(btn_row2, text="Save log (CSV)…", command=self.save_csv).pack(
            side="left", fill="x", expand=True, padx=(6, 0))

        # Settings
        settings = ttk.LabelFrame(right, text="Settings", padding=10)
        settings.pack(fill="x", pady=(10, 0))

        ttk.Label(settings, text="Blink threshold (EAR)").pack(anchor="w")
        thr_row = ttk.Frame(settings)
        thr_row.pack(fill="x")
        ttk.Scale(thr_row, from_=0.12, to=0.35, variable=self.threshold_var,
                  orient="horizontal").pack(side="left", fill="x", expand=True)
        self.thr_label = ttk.Label(thr_row, text="0.21", width=5)
        self.thr_label.pack(side="left", padx=(6, 0))
        ttk.Label(settings,
                  text="Lower = less sensitive. Raise it if blinks are missed,\n"
                       "lower it if normal eye movements are counted.",
                  foreground="#666666", justify="left").pack(anchor="w", pady=(2, 8))

        ttk.Label(settings, text="Min. closed frames (higher = fewer false positives)").pack(anchor="w")
        ttk.Spinbox(settings, from_=1, to=6, textvariable=self.min_frames_var,
                    width=6).pack(anchor="w", pady=(2, 8))

        cam_row = ttk.Frame(settings)
        cam_row.pack(fill="x")
        ttk.Label(cam_row, text="Camera index:").pack(side="left")
        ttk.Spinbox(cam_row, from_=0, to=4, textvariable=self.camera_var,
                    width=6).pack(side="left", padx=(6, 0))
        ttk.Label(cam_row, text="(try 1 if 0 shows the wrong camera)",
                  foreground="#666666").pack(side="left", padx=(6, 0))

        # 20-20-20 breaks
        breaks = ttk.LabelFrame(right, text="20-20-20 eye breaks", padding=10)
        breaks.pack(fill="x", pady=(10, 0))

        ttk.Checkbutton(breaks, text="Enable break notifications",
                        variable=self.break_enabled_var).pack(anchor="w")
        ttk.Label(breaks,
                  text="Every 20 minutes of session time, a popup will remind\n"
                       "you to look at something at least 20 feet (~6 m) away\n"
                       "for 20 seconds. The timer pauses when the camera is\n"
                       "paused, and the popup beeps and comes to the front.",
                  foreground="#666666", justify="left").pack(anchor="w", pady=(4, 8))

        brk_row = ttk.Frame(breaks)
        brk_row.pack(fill="x")
        ttk.Label(brk_row, text="Every").pack(side="left")
        ttk.Spinbox(brk_row, from_=1, to=60, textvariable=self.break_interval_var,
                    width=5).pack(side="left", padx=(4, 4))
        ttk.Label(brk_row, text="min, look away for").pack(side="left")
        ttk.Spinbox(brk_row, from_=5, to=120, textvariable=self.break_duration_var,
                    width=5).pack(side="left", padx=(4, 4))
        ttk.Label(brk_row, text="sec").pack(side="left")
        ttk.Label(breaks, text="(the rule is 20 / 20 — set the interval to 1 min to test it)",
                  foreground="#666666").pack(anchor="w", pady=(4, 8))

        brk_btns = ttk.Frame(breaks)
        brk_btns.pack(fill="x")
        ttk.Button(brk_btns, text="Take a break now", command=self.open_break_window).pack(
            side="left", fill="x", expand=True)
        ttk.Button(brk_btns, text="Reset break timer", command=self.reset_break_timer).pack(
            side="left", fill="x", expand=True, padx=(6, 0))

        # Presence detection
        presence = ttk.LabelFrame(right, text="Presence detection", padding=10)
        presence.pack(fill="x", pady=(10, 0))
        ttk.Checkbutton(presence, text="Detect when I step away",
                        variable=self.presence_enabled_var).pack(anchor="w")
        ttk.Label(presence,
                  text="If your face isn't seen for longer than the grace period,\n"
                       "you're counted as away: that stretch is shaded red on\n"
                       "the plot, minutes that are mostly away are marked\n"
                       "unreliable (red ✕, line breaks), away time is taken out\n"
                       "of the rate calculations, and the break timer pauses.",
                  foreground="#666666", justify="left").pack(anchor="w", pady=(4, 8))
        grace_row = ttk.Frame(presence)
        grace_row.pack(fill="x")
        ttk.Label(grace_row, text="Grace period before 'away':").pack(side="left")
        ttk.Spinbox(grace_row, from_=2, to=30, textvariable=self.away_grace_var,
                    width=5).pack(side="left", padx=(6, 4))
        ttk.Label(grace_row, text="seconds").pack(side="left")

    # ------------------------------------------------------------- Camera
    def start_camera(self):
        if cv2 is None or Image is None or mp is None:
            missing = []
            if cv2 is None:
                missing.append("opencv-python")
            if Image is None:
                missing.append("Pillow")
            if mp is None:
                missing.append("mediapipe")
            messagebox.showerror(
                "Missing packages",
                "These packages are not installed:\n\n"
                + ", ".join(missing)
                + "\n\nInstall them with:\n  pip install -r requirements.txt"
            )
            return

        self.cap = cv2.VideoCapture(int(self.camera_var.get()))
        if not self.cap.isOpened():
            messagebox.showerror(
                "Camera not found",
                f"Could not open camera {self.camera_var.get()}.\n"
                "Check it's not in use by another app, or try a different camera index."
            )
            self.cap = None
            return

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        if self.session_start is None:
            self.session_start = time.monotonic()
        else:
            # Resuming after a pause
            self.session_start = time.monotonic() - self.elapsed_before_pause

        self.running = True
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        # Open an activity segment unless one is already open (segments
        # stay open across pauses — the session clock freezes with them)
        if not self.activity_segments or self.activity_segments[-1][1] is not None:
            self._open_activity_segment(self.elapsed_seconds())
        self._video_loop()

    def stop_camera(self):
        self.running = False
        if self.after_id:
            self.after_cancel(self.after_id)
            self.after_id = None
        if self.session_start is not None:
            self.elapsed_before_pause = time.monotonic() - self.session_start
        # A deliberate pause is not "away" — close any open away interval here
        if self.away_start is not None:
            self.away_intervals.append((self.away_start, self.elapsed_before_pause))
            self.away_start = None
        self.no_face_since = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.face_mesh is not None:
            self.face_mesh.close()
            self.face_mesh = None
        self.video_label.config(image="", text="Camera paused\n\nPress Start Camera to resume")
        self.video_label.image = None
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Status: Paused")

    def _video_loop(self):
        if not self.running or self.cap is None:
            return

        ok, frame = self.cap.read()
        if not ok:
            self.status_var.set("Status: Camera read failed — stopping")
            self.stop_camera()
            return

        frame = cv2.flip(frame, 1)  # mirror view, feels natural
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.face_mesh.process(rgb)

        ear = None
        face_present = bool(result.multi_face_landmarks)
        if face_present:
            landmarks = result.multi_face_landmarks[0].landmark
            left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
            right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
            ear = (left_ear + right_ear) / 2.0

            self._process_ear(ear)

            # Draw eye outlines
            for indices in (LEFT_EYE, RIGHT_EYE):
                for idx in indices:
                    lm = landmarks[idx]
                    cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2, (0, 255, 0), -1)

            state = "BLINK" if ear < self.threshold_var.get() else "Open"
            self.status_var.set(f"Status: Face detected  •  EAR {ear:.3f}  •  {state}")
        else:
            self.status_var.set("Status: No face detected — look toward the camera")

        self._update_presence(face_present)
        if self.away_start is not None:
            away_for = int(self.elapsed_seconds() - self.away_start)
            self.status_var.set(
                f"Status: Away — no face for {away_for}s "
                f"(this time is shaded red and excluded from your rates)")

        if ear is not None:
            self.ear_history.append(ear)
        self._draw_ear_chart()

        # Overlay blink count on the video itself
        cv2.putText(frame, f"Blinks: {self.total_blinks}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

        # Fit frame into the label
        display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(display)
        label_w = max(self.video_label.winfo_width(), 320)
        label_h = max(self.video_label.winfo_height(), 240)
        img.thumbnail((label_w, label_h))
        photo = ImageTk.PhotoImage(image=img)
        self.video_label.config(image=photo, text="")
        self.video_label.image = photo

        self.after_id = self.after(15, self._video_loop)

    # -------------------------------------------------------------- Logic
    def _process_ear(self, ear: float):
        threshold = float(self.threshold_var.get())
        if ear < threshold:
            self.ear_below_frames += 1
        else:
            # Eyes reopened after being closed long enough -> one blink
            if self.ear_below_frames >= int(self.min_frames_var.get()):
                self.total_blinks += 1
                self.blink_timestamps.append(self.elapsed_seconds())
            self.ear_below_frames = 0

    # ---------------------------------------------------------- Presence
    def _update_presence(self, face_present: bool):
        """Track away intervals (session-elapsed seconds).

        A no-face stretch only becomes an away interval once it outlasts
        the grace period; the interval then starts at the first no-face
        frame, because that data really is missing.
        """
        elapsed = self.elapsed_seconds()
        if not self.presence_enabled_var.get():
            # Turning detection off closes any open interval at this moment
            if self.away_start is not None:
                self.away_intervals.append((self.away_start, elapsed))
                self.away_start = None
            self.no_face_since = None
            return

        if face_present:
            if self.away_start is not None:
                start = self.away_start
                self.away_intervals.append((start, elapsed))
                self.away_start = None
                # Pause the 20-20-20 timer for the time spent away
                self.next_break_at += elapsed - start
            self.no_face_since = None
            return

        if self.no_face_since is None:
            self.no_face_since = elapsed
            return
        if self.away_start is None:
            try:
                grace = int(self.away_grace_var.get())
            except (tk.TclError, ValueError):
                grace = 5
            grace = max(min(grace, 30), 2)
            if elapsed - self.no_face_since >= grace:
                self.away_start = self.no_face_since

    def _away_overlap(self, start: float, end: float, elapsed: float) -> float:
        """Seconds of away time inside [start, end], incl. an open interval."""
        total = 0.0
        for a, b in self.away_intervals:
            total += max(0.0, min(b, end) - max(a, start))
        if self.away_start is not None:
            total += max(0.0, min(elapsed, end) - max(self.away_start, start))
        return total

    def away_seconds_total(self, elapsed: float) -> float:
        return self._away_overlap(0.0, elapsed, elapsed)

    def away_seconds_per_minute(self, elapsed: float) -> list[float]:
        """Away seconds in each session minute, aligned with the plot series."""
        n = int(elapsed // 60) + 1
        return [self._away_overlap(m * 60.0, min((m + 1) * 60.0, elapsed), elapsed)
                for m in range(n)]

    def unreliable_flags(self, elapsed: float,
                         away_per_minute: list[float]) -> list[bool]:
        """A minute is unreliable if away time covers more than half of it.

        For the current (partial) minute, 'half' means half of the part
        that has actually elapsed.
        """
        flags = []
        for m, away_s in enumerate(away_per_minute):
            minute_start = m * 60.0
            minute_len = min(60.0, max(elapsed - minute_start, 0.0))
            flags.append(minute_len > 0 and away_s > minute_len / 2)
        return flags

    # ---------------------------------------------------------- Activities
    def _activity_name(self) -> str:
        return self.activity_var.get().strip() or DEFAULT_ACTIVITY

    def _activity_colors(self, name: str) -> tuple[str, str]:
        if name not in self._activity_color_idx:
            self._activity_color_idx[name] = len(self._activity_color_idx)
        return ACTIVITY_PALETTE[self._activity_color_idx[name] % len(ACTIVITY_PALETTE)]

    def _open_activity_segment(self, at: float):
        name = self._activity_name()
        self._activity_colors(name)  # assign its color in first-use order
        self.activity_segments.append([at, None, name])

    def _on_activity_changed(self, _event=None):
        """Switching activity closes the open segment and starts a new one."""
        if self.session_start is None:
            return  # no session yet — the segment opens when the camera starts
        name = self._activity_name()
        if self.activity_segments and self.activity_segments[-1][1] is None:
            if self.activity_segments[-1][2] == name:
                return
            elapsed = self.elapsed_seconds()
            self.activity_segments[-1][1] = elapsed
            self.activity_segments.append([elapsed, None, name])
            self._activity_colors(name)
        elif not self.activity_segments:
            self._open_activity_segment(self.elapsed_seconds())

    def _auto_detect_tick(self):
        """Poll the foreground window (~every 2s) and switch activity."""
        if not self.auto_activity_var.get():
            return
        if not self.running or self.session_start is None:
            return
        self._auto_tick_count += 1
        if self._auto_tick_count % 4 != 0:  # stats loop runs every 500 ms
            return
        info = foreground_window_info()
        if not info:
            return
        name = classify_activity(*info)
        if name and name != self._activity_name():
            self.activity_var.set(name)
            self._on_activity_changed()

    def activity_at(self, t: float, elapsed: float) -> str:
        for start, end, name in self.activity_segments:
            seg_end = elapsed if end is None else end
            if start <= t < seg_end or (t == seg_end == elapsed):
                return name
        return DEFAULT_ACTIVITY

    def activity_stats(self, elapsed: float) -> list[dict]:
        """Per-activity totals, merged by name, in first-appearance order.

        Durations are *present* time (away time inside a segment is not
        counted against that activity).
        """
        order: list[str] = []
        agg: dict[str, dict] = {}
        for start, end, name in self.activity_segments:
            seg_end = elapsed if end is None else min(end, elapsed)
            if seg_end <= start:
                continue
            present = (seg_end - start) - self._away_overlap(start, seg_end, elapsed)
            blinks = sum(1 for t in self.blink_timestamps if start <= t < seg_end)
            if name not in agg:
                agg[name] = {"name": name, "seconds": 0.0, "blinks": 0}
                order.append(name)
            agg[name]["seconds"] += max(present, 0.0)
            agg[name]["blinks"] += blinks
        return [agg[n] for n in order if agg[n]["seconds"] > 0 or agg[n]["blinks"] > 0]

    def _activity_bands(self, elapsed: float, window_start_min: float,
                        window_len_min: float) -> list[tuple[float, float, str]]:
        """(start, end, band color) in plot-window minute coordinates."""
        bands = []
        win_start_s = window_start_min * 60.0
        win_end_s = win_start_s + window_len_min * 60.0
        for start, end, name in self.activity_segments:
            seg_end = elapsed if end is None else end
            lo, hi = max(start, win_start_s), min(seg_end, win_end_s)
            if hi > lo:
                bands.append(((lo - win_start_s) / 60.0, (hi - win_start_s) / 60.0,
                              self._activity_colors(name)[0]))
        return bands

    def _dominant_activity(self, minute: int, elapsed: float) -> str:
        """Activity covering the most of session minute `minute` (0-based)."""
        lo, hi = minute * 60.0, min((minute + 1) * 60.0, elapsed)
        best, best_len = DEFAULT_ACTIVITY, 0.0
        for start, end, name in self.activity_segments:
            seg_end = elapsed if end is None else end
            overlap = min(seg_end, hi) - max(start, lo)
            if overlap > best_len:
                best, best_len = name, overlap
        return best

    def _update_breakdown_panel(self, elapsed: float):
        stats = self.activity_stats(elapsed)
        if not stats:
            text = "Start the camera and pick what you're doing."
            lines = [(text, None)]
        else:
            lines = []
            for s in stats:
                mins = s["seconds"] / 60.0
                bpm = s["blinks"] / mins if mins > 0 else 0.0
                dur_m, dur_s = int(s["seconds"] // 60), int(s["seconds"] % 60)
                lines.append((f"■ {s['name']} — {dur_m}:{dur_s:02d} • "
                              f"{s['blinks']} blinks • {bpm:.1f}/min "
                              f"({bpm * 60:,.0f}/hr)", s["name"]))
            text = "\n".join(l for l, _ in lines)
        if text == self._breakdown_text_cache:
            return
        self._breakdown_text_cache = text
        self.breakdown_text.config(state="normal")
        self.breakdown_text.delete("1.0", "end")
        for i, (line, name) in enumerate(lines):
            start = self.breakdown_text.index("end-1c")
            self.breakdown_text.insert("end", line + ("\n" if i < len(lines) - 1 else ""))
            if name is not None:
                tag = f"act_{self._activity_color_idx[name]}"
                self.breakdown_text.tag_config(tag, foreground=self._activity_colors(name)[1])
                self.breakdown_text.tag_add(tag, start, f"{start} + 1 chars")
        self.breakdown_text.config(state="disabled")

    def elapsed_seconds(self) -> float:
        if self.session_start is None:
            return self.elapsed_before_pause
        if self.running:
            return time.monotonic() - self.session_start
        return self.elapsed_before_pause

    def _update_stats_loop(self):
        elapsed = self.elapsed_seconds()
        hours = int(elapsed // 3600)
        mins = int((elapsed % 3600) // 60)
        secs = int(elapsed % 60)
        self.elapsed_var.set(f"{hours:02d}:{mins:02d}:{secs:02d}")
        self.total_var.set(str(self.total_blinks))
        self.thr_label.config(text=f"{self.threshold_var.get():.2f}")

        # Blink timestamps are session-elapsed seconds, so windows are too.
        last_60 = sum(1 for t in self.blink_timestamps if elapsed - t <= 60)
        last_300 = sum(1 for t in self.blink_timestamps if elapsed - t <= 300)
        self.last_min_var.set(str(last_60))

        # Away time is excluded from every rate denominator
        away_total = self.away_seconds_total(elapsed)
        away_m, away_s = int(away_total // 60), int(away_total % 60)
        self.away_var.set(f"{away_m:02d}:{away_s:02d}" if self.away_start is None
                          else f"{away_m:02d}:{away_s:02d} (away now)")
        present = max(elapsed - away_total, 0.0)

        if present >= 10:
            self.bpm_var.set(f"{self.total_blinks / (present / 60):.1f}")
            self.bph_projected_var.set(f"{self.total_blinks / (present / 3600):,.0f}")
        else:
            self.bpm_var.set("—")
            self.bph_projected_var.set("—")

        # Rolling rate: blinks in the last 300 session seconds, divided by
        # the *present* time inside that same window
        window_len = min(elapsed, 300)
        away_in_window = self._away_overlap(elapsed - window_len, elapsed, elapsed)
        present_window = window_len - away_in_window
        if present_window >= 30:
            self.bph_rolling_var.set(f"{last_300 / (present_window / 3600):,.0f}")
        else:
            self.bph_rolling_var.set("collecting…")

        # 20-20-20 break display + timer
        self.breaks_taken_var.set(
            f"{self.breaks_taken}" + (f"  (skipped {self.breaks_skipped})"
                                      if self.breaks_skipped else ""))
        if not self.break_enabled_var.get():
            self.next_break_var.set("off")
        elif self.break_window is not None:
            self.next_break_var.set("break time!")
        elif self.session_start is None:
            self.next_break_var.set("starts with camera")
        elif self.away_start is not None:
            # Timer is paused while away — show the frozen remaining time
            remaining = max(self.next_break_at - self.away_start, 0)
            self.next_break_var.set(f"{int(remaining // 60)}:{int(remaining % 60):02d} (paused)")
        else:
            remaining = max(self.next_break_at - elapsed, 0)
            self.next_break_var.set(f"{int(remaining // 60)}:{int(remaining % 60):02d}")
        self._check_break_timer(elapsed)
        self._auto_detect_tick()
        self._update_breakdown_panel(elapsed)

        self._draw_minute_chart(elapsed)
        self._refresh_plot_window(elapsed)
        self.after(500, self._update_stats_loop)

    # ------------------------------------------------------------- Charts
    def _draw_ear_chart(self):
        c = self.ear_canvas
        c.delete("all")
        w = c.winfo_width() or 400
        h = c.winfo_height() or 70
        if len(self.ear_history) < 2:
            return
        # EAR range 0.05–0.45 mapped to canvas
        def y_of(v):
            return h - ((v - 0.05) / 0.40) * (h - 8) - 4

        thr_y = y_of(float(self.threshold_var.get()))
        c.create_line(0, thr_y, w, thr_y, fill="red", dash=(4, 3))
        points = []
        n = len(self.ear_history)
        for i, v in enumerate(self.ear_history):
            x = (i / max(n - 1, 1)) * (w - 4) + 2
            points.extend([x, y_of(v)])
        if len(points) >= 4:
            c.create_line(points, fill="#1f77b4", width=2, smooth=True)

    # ------------------------------------------------------ Plot helpers
    def per_minute_series(self, elapsed: float) -> list[int]:
        """Blinks in each session minute: [minute 1, minute 2, ...].

        The last entry is the current, still-running minute (partial).
        Blink timestamps are session-elapsed seconds, so bucketing is direct.
        """
        if self.session_start is None and not self.blink_timestamps:
            return []
        current_minute = int(elapsed // 60)
        buckets: dict[int, int] = {}
        for t in self.blink_timestamps:
            minute = int(t // 60)
            if 0 <= minute <= current_minute:
                buckets[minute] = buckets.get(minute, 0) + 1
        return [buckets.get(m, 0) for m in range(current_minute + 1)]

    @staticmethod
    def _draw_line_plot(canvas, values: list[int], first_minute: int = 1,
                        empty_text: str = "",
                        away_per_minute: list[float] | None = None,
                        unreliable: list[bool] | None = None,
                        activity_bands: list[tuple[float, float, str]] | None = None):
        """Draw values as a line plot: x = session minute, y = blinks/min.

        Background bands show which activity was happening (colors match
        the "Blink rate by activity" panel). Minutes with away time get
        a pale-red background column on top of the band. Minutes flagged
        unreliable (mostly away) get a red ✕ instead of a point, and
        the line is broken around them — a fake '0 blinks' dip would
        be misleading.
        """
        c = canvas
        c.delete("all")
        w = c.winfo_width() or 360
        h = c.winfo_height() or 140
        if not values:
            if empty_text:
                c.create_text(w // 2, h // 2, text=empty_text, fill="#888888")
            return

        left, right_pad, top, bottom = 34, 12, 10, 22
        plot_w = max(w - left - right_pad, 10)
        plot_h = max(h - top - bottom, 10)
        max_v = max(max(values), 10)
        # Round the y ceiling up to a tidy step
        step = 5 if max_v <= 30 else 10
        y_max = int(math.ceil(max_v / step)) * step

        def x_of(i):
            if len(values) == 1:
                return left + plot_w / 2
            return left + (i / (len(values) - 1)) * plot_w

        def y_of(v):
            return top + plot_h - (v / y_max) * plot_h

        # Gridlines + y labels
        for gv in range(0, y_max + 1, step):
            y = y_of(gv)
            c.create_line(left, y, left + plot_w, y, fill="#e3e3e3")
            c.create_text(left - 4, y, text=str(gv), anchor="e",
                          font=("TkDefaultFont", 7), fill="#666666")
        # Axes
        c.create_line(left, top, left, top + plot_h, fill="#999999")
        c.create_line(left, top + plot_h, left + plot_w, top + plot_h, fill="#999999")
        c.create_text(4, top + 2, text="blinks/min", anchor="w",
                      font=("TkDefaultFont", 7), fill="#666666")

        # X labels: first, middle, last minute
        label_idx = {0, len(values) - 1, (len(values) - 1) // 2}
        for i in sorted(label_idx):
            c.create_text(x_of(i), top + plot_h + 6, text=f"min {first_minute + i}",
                          font=("TkDefaultFont", 7), fill="#666666")

        # Reference band for a typical resting rate (15–20/min)
        if y_max >= 20:
            c.create_rectangle(left, y_of(20), left + plot_w, y_of(15),
                               fill="#eef7ee", outline="")
            c.create_text(left + plot_w - 2, y_of(20) + 2, text="typical 15–20",
                          anchor="ne", font=("TkDefaultFont", 7), fill="#7a9a7a")

        # Activity bands: colored background for each activity segment.
        # Band coordinates are in minutes relative to the first value.
        if activity_bands and len(values) > 1:
            for band_start, band_end, color in activity_bands:
                x0 = x_of(max(band_start, 0.0))
                x1 = x_of(min(band_end, float(len(values) - 1)))
                if x1 > x0:
                    c.create_rectangle(x0, top, x1, top + plot_h,
                                       fill=color, outline="")

        # Away shading: a pale-red column behind any minute with away time
        has_away = bool(away_per_minute) and any(a > 0.5 for a in away_per_minute)
        if has_away:
            for i, away_s in enumerate(away_per_minute):
                if away_s <= 0.5:
                    continue
                x_left = left if i == 0 else (x_of(i - 1) + x_of(i)) / 2
                x_right = (left + plot_w if i == len(values) - 1
                           else (x_of(i) + x_of(i + 1)) / 2)
                c.create_rectangle(x_left, top, x_right, top + plot_h,
                                   fill="#fbdcdc", outline="")
            c.create_text(left + plot_w - 2, top + 2, text="■ away",
                          anchor="ne", font=("TkDefaultFont", 7), fill="#c0392b")

        def is_unreliable(i):
            return bool(unreliable) and i < len(unreliable) and unreliable[i]

        # Line, drawn in segments so it breaks around unreliable minutes
        segment: list[float] = []
        for i, v in enumerate(values):
            if is_unreliable(i):
                if len(segment) >= 4:
                    c.create_line(segment, fill="#1f77b4", width=2, smooth=False)
                segment = []
                continue
            segment.extend([x_of(i), y_of(v)])
        if len(segment) >= 4:
            c.create_line(segment, fill="#1f77b4", width=2, smooth=False)

        # Points (the final, still-running minute is a hollow point;
        # unreliable minutes get a red ✕ at the bottom instead)
        for i, v in enumerate(values):
            x = x_of(i)
            if is_unreliable(i):
                y = top + plot_h - 8
                c.create_line(x - 4, y - 4, x + 4, y + 4, fill="#c0392b", width=2)
                c.create_line(x - 4, y + 4, x + 4, y - 4, fill="#c0392b", width=2)
                continue
            y = y_of(v)
            if i == len(values) - 1 and len(values) > 1:
                c.create_oval(x - 3, y - 3, x + 3, y + 3, outline="#1f77b4",
                              fill="white", width=2)
            else:
                c.create_oval(x - 2.5, y - 2.5, x + 2.5, y + 2.5,
                              fill="#1f77b4", outline="")
            if v:
                c.create_text(x, y - 9, text=str(v), font=("TkDefaultFont", 7),
                              fill="#333333")

    def _draw_minute_chart(self, elapsed: float):
        series = self.per_minute_series(elapsed)
        if not series:
            self._draw_line_plot(self.chart_canvas, [],
                                 empty_text="Start the camera to see your per-minute trend")
            return
        # Small chart: most recent 30 minutes (the full plot shows everything)
        away_pm = self.away_seconds_per_minute(elapsed)
        flags = self.unreliable_flags(elapsed, away_pm)
        window = series[-30:]
        win_start_min = float(len(series) - len(window))
        self._draw_line_plot(self.chart_canvas, window,
                             first_minute=len(series) - len(window) + 1,
                             away_per_minute=away_pm[-30:],
                             unreliable=flags[-30:],
                             activity_bands=self._activity_bands(
                                 elapsed, win_start_min, float(len(window))))

    # ------------------------------------------------------- Plot window
    def open_plot_window(self):
        if self.plot_window is not None and self.plot_window.winfo_exists():
            self.plot_window.lift()
            self.plot_window.focus_set()
            return
        win = tk.Toplevel(self)
        win.title("Blinks per minute over time")
        win.geometry("920x540")
        win.minsize(640, 380)

        ttk.Label(win, text="Blinks per minute over time",
                  font=("TkDefaultFont", 14, "bold")).pack(anchor="w", padx=12, pady=(10, 0))
        self.plot_info_var = tk.StringVar(value="")
        ttk.Label(win, textvariable=self.plot_info_var, foreground="#555555"
                  ).pack(anchor="w", padx=12, pady=(2, 6))

        canvas = tk.Canvas(win, bg="white", highlightthickness=1,
                           highlightbackground="#cccccc")
        canvas.pack(fill="both", expand=True, padx=12, pady=(0, 8))
        self.plot_canvas = canvas

        btns = ttk.Frame(win)
        btns.pack(fill="x", padx=12, pady=(0, 10))
        ttk.Button(btns, text="Save per-minute data (CSV)…",
                   command=self.save_per_minute_csv).pack(side="left")
        ttk.Button(btns, text="Close", command=self._close_plot_window).pack(side="right")
        ttk.Label(btns, text="Updates live. Hollow dot = current minute. "
                             "Background color = activity (see the by-activity panel). "
                             "Red columns = away; red ✕ = unreliable minute.",
                  foreground="#666666").pack(side="left", padx=(10, 0))

        self.plot_window = win
        win.protocol("WM_DELETE_WINDOW", self._close_plot_window)
        self._refresh_plot_window(self.elapsed_seconds())

    def _close_plot_window(self):
        if self.plot_window is not None:
            self.plot_window.destroy()
        self.plot_window = None
        self.plot_canvas = None
        self.plot_info_var = None

    def _refresh_plot_window(self, elapsed: float):
        if self.plot_window is None or self.plot_canvas is None:
            return
        if not self.plot_window.winfo_exists():
            self.plot_window = None
            self.plot_canvas = None
            return
        series = self.per_minute_series(elapsed)
        if not series:
            self._draw_line_plot(self.plot_canvas, [],
                                 empty_text="Start the camera — your per-minute plot will appear here")
            if self.plot_info_var is not None:
                self.plot_info_var.set("No data yet.")
            return
        away_pm = self.away_seconds_per_minute(elapsed)
        flags = self.unreliable_flags(elapsed, away_pm)
        self._draw_line_plot(self.plot_canvas, series, first_minute=1,
                             away_per_minute=away_pm, unreliable=flags,
                             activity_bands=self._activity_bands(
                                 elapsed, 0.0, float(len(series))))
        # Average over completed, reliable minutes only
        completed = series[:-1] if len(series) > 1 else series
        reliable_completed = [v for i, v in enumerate(completed)
                              if not (i < len(flags) and flags[i])]
        avg = (sum(reliable_completed) / len(reliable_completed)
               if reliable_completed else 0.0)
        peak = max(series)
        peak_minute = series.index(peak) + 1
        away_total = self.away_seconds_total(elapsed)
        n_unreliable = sum(1 for f in flags if f)
        if self.plot_info_var is not None:
            act = self.activity_stats(elapsed)
            act_text = ""
            if act:
                parts = []
                for s in act:
                    mins = s["seconds"] / 60.0
                    bpm = s["blinks"] / mins if mins > 0 else 0.0
                    parts.append(f"{s['name']}: {bpm:.1f}/min")
                act_text = "  •  by activity — " + ", ".join(parts)
            self.plot_info_var.set(
                f"{len(series)} minutes so far  •  average {avg:.1f} blinks/min "
                f"(completed, reliable minutes)  •  peak {peak} blinks/min in minute "
                f"{peak_minute}  •  total {self.total_blinks} blinks  •  "
                f"away {int(away_total // 60)}m {int(away_total % 60)}s"
                + (f"  •  {n_unreliable} unreliable minute(s) marked ✕"
                   if n_unreliable else "")
                + act_text)

    def save_per_minute_csv(self):
        elapsed = self.elapsed_seconds()
        series = self.per_minute_series(elapsed)
        if not series:
            messagebox.showinfo("Nothing to save", "No per-minute data yet — start the camera first.")
            return
        default = f"blinks_per_minute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=default,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        away_pm = self.away_seconds_per_minute(elapsed)
        flags = self.unreliable_flags(elapsed, away_pm)
        cumulative = 0
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["minute", "blinks_in_minute", "cumulative_blinks",
                             "away_seconds_in_minute", "reliable", "activity",
                             "is_current_partial_minute"])
            for i, v in enumerate(series, start=1):
                cumulative += v
                away_s = away_pm[i - 1] if i - 1 < len(away_pm) else 0.0
                reliable = not (i - 1 < len(flags) and flags[i - 1])
                writer.writerow([i, v, cumulative, f"{away_s:.1f}",
                                 "yes" if reliable else "no",
                                 self._dominant_activity(i - 1, elapsed),
                                 "yes" if i == len(series) and len(series) > 1 else ""])
        messagebox.showinfo("Saved", f"Saved per-minute data to:\n{path}")

    # ----------------------------------------------------- 20-20-20 breaks
    def _break_interval_seconds(self) -> float:
        try:
            minutes = int(self.break_interval_var.get())
        except (tk.TclError, ValueError):
            minutes = 20
        return max(min(minutes, 60), 1) * 60.0

    def _break_duration_seconds(self) -> int:
        try:
            seconds = int(self.break_duration_var.get())
        except (tk.TclError, ValueError):
            seconds = 20
        return max(min(seconds, 120), 5)

    def reset_break_timer(self):
        self.next_break_at = self.elapsed_seconds() + self._break_interval_seconds()

    def _check_break_timer(self, elapsed: float):
        if not self.break_enabled_var.get():
            return
        if self.break_window is not None:
            return  # a break popup is already open
        if self.session_start is None or not self.running:
            return  # timer runs on active (camera-on) session time
        if self.away_start is not None:
            return  # paused while the user is away (they're on a break already)
        if elapsed >= self.next_break_at:
            self.open_break_window()

    def open_break_window(self):
        if self.break_window is not None and self.break_window.winfo_exists():
            self.break_window.lift()
            self.break_window.focus_set()
            return

        duration = self._break_duration_seconds()
        self.break_countdown_left = duration

        win = tk.Toplevel(self)
        win.title("20-20-20 break")
        win.geometry("460x340")
        win.resizable(False, False)
        win.attributes("-topmost", True)
        self.break_window = win
        self.bell()
        win.lift()
        win.focus_set()

        ttk.Label(win, text="👀  Time for a 20-20-20 break",
                  font=("TkDefaultFont", 15, "bold")).pack(pady=(18, 6))
        ttk.Label(win,
                  text="Look at something at least 20 feet away\n"
                       "(about 6 metres — out a window is perfect)\n"
                       "and keep looking until the countdown ends.",
                  justify="center").pack(pady=(0, 10))

        self.break_countdown_var = tk.StringVar(value=str(duration))
        ttk.Label(win, textvariable=self.break_countdown_var,
                  font=("TkDefaultFont", 40, "bold")).pack()
        ttk.Label(win, text="seconds left", foreground="#666666").pack(pady=(0, 8))

        self.break_progress = ttk.Progressbar(win, maximum=duration,
                                              value=duration, length=360)
        self.break_progress.pack(pady=(0, 14))

        btns = ttk.Frame(win)
        btns.pack(pady=(0, 8))
        ttk.Button(btns, text="Snooze 5 min",
                   command=lambda: self._finish_break("snooze")).pack(side="left", padx=4)
        ttk.Button(btns, text="Skip this break",
                   command=lambda: self._finish_break("skip")).pack(side="left", padx=4)
        self.break_done_btn = ttk.Button(btns, text="Done early",
                                         command=lambda: self._finish_break("done"))
        self.break_done_btn.pack(side="left", padx=4)

        ttk.Label(win, text="Blinking and looking far away lets your eyes reset.",
                  foreground="#666666").pack(pady=(6, 0))

        win.protocol("WM_DELETE_WINDOW", lambda: self._finish_break("skip"))
        self.break_countdown_id = win.after(1000, self._break_tick)

    def _break_tick(self):
        if self.break_window is None or not self.break_window.winfo_exists():
            return
        self.break_countdown_left -= 1
        if self.break_countdown_left <= 0:
            self._finish_break("completed")
            return
        self.break_countdown_var.set(str(self.break_countdown_left))
        self.break_progress.config(value=self.break_countdown_left)
        self.break_countdown_id = self.break_window.after(1000, self._break_tick)

    def _finish_break(self, how: str):
        # Schedule the next break from *now* (current session elapsed time)
        elapsed = self.elapsed_seconds()
        if how == "completed":
            self.breaks_taken += 1
            self.next_break_at = elapsed + self._break_interval_seconds()
        elif how == "done":  # user ended the countdown early — count it
            self.breaks_taken += 1
            self.next_break_at = elapsed + self._break_interval_seconds()
        elif how == "snooze":
            self.next_break_at = elapsed + 5 * 60.0
        else:  # skip / window closed
            self.breaks_skipped += 1
            self.next_break_at = elapsed + self._break_interval_seconds()

        win = self.break_window
        self.break_window = None
        self.break_countdown_id = None
        if win is not None and win.winfo_exists():
            if how == "completed":
                # Brief "well done" state instead of vanishing instantly
                for child in win.winfo_children():
                    child.destroy()
                ttk.Label(win, text="✓  Break complete",
                          font=("TkDefaultFont", 16, "bold")).pack(pady=(50, 8))
                ttk.Label(win, text="Nice — your eyes got their reset.\n"
                                    "This closes in a few seconds.").pack()
                self.bell()
                self.break_window = win  # keep the slot busy until it closes
                win.protocol("WM_DELETE_WINDOW", lambda: self._clear_break_window(win))
                # _check_break_timer won't fire while this is up, and the
                # next break is already scheduled a full interval away.
                win.after(2500, lambda: self._clear_break_window(win))
                return
            win.destroy()

    def _clear_break_window(self, win):
        if self.break_window is win:
            self.break_window = None
        if win.winfo_exists():
            win.destroy()

    # ------------------------------------------------------------ Actions
    def reset_stats(self):
        self.total_blinks = 0
        self.blink_timestamps.clear()
        self.ear_below_frames = 0
        self.elapsed_before_pause = 0.0
        self.session_start = time.monotonic() if self.running else None
        self.breaks_taken = 0
        self.breaks_skipped = 0
        self.next_break_at = self._break_interval_seconds()
        self.away_intervals.clear()
        self.away_start = None
        self.no_face_since = None
        self.activity_segments.clear()
        self._activity_color_idx.clear()
        if self.running:
            self._open_activity_segment(0.0)

    def save_csv(self):
        if not self.blink_timestamps:
            messagebox.showinfo("Nothing to save", "No blinks have been recorded yet.")
            return
        default = f"blink_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path = filedialog.asksaveasfilename(
            defaultextension=".csv", initialfile=default,
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
        )
        if not path:
            return
        ref = self.blink_timestamps[0]
        elapsed = self.elapsed_seconds()
        now_ts = datetime.now().timestamp()
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["blink_number", "seconds_since_first_blink",
                             "approx_wall_time", "activity", "cumulative_blinks"])
            for i, t in enumerate(self.blink_timestamps, start=1):
                wall = datetime.fromtimestamp(now_ts - (elapsed - t))
                writer.writerow([i, f"{t - ref:.2f}", wall.strftime("%Y-%m-%d %H:%M:%S"),
                                 self.activity_at(t, elapsed), i])

        # Also write a summary line the user can keep
        elapsed = self.elapsed_seconds()
        summary_path = Path(path).with_suffix(".summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Blink session — {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            away_total = self.away_seconds_total(elapsed)
            present = max(elapsed - away_total, 0.0)
            f.write(f"Duration: {elapsed/60:.1f} minutes "
                    f"(present: {present/60:.1f}, away excluded: {away_total/60:.1f})\n")
            f.write(f"Total blinks: {self.total_blinks}\n")
            f.write(f"20-20-20 breaks taken: {self.breaks_taken} "
                    f"(skipped: {self.breaks_skipped})\n")
            if present > 0:
                f.write(f"Average (present time): {self.total_blinks/(present/60):.1f} blinks/min "
                        f"({self.total_blinks/(present/3600):,.0f} blinks/hour)\n")
            act = self.activity_stats(elapsed)
            if act:
                f.write("\nBy activity (present time only):\n")
                for s in act:
                    mins = s["seconds"] / 60.0
                    bpm = s["blinks"] / mins if mins > 0 else 0.0
                    f.write(f"  {s['name']}: {mins:.1f} min, {s['blinks']} blinks, "
                            f"{bpm:.1f} blinks/min ({bpm * 60:,.0f} blinks/hour)\n")
        messagebox.showinfo("Saved", f"Saved:\n{path}\n{summary_path}")

    def on_close(self):
        if self.break_window is not None and self.break_window.winfo_exists():
            self.break_window.destroy()
        self.break_window = None
        self._close_plot_window()
        self.stop_camera()
        self.destroy()


if __name__ == "__main__":
    app = BlinkDetectorApp()
    app.mainloop()
