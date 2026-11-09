#!/usr/bin/env python3
"""
Blink Rate Detector — OpenCV-only fallback (no MediaPipe)
=========================================================
Same GUI and stats as blink_detector.py, but it detects blinks with
OpenCV Haar cascades only, so it runs on Python 3.13+ where MediaPipe
has no prebuilt wheel.

Includes activity categories: tag what you are doing (video games,
watching a show, YouTube, your own custom labels…) and the session
splits into activity segments with a blink-rate breakdown per
activity, per-minute bars colored by activity, and an activity
column in the CSV export.

How it works:
  * Haar face cascade finds the face, Haar eye cascade finds the eyes.
  * Eyes visible  -> open.  Eyes missing for >= min_closed_frames
    while a face is still visible -> closed. Eyes coming back after a
    closed run -> one blink.
  * The "eye openness" graph is a 0/1 proxy (1 = eyes found,
    0 = eyes not found), smoothed over the last few frames, so the
    red threshold line is at 0.5. It is NOT the EAR from Face Mesh.

Accuracy note — be honest about the trade-off:
  This is less precise than the MediaPipe version. It can miss blinks
  in poor light, with glasses, or when you look down / turn your head
  (the eye cascade loses the eyes and that can look like a blink).
  Use it as a fallback if you cannot use Python 3.11/3.12 with
  MediaPipe; prefer blink_detector.py when MediaPipe installs.

Requirements:
  pip install -r requirements_opencv.txt
  (opencv-python and Pillow only — no mediapipe)

Run:
  python blink_detector_opencv.py

All processing is local. No video is recorded or sent anywhere.
The only file written is the CSV/summary you explicitly save.
"""

from __future__ import annotations

import csv
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


# Activity categories: pick one (or type your own) to tag what you are
# doing; the session splits into segments and blink rates are reported
# per activity. Each activity gets a (band, legend) color pair,
# assigned in the order activities first appear.
ACTIVITY_PRESETS = ["Video games", "Watching a show", "YouTube", "Reading",
                    "Coding / work", "Video call", "Browsing", "Other"]
ACTIVITY_PALETTE = [("#dceeff", "#1f77b4"), ("#ffe9d6", "#e67e22"),
                    ("#e6f4d7", "#2e9e4f"), ("#f3e2fd", "#8e44ad"),
                    ("#fdf3c4", "#b7950b"), ("#d9f2f0", "#148f77"),
                    ("#ffe0ec", "#c2185b"), ("#e8e8e8", "#666666")]
DEFAULT_ACTIVITY = "Uncategorized"


class BlinkDetectorApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Blink Rate Detector (OpenCV-only, no MediaPipe)")
        self.geometry("1120x720")
        self.minsize(980, 640)
        self.protocol("WM_DELETE_WINDOW", self.on_close)

        # --- State ---
        self.cap = None
        self.face_cascade = None
        self.eye_cascade = None
        self.running = False
        self.after_id = None

        self.total_blinks = 0
        self.blink_timestamps: list[float] = []
        self.session_start: float | None = None
        self.elapsed_before_pause = 0.0

        self.eyes_closed_frames = 0
        # Openness proxy: 1.0 = eyes found, 0.0 = not found
        self.openness_history: deque[float] = deque(maxlen=90)

        # --- Activity segments (session-elapsed seconds) ---
        # [start, end_or_None, name]; the open segment runs to "now".
        self.activity_var = tk.StringVar(value=DEFAULT_ACTIVITY)
        self.activity_segments: list[list] = []
        self._activity_color_idx: dict[str, int] = {}
        self._breakdown_text_cache = ""

        # Settings (bound to sliders)
        self.threshold_var = tk.DoubleVar(value=0.5)
        self.min_frames_var = tk.IntVar(value=2)
        self.camera_var = tk.IntVar(value=0)

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

        ttk.Label(left, text="Eye openness (proxy: 1 = eyes found, 0 = eyes not found):"
                  ).pack(anchor="w", pady=(8, 2))
        self.ear_canvas = tk.Canvas(left, height=70, bg="white", highlightthickness=1,
                                    highlightbackground="#cccccc")
        self.ear_canvas.pack(fill="x")

        # Right: stats + controls
        right = ttk.Frame(root)
        right.grid(row=0, column=1, sticky="nsew")
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

        rows = [
            ("Total blinks:", self.total_var, 16),
            ("Session time:", self.elapsed_var, 12),
            ("Blinks / minute:", self.bpm_var, 16),
            ("Blinks / hour (projected):", self.bph_projected_var, 16),
            ("Blinks / hour (last 5 min):", self.bph_rolling_var, 16),
            ("Blinks in last 60 seconds:", self.last_min_var, 12),
        ]
        for i, (label, var, size) in enumerate(rows):
            ttk.Label(stats, text=label).grid(row=i, column=0, sticky="w", pady=3)
            ttk.Label(stats, textvariable=var,
                      font=("TkDefaultFont", size, "bold")).grid(row=i, column=1, sticky="e", pady=3)

        ttk.Label(stats,
                  text="Projected = whole-session average scaled to 1 hour.\n"
                       "Last 5 min = recent rate × 12, steadier after a few minutes.\n"
                       "OpenCV-only fallback: less precise than the MediaPipe version.",
                  foreground="#666666", wraplength=360, justify="left"
                  ).grid(row=len(rows), column=0, columnspan=2, sticky="w", pady=(6, 0))

        # Per-minute chart
        chart_box = ttk.LabelFrame(right, text="Blinks per minute (last 15 minutes)", padding=8)
        chart_box.pack(fill="x", pady=(10, 0))
        self.chart_canvas = tk.Canvas(chart_box, height=110, bg="white",
                                      highlightthickness=1, highlightbackground="#cccccc")
        self.chart_canvas.pack(fill="x")
        ttk.Label(chart_box, text="Bar color = the activity you were doing that minute.",
                  foreground="#666666").pack(anchor="w", pady=(4, 0))

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
        ttk.Label(activity_box,
                  text="Pick what you're doing — or type your own label and\n"
                       "press Enter. Switching starts a new segment, and the\n"
                       "breakdown below keeps a separate blink rate for each.",
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

        ttk.Label(settings, text="Blink threshold (openness proxy)").pack(anchor="w")
        thr_row = ttk.Frame(settings)
        thr_row.pack(fill="x")
        ttk.Scale(thr_row, from_=0.1, to=0.9, variable=self.threshold_var,
                  orient="horizontal").pack(side="left", fill="x", expand=True)
        self.thr_label = ttk.Label(thr_row, text="0.50", width=5)
        self.thr_label.pack(side="left", padx=(6, 0))
        ttk.Label(settings,
                  text="Eyes are counted as closed when the openness proxy\n"
                       "drops below this line. Leave at 0.50 unless counts look off.",
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

    # ------------------------------------------------------------- Camera
    def start_camera(self):
        if cv2 is None or Image is None:
            missing = []
            if cv2 is None:
                missing.append("opencv-python")
            if Image is None:
                missing.append("Pillow")
            messagebox.showerror(
                "Missing packages",
                "These packages are not installed:\n\n"
                + ", ".join(missing)
                + "\n\nInstall them with:\n  pip install -r requirements_opencv.txt"
            )
            return

        self.face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
        self.eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_eye.xml")
        if self.face_cascade.empty() or self.eye_cascade.empty():
            messagebox.showerror(
                "Cascades not found",
                "OpenCV could not load its built-in Haar cascades.\n"
                "Reinstall opencv-python and try again."
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

        if self.session_start is None:
            self.session_start = time.monotonic()
        else:
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
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.face_cascade = None
        self.eye_cascade = None
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

        frame = cv2.flip(frame, 1)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))

        openness = None
        if len(faces):
            # Largest face
            x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # Eyes live in the upper half of the face
            roi_gray = gray[y:y + h // 2, x:x + w]
            eyes = self.eye_cascade.detectMultiScale(
                roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(18, 18))
            eyes_found = len(eyes) >= 1
            openness = 1.0 if eyes_found else 0.0
            for (ex, ey, ew, eh) in eyes:
                cv2.rectangle(frame, (x + ex, y + ey),
                              (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
            self._process_openness(openness)
            state = "Open" if eyes_found else "Closed?"
            self.status_var.set(
                f"Status: Face detected  •  Eyes {'found' if eyes_found else 'not found'}  •  {state}")
        else:
            # No face: do not count this as a blink; just wait.
            self.eyes_closed_frames = 0
            self.status_var.set("Status: No face detected — look toward the camera")

        if openness is not None:
            self.openness_history.append(openness)
        self._draw_ear_chart()

        cv2.putText(frame, f"Blinks: {self.total_blinks}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)

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
    def _process_openness(self, openness: float):
        threshold = float(self.threshold_var.get())
        if openness < threshold:
            self.eyes_closed_frames += 1
        else:
            if self.eyes_closed_frames >= int(self.min_frames_var.get()):
                self.total_blinks += 1
                # Store session-elapsed seconds (not wall clock), so the
                # stats freeze when the camera is stopped instead of the
                # last-minute / rolling windows draining away in real time.
                self.blink_timestamps.append(self.elapsed_seconds())
            self.eyes_closed_frames = 0

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

    def activity_at(self, t: float, elapsed: float) -> str:
        for start, end, name in self.activity_segments:
            seg_end = elapsed if end is None else end
            if start <= t < seg_end or (t == seg_end == elapsed):
                return name
        return DEFAULT_ACTIVITY

    def activity_stats(self, elapsed: float) -> list[dict]:
        """Per-activity totals, merged by name, in first-appearance order."""
        order: list[str] = []
        agg: dict[str, dict] = {}
        for start, end, name in self.activity_segments:
            seg_end = elapsed if end is None else min(end, elapsed)
            if seg_end <= start:
                continue
            blinks = sum(1 for t in self.blink_timestamps if start <= t < seg_end)
            if name not in agg:
                agg[name] = {"name": name, "seconds": 0.0, "blinks": 0}
                order.append(name)
            agg[name]["seconds"] += seg_end - start
            agg[name]["blinks"] += blinks
        return [agg[n] for n in order if agg[n]["seconds"] > 0 or agg[n]["blinks"] > 0]

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

        # Timestamps are session-elapsed seconds, so windows are measured
        # against the (freezable) session clock, never wall-clock time.
        last_60 = sum(1 for t in self.blink_timestamps if elapsed - t <= 60)
        last_300 = sum(1 for t in self.blink_timestamps if elapsed - t <= 300)
        self.last_min_var.set(str(last_60))

        if elapsed >= 10:
            self.bpm_var.set(f"{self.total_blinks / (elapsed / 60):.1f}")
            self.bph_projected_var.set(f"{self.total_blinks / (elapsed / 3600):,.0f}")
        else:
            self.bpm_var.set("—")
            self.bph_projected_var.set("—")

        window = min(elapsed, 300)
        if window >= 30:
            self.bph_rolling_var.set(f"{last_300 / (window / 3600):,.0f}")
        else:
            self.bph_rolling_var.set("collecting…")

        self._update_breakdown_panel(elapsed)
        self._draw_minute_chart(elapsed)
        self.after(500, self._update_stats_loop)

    # ------------------------------------------------------------- Charts
    def _draw_ear_chart(self):
        c = self.ear_canvas
        c.delete("all")
        w = c.winfo_width() or 400
        h = c.winfo_height() or 70
        if len(self.openness_history) < 2:
            return

        def y_of(v):
            return h - (v / 1.0) * (h - 8) - 4

        thr_y = y_of(float(self.threshold_var.get()))
        c.create_line(0, thr_y, w, thr_y, fill="red", dash=(4, 3))
        points = []
        n = len(self.openness_history)
        for i, v in enumerate(self.openness_history):
            x = (i / max(n - 1, 1)) * (w - 4) + 2
            points.extend([x, y_of(v)])
        if len(points) >= 4:
            c.create_line(points, fill="#1f77b4", width=2, smooth=True)

    def _draw_minute_chart(self, elapsed: float):
        c = self.chart_canvas
        c.delete("all")
        w = c.winfo_width() or 360
        h = c.winfo_height() or 110
        if self.session_start is None and not self.blink_timestamps:
            c.create_text(w // 2, h // 2, text="Start the camera to see your per-minute trend",
                          fill="#888888")
            return

        buckets: dict[int, int] = {}
        for t in self.blink_timestamps:
            minute = int(t // 60)
            buckets[minute] = buckets.get(minute, 0) + 1

        current_minute = int(elapsed // 60)
        start_minute = max(0, current_minute - 14)
        values = [buckets.get(m, 0) for m in range(start_minute, current_minute + 1)]
        if not values:
            return
        max_v = max(max(values), 5)
        bar_w = (w - 20) / len(values)
        for i, v in enumerate(values):
            x0 = 10 + i * bar_w + 2
            x1 = 10 + (i + 1) * bar_w - 2
            bar_h = (v / max_v) * (h - 28)
            dom = self._dominant_activity(start_minute + i, elapsed)
            color = (self._activity_colors(dom)[1]
                     if dom != DEFAULT_ACTIVITY else "#4a90d9")
            c.create_rectangle(x0, h - 18 - bar_h, x1, h - 18, fill=color, outline="")
            if v:
                c.create_text((x0 + x1) / 2, h - 22 - bar_h, text=str(v), font=("TkDefaultFont", 8))
            c.create_text((x0 + x1) / 2, h - 8, text=f"{start_minute + i + 1}",
                          font=("TkDefaultFont", 7), fill="#666666")

    # ------------------------------------------------------------ Actions
    def reset_stats(self):
        self.total_blinks = 0
        self.blink_timestamps.clear()
        self.eyes_closed_frames = 0
        self.elapsed_before_pause = 0.0
        self.session_start = time.monotonic() if self.running else None
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
                # Timestamps are session-elapsed seconds; count back from
                # the session's elapsed time to estimate wall-clock time.
                wall = datetime.fromtimestamp(now_ts - (elapsed - t))
                writer.writerow([i, f"{t - ref:.2f}", wall.strftime("%Y-%m-%d %H:%M:%S"),
                                 self.activity_at(t, elapsed), i])

        summary_path = Path(path).with_suffix(".summary.txt")
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(f"Blink session (OpenCV-only) — {datetime.now():%Y-%m-%d %H:%M:%S}\n")
            f.write(f"Duration: {elapsed/60:.1f} minutes\n")
            f.write(f"Total blinks: {self.total_blinks}\n")
            if elapsed > 0:
                f.write(f"Average: {self.total_blinks/(elapsed/60):.1f} blinks/min "
                        f"({self.total_blinks/(elapsed/3600):,.0f} blinks/hour)\n")
            act = self.activity_stats(elapsed)
            if act:
                f.write("\nBy activity:\n")
                for s in act:
                    mins = s["seconds"] / 60.0
                    bpm = s["blinks"] / mins if mins > 0 else 0.0
                    f.write(f"  {s['name']}: {mins:.1f} min, {s['blinks']} blinks, "
                            f"{bpm:.1f} blinks/min ({bpm * 60:,.0f} blinks/hour)\n")
        messagebox.showinfo("Saved", f"Saved:\n{path}\n{summary_path}")

    def on_close(self):
        self.stop_camera()
        self.destroy()


if __name__ == "__main__":
    app = BlinkDetectorApp()
    app.mainloop()
