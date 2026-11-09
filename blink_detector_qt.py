#!/usr/bin/env python3
"""
Blink Rate Detector — GUI (PySide6 / Qt edition)
================================================
Detects blinks from your webcam in real time and reports:
  • Total blinks
  • Blinks per minute
  • Blinks per hour (projected from the whole session)
  • Blinks per hour (rolling, based on the last 5 minutes — more stable)
  • A live plot of blinks per minute over time (plus a full-session
    plot dialog and a per-minute CSV export)
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
  • Low blink rate alerts: if your rate over the last minute
    stays below a threshold (default 8/min) for a sustained
    period (default 60 s), a badge with a pair of blinking dry
    eyes pulls down from the top of the screen. It hides itself
    when your rate recovers; Dismiss silences it for 5 minutes.
  • Breathing rate: the subtle rise and fall of your chest and
    shoulders in the region below your face is tracked frame to
    frame (optical flow) and turned into an estimated breaths-per-
    minute readout with a live waveform. It needs about 25 s of
    sitting fairly still, with your chest in frame, to lock on.
  • Optional auto-detection (Windows): the foreground window's
    program and title are checked every ~2 seconds and matched
    to an activity (YouTube/Netflix/etc. in a browser, known
    coding and call apps, fullscreen non-browser app = video
    games). Titles are read locally and never stored.

How it works:
  MediaPipe Face Mesh locates 6 landmarks around each eye.
  The Eye Aspect Ratio (EAR) drops sharply when you blink.
  When EAR stays below a threshold for a few frames and then rises
  again, that counts as one blink. If MediaPipe is not installed,
  the app automatically falls back to an OpenCV Haar-cascade
  engine (eyes found / not found as an openness proxy), the same
  approach as the old OpenCV-only Tkinter version.

Requirements:
  pip install -r requirements_qt.txt
  (mediapipe is optional — without it the OpenCV engine is used)

Run:
  python blink_detector_qt.py

Notes:
  - All processing is local. Frames are processed and discarded —
    no video or images are ever recorded, stored or sent anywhere.
    Breathing is estimated from chest/shoulder motion in those
    same discarded frames; nothing extra is captured or stored.
    The only files written are the CSV/summary logs you explicitly
    save.
  - Good, even lighting and looking roughly toward the camera works best.
  - Normal resting blink rate is roughly 900–1,200/hour (15–20/min),
    and it usually falls a lot during concentrated screen work.
"""

from __future__ import annotations

import csv
import math
import sys
import time
from collections import deque
from datetime import datetime
from pathlib import Path

try:
    from PySide6.QtCore import (
        QObject, QPropertyAnimation, QEasingCurve, QRect, Qt, QThread,
        QTimer, Signal, Slot,
    )
    from PySide6.QtGui import QColor, QImage, QPainter, QPen, QPixmap
    from PySide6.QtWidgets import (
        QApplication, QCheckBox, QComboBox, QDialog, QDoubleSpinBox,
        QFileDialog, QFrame, QGridLayout, QGroupBox, QHBoxLayout, QLabel,
        QListWidget, QListWidgetItem, QMainWindow, QMessageBox,
        QProgressBar, QPushButton, QScrollArea, QSpinBox, QTableWidget,
        QTableWidgetItem, QVBoxLayout, QWidget,
    )
except ImportError:
    print("PySide6 is not installed.\n\n"
          "Install the Qt version's requirements with:\n"
          "  pip install -r requirements_qt.txt\n"
          "(or just:  pip install PySide6)")
    sys.exit(1)

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None

try:
    import numpy as np
except ImportError:  # pragma: no cover
    np = None

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

ACCENT = "#38bdf8"

DARK_STYLESHEET = """
QMainWindow, QDialog { background: #0f172a; }
QWidget { color: #e2e8f0; font-family: "Segoe UI", "Inter", sans-serif;
          font-size: 13px; }
QGroupBox { background: #1e293b; border: 1px solid #334155;
            border-radius: 10px; margin-top: 16px; padding: 14px 12px 12px 12px;
            font-weight: 600; }
QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
                   left: 12px; padding: 0 5px; color: #7dd3fc; }
QPushButton { background: #334155; border: 1px solid #475569;
              border-radius: 8px; padding: 8px 14px; font-weight: 600; }
QPushButton:hover { background: #475569; }
QPushButton:pressed { background: #1e293b; }
QPushButton:disabled { background: #1e293b; color: #64748b; border-color: #334155; }
QPushButton#accent { background: #0284c7; border-color: #0284c7; color: #ffffff; }
QPushButton#accent:hover { background: #0369a1; }
QPushButton#accent:disabled { background: #1e293b; color: #64748b; }
QSpinBox, QDoubleSpinBox, QComboBox { background: #0f172a; border: 1px solid #334155;
              border-radius: 6px; padding: 5px 7px; selection-background-color: #0284c7; }
QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus { border-color: #38bdf8; }
QCheckBox { spacing: 7px; }
QCheckBox::indicator { width: 16px; height: 16px; border-radius: 4px;
              border: 1px solid #475569; background: #0f172a; }
QCheckBox::indicator:checked { background: #0284c7; border-color: #38bdf8; }
QLabel#muted { color: #94a3b8; font-size: 12px; font-weight: 400; }
QLabel#statName { color: #94a3b8; font-weight: 400; }
QLabel#statValue { color: #f8fafc; font-size: 16px; font-weight: 700; }
QLabel#statValueBig { color: #38bdf8; font-size: 20px; font-weight: 800; }
QLabel#videoLabel { background: #020617; border: 1px solid #334155;
              border-radius: 10px; color: #94a3b8; font-size: 14px; }
QLabel#headerTitle { font-size: 19px; font-weight: 800; color: #f8fafc; }
QLabel#enginePill { background: #164e63; color: #a5f3fc; border-radius: 9px;
              padding: 3px 10px; font-size: 12px; font-weight: 600; }
QScrollArea { border: none; background: transparent; }
QScrollBar:vertical { background: #0f172a; width: 10px; border: none; }
QScrollBar::handle:vertical { background: #334155; border-radius: 5px; min-height: 30px; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QTableWidget { background: #0f172a; border: 1px solid #334155; border-radius: 8px;
              gridline-color: #1e293b; font-weight: 400; }
QHeaderView::section { background: #1e293b; color: #94a3b8; border: none;
              padding: 5px; font-size: 12px; font-weight: 600; }
QListWidget { background: #0f172a; border: 1px solid #334155; border-radius: 8px; }
QProgressBar { background: #0f172a; border: 1px solid #334155; border-radius: 7px;
              height: 16px; text-align: center; color: #e2e8f0; }
QProgressBar::chunk { background: #0284c7; border-radius: 6px; }
QFrame#badge { background: #2b1620; border: 2px solid #e0657a; border-radius: 12px; }
QLabel#badgeText { color: #ffe4e6; font-size: 13px; font-weight: 700; }
"""

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


def format_hms(seconds: float) -> str:
    seconds = int(seconds)
    return f"{seconds // 3600:02d}:{(seconds % 3600) // 60:02d}:{seconds % 60:02d}"



# --------------------------------------------------------------------------
# Custom painted widgets
# --------------------------------------------------------------------------

class LinePlotWidget(QWidget):
    """Blinks-per-minute line plot, painted with QPainter.

    Mirrors the Tkinter version: x = session minute, y = blinks/min,
    a shaded 15–20/min "typical" band, activity background bands,
    pale-red columns for minutes with away time, a red ✕ (and a break
    in the line) for unreliable minutes, and a hollow dot for the
    current, still-running minute.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(120)
        self._values: list[int] = []
        self._first_minute = 1
        self._away: list[float] | None = None
        self._unreliable: list[bool] | None = None
        self._bands: list[tuple[float, float, str]] | None = None
        self._empty_text = ""

    def set_data(self, values, first_minute=1, away_per_minute=None,
                 unreliable=None, activity_bands=None, empty_text=""):
        self._values = list(values)
        self._first_minute = first_minute
        self._away = away_per_minute
        self._unreliable = unreliable
        self._bands = activity_bands
        self._empty_text = empty_text
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#0b1220"))
        values = self._values
        if not values:
            if self._empty_text:
                p.setPen(QColor("#64748b"))
                p.drawText(0, 0, w, h, Qt.AlignmentFlag.AlignCenter,
                           self._empty_text)
            return

        left, right_pad, top, bottom = 36, 12, 12, 24
        plot_w = max(w - left - right_pad, 10)
        plot_h = max(h - top - bottom, 10)
        max_v = max(max(values), 10)
        step = 5 if max_v <= 30 else 10
        y_max = int(math.ceil(max_v / step)) * step

        def x_of(i):
            if len(values) == 1:
                return left + plot_w / 2
            return left + (i / (len(values) - 1)) * plot_w

        def y_of(v):
            return top + plot_h - (v / y_max) * plot_h

        small = p.font()
        small.setPointSize(8)
        p.setFont(small)

        # Gridlines + y labels
        p.setPen(QColor("#94a3b8"))
        p.drawText(2, top + 8, "blinks/min")
        for gv in range(0, y_max + 1, step):
            y = y_of(gv)
            p.setPen(QColor("#1e293b"))
            p.drawLine(int(left), int(y), int(left + plot_w), int(y))
            p.setPen(QColor("#94a3b8"))
            p.drawText(0, int(y) - 6, left - 5, 12,
                       Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                       str(gv))

        # X labels: first, middle, last minute
        p.setPen(QColor("#94a3b8"))
        for i in sorted({0, len(values) - 1, (len(values) - 1) // 2}):
            p.drawText(int(x_of(i)) - 20, int(top + plot_h + 4), 40, 14,
                       Qt.AlignmentFlag.AlignCenter,
                       f"min {self._first_minute + i}")

        # Reference band for a typical resting rate (15–20/min)
        if y_max >= 20:
            band = QColor("#22c55e")
            band.setAlpha(28)
            p.fillRect(int(left), int(y_of(20)), int(plot_w),
                       int(y_of(15) - y_of(20)), band)
            p.setPen(QColor("#4ade80"))
            p.drawText(int(left + plot_w) - 90, int(y_of(20)) + 2, 86, 12,
                       Qt.AlignmentFlag.AlignRight, "typical 15–20")

        # Activity bands (band coords are minutes relative to first value)
        if self._bands and len(values) > 1:
            for band_start, band_end, color in self._bands:
                x0 = x_of(max(band_start, 0.0))
                x1 = x_of(min(band_end, float(len(values) - 1)))
                if x1 > x0:
                    c = QColor(color)
                    c.setAlpha(48)
                    p.fillRect(int(x0), int(top), int(x1 - x0), int(plot_h), c)

        # Away shading: pale-red column behind any minute with away time
        has_away = bool(self._away) and any(a > 0.5 for a in self._away)
        if has_away:
            for i, away_s in enumerate(self._away):
                if away_s <= 0.5:
                    continue
                x_left = left if i == 0 else (x_of(i - 1) + x_of(i)) / 2
                x_right = (left + plot_w if i == len(values) - 1
                           else (x_of(i) + x_of(i + 1)) / 2)
                c = QColor("#ef4444")
                c.setAlpha(36)
                p.fillRect(int(x_left), int(top), int(x_right - x_left),
                           int(plot_h), c)
            p.setPen(QColor("#f87171"))
            p.drawText(int(left + plot_w) - 50, int(top) + 2, 46, 12,
                       Qt.AlignmentFlag.AlignRight, "■ away")

        def is_unreliable(i):
            return bool(self._unreliable) and i < len(self._unreliable) \
                and self._unreliable[i]

        # Line, broken around unreliable minutes
        p.setPen(QPen(QColor(ACCENT), 2))
        segment: list[tuple[float, float]] = []
        for i, v in enumerate(values):
            if is_unreliable(i):
                for a, b in zip(segment, segment[1:]):
                    p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
                segment = []
                continue
            segment.append((x_of(i), y_of(v)))
        for a, b in zip(segment, segment[1:]):
            p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))

        # Points
        for i, v in enumerate(values):
            x = x_of(i)
            if is_unreliable(i):
                y = top + plot_h - 9
                p.setPen(QPen(QColor("#ef4444"), 2))
                p.drawLine(int(x - 4), int(y - 4), int(x + 4), int(y + 4))
                p.drawLine(int(x - 4), int(y + 4), int(x + 4), int(y - 4))
                continue
            y = y_of(v)
            if i == len(values) - 1 and len(values) > 1:
                p.setPen(QPen(QColor(ACCENT), 2))
                p.setBrush(QColor("#0b1220"))
                p.drawEllipse(int(x) - 4, int(y) - 4, 8, 8)
            else:
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor(ACCENT))
                p.drawEllipse(int(x) - 3, int(y) - 3, 6, 6)
            if v:
                p.setPen(QColor("#e2e8f0"))
                p.drawText(int(x) - 12, int(y) - 20, 24, 12,
                           Qt.AlignmentFlag.AlignCenter, str(v))
        p.end()


class EarSparkline(QWidget):
    """Last ~12 s of EAR / openness values with the threshold line."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(64)
        self.setMaximumHeight(80)
        self._values: list[float] = []
        self._threshold = 0.21
        self._ear_mode = True  # False → 0..1 openness proxy range

    def set_data(self, values, threshold, ear_mode=True):
        self._values = list(values)
        self._threshold = threshold
        self._ear_mode = ear_mode
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#0b1220"))
        if len(self._values) < 2:
            p.setPen(QColor("#64748b"))
            p.drawText(10, 0, w, h, Qt.AlignmentFlag.AlignVCenter,
                       "Eye openness will appear here once the camera starts.")
            return
        lo, hi = (0.05, 0.45) if self._ear_mode else (0.0, 1.0)

        def y_of(v):
            v = max(min(v, hi), lo)
            return h - ((v - lo) / (hi - lo)) * (h - 8) - 4

        thr_y = y_of(self._threshold)
        pen = QPen(QColor("#ef4444"), 1, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(0, int(thr_y), w, int(thr_y))
        p.setPen(QPen(QColor(ACCENT), 2))
        n = len(self._values)
        pts = [(2 + (i / max(n - 1, 1)) * (w - 4), y_of(v))
               for i, v in enumerate(self._values)]
        for a, b in zip(pts, pts[1:]):
            p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
        p.end()


class BreathWaveform(QWidget):
    """Filtered breathing waveform for the last ~30 s (10 Hz points).

    Values are in arbitrary units (integrated chest motion), so the
    plot auto-scales to the largest excursion in the window.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(64)
        self.setMaximumHeight(80)
        self._values: list[float] = []

    def set_data(self, values):
        self._values = [float(v) for v in values]
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        w, h = self.width(), self.height()
        p.fillRect(0, 0, w, h, QColor("#0b1220"))
        if len(self._values) < 2:
            p.setPen(QColor("#64748b"))
            p.drawText(10, 0, w, h, Qt.AlignmentFlag.AlignVCenter,
                       "Your breathing waveform will appear here once "
                       "the signal locks on.")
            return
        scale = max(max(abs(v) for v in self._values), 0.05)
        mid = h / 2.0
        pen = QPen(QColor("#475569"), 1, Qt.PenStyle.DashLine)
        p.setPen(pen)
        p.drawLine(0, int(mid), w, int(mid))
        p.setPen(QPen(QColor(ACCENT), 2))
        n = len(self._values)
        pts = [(2 + (i / max(n - 1, 1)) * (w - 4),
                mid - (v / scale) * (mid - 4))
               for i, v in enumerate(self._values)]
        for a, b in zip(pts, pts[1:]):
            p.drawLine(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
        p.end()


class DryEyesWidget(QWidget):
    """A pair of dry, blinking eyes (as in the Tkinter badge).

    Pinkish whites with small red veins; the lids close for a blink
    every ~2.4 s, driven by a QTimer phase clock.
    """

    WIDTH, HEIGHT = 104, 64

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedSize(self.WIDTH, self.HEIGHT)
        self._openness = 1.0
        self._phase_ms = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(50)

    def _tick(self):
        self._phase_ms = (self._phase_ms + 50) % 2400
        self._openness = 0.08 if self._phase_ms >= 2200 else 1.0
        self.update()

    def paintEvent(self, _event):
        p = QPainter(self)
        p.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        openness = self._openness
        for cx in (30, 74):
            cy_top, cy_bot = 18, 46
            p.setPen(QPen(QColor("#dd6677"), 2))
            p.setBrush(QColor("#ffecec"))
            p.drawEllipse(cx - 15, cy_top, 30, cy_bot - cy_top)
            p.setPen(QPen(QColor("#e08a8a"), 1))
            p.drawLine(cx - 14, 26, cx - 7, 30)
            p.drawLine(cx + 14, 38, cx + 8, 34)
            if openness > 0.25:
                r = 6.5 * openness
                p.setPen(Qt.PenStyle.NoPen)
                p.setBrush(QColor("#5b3a29"))
                p.drawEllipse(int(cx - r), int(32 - r), int(2 * r), int(2 * r))
                p.setBrush(QColor("#1a1a1a"))
                p.drawEllipse(cx - 2, 30, 4, 4)
            lid_y = 19 + (1.0 - openness) * 13
            p.setPen(QPen(QColor("#dd6677"), 2))
            p.drawLine(cx - 15, int(lid_y), cx + 15, int(lid_y))
        p.end()


class LowRateBadge(QWidget):
    """Frameless topmost badge that slides down from the top of the
    primary screen when the blink rate stays low (and slides back up
    on recovery / dismiss)."""

    dismissed = Signal()

    WIDTH, HEIGHT = 560, 92

    def __init__(self):
        super().__init__(None)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint
                            | Qt.WindowType.WindowStaysOnTopHint
                            | Qt.WindowType.Tool)
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, False)
        self.shown = False
        self._anim: QPropertyAnimation | None = None

        frame = QFrame(self)
        frame.setObjectName("badge")
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(frame)
        row = QHBoxLayout(frame)
        row.setContentsMargins(14, 10, 12, 10)
        self._eyes = DryEyesWidget()
        row.addWidget(self._eyes)
        self._label = QLabel("")
        self._label.setObjectName("badgeText")
        self._label.setWordWrap(True)
        row.addWidget(self._label, stretch=1)
        btn = QPushButton("Dismiss")
        btn.clicked.connect(self._on_dismiss)
        row.addWidget(btn)
        self.setFixedSize(self.WIDTH, self.HEIGHT)

    def set_text(self, text: str):
        self._label.setText(text)

    def _target_geometry(self, y: int) -> QRect:
        screen = QApplication.primaryScreen().availableGeometry()
        x = screen.x() + (screen.width() - self.WIDTH) // 2
        return QRect(x, screen.y() + y, self.WIDTH, self.HEIGHT)

    def slide_down(self, text: str):
        self.set_text(text)
        if self.shown:
            return
        self.shown = True
        self.setGeometry(self._target_geometry(-self.HEIGHT))
        self.show()
        self.raise_()
        self._animate_to(0)

    def slide_up(self):
        if not self.shown:
            self.hide()
            return
        self.shown = False
        self._animate_to(-self.HEIGHT, hide_after=True)

    def _animate_to(self, y: int, hide_after: bool = False):
        if self._anim is not None:
            self._anim.stop()
        anim = QPropertyAnimation(self, b"geometry", self)
        anim.setDuration(380)
        anim.setEasingCurve(QEasingCurve.Type.OutCubic)
        anim.setStartValue(self.geometry())
        anim.setEndValue(self._target_geometry(y))
        if hide_after:
            anim.finished.connect(self.hide)
        anim.start()
        self._anim = anim

    def _on_dismiss(self):
        self.dismissed.emit()
        self.slide_up()


class BreakDialog(QDialog):
    """Topmost 20-20-20 break popup with countdown + progress bar."""

    break_finished = Signal(str)  # "completed" | "done" | "snooze" | "skip"

    def __init__(self, duration: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("20-20-20 break")
        self.setModal(False)
        self.setWindowFlags(self.windowFlags()
                            | Qt.WindowType.WindowStaysOnTopHint)
        self.setFixedSize(460, 360)
        self._duration = duration
        self._left = duration
        self._finished = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 18)
        title = QLabel("👀  Time for a 20-20-20 break")
        title.setStyleSheet("font-size: 17px; font-weight: 800; color: #f8fafc;")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        self._body = QLabel("Look at something at least 20 feet away\n"
                            "(about 6 metres — out a window is perfect)\n"
                            "and keep looking until the countdown ends.")
        self._body.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._body)
        self._count_label = QLabel(str(duration))
        self._count_label.setStyleSheet(
            "font-size: 44px; font-weight: 800; color: #38bdf8;")
        self._count_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._count_label)
        secs = QLabel("seconds left")
        secs.setObjectName("muted")
        secs.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(secs)
        self._progress = QProgressBar()
        self._progress.setMaximum(duration)
        self._progress.setValue(duration)
        self._progress.setTextVisible(False)
        layout.addWidget(self._progress)
        self._btn_row = QHBoxLayout()
        for text, how in (("Snooze 5 min", "snooze"),
                          ("Skip this break", "skip"),
                          ("Done early", "done")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, h=how: self._finish(h))
            self._btn_row.addWidget(b)
        layout.addLayout(self._btn_row)
        foot = QLabel("Blinking and looking far away lets your eyes reset.")
        foot.setObjectName("muted")
        foot.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(foot)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        QApplication.beep()
        self._timer.start(1000)

    def _tick(self):
        self._left -= 1
        if self._left <= 0:
            self._timer.stop()
            self._finished = True
            self.break_finished.emit("completed")
            self._show_completed()
            return
        self._count_label.setText(str(self._left))
        self._progress.setValue(self._left)

    def _show_completed(self):
        self._body.setText("")
        self._count_label.setText("✓")
        self._count_label.setStyleSheet(
            "font-size: 44px; font-weight: 800; color: #4ade80;")
        title_done = QLabel("Break complete — nice, your eyes got their reset.\n"
                            "This closes in a few seconds.")
        title_done.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout().addWidget(title_done)
        QApplication.beep()
        QTimer.singleShot(2500, self.close)

    def _finish(self, how: str):
        if self._finished:
            return
        self._finished = True
        self._timer.stop()
        self.break_finished.emit(how)
        self.close()

    def closeEvent(self, event):
        # Closing the window by hand counts as skipping the break
        if not self._finished:
            self._finished = True
            self._timer.stop()
            self.break_finished.emit("skip")
        super().closeEvent(event)


class FullPlotDialog(QDialog):
    """Full-session per-minute plot with its summary info line."""

    save_per_minute_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Blinks per minute over time")
        self.resize(920, 540)
        self.setMinimumSize(640, 380)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 12)
        title = QLabel("Blinks per minute over time")
        title.setStyleSheet("font-size: 16px; font-weight: 800; color: #f8fafc;")
        layout.addWidget(title)
        self.info_label = QLabel("")
        self.info_label.setObjectName("muted")
        self.info_label.setWordWrap(True)
        layout.addWidget(self.info_label)
        self.plot = LinePlotWidget()
        layout.addWidget(self.plot, stretch=1)
        row = QHBoxLayout()
        save_btn = QPushButton("Save per-minute data (CSV)…")
        save_btn.clicked.connect(self.save_per_minute_requested.emit)
        row.addWidget(save_btn)
        hint = QLabel("Updates live. Hollow dot = current minute. "
                      "Background color = activity. Red columns = away; "
                      "red ✕ = unreliable minute.")
        hint.setObjectName("muted")
        row.addWidget(hint, stretch=1)
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        row.addWidget(close_btn)
        layout.addLayout(row)



# --------------------------------------------------------------------------
# Camera / detection worker (runs in its own QThread)
# --------------------------------------------------------------------------

class CaptureWorker(QObject):
    """Captures webcam frames and runs blink detection off the UI thread.

    Emits each frame as a QImage for display, per-frame metrics
    (EAR / openness value, face-present flag), and a blink signal the
    moment a blink completes. The main window assigns the blink's
    timestamp from its own session clock when the signal arrives.
    Frames are processed and discarded — never recorded.
    """

    frame_ready = Signal(QImage)
    metrics = Signal(object, bool)       # (value: float | None, face_present)
    blink = Signal()
    breath_sample = Signal(float)        # median vertical chest motion (px/frame)
    camera_error = Signal(str)

    def __init__(self, engine: str):
        super().__init__()
        self.engine = engine  # "mediapipe" | "opencv"
        self.threshold = 0.21
        self.min_frames = 2
        self._timer: QTimer | None = None
        self._cap = None
        self._face_mesh = None
        self._face_cascade = None
        self._eye_cascade = None
        self._below_frames = 0
        self._blink_count = 0
        # Breathing (chest-motion) tracking state. Points live in a
        # half-resolution grayscale copy of the frame (see
        # _update_breathing); _last_face_box is (x, y, w, h) in
        # full-frame pixels, refreshed by both detection engines.
        self._last_face_box: tuple[int, int, int, int] | None = None
        self._breath_prev_gray = None
        self._breath_points = None
        self._breath_last_detect = 0.0

    @Slot()
    def reset_counter(self):
        self._blink_count = 0
        self._below_frames = 0

    @Slot(float, int)
    def set_params(self, threshold: float, min_frames: int):
        self.threshold = float(threshold)
        self.min_frames = int(min_frames)

    @Slot(int)
    def start(self, camera_index: int):
        if cv2 is None:
            self.camera_error.emit("opencv-python is not installed.")
            return
        if self.engine == "opencv":
            self._face_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self._eye_cascade = cv2.CascadeClassifier(
                cv2.data.haarcascades + "haarcascade_eye.xml")
            if self._face_cascade.empty() or self._eye_cascade.empty():
                self.camera_error.emit(
                    "OpenCV could not load its built-in Haar cascades.\n"
                    "Reinstall opencv-python and try again.")
                return
        self._cap = cv2.VideoCapture(int(camera_index))
        if not self._cap.isOpened():
            self._cap = None
            self.camera_error.emit(
                f"Could not open camera {camera_index}.\n"
                "Check it's not in use by another app, or try a different "
                "camera index.")
            return
        if self.engine == "mediapipe":
            self._face_mesh = mp.solutions.face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            )
        self._below_frames = 0
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        self._timer.start(15)

    @Slot()
    def stop(self):
        if self._timer is not None:
            self._timer.stop()
            self._timer = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self._face_mesh is not None:
            self._face_mesh.close()
            self._face_mesh = None
        self._face_cascade = None
        self._eye_cascade = None
        self._last_face_box = None
        self._breath_prev_gray = None
        self._breath_points = None

    # ------------------------------------------------------------ frames
    def _tick(self):
        if self._cap is None:
            return
        ok, frame = self._cap.read()
        if not ok:
            self.stop()
            self.camera_error.emit("Camera read failed — stopping.")
            return
        frame = cv2.flip(frame, 1)  # mirror view, feels natural
        if self.engine == "mediapipe":
            value, face_present = self._process_mediapipe(frame)
        else:
            value, face_present = self._process_opencv(frame)

        # Breathing: track chest/shoulder motion below the face and
        # outline the watched region so the user can frame themselves.
        roi = self._update_breathing(frame)
        if roi is not None:
            rx, ry, rw, rh = roi
            # ACCENT #38bdf8 in BGR
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh),
                          (248, 189, 56), 1)

        cv2.putText(frame, f"Blinks: {self._blink_count}", (12, 32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = rgb.shape[:2]
        image = QImage(rgb.data, w, h, rgb.strides[0],
                       QImage.Format.Format_RGB888).copy()
        self.frame_ready.emit(image)
        self.metrics.emit(value, face_present)

    def _process_mediapipe(self, frame):
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self._face_mesh.process(rgb)
        if not result.multi_face_landmarks:
            self._last_face_box = None
            return None, False
        landmarks = result.multi_face_landmarks[0].landmark
        xs = [lm.x * w for lm in landmarks]
        ys = [lm.y * h for lm in landmarks]
        fx, fy = int(min(xs)), int(min(ys))
        self._last_face_box = (fx, fy, int(max(xs)) - fx, int(max(ys)) - fy)
        left_ear = eye_aspect_ratio(landmarks, LEFT_EYE, w, h)
        right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0
        for indices in (LEFT_EYE, RIGHT_EYE):
            for idx in indices:
                lm = landmarks[idx]
                cv2.circle(frame, (int(lm.x * w), int(lm.y * h)), 2,
                           (0, 255, 0), -1)
        self._process_value(ear)
        return ear, True

    def _process_opencv(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self._face_cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(80, 80))
        if not len(faces):
            # No face: do not count this as a blink; just wait.
            self._below_frames = 0
            self._last_face_box = None
            return None, False
        x, y, w, h = max(faces, key=lambda r: r[2] * r[3])
        self._last_face_box = (int(x), int(y), int(w), int(h))
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y + h // 2, x:x + w]
        eyes = self._eye_cascade.detectMultiScale(
            roi_gray, scaleFactor=1.1, minNeighbors=5, minSize=(18, 18))
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(frame, (x + ex, y + ey),
                          (x + ex + ew, y + ey + eh), (255, 0, 0), 2)
        openness = 1.0 if len(eyes) >= 1 else 0.0
        self._process_value(openness)
        return openness, True

    def _process_value(self, value: float):
        """Blink edge detection: closed for >= min_frames, then reopen."""
        if value < self.threshold:
            self._below_frames += 1
        else:
            if self._below_frames >= self.min_frames:
                self._blink_count += 1
                self.blink.emit()
            self._below_frames = 0

    # --------------------------------------------------------- breathing
    def _update_breathing(self, frame):
        """Track chest/shoulder motion below the face (breathing).

        Shi-Tomasi points inside a "chest ROI" (the frame region below
        the face, widened ~40% each side) are followed with Lucas-Kanade
        optical flow, and the median vertical displacement is emitted
        via `breath_sample`. To keep the CPU cost modest the flow runs
        on a half-resolution grayscale copy; the emitted dy is scaled
        back to full-frame pixels. Returns the ROI as (x, y, w, h) in
        full-frame pixels so the caller can draw it, or None when
        breathing can't be tracked this frame (no face, or the region
        below the face is too short) — in that case nothing is emitted
        and the main window treats it as a gap in the data.
        """
        box = self._last_face_box
        if box is None or np is None:
            self._breath_points = None
            self._breath_prev_gray = None
            return None
        frame_h, frame_w = frame.shape[:2]
        fx, fy, fw, fh = box
        x0 = max(0, int(fx - 0.4 * fw))
        x1 = min(frame_w, int(fx + fw + 0.4 * fw))
        y0 = min(frame_h, int(fy + fh))  # just below the chin
        y1 = frame_h
        if y1 - y0 < 40 or x1 - x0 < 20:
            self._breath_points = None
            self._breath_prev_gray = None
            return None
        roi_rect = (x0, y0, x1 - x0, y1 - y0)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (frame_w // 2, frame_h // 2),
                           interpolation=cv2.INTER_AREA)
        hx0, hy0, hx1, hy1 = x0 // 2, y0 // 2, x1 // 2, y1 // 2

        now = time.monotonic()
        need_detect = (self._breath_points is None
                       or len(self._breath_points) < 15
                       or self._breath_prev_gray is None
                       or now - self._breath_last_detect > 5.0)
        if need_detect:
            corners = cv2.goodFeaturesToTrack(
                small[hy0:hy1, hx0:hx1], 60, 0.01, 5, blockSize=5)
            if corners is not None and len(corners) >= 8:
                pts = corners.reshape(-1, 1, 2).astype(np.float32)
                pts[:, :, 0] += hx0
                pts[:, :, 1] += hy0
                self._breath_points = pts
                self._breath_last_detect = now
            else:
                self._breath_points = None
            # No displacement yet — flow needs two frames.
            self._breath_prev_gray = small
            return roi_rect

        new_pts, status, _err = cv2.calcOpticalFlowPyrLK(
            self._breath_prev_gray, small, self._breath_points, None,
            winSize=(15, 15), maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                      10, 0.03))
        self._breath_prev_gray = small
        if new_pts is None or status is None:
            self._breath_points = None
            return roi_rect
        ok = status.reshape(-1) == 1
        if int(ok.sum()) >= 8:
            old = self._breath_points[ok].reshape(-1, 2)
            new = new_pts[ok].reshape(-1, 2)
            dy = float(np.median(new[:, 1] - old[:, 1])) * 2.0
            self.breath_sample.emit(dy)
            self._breath_points = new_pts[ok].reshape(-1, 1, 2)
        else:
            # Too few points survived — re-detect next frame.
            self._breath_points = None
        return roi_rect



# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class MainWindow(QMainWindow):
    start_requested = Signal(int)
    stop_requested = Signal()
    params_changed = Signal(float, int)
    reset_counter_requested = Signal()

    def __init__(self, engine: str):
        super().__init__()
        self.engine = engine  # "mediapipe" | "opencv"
        self.setWindowTitle("Blink Rate Detector")
        self.resize(1180, 760)
        self.setMinimumSize(1020, 660)

        # --- State (same model as the Tkinter version) ---
        self.running = False
        self.total_blinks = 0
        # Blink times are stored in *session-elapsed* seconds (the clock
        # that freezes while the camera is paused), so pauses and away
        # periods line up with the plot and the per-minute buckets.
        self.blink_timestamps: list[float] = []
        self.session_start: float | None = None
        self.elapsed_before_pause = 0.0

        # --- Presence / away state (session-elapsed seconds) ---
        self.away_intervals: list[tuple[float, float]] = []
        self.away_start: float | None = None   # open (confirmed) away interval
        self.no_face_since: float | None = None  # first frame with no face

        self.ear_history: deque[float] = deque(maxlen=90)

        # --- Breathing state ---
        # (session_elapsed, median vertical chest motion px) samples
        # from the worker; stamped on the same session clock as blinks
        # so pauses line up. Only the last ~95 s are kept.
        self.breath_samples: deque[tuple[float, float]] = deque()
        self.breath_rate_smooth: float | None = None  # EMA'd display rate
        self._breath_sum = 0.0        # accepted per-tick estimates (for avg)
        self._breath_ticks = 0
        self._breath_usable_seconds = 0.0

        # --- Activity segments (session-elapsed seconds) ---
        # [start, end_or_None, name]; the open segment runs to "now".
        self.activity_segments: list[list] = []
        self._activity_color_idx: dict[str, int] = {}
        self._breakdown_text_cache = ""
        self._auto_tick_count = 0

        # --- Low blink rate alert state ---
        self._low_since: float | None = None
        self._alert_cooldown_until = 0.0

        # --- 20-20-20 break state ---
        self.breaks_taken = 0
        self.breaks_skipped = 0
        # Session-elapsed seconds at which the next break is due.
        self.next_break_at = 20 * 60.0
        self.break_dialog: BreakDialog | None = None

        self._full_plot: FullPlotDialog | None = None
        self._last_image: QImage | None = None

        self._build_ui()

        # --- Camera worker in its own thread ---
        self._thread = QThread(self)
        self.worker = CaptureWorker(engine)
        self.worker.moveToThread(self._thread)
        self.start_requested.connect(self.worker.start)
        self.stop_requested.connect(self.worker.stop)
        self.params_changed.connect(self.worker.set_params)
        self.reset_counter_requested.connect(self.worker.reset_counter)
        self.worker.frame_ready.connect(self._on_frame)
        self.worker.metrics.connect(self._on_metrics)
        self.worker.blink.connect(self._on_blink)
        self.worker.breath_sample.connect(self._on_breath_sample)
        self.worker.camera_error.connect(self._on_camera_error)
        self._thread.start()
        self._push_params()

        # Low-rate badge (created hidden) — must exist before the first
        # stats tick, which may touch it.
        self.badge = LowRateBadge()
        self.badge.dismissed.connect(self._dismiss_low_rate_alert)

        # Stats tick (replaces the Tkinter after(500) loop)
        self._stats_timer = QTimer(self)
        self._stats_timer.timeout.connect(self._update_stats)
        self._stats_timer.start(500)
        self._update_stats()

    # ---------------------------------------------------------------- UI
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(14, 12, 14, 14)
        root.setSpacing(10)

        # Header bar
        header = QHBoxLayout()
        title = QLabel("👁  Blink Rate Detector")
        title.setObjectName("headerTitle")
        header.addWidget(title)
        header.addStretch(1)
        engine_text = ("Engine: MediaPipe" if self.engine == "mediapipe"
                       else "Engine: OpenCV (fallback)")
        self.engine_label = QLabel(engine_text)
        self.engine_label.setObjectName("enginePill")
        header.addWidget(self.engine_label)
        root.addLayout(header)

        body = QHBoxLayout()
        body.setSpacing(12)
        root.addLayout(body, stretch=1)

        # ---- Left column: video + controls + EAR sparkline ----
        left = QVBoxLayout()
        left.setSpacing(8)
        body.addLayout(left, stretch=3)

        self.video_label = QLabel("Camera is off\n\nPress  Start Camera  to begin")
        self.video_label.setObjectName("videoLabel")
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setMinimumSize(480, 340)
        left.addWidget(self.video_label, stretch=1)

        self.status_label = QLabel("Status: Idle  •  No face detected")
        left.addWidget(self.status_label)

        ear_caption = QLabel("Eye openness (EAR) — blink when it dips below the red line:"
                             if self.engine == "mediapipe" else
                             "Eye openness (proxy: 1 = eyes found, 0 = eyes not found):")
        ear_caption.setObjectName("muted")
        left.addWidget(ear_caption)
        self.ear_spark = EarSparkline()
        left.addWidget(self.ear_spark)

        btn_row = QHBoxLayout()
        self.start_btn = QPushButton("▶  Start Camera")
        self.start_btn.setObjectName("accent")
        self.start_btn.clicked.connect(self._toggle_camera)
        btn_row.addWidget(self.start_btn)
        reset_btn = QPushButton("Reset stats")
        reset_btn.clicked.connect(self.reset_stats)
        btn_row.addWidget(reset_btn)
        save_btn = QPushButton("Save log (CSV)…")
        save_btn.clicked.connect(self.save_csv)
        btn_row.addWidget(save_btn)
        left.addLayout(btn_row)

        # ---- Right column: scrollable stack of cards ----
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFixedWidth(448)
        body.addWidget(scroll)
        right = QWidget()
        scroll.setWidget(right)
        col = QVBoxLayout(right)
        col.setContentsMargins(2, 2, 8, 2)
        col.setSpacing(4)

        col.addWidget(self._build_stats_card())
        col.addWidget(self._build_breath_card())
        col.addWidget(self._build_plot_card())
        col.addWidget(self._build_activity_card())
        col.addWidget(self._build_alert_card())
        col.addWidget(self._build_breaks_card())
        col.addWidget(self._build_presence_card())
        col.addWidget(self._build_settings_card())
        col.addStretch(1)

    def _stat_row(self, grid: QGridLayout, row: int, name: str,
                  big: bool = False) -> QLabel:
        label = QLabel(name)
        label.setObjectName("statName")
        value = QLabel("—")
        value.setObjectName("statValueBig" if big else "statValue")
        value.setAlignment(Qt.AlignmentFlag.AlignRight)
        grid.addWidget(label, row, 0)
        grid.addWidget(value, row, 1)
        return value

    def _build_stats_card(self) -> QGroupBox:
        box = QGroupBox("Your blink rate")
        grid = QGridLayout(box)
        grid.setColumnStretch(0, 1)
        self.stat_total = self._stat_row(grid, 0, "Total blinks:", big=True)
        self.stat_elapsed = self._stat_row(grid, 1, "Session time:")
        self.stat_away = self._stat_row(grid, 2, "Time away (excluded):")
        self.stat_bpm = self._stat_row(grid, 3, "Blinks / minute:", big=True)
        self.stat_bph = self._stat_row(grid, 4, "Blinks / hour (projected):")
        self.stat_rolling = self._stat_row(grid, 5, "Blinks / hour (last 5 min):")
        self.stat_last60 = self._stat_row(grid, 6, "Blinks in last 60 seconds:")
        note = QLabel("All rates use time present only — away time is excluded. "
                      "Projected = whole-session average scaled to 1 hour. "
                      "Last 5 min = recent rate over present time in that window.")
        note.setObjectName("muted")
        note.setWordWrap(True)
        grid.addWidget(note, 7, 0, 1, 2)
        return box

    def _build_breath_card(self) -> QGroupBox:
        box = QGroupBox("Breathing")
        layout = QVBoxLayout(box)
        self.breath_value = QLabel("— breaths/min")
        self.breath_value.setObjectName("statValueBig")
        layout.addWidget(self.breath_value)
        self.breath_status = QLabel(
            "Watching your chest and shoulders — sit so they're in frame")
        self.breath_status.setObjectName("muted")
        self.breath_status.setWordWrap(True)
        layout.addWidget(self.breath_status)
        self.breath_wave = BreathWaveform()
        layout.addWidget(self.breath_wave)
        self.breath_avg = QLabel("Session avg: —")
        self.breath_avg.setObjectName("muted")
        self.breath_avg.setWordWrap(True)
        layout.addWidget(self.breath_avg)
        return box

    def _build_plot_card(self) -> QGroupBox:
        box = QGroupBox("Blinks per minute over time")
        layout = QVBoxLayout(box)
        self.mini_plot = LinePlotWidget()
        self.mini_plot.setMinimumHeight(150)
        layout.addWidget(self.mini_plot)
        row = QHBoxLayout()
        full_btn = QPushButton("Full session plot…")
        full_btn.clicked.connect(self.open_full_plot)
        row.addWidget(full_btn)
        per_min_btn = QPushButton("Save per-minute data (CSV)…")
        per_min_btn.clicked.connect(self.save_per_minute_csv)
        row.addWidget(per_min_btn)
        layout.addLayout(row)
        return box

    def _build_activity_card(self) -> QGroupBox:
        box = QGroupBox("What are you doing?")
        layout = QVBoxLayout(box)
        self.activity_combo = QComboBox()
        self.activity_combo.setEditable(True)
        self.activity_combo.addItems(ACTIVITY_PRESETS)
        self.activity_combo.setCurrentText(DEFAULT_ACTIVITY)
        self.activity_combo.activated.connect(lambda _i: self._on_activity_changed())
        line_edit = self.activity_combo.lineEdit()
        if line_edit is not None:
            line_edit.editingFinished.connect(self._on_activity_changed)
        layout.addWidget(self.activity_combo)
        self.auto_check = QCheckBox("Auto-detect from the active window (Windows)")
        layout.addWidget(self.auto_check)
        hint = QLabel("Pick what you're doing — or type your own label and press "
                      "Enter. Switching starts a new segment; away time is not "
                      "counted against any activity.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        sub = QLabel("Blink rate by activity")
        sub.setStyleSheet("font-weight: 700; color: #7dd3fc;")
        layout.addWidget(sub)
        self.breakdown_list = QListWidget()
        self.breakdown_list.setMaximumHeight(132)
        layout.addWidget(self.breakdown_list)
        return box

    def _build_alert_card(self) -> QGroupBox:
        box = QGroupBox("Low blink rate alerts")
        layout = QVBoxLayout(box)
        self.alert_enabled = QCheckBox("Badge drops down from the top of the screen")
        self.alert_enabled.setChecked(True)
        layout.addWidget(self.alert_enabled)
        row = QHBoxLayout()
        row.addWidget(QLabel("Warn below"))
        self.alert_threshold = QSpinBox()
        self.alert_threshold.setRange(3, 20)
        self.alert_threshold.setValue(8)
        row.addWidget(self.alert_threshold)
        row.addWidget(QLabel("blinks/min, sustained for"))
        self.alert_sustain = QSpinBox()
        self.alert_sustain.setRange(15, 300)
        self.alert_sustain.setSingleStep(15)
        self.alert_sustain.setValue(60)
        row.addWidget(self.alert_sustain)
        row.addWidget(QLabel("sec"))
        row.addStretch(1)
        layout.addLayout(row)
        hint = QLabel("A badge with a pair of blinking dry eyes slides down from "
                      "the top of the screen. It hides itself once your rate "
                      "recovers; Dismiss silences it for 5 minutes.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return box

    def _build_breaks_card(self) -> QGroupBox:
        box = QGroupBox("20-20-20 eye breaks")
        layout = QVBoxLayout(box)
        self.break_enabled = QCheckBox("Enable break notifications")
        self.break_enabled.setChecked(True)
        layout.addWidget(self.break_enabled)
        grid = QGridLayout()
        grid.setColumnStretch(0, 1)
        self.stat_next_break = self._stat_row(grid, 0, "Next break in:")
        self.stat_breaks = self._stat_row(grid, 1, "Breaks taken:")
        layout.addLayout(grid)
        row = QHBoxLayout()
        row.addWidget(QLabel("Every"))
        self.break_interval = QSpinBox()
        self.break_interval.setRange(1, 60)
        self.break_interval.setValue(20)
        row.addWidget(self.break_interval)
        row.addWidget(QLabel("min, look away for"))
        self.break_duration = QSpinBox()
        self.break_duration.setRange(5, 120)
        self.break_duration.setValue(20)
        row.addWidget(self.break_duration)
        row.addWidget(QLabel("sec"))
        row.addStretch(1)
        layout.addLayout(row)
        hint = QLabel("The timer pauses when the camera is paused or you're away. "
                      "(Set the interval to 1 min to test it.)")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        btns = QHBoxLayout()
        now_btn = QPushButton("Take a break now")
        now_btn.clicked.connect(self.open_break_window)
        btns.addWidget(now_btn)
        reset_btn = QPushButton("Reset break timer")
        reset_btn.clicked.connect(self.reset_break_timer)
        btns.addWidget(reset_btn)
        layout.addLayout(btns)
        return box

    def _build_presence_card(self) -> QGroupBox:
        box = QGroupBox("Presence detection")
        layout = QVBoxLayout(box)
        self.presence_enabled = QCheckBox("Detect when I step away")
        self.presence_enabled.setChecked(True)
        layout.addWidget(self.presence_enabled)
        row = QHBoxLayout()
        row.addWidget(QLabel("Grace period before 'away':"))
        self.away_grace = QSpinBox()
        self.away_grace.setRange(2, 30)
        self.away_grace.setValue(5)
        row.addWidget(self.away_grace)
        row.addWidget(QLabel("seconds"))
        row.addStretch(1)
        layout.addLayout(row)
        status_row = QHBoxLayout()
        name = QLabel("Presence:")
        name.setObjectName("statName")
        status_row.addWidget(name)
        self.presence_status = QLabel("—")
        self.presence_status.setObjectName("statValue")
        self.presence_status.setAlignment(Qt.AlignmentFlag.AlignRight)
        status_row.addWidget(self.presence_status, stretch=1)
        layout.addLayout(status_row)
        hint = QLabel("Away stretches are shaded red on the plot, mostly-away "
                      "minutes are marked unreliable, away time is taken out of "
                      "the rate calculations, and the break timer pauses.")
        hint.setObjectName("muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        return box

    def _build_settings_card(self) -> QGroupBox:
        box = QGroupBox("Settings")
        layout = QVBoxLayout(box)
        layout.addWidget(QLabel("Blink threshold (EAR)" if self.engine == "mediapipe"
                                else "Blink threshold (openness proxy)"))
        thr_row = QHBoxLayout()
        self.threshold_spin = QDoubleSpinBox()
        self.threshold_spin.setRange(0.05, 0.95)
        self.threshold_spin.setSingleStep(0.01)
        self.threshold_spin.setDecimals(2)
        self.threshold_spin.setValue(0.21 if self.engine == "mediapipe" else 0.50)
        self.threshold_spin.valueChanged.connect(lambda _v: self._push_params())
        thr_row.addWidget(self.threshold_spin)
        thr_hint = QLabel("Lower = less sensitive. Raise it if blinks are missed, "
                          "lower it if normal eye movements are counted."
                          if self.engine == "mediapipe" else
                          "Eyes count as closed when the openness proxy drops "
                          "below this. Leave at 0.50 unless counts look off.")
        thr_hint.setObjectName("muted")
        thr_hint.setWordWrap(True)
        thr_row.addWidget(thr_hint, stretch=1)
        layout.addLayout(thr_row)
        frames_row = QHBoxLayout()
        frames_row.addWidget(QLabel("Min. closed frames:"))
        self.min_frames_spin = QSpinBox()
        self.min_frames_spin.setRange(1, 6)
        self.min_frames_spin.setValue(2)
        self.min_frames_spin.valueChanged.connect(lambda _v: self._push_params())
        frames_row.addWidget(self.min_frames_spin)
        frames_hint = QLabel("(higher = fewer false positives)")
        frames_hint.setObjectName("muted")
        frames_row.addWidget(frames_hint)
        frames_row.addStretch(1)
        layout.addLayout(frames_row)
        cam_row = QHBoxLayout()
        cam_row.addWidget(QLabel("Camera index:"))
        self.camera_spin = QSpinBox()
        self.camera_spin.setRange(0, 4)
        self.camera_spin.setValue(0)
        cam_row.addWidget(self.camera_spin)
        cam_hint = QLabel("(try 1 if 0 shows the wrong camera)")
        cam_hint.setObjectName("muted")
        cam_row.addWidget(cam_hint)
        cam_row.addStretch(1)
        layout.addLayout(cam_row)
        return box

    def _push_params(self):
        self.params_changed.emit(self.threshold_spin.value(),
                                 self.min_frames_spin.value())


    # ------------------------------------------------------------- Camera
    def _toggle_camera(self):
        if self.running:
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if cv2 is None or np is None:
            missing = []
            if cv2 is None:
                missing.append("opencv-python")
            if np is None:
                missing.append("numpy")
            QMessageBox.critical(
                self, "Missing packages",
                "These packages are not installed:\n\n"
                + ", ".join(missing)
                + "\n\nInstall them with:\n  pip install -r requirements_qt.txt")
            return
        if self.session_start is None:
            self.session_start = time.monotonic()
        else:
            # Resuming after a pause
            self.session_start = time.monotonic() - self.elapsed_before_pause
        self.running = True
        self.start_btn.setText("⏸  Pause")
        # Open an activity segment unless one is already open (segments
        # stay open across pauses — the session clock freezes with them)
        if not self.activity_segments or self.activity_segments[-1][1] is not None:
            self._open_activity_segment(self.elapsed_seconds())
        self.start_requested.emit(int(self.camera_spin.value()))

    def stop_camera(self):
        if not self.running:
            return
        self.running = False
        self.stop_requested.emit()
        if self.session_start is not None:
            self.elapsed_before_pause = time.monotonic() - self.session_start
        # A deliberate pause is not "away" — close any open away interval here
        if self.away_start is not None:
            self.away_intervals.append((self.away_start, self.elapsed_before_pause))
            self.away_start = None
        self.no_face_since = None
        self._last_image = None
        self.video_label.setPixmap(QPixmap())
        self.video_label.setText("Camera paused\n\nPress Start Camera to resume")
        self.start_btn.setText("▶  Start Camera")
        self._hide_low_rate_alert()
        self.status_label.setText("Status: Paused")

    def _on_camera_error(self, message: str):
        was_running = self.running
        if was_running:
            self.stop_camera()
        QMessageBox.warning(self, "Camera problem", message)

    # -------------------------------------------------------- Worker slots
    def _on_frame(self, image: QImage):
        self._last_image = image
        pix = QPixmap.fromImage(image)
        size = self.video_label.size()
        self.video_label.setPixmap(pix.scaled(
            size, Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _on_metrics(self, value, face_present: bool):
        self._update_presence(face_present)
        threshold = self.threshold_spin.value()
        if value is not None:
            self.ear_history.append(float(value))
            state = "BLINK" if value < threshold else "Open"
            kind = "EAR" if self.engine == "mediapipe" else "Openness"
            self.status_label.setText(
                f"Status: Face detected  •  {kind} {value:.3f}  •  {state}")
        else:
            self.status_label.setText(
                "Status: No face detected — look toward the camera")
        if self.away_start is not None:
            away_for = int(self.elapsed_seconds() - self.away_start)
            self.status_label.setText(
                f"Status: Away — no face for {away_for}s "
                f"(this time is shaded red and excluded from your rates)")
        self.ear_spark.set_data(self.ear_history, threshold,
                                ear_mode=self.engine == "mediapipe")

    def _on_blink(self):
        # The worker detected the blink; the timestamp comes from the
        # main window's session clock, exactly as in the Tkinter version.
        self.total_blinks += 1
        self.blink_timestamps.append(self.elapsed_seconds())

    # --------------------------------------------------------- Breathing
    def _on_breath_sample(self, dy: float):
        """One chest-motion sample from the worker.

        The worker only emits while a face is present; samples are
        stamped with the session-elapsed clock (which freezes on
        pause), exactly like blink timestamps.
        """
        if not self.running:
            return
        now = self.elapsed_seconds()
        self.breath_samples.append((now, float(dy)))
        while self.breath_samples and self.breath_samples[0][0] < now - 95.0:
            self.breath_samples.popleft()

    def estimate_breath_rate(self):
        """Estimate breathing rate from the buffered chest-motion samples.

        Returns (breaths_per_min | None, waveform, quality) where
        waveform is the band-passed breathing signal for the last
        30 s (10 Hz points, arbitrary units) and quality is one of
        "good", "low signal", "warming up".

        Method: resample the per-frame vertical displacements onto a
        uniform 10 Hz grid, detrend with a ~10 s moving average, then
        look for the FFT peak in the breathing band. (The integrated
        chest-position signal is used only for the display waveform —
        integrating the rate signal itself would let white motion
        noise masquerade as slow breathing.) The peak must stand out from the
        band's median magnitude (ratio >= 2) to be believed at all,
        and >= 3 for "good". The returned rate is smoothed across
        calls (EMA, alpha 0.3, changes clamped to ±3/min per call).
        """
        if np is None or not self.breath_samples:
            return None, [], "warming up"
        elapsed = self.elapsed_seconds()
        samples = [(t, v) for t, v in self.breath_samples
                   if t >= elapsed - 45.0]
        if not samples or elapsed - samples[-1][0] > 2.0:
            return None, [], "warming up"
        # Use the contiguous tail segment (no gap over 2 s)
        start = len(samples) - 1
        while start > 0 and samples[start][0] - samples[start - 1][0] <= 2.0:
            start -= 1
        seg = samples[start:]
        if seg[-1][0] - seg[0][0] < 25.0:
            return None, [], "warming up"

        times = np.array([t for t, _ in seg], dtype=float)
        dys = np.array([v for _, v in seg], dtype=float)
        n = int((times[-1] - times[0]) * 10) + 1
        grid = times[0] + np.arange(n) / 10.0
        # Detrend: subtract a ~10 s moving average (edge-padded)
        kernel = np.ones(101) / 101.0

        def detrended(values):
            return values - np.convolve(
                np.pad(values, 50, mode="edge"), kernel, mode="valid")

        # Display waveform: integrated chest position, band-passed.
        pos = detrended(np.interp(grid, times, np.cumsum(dys)))
        # Rate spectrum: use the per-frame displacement (dy) signal
        # itself, NOT the integrated position — integrating turns white
        # motion noise into Brownian noise whose spectrum piles up at
        # the lowest frequencies and can fake a confident slow "breath".
        sig = detrended(np.interp(grid, times, dys))

        freqs = np.fft.rfftfreq(n, d=0.1)
        # Band-passed waveform for display: keep 0.10–0.50 Hz (6–30/min)
        spec = np.fft.rfft(pos)
        spec[(freqs < 0.10) | (freqs > 0.50)] = 0.0
        wave = np.fft.irfft(spec, n=n)
        wave_points = [float(v) for v in wave[-300:]]

        # Rate: strongest peak in 0.12–0.45 Hz on a Hann-windowed FFT
        mags = np.abs(np.fft.rfft(sig * np.hanning(n)))
        band = (freqs >= 0.12) & (freqs <= 0.45)
        if not band.any():
            return None, wave_points, "low signal"
        band_mags = mags[band]
        peak = int(np.argmax(band_mags))
        median_mag = float(np.median(band_mags))
        ratio = float(band_mags[peak]) / median_mag if median_mag > 0 else 0.0
        if ratio < 2.0:
            return None, wave_points, "low signal"
        bpm_raw = float(freqs[band][peak]) * 60.0
        quality = "good" if ratio >= 3.0 else "low signal"
        if quality == "good":
            # Stationarity: both halves of the window must independently
            # agree on the rate — random motion gets a lucky peak in one
            # stretch, rarely the same one twice.
            half = n // 2
            half_bpms = []
            for seg in (sig[:half], sig[half:]):
                sm = np.abs(np.fft.rfft(seg * np.hanning(len(seg))))
                sf = np.fft.rfftfreq(len(seg), d=0.1)
                sb = (sf >= 0.12) & (sf <= 0.45)
                if not sb.any():
                    half_bpms = []
                    break
                half_bpms.append(float(sf[sb][int(np.argmax(sm[sb]))] * 60.0))
            if not (len(half_bpms) == 2
                    and abs(half_bpms[0] - half_bpms[1]) <= 2.5
                    and abs(half_bpms[0] - bpm_raw) <= 3.0
                    and abs(half_bpms[1] - bpm_raw) <= 3.0):
                quality = "low signal"
        prev = self.breath_rate_smooth
        if prev is None:
            smoothed = bpm_raw
        else:
            target = prev + max(-3.0, min(3.0, bpm_raw - prev))
            smoothed = prev + 0.3 * (target - prev)
        self.breath_rate_smooth = smoothed
        return smoothed, wave_points, quality

    def _update_breath_card(self):
        """Refresh the Breathing card from the latest estimate (per tick)."""
        bpm, waveform, quality = self.estimate_breath_rate()
        if waveform:
            self.breath_wave.set_data(waveform)
        if bpm is None:
            self.breath_value.setText("— breaths/min")
            if quality == "low signal":
                self.breath_status.setText(
                    "Low signal — try sitting still, chest in frame")
            elif self.running:
                self.breath_status.setText(
                    "Warming up… needs ~25 s of you sitting fairly still")
            else:
                self.breath_status.setText(
                    "Watching your chest and shoulders — sit so they're "
                    "in frame")
        else:
            self.breath_value.setText(f"{bpm:.1f} breaths/min")
            self.breath_status.setText(
                "Good signal" if quality == "good" else
                "Low signal — try sitting still, chest in frame")
            self._breath_sum += bpm
            self._breath_ticks += 1
            self._breath_usable_seconds += 0.5  # one stats tick
        if self._breath_ticks:
            self.breath_avg.setText(
                f"Session avg: "
                f"{self._breath_sum / self._breath_ticks:.1f} breaths/min "
                f"({self._breath_usable_seconds / 60:.1f} min of usable "
                f"signal)")
        else:
            self.breath_avg.setText("Session avg: —")

    # ------------------------------------------------------------- Clock
    def elapsed_seconds(self) -> float:
        if self.session_start is None:
            return self.elapsed_before_pause
        if self.running:
            return time.monotonic() - self.session_start
        return self.elapsed_before_pause

    # ---------------------------------------------------------- Presence
    def _update_presence(self, face_present: bool):
        """Track away intervals (session-elapsed seconds).

        A no-face stretch only becomes an away interval once it outlasts
        the grace period; the interval then starts at the first no-face
        frame, because that data really is missing.
        """
        elapsed = self.elapsed_seconds()
        if not self.presence_enabled.isChecked():
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
            grace = max(min(int(self.away_grace.value()), 30), 2)
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
        return self.activity_combo.currentText().strip() or DEFAULT_ACTIVITY

    def _activity_colors(self, name: str) -> tuple[str, str]:
        if name not in self._activity_color_idx:
            self._activity_color_idx[name] = len(self._activity_color_idx)
        return ACTIVITY_PALETTE[self._activity_color_idx[name] % len(ACTIVITY_PALETTE)]

    def _open_activity_segment(self, at: float):
        name = self._activity_name()
        self._activity_colors(name)  # assign its color in first-use order
        self.activity_segments.append([at, None, name])

    def _on_activity_changed(self):
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
        if not self.auto_check.isChecked():
            return
        if not self.running or self.session_start is None:
            return
        self._auto_tick_count += 1
        if self._auto_tick_count % 4 != 0:  # stats tick runs every 500 ms
            return
        info = foreground_window_info()
        if not info:
            return
        name = classify_activity(*info)
        if name and name != self._activity_name():
            self.activity_combo.setCurrentText(name)
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
            lines = [("Start the camera and pick what you're doing.", None)]
        else:
            lines = []
            for s in stats:
                mins = s["seconds"] / 60.0
                bpm = s["blinks"] / mins if mins > 0 else 0.0
                dur_m, dur_s = int(s["seconds"] // 60), int(s["seconds"] % 60)
                lines.append((f"● {s['name']} — {dur_m}:{dur_s:02d} • "
                              f"{s['blinks']} blinks • {bpm:.1f}/min "
                              f"({bpm * 60:,.0f}/hr)", s["name"]))
        text = "\n".join(line for line, _ in lines)
        if text == self._breakdown_text_cache:
            return
        self._breakdown_text_cache = text
        self.breakdown_list.clear()
        for line, name in lines:
            item = QListWidgetItem(line)
            if name is not None:
                item.setForeground(QColor(self._activity_colors(name)[1]))
            self.breakdown_list.addItem(item)

    # ------------------------------------------------------- Per-minute
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

    # ---------------------------------------------------------- Stats tick
    def _update_stats(self):
        elapsed = self.elapsed_seconds()
        self.stat_elapsed.setText(format_hms(elapsed))
        self.stat_total.setText(str(self.total_blinks))

        # Blink timestamps are session-elapsed seconds, so windows are too.
        last_60 = sum(1 for t in self.blink_timestamps if elapsed - t <= 60)
        last_300 = sum(1 for t in self.blink_timestamps if elapsed - t <= 300)
        self.stat_last60.setText(str(last_60))

        # Away time is excluded from every rate denominator
        away_total = self.away_seconds_total(elapsed)
        away_m, away_s = int(away_total // 60), int(away_total % 60)
        self.stat_away.setText(f"{away_m:02d}:{away_s:02d}" if self.away_start is None
                               else f"{away_m:02d}:{away_s:02d} (away now)")
        present = max(elapsed - away_total, 0.0)

        if present >= 10:
            self.stat_bpm.setText(f"{self.total_blinks / (present / 60):.1f}")
            self.stat_bph.setText(f"{self.total_blinks / (present / 3600):,.0f}")
        else:
            self.stat_bpm.setText("—")
            self.stat_bph.setText("—")

        # Rolling rate: blinks in the last 300 session seconds, divided by
        # the *present* time inside that same window
        window_len = min(elapsed, 300)
        away_in_window = self._away_overlap(elapsed - window_len, elapsed, elapsed)
        present_window = window_len - away_in_window
        if present_window >= 30:
            self.stat_rolling.setText(f"{last_300 / (present_window / 3600):,.0f}")
        else:
            self.stat_rolling.setText("collecting…")

        # Presence status
        if not self.presence_enabled.isChecked():
            self.presence_status.setText("Off")
        elif self.away_start is not None:
            self.presence_status.setText(
                f"Away — {int(elapsed - self.away_start)}s (excluded)")
        elif self.session_start is not None:
            self.presence_status.setText("Present")
        else:
            self.presence_status.setText("—")

        # 20-20-20 break display + timer
        self.stat_breaks.setText(
            f"{self.breaks_taken}" + (f"  (skipped {self.breaks_skipped})"
                                      if self.breaks_skipped else ""))
        if not self.break_enabled.isChecked():
            self.stat_next_break.setText("off")
        elif self.break_dialog is not None:
            self.stat_next_break.setText("break time!")
        elif self.session_start is None:
            self.stat_next_break.setText("starts with camera")
        elif self.away_start is not None:
            # Timer is paused while away — show the frozen remaining time
            remaining = max(self.next_break_at - self.away_start, 0)
            self.stat_next_break.setText(
                f"{int(remaining // 60)}:{int(remaining % 60):02d} (paused)")
        else:
            remaining = max(self.next_break_at - elapsed, 0)
            self.stat_next_break.setText(
                f"{int(remaining // 60)}:{int(remaining % 60):02d}")
        self._check_break_timer(elapsed)
        self._auto_detect_tick()
        recent_present = min(elapsed, 60.0) - self._away_overlap(
            max(0.0, elapsed - 60.0), elapsed, elapsed)
        self._update_low_rate_alert(elapsed, last_60, present, recent_present)
        self._update_breath_card()
        self._update_breakdown_panel(elapsed)
        self._refresh_mini_plot(elapsed)
        self._refresh_full_plot(elapsed)

    def _refresh_mini_plot(self, elapsed: float):
        series = self.per_minute_series(elapsed)
        if not series:
            self.mini_plot.set_data(
                [], empty_text="Start the camera to see your per-minute trend")
            return
        # Small chart: most recent 30 minutes (the full plot shows everything)
        away_pm = self.away_seconds_per_minute(elapsed)
        flags = self.unreliable_flags(elapsed, away_pm)
        window = series[-30:]
        win_start_min = float(len(series) - len(window))
        self.mini_plot.set_data(
            window, first_minute=len(series) - len(window) + 1,
            away_per_minute=away_pm[-30:], unreliable=flags[-30:],
            activity_bands=self._activity_bands(
                elapsed, win_start_min, float(len(window))))

    # -------------------------------------------------------- Full plot
    def open_full_plot(self):
        if self._full_plot is None:
            self._full_plot = FullPlotDialog(self)
            self._full_plot.save_per_minute_requested.connect(
                self.save_per_minute_csv)
        self._full_plot.show()
        self._full_plot.raise_()
        self._full_plot.activateWindow()
        self._refresh_full_plot(self.elapsed_seconds())

    def _refresh_full_plot(self, elapsed: float):
        if self._full_plot is None or not self._full_plot.isVisible():
            return
        series = self.per_minute_series(elapsed)
        if not series:
            self._full_plot.plot.set_data(
                [], empty_text="Start the camera — your per-minute plot will appear here")
            self._full_plot.info_label.setText("No data yet.")
            return
        away_pm = self.away_seconds_per_minute(elapsed)
        flags = self.unreliable_flags(elapsed, away_pm)
        self._full_plot.plot.set_data(
            series, first_minute=1, away_per_minute=away_pm, unreliable=flags,
            activity_bands=self._activity_bands(elapsed, 0.0, float(len(series))))
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
        act = self.activity_stats(elapsed)
        act_text = ""
        if act:
            parts = []
            for s in act:
                mins = s["seconds"] / 60.0
                bpm = s["blinks"] / mins if mins > 0 else 0.0
                parts.append(f"{s['name']}: {bpm:.1f}/min")
            act_text = "  •  by activity — " + ", ".join(parts)
        self._full_plot.info_label.setText(
            f"{len(series)} minutes so far  •  average {avg:.1f} blinks/min "
            f"(completed, reliable minutes)  •  peak {peak} blinks/min in minute "
            f"{peak_minute}  •  total {self.total_blinks} blinks  •  "
            f"away {int(away_total // 60)}m {int(away_total % 60)}s"
            + (f"  •  {n_unreliable} unreliable minute(s) marked ✕"
               if n_unreliable else "")
            + act_text)

    # ----------------------------------------------------- 20-20-20 breaks
    def _break_interval_seconds(self) -> float:
        minutes = max(min(int(self.break_interval.value()), 60), 1)
        return minutes * 60.0

    def _break_duration_seconds(self) -> int:
        return max(min(int(self.break_duration.value()), 120), 5)

    def reset_break_timer(self):
        self.next_break_at = self.elapsed_seconds() + self._break_interval_seconds()

    def _check_break_timer(self, elapsed: float):
        if not self.break_enabled.isChecked():
            return
        if self.break_dialog is not None:
            return  # a break popup is already open
        if self.session_start is None or not self.running:
            return  # timer runs on active (camera-on) session time
        if self.away_start is not None:
            return  # paused while the user is away (they're on a break already)
        if elapsed >= self.next_break_at:
            self.open_break_window()

    def open_break_window(self):
        if self.break_dialog is not None:
            self.break_dialog.raise_()
            self.break_dialog.activateWindow()
            return
        dialog = BreakDialog(self._break_duration_seconds(), self)
        self.break_dialog = dialog
        dialog.break_finished.connect(self._on_break_finished)
        dialog.finished.connect(lambda _r: setattr(self, "break_dialog", None))
        dialog.show()

    def _on_break_finished(self, how: str):
        # Schedule the next break from *now* (current session elapsed time)
        elapsed = self.elapsed_seconds()
        if how in ("completed", "done"):  # countdown done / ended early — count it
            self.breaks_taken += 1
            self.next_break_at = elapsed + self._break_interval_seconds()
        elif how == "snooze":
            self.next_break_at = elapsed + 5 * 60.0
        else:  # skip / window closed
            self.breaks_skipped += 1
            self.next_break_at = elapsed + self._break_interval_seconds()

    # ------------------------------------------------- Low-rate alerts
    def _update_low_rate_alert(self, elapsed: float, last_60: int,
                               present_total: float, recent_present: float):
        """Show/hide the low-blink-rate badge.

        Fires only when the rate over the last 60 session seconds stays
        below the threshold for the whole sustain period, with enough
        present time in the window (so stepping away can't trigger it).
        Recovers with hysteresis (+2/min) so it doesn't flicker.
        """
        threshold = int(self.alert_threshold.value())
        sustain = int(self.alert_sustain.value())
        if self.badge.shown:
            self.badge.set_text(self._alert_text(last_60))
        enabled = self.alert_enabled.isChecked()
        low_now = (enabled and self.running and present_total >= 120
                   and recent_present >= 45 and last_60 < threshold)
        if low_now:
            if self._low_since is None:
                self._low_since = elapsed
            elif (elapsed - self._low_since >= sustain
                  and not self.badge.shown
                  and elapsed >= self._alert_cooldown_until):
                self.badge.slide_down(self._alert_text(last_60))
        else:
            recovered = last_60 >= threshold + 2
            if recovered or not enabled or not self.running:
                self._low_since = None
                self._hide_low_rate_alert()

    @staticmethod
    def _alert_text(last_60: int) -> str:
        return (f"Low blink rate — {last_60} in the last minute "
                f"(healthy is ~15–20). Blink a few times, look far away.")

    def _dismiss_low_rate_alert(self):
        # Dismiss = hide now and stay quiet for 5 session minutes
        self._alert_cooldown_until = self.elapsed_seconds() + 300.0
        self._low_since = None
        self._hide_low_rate_alert()

    def _hide_low_rate_alert(self):
        if self.badge.shown:
            self.badge.slide_up()

    # ------------------------------------------------------------ Actions
    def reset_stats(self):
        self.total_blinks = 0
        self.blink_timestamps.clear()
        self.ear_history.clear()
        self.breath_samples.clear()
        self.breath_rate_smooth = None
        self._breath_sum = 0.0
        self._breath_ticks = 0
        self._breath_usable_seconds = 0.0
        self.breath_wave.set_data([])
        self.elapsed_before_pause = 0.0
        self.session_start = time.monotonic() if self.running else None
        self.breaks_taken = 0
        self.breaks_skipped = 0
        self.next_break_at = self._break_interval_seconds()
        self.away_intervals.clear()
        self.away_start = None
        self.no_face_since = None
        self.activity_segments.clear()
        self._low_since = None
        self._alert_cooldown_until = 0.0
        self._hide_low_rate_alert()
        self._activity_color_idx.clear()
        self._breakdown_text_cache = ""
        self.reset_counter_requested.emit()
        if self.running:
            self._open_activity_segment(0.0)
        self._update_stats()

    def save_csv(self):
        if not self.blink_timestamps:
            QMessageBox.information(self, "Nothing to save",
                                    "No blinks have been recorded yet.")
            return
        default = f"blink_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save blink log", default,
            "CSV files (*.csv);;All files (*.*)")
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

        # Also write a summary the user can keep
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
            if self._breath_ticks:
                f.write(f"\nBreathing: session average "
                        f"{self._breath_sum / self._breath_ticks:.1f} breaths/min "
                        f"over {self._breath_usable_seconds / 60:.1f} min of "
                        f"usable signal (estimated from chest/shoulder motion)\n")
        QMessageBox.information(self, "Saved", f"Saved:\n{path}\n{summary_path}")

    def save_per_minute_csv(self):
        elapsed = self.elapsed_seconds()
        series = self.per_minute_series(elapsed)
        if not series:
            QMessageBox.information(
                self, "Nothing to save",
                "No per-minute data yet — start the camera first.")
            return
        default = f"blinks_per_minute_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save per-minute data", default,
            "CSV files (*.csv);;All files (*.*)")
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
        QMessageBox.information(self, "Saved", f"Saved per-minute data to:\n{path}")

    # -------------------------------------------------------------- Close
    def closeEvent(self, event):
        self._stats_timer.stop()
        if self.break_dialog is not None:
            self.break_dialog.close()
            self.break_dialog = None
        if self._full_plot is not None:
            self._full_plot.close()
        self.badge.hide()
        if self.running:
            self.stop_camera()
        self.stop_requested.emit()
        self._thread.quit()
        self._thread.wait(3000)
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("Blink Rate Detector")
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLESHEET)

    if cv2 is None or np is None:
        missing = []
        if cv2 is None:
            missing.append("opencv-python")
        if np is None:
            missing.append("numpy")
        QMessageBox.critical(
            None, "Missing packages",
            "These packages are not installed:\n\n"
            + ", ".join(missing)
            + "\n\nInstall them with:\n  pip install -r requirements_qt.txt")
        return 1

    # Engine selection: MediaPipe if available, OpenCV Haar fallback otherwise
    engine = "mediapipe" if mp is not None else "opencv"
    window = MainWindow(engine)
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
