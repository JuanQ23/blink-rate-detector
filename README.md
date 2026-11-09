# Blink Rate Detector

A Python GUI program that uses your webcam to detect blinks in real time and shows your blink rate per minute and **per hour**.

## Files in this repo

- **`blink_detector.py`** — the main program (MediaPipe Face Mesh). Includes the live per-minute plot, 20-20-20 break notifications, presence detection, and activity categories described below. Use this one if you can.
- **`requirements.txt`** — dependencies for the main program (pinned `mediapipe==0.10.21`, Python 3.11/3.12 recommended).
- **`blink_detector_opencv.py`** — a fallback that uses only OpenCV Haar cascades, for Python versions where MediaPipe won't install (3.13+). It has the core GUI, blink stats, and activity categories, but not the line plot, break notifications, or presence detection, and it is less accurate.
- **`requirements_opencv.txt`** — dependencies for the fallback (OpenCV + Pillow only).

![No screenshot — run it locally to see the live GUI]

## What it shows

- **Total blinks** and session timer
- **Blinks / minute**
- **Blinks / hour (projected)** — whole-session average scaled to an hour
- **Blinks / hour (last 5 min)** — recent, rolling rate (steadier for long sessions)
- **Live EAR graph** — watch your eye-openness dip below the blink threshold
- **Blinks per minute over time (plot)** — a live line plot in the main window (most recent 30 minutes), with a shaded "typical 15–20/min" band. **Open full plot** shows the whole session in a large window with average/peak stats, and **Save per-minute data (CSV)** exports minute-by-minute counts
- **20-20-20 break notifications** — every 20 minutes of camera-on session time, a popup (with a beep, brought to the front) reminds you to look at something 20+ feet away for 20 seconds, with a live countdown, **Snooze 5 min**, **Skip**, and a breaks-taken counter. The timer pauses when the camera is paused. Interval and look-away duration are adjustable — set the interval to 1 minute to test it
- **Presence detection** — if your face isn't seen for more than the grace period (default 5 seconds, adjustable 2–30s), you're counted as *away*: that stretch is shaded red on the plot, a minute that is more than half away is marked unreliable (red ✕, line breaks around it instead of showing a fake dip), away time is excluded from all rate calculations, the 20-20-20 timer pauses, and the stats panel shows your total *Time away (excluded)*. Pausing the camera manually is not counted as away
- **Activity categories** — pick what you're doing in the *What are you doing?* dropdown (Video games, Watching a show, YouTube, Reading, Coding / work, Video call, Browsing, Other) or type your own label. Switching splits the session into activity segments: the **Blink rate by activity** panel keeps a separate time / blinks / blinks-per-minute (and per hour) for each, segments are drawn as colored background bands on the plot (bars are colored per minute in the fallback), switching back to an earlier activity merges its totals, and away time is not counted against any activity. **Auto-detect (optional, Windows):** tick *Auto-detect from the active window* and the foreground window's program + title are checked every ~2 seconds — YouTube or a streaming service in the browser tab title, coding apps, call apps, video players, and any fullscreen non-browser app (treated as a video game) switch the activity for you; unknown windows keep the current activity instead of guessing. Window titles are read locally and never stored or sent anywhere
- **Save log (CSV)** — every blink with a timestamp and the activity at that moment, plus a small `.summary.txt` file (including breaks taken/skipped, present vs. away time, and a by-activity breakdown). The per-minute CSV includes `away_seconds_in_minute`, a `reliable` yes/no column, and the activity that covered most of each minute

For context, a typical resting rate is about 15–20 blinks/min (900–1,200/hour). It often drops to 5–7/min during intense screen focus, which is linked to dry eyes.

## Install

You need a webcam and **Python 3.11 or 3.12** (recommended). MediaPipe ships prebuilt wheels for Python 3.9–3.12; on Python 3.13+ the install can fail with "No matching distribution found" — if that's your version, create the venv below with Python 3.12 specifically (e.g. `py -3.12 -m venv .venv` on Windows).

```bash
cd blink_rate_detector
python -m venv .venv

# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> Ubuntu/Debian: if you get `No module named tkinter`, run `sudo apt install python3-tk`.

## Run

```bash
python blink_detector.py
```

1. Press **Start Camera** (allow camera access if your OS asks).
2. Sit in decent, even light and look roughly toward the camera.
3. Your stats update live. Give the *last 5 min* rate a few minutes to stabilize.
4. Press **Save log (CSV)…** any time to export your session.

## Tuning it

- **Blink threshold (EAR)** — default `0.21`. If blinks are being missed, raise it slightly (e.g. 0.24). If normal squints/looks-down are counted, lower it (e.g. 0.19). The red dashed line on the EAR graph shows the threshold — a real blink should dip clearly below it and come back up.
- **Min. closed frames** — default `2`. Raise to `3` if you get false positives.
- **Camera index** — `0` is your default webcam. Try `1` for an external/USB camera.

## How it works

MediaPipe Face Mesh finds 6 landmarks around each eye. The app computes the Eye Aspect Ratio (EAR):

```
EAR = (|p2-p6| + |p3-p5|) / (2 * |p1-p4|)
```

EAR is ~0.25–0.35 with eyes open and drops toward 0 during a blink. When EAR stays below the threshold for the minimum number of frames and then rises again, one blink is counted and timestamped. Rates per hour are derived from those timestamps.

## Privacy

Everything runs locally on your computer. No video is recorded, stored, or sent anywhere. The only file written is the CSV/summary you explicitly save.

## Troubleshooting

- **Camera not found** — close other apps using the camera (Zoom, Teams, browser tabs), or try camera index 1.
- **No face detected** — improve lighting, move a bit closer, remove heavy backlight.
- **Counts seem low/high** — adjust the EAR threshold using the graph as a guide (see Tuning).
