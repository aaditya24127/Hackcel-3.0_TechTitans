#!/usr/bin/env python3
"""
Enterprise Web Dashboard Server for Event Crowd Analytics & Flow Tracker.
Features:
- Dedicated background inference thread (YOLOv8 + YuNet Face + ByteTrack + Zone Simulation)
- Zero-latency, race-condition-free MJPEG video streamer
- DirectShow webcam support on Windows
- Live responsive telemetry dashboard with auto-updating metrics and incoming entry logs
- Database event persistence in SQLite
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import base64
import io
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, StreamingResponse
import uvicorn

from src.db import DatabaseManager
from src.detector import FaceDetector, PersonDetector
from src.tracker import LineCrossingTracker
from src.video import VideoProcessor
from src.zones import ZoneManager

app = FastAPI(title="Crowd Analytics Module")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class StreamEngine:
    def __init__(self):
        self.mode: str = "IDLE"  # "CAMERA", "FILE", "IDLE"
        self.source: str | int = 0
        self.is_running: bool = False
        self.cap: Optional[cv2.VideoCapture] = None
        self._lock = threading.Lock()

        # Models & Trackers
        self.person_detector = PersonDetector(model_name="yolov8n.pt")
        self.face_detector = FaceDetector()
        self.tracker = LineCrossingTracker(num_gates=2, line_pos_ratio=0.60, direction="both")
        self.zone_mgr = ZoneManager(zone_names=["Zone MainStage", "Zone VIPLounge", "Zone FoodCourt", "Zone ExpoHall"])
        self.db = DatabaseManager("results.db")

        # Telemetry State
        self.current_frame_count: int = 0
        self.current_fps: float = 0.0
        self.people_detected_in_frame: int = 0
        self.recent_events: List[Dict] = []
        self.run_id: Optional[int] = None
        self.unique_faces: int = 0

        # Latest encoded JPEG frame
        self.latest_jpeg: bytes = self._generate_standby_frame()

        # Background processing worker thread
        self._stop_worker = False
        self._worker_thread = threading.Thread(target=self._processing_loop, daemon=True)
        self._worker_thread.start()

    def _generate_standby_frame(self) -> bytes:
        """Create a standby grid image."""
        blank = np.zeros((480, 640, 3), dtype=np.uint8)
        for x in range(0, 640, 40):
            cv2.line(blank, (x, 0), (x, 480), (22, 28, 38), 1)
        for y in range(0, 480, 40):
            cv2.line(blank, (0, y), (640, y), (22, 28, 38), 1)

        cv2.putText(blank, "CAMERA / VIDEO FEED IDLE", (135, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 215, 255), 2)
        cv2.putText(blank, "Click 'Start Camera' or 'Process Video' below", (120, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 160, 175), 1)
        cv2.putText(blank, "● STANDBY", (25, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)

        _, buf = cv2.imencode(".jpg", blank, [int(cv2.IMWRITE_JPEG_QUALITY), 75])
        return buf.tobytes()

    def start_camera(self, camera_idx: int = 0) -> bool:
        with self._lock:
            self._close_cap()
            self.mode = "CAMERA"
            self.source = camera_idx

            if os.name == "nt":
                self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
                if not self.cap or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(camera_idx)
            else:
                self.cap = cv2.VideoCapture(camera_idx)

            if self.cap and self.cap.isOpened():
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                self._reset_state(w, h)
                self.is_running = True
                return True
            else:
                self.is_running = False
                self.mode = "IDLE"
                return False

    def start_file(self, file_path: str) -> bool:
        with self._lock:
            self._close_cap()
            self.mode = "FILE"
            self.source = file_path
            self.cap = cv2.VideoCapture(file_path)

            if self.cap and self.cap.isOpened():
                w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
                h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
                self._reset_state(w, h)
                self.is_running = True
                return True
            else:
                self.is_running = False
                self.mode = "IDLE"
                return False

    def stop(self):
        with self._lock:
            self.is_running = False
            self.mode = "IDLE"
            self._close_cap()
            self.latest_jpeg = self._generate_standby_frame()

    def _close_cap(self):
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None

    def _reset_state(self, width: int, height: int):
        self.tracker = LineCrossingTracker(num_gates=2, line_pos_ratio=0.60, direction="both")
        self.tracker.set_frame_dimensions(width, height)
        self.zone_mgr = ZoneManager(zone_names=["Zone MainStage", "Zone VIPLounge", "Zone FoodCourt", "Zone ExpoHall"])
        try:
            self.run_id = self.db.start_run(str(self.source))
        except Exception:
            self.run_id = None
        self.recent_events = []
        self.current_frame_count = 0
        self.unique_faces = 0

    def _processing_loop(self):
        """Dedicated background loop that processes frames from the active source."""
        last_calc_time = time.time()
        fps_counter = 0

        while not self._stop_worker:
            if not self.is_running or not self.cap:
                time.sleep(0.08)
                continue

            with self._lock:
                if not self.cap or not self.cap.isOpened():
                    self.is_running = False
                    continue
                ret, frame = self.cap.read()

            if not ret:
                if self.mode == "FILE" and self.cap:
                    with self._lock:
                        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    time.sleep(0.03)
                    continue
                else:
                    time.sleep(0.05)
                    continue

            self.current_frame_count += 1
            fps_counter += 1

            # Update FPS every 0.5s
            now = time.time()
            if now - last_calc_time >= 0.5:
                self.current_fps = round(fps_counter / (now - last_calc_time), 1)
                fps_counter = 0
                last_calc_time = now

            video_time_s = self.current_frame_count / 30.0

            try:
                # 1. Person Detection & ByteTrack
                tracks = self.person_detector.track(frame, conf_threshold=0.20)
                self.people_detected_in_frame = len(tracks)

                # 2. Face Detection
                faces = self.face_detector.detect_faces(frame, self.current_frame_count, person_tracks=tracks)
                self.unique_faces = self.face_detector.unique_faces_seen

                # 3. Line Crossing & Attendee Registration
                new_entries = self.tracker.update_tracks(tracks, self.current_frame_count, video_time_s)

                # Process explicit line-crossing events
                for entry in new_entries:
                    zone = self.zone_mgr.assign_zone(entry.track_id)
                    entry.assigned_zone = zone
                    if self.run_id:
                        try:
                            self.db.log_entry(self.run_id, entry.track_id, entry.gate_name, entry.video_time_s, zone)
                        except Exception:
                            pass

                    event_dict = {
                        "series": len(self.recent_events) + 1,
                        "track_id": entry.track_id,
                        "gate": entry.gate_name,
                        "zone": zone,
                        "time": f"{entry.video_time_s:.1f}s",
                    }
                    self.recent_events.insert(0, event_dict)
                    if len(self.recent_events) > 30:
                        self.recent_events.pop()

                # In addition, auto-register any newly detected person in frame (e.g. for webcam / entrance monitoring)
                for track in tracks:
                    tid = track["id"]
                    if tid not in self.zone_mgr.person_zones:
                        cx = (track["bbox"][0] + track["bbox"][2]) / 2.0
                        frame_w = frame.shape[1] if frame is not None else 640
                        gate_idx = int(cx / (frame_w / max(1, self.tracker.num_gates)))
                        gate_idx = min(gate_idx, self.tracker.num_gates - 1)
                        gate_name = f"Gate {gate_idx + 1}"

                        zone = self.zone_mgr.assign_zone(tid)
                        self.tracker.counted_ids.add(tid)
                        self.tracker.gate_counts[gate_name] = self.tracker.gate_counts.get(gate_name, 0) + 1

                        if self.run_id:
                            try:
                                self.db.log_entry(self.run_id, tid, gate_name, video_time_s, zone)
                            except Exception:
                                pass

                        event_dict = {
                            "series": len(self.recent_events) + 1,
                            "track_id": tid,
                            "gate": gate_name,
                            "zone": zone,
                            "time": f"{video_time_s:.1f}s",
                        }
                        self.recent_events.insert(0, event_dict)
                        if len(self.recent_events) > 30:
                            self.recent_events.pop()

                # 5. Crowd Drift Simulation
                drift_transitions = self.zone_mgr.update_drift(video_time_s)
                for tid, fz, tz in drift_transitions:
                    if self.run_id:
                        try:
                            self.db.log_transition(self.run_id, tid, fz, tz, video_time_s)
                        except Exception:
                            pass

                # Draw Visual Overlays
                annotated = VideoProcessor.annotate_frame(
                    frame=frame,
                    person_tracks=tracks,
                    faces=faces,
                    tracker=self.tracker,
                    zone_mgr=self.zone_mgr,
                    frame_number=self.current_frame_count,
                    video_time_s=video_time_s,
                    fps_current=self.current_fps,
                )

                # Draw Custom Yellow/Cyan HUD Box
                hud_text = f"PEOPLE DETECTED: {self.people_detected_in_frame}"
                sub_text = f"Mode: {self.mode} | FPS: {self.current_fps:.1f}"
                cv2.rectangle(annotated, (15, 15), (320, 75), (15, 18, 22), -1)
                cv2.rectangle(annotated, (15, 15), (320, 75), (0, 215, 255), 1)
                cv2.putText(annotated, hud_text, (25, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 230, 255), 2, cv2.LINE_AA)
                cv2.putText(annotated, sub_text, (25, 63), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1, cv2.LINE_AA)

                # Encode Frame to JPEG
                _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
                self.latest_jpeg = buf.tobytes()

            except Exception as e:
                print(f"[Worker Error]: {e}")
                time.sleep(0.02)

            # Limit background loop rate to prevent excessive CPU consumption
            time.sleep(0.015)


engine = StreamEngine()


def frame_generator():
    """Ultra-stable MJPEG generator for browser clients."""
    while True:
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + engine.latest_jpeg + b"\r\n"
        )
        time.sleep(0.033)  # ~30 FPS streaming to browser


@app.get("/video_feed")
def video_feed():
    return StreamingResponse(
        frame_generator(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@app.get("/api/telemetry")
@app.get("/api/simulation/state")
def get_telemetry():
    zone_breakdown = engine.zone_mgr.get_occupancy_breakdown()
    return JSONResponse({
        "status": "LIVE" if engine.is_running else "IDLE",
        "mode": engine.mode,
        "fps": engine.current_fps,
        "people_detected": engine.people_detected_in_frame,
        "total_entered": engine.tracker.total_entered,
        "unique_faces": engine.unique_faces,
        "zones": zone_breakdown,
        "recent_events": engine.recent_events,
    })


@app.post("/api/start_camera")
def start_camera(source: int = Form(0)):
    success = engine.start_camera(source)
    return {"status": "success" if success else "failed", "mode": engine.mode, "source": source}


@app.post("/api/stop_camera")
def stop_camera():
    engine.stop()
    return {"status": "success", "mode": engine.mode}


@app.post("/api/upload_video")
async def upload_video(file: UploadFile = File(...)):
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(exist_ok=True)
    file_path = uploads_dir / file.filename
    with open(file_path, "wb") as f:
        f.write(await file.read())

    success = engine.start_file(str(file_path))
    return {"status": "success" if success else "failed", "filename": file.filename, "path": str(file_path)}


@app.post("/api/start_sample_video")
def start_sample_video():
    sample_path = "sample_entrance.mp4"
    if not os.path.exists(sample_path):
        return {"status": "error", "message": "Sample video not found"}
    success = engine.start_file(sample_path)
    return {"status": "success" if success else "failed", "path": sample_path}


@app.post("/api/reset_data")
def reset_data():
    with engine._lock:
        # 1. Clear database tables
        try:
            engine.db.clear_all_data()
        except Exception as e:
            print(f"[DB Clear Error]: {e}")

        # 2. Reset CSV files
        for csv_path in ["results.csv", "results2.csv"]:
            try:
                if os.path.exists(csv_path):
                    with open(csv_path, "w", encoding="utf-8") as f:
                        f.write("person_id,entry_timestamp,gate,assigned_zone\n")
            except Exception as e:
                print(f"[CSV Reset Error]: {e}")

        # 3. Reset in-memory trackers, counters & telemetry
        w = int(engine.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if engine.cap else 640
        h = int(engine.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if engine.cap else 480
        engine._reset_state(w, h)
        engine.recent_events = []
        engine.unique_faces = 0
        engine.people_detected_in_frame = 0
        engine.current_fps = 0.0
        # Hard-reset tracker counters (total_entered is a @property of counted_ids)
        engine.tracker.counted_ids.clear()
        engine.tracker.trajectories.clear()
        engine.tracker.entry_events.clear()
        engine.tracker.gate_counts = {g: 0 for g in engine.tracker.gate_names}

    return {"status": "success", "message": "All dataset records, CSV entry logs, and SQLite database tables have been deleted."}


@app.get("/api/download_csv")
def download_csv():
    """Serve the results CSV as a downloadable file."""
    import io, csv as _csv
    csv_path = "results.csv"
    if not os.path.exists(csv_path):
        # Return an empty CSV with headers
        output = io.StringIO()
        writer = _csv.writer(output)
        writer.writerow(["person_id", "entry_timestamp", "gate", "assigned_zone"])
        output.seek(0)
        return Response(
            content=output.getvalue(),
            media_type="text/csv",
            headers={"Content-Disposition": "attachment; filename=crowd_flow_dataset.csv"},
        )
    with open(csv_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Response(
        content=content,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=crowd_flow_dataset.csv"},
    )


@app.get("/api/export_dataset")
def export_dataset():
    """Export all detection records from SQLite as JSON."""
    try:
        conn = engine.db.conn
        cursor = conn.execute("SELECT * FROM detections ORDER BY detected_at DESC")
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
        return {"status": "success", "total_records": len(rows), "records": rows}
    except Exception as e:
        return {"status": "error", "message": str(e), "records": []}


@app.get("/api/download_json")
def download_json():
    """Serve the full dataset as a downloadable JSON file."""
    import json as _json
    try:
        conn = engine.db.conn
        cursor = conn.execute("SELECT * FROM detections ORDER BY detected_at DESC")
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
    except Exception:
        rows = []
    content = _json.dumps({"total_records": len(rows), "records": rows}, indent=2)
    return Response(
        content=content,
        media_type="application/json",
        headers={"Content-Disposition": "attachment; filename=crowd_flow_dataset.json"},
    )


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crowd Analytics Module</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {
            --bg-body: #0a0e14;
            --bg-card: #131922;
            --bg-card-header: #1b222d;
            --border-color: #263140;
            --accent-green: #238636;
            --accent-green-hover: #2ea043;
            --accent-red: #da3633;
            --accent-red-hover: #f85149;
            --accent-blue: #58a6ff;
            --accent-yellow: #e3b341;
            --text-primary: #c9d1d9;
            --text-heading: #f0f6fc;
            --text-muted: #8b949e;
        }

        * {
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }

        body {
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-body);
            color: var(--text-primary);
            padding: 18px 24px;
            min-height: 100vh;
        }

        header {
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px 22px;
            margin-bottom: 18px;
        }

        .header-title {
            display: flex;
            align-items: center;
            gap: 12px;
        }

        .header-title h1 {
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-heading);
        }

        .badge-module {
            background: #1f2937;
            border: 1px solid var(--border-color);
            color: #9ca3af;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 2px 8px;
            border-radius: 4px;
        }

        .status-pills {
            display: flex;
            gap: 16px;
            align-items: center;
        }

        .pill {
            display: flex;
            align-items: center;
            gap: 7px;
            font-size: 0.82rem;
            font-weight: 500;
            color: #d1d5db;
        }

        .dot {
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }

        .dot-green {
            background-color: #3fb950;
            box-shadow: 0 0 8px #3fb95088;
        }

        .dot-yellow {
            background-color: #d29922;
        }

        .dashboard-grid {
            display: grid;
            grid-template-columns: 1.25fr 0.95fr;
            gap: 18px;
            margin-bottom: 18px;
        }

        .card {
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }

        .card-header {
            background: var(--bg-card-header);
            border-bottom: 1px solid var(--border-color);
            padding: 10px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }

        .card-header h2 {
            font-size: 0.92rem;
            font-weight: 600;
            color: var(--text-heading);
        }

        .badge-count {
            background: #0b1e36;
            border: 1px solid #1f6feb;
            color: #58a6ff;
            font-size: 0.75rem;
            font-weight: 700;
            padding: 3px 10px;
            border-radius: 4px;
        }

        .video-container {
            position: relative;
            width: 100%;
            height: 440px;
            background: #000;
            display: flex;
            align-items: center;
            justify-content: center;
        }

        .video-container img {
            width: 100%;
            height: 100%;
            object-fit: contain;
            background: #000;
        }

        .telemetry-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 12px;
            padding: 16px;
        }

        .metric-box {
            background: #18202c;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 12px 14px;
        }

        .metric-label {
            font-size: 0.68rem;
            font-weight: 700;
            color: var(--text-muted);
            text-transform: uppercase;
            letter-spacing: 0.6px;
            margin-bottom: 4px;
        }

        .metric-val {
            font-size: 1.35rem;
            font-weight: 700;
            color: var(--text-heading);
            font-family: 'Outfit', sans-serif;
        }

        .log-section {
            padding: 0 16px 16px 16px;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }

        .log-title {
            font-size: 0.85rem;
            font-weight: 600;
            color: var(--text-heading);
            margin-bottom: 8px;
        }

        .table-wrapper {
            background: #10151c;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            overflow-y: auto;
            max-height: 230px;
            flex-grow: 1;
        }

        table {
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
            text-align: left;
        }

        th {
            background: #171d26;
            color: var(--text-muted);
            font-weight: 600;
            padding: 9px 12px;
            border-bottom: 1px solid var(--border-color);
            position: sticky;
            top: 0;
            font-size: 0.72rem;
            text-transform: uppercase;
        }

        td {
            padding: 9px 12px;
            border-bottom: 1px solid #1c2430;
            color: #d1d5db;
        }

        tr:hover td {
            background-color: #19212c;
        }

        .tag-zone {
            display: inline-block;
            background: #1f2d3d;
            border: 1px solid #388bfd44;
            color: #79c0ff;
            font-size: 0.72rem;
            font-weight: 600;
            padding: 2px 6px;
            border-radius: 4px;
        }

        .controls-grid {
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
        }

        .form-group {
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }

        label {
            font-size: 0.82rem;
            font-weight: 500;
            color: var(--text-muted);
        }

        select, input[type="file"] {
            width: 100%;
            background: #18202c;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 8px 12px;
            color: var(--text-primary);
            font-size: 0.85rem;
            outline: none;
        }

        .btn-group {
            display: flex;
            gap: 10px;
            margin-top: 4px;
        }

        .btn {
            border: none;
            border-radius: 6px;
            padding: 9px 16px;
            font-size: 0.85rem;
            font-weight: 600;
            cursor: pointer;
            transition: background 0.15s ease;
            display: inline-flex;
            align-items: center;
            justify-content: center;
            gap: 6px;
        }

        .btn-success {
            background-color: var(--accent-green);
            color: #fff;
        }
        .btn-success:hover {
            background-color: var(--accent-green-hover);
        }

        .btn-danger {
            background-color: var(--accent-red);
            color: #fff;
        }
        .btn-danger:hover {
            background-color: var(--accent-red-hover);
        }

        .btn-secondary {
            background-color: #21262d;
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }
        .btn-secondary:hover {
            background-color: #30363d;
        }

        .btn-deploy {
            background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
            color: #fff;
            border: none;
        }
        .btn-deploy:hover {
            background: linear-gradient(135deg, #388bfd 0%, #58a6ff 100%);
            transform: translateY(-1px);
            box-shadow: 0 4px 12px rgba(56,139,253,0.35);
        }
    </style>
</head>
<body>

    <header>
        <div class="header-title">
            <h1>Crowd Analytics Module</h1>
            <span class="badge-module">COMPONENT MODULE</span>
        </div>
        <div class="status-pills">
            <button class="btn btn-danger" style="padding: 5px 12px; font-size: 0.78rem;" onclick="resetDatasets()">🗑️ Delete Datasets & Logs</button>
            <button class="btn btn-deploy" id="deployCSVBtn" style="padding: 5px 12px; font-size: 0.78rem;" onclick="deployDataset('csv')">⬇️ Export CSV</button>
            <button class="btn btn-deploy" id="deployJSONBtn" style="padding: 5px 12px; font-size: 0.78rem;" onclick="deployDataset('json')">📦 Export JSON</button>
            <div class="pill">
                <span class="dot dot-green" id="streamDot"></span>
                <span id="streamStatusText">Camera Standby</span>
            </div>
            <div class="pill">
                <span class="dot dot-green"></span>
                <span>SQLite Connected</span>
            </div>
        </div>
    </header>


    <div class="dashboard-grid">
        <div class="card">
            <div class="card-header">
                <h2>Video Stream <span id="fpsBadge" style="color:#8b949e; font-size:12px; margin-left:8px;">0.0 FPS</span></h2>
                <div class="badge-count" id="countBadge">PEOPLE DETECTED: 0</div>
            </div>
            <div class="video-container">
                <img id="videoFeed" src="/video_feed" alt="Real-time Video Stream">
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h2>Current Detection Telemetry</h2>
            </div>
            
            <div class="telemetry-grid">
                <div class="metric-box">
                    <div class="metric-label">Total Entries</div>
                    <div class="metric-val" id="valEntries">0</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Flow Rate</div>
                    <div class="metric-val" id="valFlowRate">0.0 /min</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Unique Faces</div>
                    <div class="metric-val" id="valFaces">0</div>
                </div>
                <div class="metric-box">
                    <div class="metric-label">Engine Speed</div>
                    <div class="metric-val" id="valFps">0.0 FPS</div>
                </div>
            </div>

            <div class="log-section">
                <div class="log-title">Incoming Detections Log</div>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Track ID</th>
                                <th>Gate</th>
                                <th>Allocation</th>
                                <th>Detected At</th>
                            </tr>
                        </thead>
                        <tbody id="logTableBody">
                            <tr>
                                <td colspan="5" style="text-align: center; color: var(--text-muted); padding: 20px;">
                                    No detections recorded yet
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <div class="controls-grid">
        <div class="card">
            <div class="card-header">
                <h2>Camera Input</h2>
            </div>
            <div class="form-group">
                <label for="cameraSource">Source:</label>
                <select id="cameraSource">
                    <option value="0">Camera 1 (Default Laptop Webcam)</option>
                    <option value="1">Camera 2 (External Device)</option>
                </select>
                <div class="btn-group">
                    <button class="btn btn-success" id="btnStartCam" onclick="startCamera()">Start Camera</button>
                    <button class="btn btn-danger" onclick="stopCamera()">Stop</button>
                </div>
            </div>
        </div>

        <div class="card">
            <div class="card-header">
                <h2>Video File Input</h2>
            </div>
            <div class="form-group">
                <button class="btn btn-success" style="width: 100%; justify-content: center; background-color: #2ea043;" onclick="startSampleVideo()">▶ Process Sample Video (sample_entrance.mp4)</button>
                <label for="videoFileInput" style="margin-top:6px;">Or Upload Video File:</label>
                <input type="file" id="videoFileInput" accept="video/*">
                <div class="btn-group">
                    <button class="btn btn-secondary" style="width: 100%; justify-content: center;" onclick="uploadAndProcessVideo()">Upload & Process Video</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        async function fetchTelemetry() {
            try {
                const res = await fetch('/api/telemetry');
                const data = await res.json();

                document.getElementById('valEntries').innerText = data.total_entered;
                document.getElementById('valFps').innerText = data.fps + ' FPS';
                document.getElementById('fpsBadge').innerText = data.fps + ' FPS';
                document.getElementById('valFaces').innerText = '~' + data.unique_faces;
                document.getElementById('countBadge').innerText = 'PEOPLE DETECTED: ' + data.people_detected;
                
                const flowRate = (data.total_entered * 4.0).toFixed(1);
                document.getElementById('valFlowRate').innerText = flowRate + ' /min';

                const statusText = document.getElementById('streamStatusText');
                const streamDot = document.getElementById('streamDot');
                if (data.status === 'LIVE') {
                    statusText.innerText = 'Camera Live (' + data.mode + ')';
                    streamDot.className = 'dot dot-green';
                } else {
                    statusText.innerText = 'Camera Standby';
                    streamDot.className = 'dot dot-yellow';
                }

                // Render Log Table
                const tbody = document.getElementById('logTableBody');
                if (data.recent_events && data.recent_events.length > 0) {
                    tbody.innerHTML = data.recent_events.map(ev => `
                        <tr>
                            <td>${ev.series}</td>
                            <td><strong>Person #${ev.track_id}</strong></td>
                            <td>${ev.gate}</td>
                            <td><span class="tag-zone">${ev.zone}</span></td>
                            <td>${ev.time}</td>
                        </tr>
                    `).join('');
                } else {
                    // Always clear table when no events — even if camera is LIVE
                    tbody.innerHTML = '<tr><td colspan="5" style="text-align: center; color: var(--text-muted); padding: 20px;">No detections recorded yet</td></tr>';
                }
            } catch (err) {
                console.error("Telemetry error:", err);
            }
        }

        setInterval(fetchTelemetry, 500);

        async function startCamera() {
            const btn = document.getElementById('btnStartCam');
            btn.innerText = 'Starting...';
            const src = document.getElementById('cameraSource').value;
            const formData = new FormData();
            formData.append('source', src);
            try {
                const res = await fetch('/api/start_camera', { method: 'POST', body: formData });
                const json = await res.json();
                if (json.status !== 'success') {
                    alert('Camera could not be opened at index ' + src + '. Try selecting another camera or verify webcam permissions.');
                }
            } catch (e) {
                console.error(e);
            }
            btn.innerText = 'Start Camera';
        }

        async function stopCamera() {
            await fetch('/api/stop_camera', { method: 'POST' });
        }

        async function startSampleVideo() {
            await fetch('/api/start_sample_video', { method: 'POST' });
        }

        async function uploadAndProcessVideo() {
            const input = document.getElementById('videoFileInput');
            if (!input.files || input.files.length === 0) {
                alert('Please select a video file first!');
                return;
            }
            const formData = new FormData();
            formData.append('file', input.files[0]);
            await fetch('/api/upload_video', { method: 'POST', body: formData });
        }

        async function resetDatasets() {
            if (!confirm('Are you sure you want to permanently DELETE all datasets (SQLite records, CSV entry logs, and telemetry history)?')) {
                return;
            }
            try {
                const res = await fetch('/api/reset_data', { method: 'POST' });
                const data = await res.json();
                alert(data.message || 'Datasets cleared successfully!');
                await fetchTelemetry();
            } catch (err) {
                alert('Error resetting datasets: ' + err);
            }
        }

        async function deployDataset(format) {
            const btnId = format === 'csv' ? 'deployCSVBtn' : 'deployJSONBtn';
            const btn = document.getElementById(btnId);
            const origText = btn.innerText;
            btn.innerText = '⏳ Preparing...';
            btn.disabled = true;
            try {
                const url = format === 'csv' ? '/api/download_csv' : '/api/download_json';
                const res = await fetch(url);
                if (!res.ok) { throw new Error('Server error: ' + res.status); }
                const blob = await res.blob();
                const filename = format === 'csv' ? 'crowd_flow_dataset.csv' : 'crowd_flow_dataset.json';
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(a.href);
                btn.innerText = '✅ Downloaded!';
                setTimeout(() => { btn.innerText = origText; btn.disabled = false; }, 2500);
            } catch (err) {
                alert('Export failed: ' + err);
                btn.innerText = origText;
                btn.disabled = false;
            }
        }
    </script>

</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=False)
