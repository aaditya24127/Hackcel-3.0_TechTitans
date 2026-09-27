#!/usr/bin/env python3
"""
Enterprise Web Dashboard Server for Event Crowd Analytics & Flow Tracker.
Features:
- Zero-Latency Multi-Source Video Engine:
    1. Direct Browser / Phone Camera (High-FPS non-blocking WebRTC push)
    2. PC / USB Webcam (DirectShow / MSMF with FreshFrameGrabber)
    3. Mobile Stream (DroidCam / IP Webcam / RTSP with zero buffer lag)
    4. Video File Processing & Sample Entrance Video
- Real-time MJPEG video streamer & live responsive telemetry
- Database event persistence in SQLite + Export CSV/JSON
"""

from __future__ import annotations

import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp|fflags;nobuffer|flags;low_delay|max_delay;0"

import asyncio
import base64
import functools
import io
import socket
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response, StreamingResponse
import uvicorn

from src.db import DatabaseManager
from src.detector import FaceDetector, PersonDetector
from src.tracker import LineCrossingTracker
from src.video import VideoProcessor
from src.zones import ZoneManager


def get_local_ip() -> str:
    """Get the primary local LAN IP address of this machine."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    except Exception:
        ip = "127.0.0.1"
    finally:
        s.close()
    return ip


app = FastAPI(title="Crowd Analytics Module")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DirectHttpMjpegGrabber:
    """True zero-latency stream reader for DroidCam, IP Webcam, and HTTP MJPEG streams.
    Bypasses FFmpeg buffer entirely and always grabs the latest frame from the HTTP socket.
    """

    def __init__(self, url: str):
        self.url = url
        self.latest_frame: Optional[np.ndarray] = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()

    def _stream_loop(self):
        import urllib.request
        while self.running:
            try:
                req = urllib.request.Request(self.url, headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"})
                with urllib.request.urlopen(req, timeout=5) as stream:
                    buffer = b""
                    while self.running:
                        chunk = stream.read(8192)
                        if not chunk:
                            break
                        buffer += chunk

                        # Find the newest complete JPEG image in the received buffer
                        b = buffer.rfind(b"\xff\xd9")
                        if b != -1:
                            a = buffer[:b].rfind(b"\xff\xd8")
                            if a != -1 and b > a:
                                jpg = buffer[a : b + 2]
                                buffer = buffer[b + 2 :]
                                frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
                                if frame is not None:
                                    with self.lock:
                                        self.latest_frame = frame

                        if len(buffer) > 65536:
                            buffer = buffer[-32768:]
            except Exception:
                time.sleep(0.3)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame
            return False, None

    def release(self):
        self.running = False


class FreshFrameGrabber:
    """Continuously drains frames in a dedicated thread to ensure zero buffer delay for RTSP/HTTP/Webcam."""

    def __init__(self, cap: cv2.VideoCapture):
        self.cap = cap
        self.latest_frame: Optional[np.ndarray] = None
        self.running = True
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._grab_loop, daemon=True)
        self.thread.start()

    def _grab_loop(self):
        while self.running:
            if not self.cap:
                time.sleep(0.01)
                continue
            try:
                if not self.cap.isOpened():
                    time.sleep(0.01)
                    continue
                ret = self.cap.grab()
                if ret:
                    success, frame = self.cap.retrieve()
                    if success and frame is not None:
                        with self.lock:
                            self.latest_frame = frame
                else:
                    time.sleep(0.005)
            except Exception:
                time.sleep(0.01)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        with self.lock:
            if self.latest_frame is not None:
                return True, self.latest_frame
            return False, None

    def release(self):
        self.running = False
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None


class StreamEngine:
    def __init__(self):
        self.mode: str = "IDLE"  # "CAMERA", "CLIENT_CAM", "IP_CAM", "FILE", "IDLE"
        self.source: str | int = 0
        self.is_running: bool = False
        self.cap: Optional[cv2.VideoCapture] = None
        self.grabber: Optional[FreshFrameGrabber] = None
        self.http_grabber: Optional[DirectHttpMjpegGrabber] = None
        self._lock = threading.Lock()

        # Client-pushed frame buffer
        self._client_frame: Optional[np.ndarray] = None
        self._has_new_client_frame: bool = False

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
        cv2.putText(blank, "Select an input source below to start tracking", (130, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (150, 160, 175), 1)
        cv2.putText(blank, "● STANDBY", (25, 455), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 215, 255), 1)

        _, buf = cv2.imencode(".jpg", blank, [int(cv2.IMWRITE_JPEG_QUALITY), 70])
        return buf.tobytes()

    def start_camera(self, camera_idx: int = 0) -> Tuple[bool, str]:
        with self._lock:
            self._close_sources()
            self.mode = "CAMERA"
            self.source = camera_idx

            if os.name == "nt":
                self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_DSHOW)
                if not self.cap or not self.cap.isOpened():
                    self.cap = cv2.VideoCapture(camera_idx, cv2.CAP_MSMF)
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
                self.grabber = FreshFrameGrabber(self.cap)
                self._reset_state(w, h)
                self.is_running = True
                return True, f"Camera {camera_idx} started successfully."
            else:
                self.is_running = False
                self.mode = "IDLE"
                return False, f"Could not open Camera {camera_idx}. Ensure no other app has exclusive lock."

    def start_client_stream(self, width: int = 640, height: int = 480):
        """Enable client browser camera streaming mode."""
        with self._lock:
            self._close_sources()
            self.mode = "CLIENT_CAM"
            self.source = "Browser / Phone Camera"
            self._reset_state(width, height)
            self.is_running = True

    def push_client_frame(self, frame: np.ndarray):
        with self._lock:
            if not self.is_running or self.mode != "CLIENT_CAM":
                self.mode = "CLIENT_CAM"
                self.source = "Browser / Phone Camera"
                self._reset_state(frame.shape[1], frame.shape[0])
                self.is_running = True
            self._client_frame = frame
            self._has_new_client_frame = True

    def start_file(self, file_path: str) -> bool:
        with self._lock:
            self._close_sources()
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

    def start_ip_camera(self, stream_url: str) -> bool:
        """Connect to a mobile/IP camera stream (DroidCam / IP Webcam / HTTP / RTSP) with zero latency."""
        stream_url = stream_url.strip()

        # 1. Fast direct HTTP reader for DroidCam / IP Webcam
        if stream_url.startswith("http://") or stream_url.startswith("https://"):
            with self._lock:
                self._close_sources()
                self.mode = "IP_CAM"
                self.source = stream_url
                self.http_grabber = DirectHttpMjpegGrabber(stream_url)

            # Wait up to 3 seconds for first frame
            for _ in range(30):
                time.sleep(0.1)
                ret, frame = self.http_grabber.read()
                if ret and frame is not None:
                    w, h = frame.shape[1], frame.shape[0]
                    self._reset_state(w, h)
                    self.is_running = True
                    return True

            with self._lock:
                if self.http_grabber:
                    self.http_grabber.release()
                    self.http_grabber = None

        # 2. RTSP or OpenCV Fallback
        cap = cv2.VideoCapture(stream_url, cv2.CAP_FFMPEG)
        if not cap or not cap.isOpened():
            cap = cv2.VideoCapture(stream_url)

        if not cap or not cap.isOpened():
            self.is_running = False
            self.mode = "IDLE"
            return False

        with self._lock:
            self._close_sources()
            self.mode = "IP_CAM"
            self.source = stream_url
            self.cap = cap
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
            h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
            self.grabber = FreshFrameGrabber(self.cap)
            self._reset_state(w, h)
            self.is_running = True
            return True

    def process_image(self, image: np.ndarray, filename: str = "image.jpg") -> Dict[str, Any]:
        """Process a single uploaded static image, detect people & faces, and update telemetry."""
        with self._lock:
            self._close_sources()
            self.mode = "IMAGE"
            self.source = filename
            h, w = image.shape[:2]
            self._reset_state(w, h)
            self.is_running = True

            # 1. Person detection
            results = self.person_detector.model.predict(
                source=image,
                classes=[0],
                conf=0.25,
                device=self.person_detector.device,
                verbose=False,
            )
            tracks = []
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                for i in range(len(boxes)):
                    tid = i + 1
                    x1, y1, x2, y2 = map(float, boxes[i])
                    tracks.append({
                        "id": tid,
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(confs[i]),
                    })

            # 2. Face detection
            faces = self.face_detector.detect_faces(image, 1, person_tracks=tracks)
            matched_face_ids = {f.matched_track_id for f in faces if f.matched_track_id is not None}

            # 3. Update telemetry metrics
            self.people_detected_in_frame = len(tracks)
            self.tracker.counted_ids = set([t["id"] for t in tracks])
            self.unique_faces = min(len(faces) if faces else len(matched_face_ids), len(tracks))
            self.current_fps = 0.0

            # 4. Render visual annotations on image
            annotated = image.copy()
            for t in tracks:
                tid = t["id"]
                x1, y1, x2, y2 = map(int, t["bbox"])
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 215, 255), 2)
                has_face = tid in matched_face_ids
                label = f"Person #{tid} [Face OK]" if has_face else f"Person #{tid}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                cv2.rectangle(annotated, (x1, max(0, y1 - th - 8)), (x1 + tw + 8, y1), (0, 215, 255), -1)
                cv2.putText(annotated, label, (x1 + 4, max(th + 2, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (10, 14, 20), 1, cv2.LINE_AA)

            for f in faces:
                fx1, fy1, fx2, fy2 = map(int, f.bbox)
                cv2.rectangle(annotated, (fx1, fy1), (fx2, fy2), (255, 100, 200), 2)
                flabel = f"Face #{f.matched_track_id}" if f.matched_track_id else "Face"
                (tw, th), _ = cv2.getTextSize(flabel, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
                cv2.rectangle(annotated, (fx1, max(0, fy1 - th - 6)), (fx1 + tw + 6, fy1), (255, 100, 200), -1)
                cv2.putText(annotated, flabel, (fx1 + 3, max(th + 1, fy1 - 3)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)

            badge_text = f"IMAGE ANALYSIS: {len(tracks)} People Detected | {self.unique_faces} Faces"
            cv2.rectangle(annotated, (15, 15), (400, 48), (22, 27, 34), -1)
            cv2.rectangle(annotated, (15, 15), (400, 48), (56, 139, 253), 1)
            cv2.putText(annotated, badge_text, (25, 38), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (240, 240, 240), 1, cv2.LINE_AA)

            _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
            self.latest_jpeg = buf.tobytes()

            # Record detection events
            self.recent_events = []
            for t in tracks:
                tid = t["id"]
                self.recent_events.append({
                    "series": tid,
                    "track_id": tid,
                    "time": f"Conf {int(t['conf'] * 100)}%",
                })

            return {
                "status": "success",
                "people_count": len(tracks),
                "faces_count": len(faces),
                "filename": filename,
            }

    def stop(self):
        with self._lock:
            self.is_running = False
            self.mode = "IDLE"
            self._close_sources()
            self.latest_jpeg = self._generate_standby_frame()

    def _close_sources(self):
        if self.http_grabber:
            try:
                self.http_grabber.release()
            except Exception:
                pass
            self.http_grabber = None
        if self.grabber:
            try:
                self.grabber.release()
            except Exception:
                pass
            self.grabber = None
        if self.cap:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
        self._client_frame = None
        self._has_new_client_frame = False

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
        self.face_detector.reset()
        self.unique_faces = 0

    def _process_frame_tensor(self, frame: np.ndarray):
        """High-speed inference and HUD overlay pipeline."""
        self.current_frame_count += 1
        video_time_s = self.current_frame_count / 30.0

        try:
            # 1. Person Detection & ByteTrack (Optimized at imgsz=384 for high FPS)
            tracks = self.person_detector.track(frame, conf_threshold=0.20, imgsz=384)
            self.people_detected_in_frame = len(tracks)

            # 2. Face Detection (Every 2nd frame for low CPU overhead)
            if self.current_frame_count % 2 == 0:
                faces = self.face_detector.detect_faces(frame, self.current_frame_count, person_tracks=tracks)
            else:
                faces = []

            # 3. Line Crossing & Attendee Registration
            new_entries = self.tracker.update_tracks(tracks, self.current_frame_count, video_time_s)

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
                    "zone": zone,
                    "time": f"{entry.video_time_s:.1f}s",
                }
                self.recent_events.insert(0, event_dict)
                if len(self.recent_events) > 30:
                    self.recent_events.pop()

            # Auto-register newly detected persons
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
                        "zone": zone,
                        "time": f"{video_time_s:.1f}s",
                    }
                    self.recent_events.insert(0, event_dict)
                    if len(self.recent_events) > 30:
                        self.recent_events.pop()

            # Bounded Unique Faces: Can never exceed total entries
            if self.tracker.total_entered > 0:
                matched_face_attendees = self.face_detector.seen_face_track_ids.intersection(self.tracker.counted_ids)
                if len(matched_face_attendees) > 0:
                    self.unique_faces = min(len(matched_face_attendees), self.tracker.total_entered)
                else:
                    self.unique_faces = min(len(self.face_detector.seen_face_track_ids), self.tracker.total_entered)
            else:
                self.unique_faces = 0

            # 4. Crowd Drift Simulation
            drift_transitions = self.zone_mgr.update_drift(video_time_s)
            for tid, fz, tz in drift_transitions:
                if self.run_id:
                    try:
                        self.db.log_transition(self.run_id, tid, fz, tz, video_time_s)
                    except Exception:
                        pass

            # 5. Visual Overlays
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

            # Encode directly to JPEG
            _, buf = cv2.imencode(".jpg", annotated, [int(cv2.IMWRITE_JPEG_QUALITY), 68])
            self.latest_jpeg = buf.tobytes()

        except Exception as e:
            print(f"[Inference Error]: {e}")

    def _processing_loop(self):
        """Dedicated background loop that processes frames from the active source."""
        last_calc_time = time.time()
        fps_counter = 0

        while not self._stop_worker:
            if not self.is_running:
                time.sleep(0.04)
                continue

            frame = None

            if self.mode == "CAMERA":
                if self.grabber:
                    ret, frame = self.grabber.read()
                    if not ret or frame is None:
                        time.sleep(0.01)
                        continue
                else:
                    time.sleep(0.02)
                    continue

            elif self.mode == "IP_CAM":
                if self.http_grabber:
                    ret, frame = self.http_grabber.read()
                elif self.grabber:
                    ret, frame = self.grabber.read()
                else:
                    ret, frame = False, None

                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

            elif self.mode == "CLIENT_CAM":
                with self._lock:
                    if self._has_new_client_frame and self._client_frame is not None:
                        frame = self._client_frame
                        self._has_new_client_frame = False
                    else:
                        frame = None
                if frame is None:
                    time.sleep(0.01)
                    continue

            elif self.mode == "FILE":
                if self.cap and self.cap.isOpened():
                    with self._lock:
                        ret, frame = self.cap.read()
                    if not ret:
                        with self._lock:
                            if self.cap:
                                self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                        time.sleep(0.03)
                        continue
                else:
                    time.sleep(0.02)
                    continue

            if frame is None:
                time.sleep(0.01)
                continue

            fps_counter += 1
            now = time.time()
            if now - last_calc_time >= 0.5:
                self.current_fps = round(fps_counter / (now - last_calc_time), 1)
                fps_counter = 0
                last_calc_time = now

            self._process_frame_tensor(frame)
            time.sleep(0.005)


engine = StreamEngine()


def frame_generator():
    """Ultra-stable MJPEG generator for browser clients."""
    while True:
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + engine.latest_jpeg + b"\r\n"
        )
        time.sleep(0.030)


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
    success, msg = engine.start_camera(source)
    return {"status": "success" if success else "failed", "mode": engine.mode, "source": source, "message": msg}


@app.post("/api/stop_camera")
def stop_camera():
    engine.stop()
    return {"status": "success", "mode": engine.mode}


@app.post("/api/push_frame")
async def push_frame(file: UploadFile = File(...)):
    """Receives a frame pushed from browser/phone client with zero-latency handoff."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"status": "error", "message": "Invalid image"}, status_code=400)

    engine.push_client_frame(frame)
    return {"status": "ok"}


@app.post("/api/start_ip_camera")
async def start_ip_camera_route(url: str = Form(...)):
    """Connect to a mobile IP camera or RTSP/HTTP stream."""
    url = url.strip()
    if not url:
        return {"status": "error", "message": "No URL provided"}
    print(f"[IP CAM] Connecting with zero-lag grabber: {url}")
    loop = asyncio.get_event_loop()
    success = await loop.run_in_executor(None, functools.partial(engine.start_ip_camera, url))
    if success:
        return {"status": "success", "mode": engine.mode, "url": url}
    return {
        "status": "error",
        "message": f"Could not connect to {url}. Check Wi-Fi connection and ensure DroidCam / IP Webcam server is running."
    }


@app.post("/api/upload_image")
async def upload_image(file: UploadFile = File(...)):
    """Upload and recognize persons and faces in a static image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    if frame is None:
        return JSONResponse({"status": "error", "message": "Could not decode uploaded image."}, status_code=400)

    result = engine.process_image(frame, file.filename or "uploaded_image.jpg")
    return JSONResponse(result)


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
        try:
            engine.db.clear_all_data()
        except Exception as e:
            print(f"[DB Clear Error]: {e}")

        for csv_path in ["results.csv", "results2.csv"]:
            try:
                if os.path.exists(csv_path):
                    with open(csv_path, "w", encoding="utf-8") as f:
                        f.write("person_id,entry_timestamp,gate,assigned_zone\n")
            except Exception as e:
                print(f"[CSV Reset Error]: {e}")

        w = int(engine.cap.get(cv2.CAP_PROP_FRAME_WIDTH)) if engine.cap else 640
        h = int(engine.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) if engine.cap else 480
        engine._reset_state(w, h)
        engine.recent_events = []
        engine.unique_faces = 0
        engine.people_detected_in_frame = 0
        engine.current_fps = 0.0
        engine.tracker.counted_ids.clear()
        engine.tracker.trajectories.clear()
        engine.tracker.entry_events.clear()
        engine.tracker.gate_counts = {g: 0 for g in engine.tracker.gate_names}

    return {"status": "success", "message": "All dataset records, CSV logs, and database entries deleted."}


@app.get("/api/download_csv")
def download_csv():
    csv_path = "results.csv"
    if not os.path.exists(csv_path):
        output = io.StringIO()
        output.write("person_id,entry_timestamp,gate,assigned_zone\n")
        return Response(content=output.getvalue(), media_type="text/csv", headers={"Content-Disposition": "attachment; filename=crowd_flow_dataset.csv"})
    with open(csv_path, "r", encoding="utf-8") as f:
        content = f.read()
    return Response(content=content, media_type="text/csv", headers={"Content-Disposition": "attachment; filename=crowd_flow_dataset.csv"})


@app.get("/api/download_json")
def download_json():
    import json as _json
    try:
        conn = engine.db.conn
        cursor = conn.execute("SELECT * FROM detections ORDER BY detected_at DESC")
        cols = [d[0] for d in cursor.description]
        rows = [dict(zip(cols, row)) for row in cursor.fetchall()]
    except Exception:
        rows = []
    content = _json.dumps({"total_records": len(rows), "records": rows}, indent=2)
    return Response(content=content, media_type="application/json", headers={"Content-Disposition": "attachment; filename=crowd_flow_dataset.json"})


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    local_ip = get_local_ip()
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Crowd Analytics & Flow Tracker</title>
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@400;600;700&display=swap" rel="stylesheet">
    <style>
        :root {{
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
        }}

        * {{
            box-sizing: border-box;
            margin: 0;
            padding: 0;
        }}

        body {{
            font-family: 'Inter', sans-serif;
            background-color: var(--bg-body);
            color: var(--text-primary);
            padding: 18px 24px;
            min-height: 100vh;
        }}

        header {{
            display: flex;
            justify-content: space-between;
            align-items: center;
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            padding: 14px 22px;
            margin-bottom: 18px;
            flex-wrap: wrap;
            gap: 12px;
        }}

        .header-title {{
            display: flex;
            align-items: center;
            gap: 12px;
        }}

        .header-title h1 {{
            font-size: 1.15rem;
            font-weight: 700;
            color: var(--text-heading);
        }}

        .badge-module {{
            font-size: 0.65rem;
            background: #1f6feb;
            color: #fff;
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: 600;
            letter-spacing: 0.5px;
        }}

        .status-pills {{
            display: flex;
            gap: 10px;
            align-items: center;
            flex-wrap: wrap;
        }}

        .pill {{
            display: flex;
            align-items: center;
            gap: 6px;
            background: #161b22;
            border: 1px solid var(--border-color);
            padding: 5px 12px;
            border-radius: 20px;
            font-size: 0.78rem;
            font-weight: 500;
        }}

        .dot {{
            width: 8px;
            height: 8px;
            border-radius: 50%;
        }}
        .dot-green {{ background-color: #3fb950; box-shadow: 0 0 6px #3fb950; }}
        .dot-yellow {{ background-color: #e3b341; }}
        .dot-blue {{ background-color: #58a6ff; box-shadow: 0 0 6px #58a6ff; }}

        .network-banner {{
            background: rgba(88, 166, 255, 0.08);
            border: 1px solid rgba(88, 166, 255, 0.25);
            border-radius: 8px;
            padding: 10px 16px;
            margin-bottom: 18px;
            display: flex;
            align-items: center;
            justify-content: space-between;
            font-size: 0.85rem;
            flex-wrap: wrap;
            gap: 8px;
        }}

        .network-banner code {{
            background: #1f242c;
            color: #79c0ff;
            padding: 3px 8px;
            border-radius: 4px;
            font-family: monospace;
            font-weight: bold;
        }}

        .dashboard-grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 18px;
            margin-bottom: 18px;
        }}

        @media (max-width: 900px) {{
            .dashboard-grid {{
                grid-template-columns: 1fr;
            }}
        }}

        .card {{
            background: var(--bg-card);
            border: 1px solid var(--border-color);
            border-radius: 8px;
            overflow: hidden;
            display: flex;
            flex-direction: column;
        }}

        .card-header {{
            background: var(--bg-card-header);
            border-bottom: 1px solid var(--border-color);
            padding: 10px 16px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }}

        .card-header h2 {{
            font-size: 0.92rem;
            font-weight: 600;
            color: var(--text-heading);
            display: flex;
            align-items: center;
            gap: 8px;
        }}

        .badge-count {{
            font-size: 0.72rem;
            background: #238636;
            color: #fff;
            padding: 2px 8px;
            border-radius: 12px;
            font-weight: bold;
        }}

        .video-container {{
            background: #000;
            width: 100%;
            aspect-ratio: 4/3;
            max-height: 480px;
            display: flex;
            align-items: center;
            justify-content: center;
            position: relative;
        }}

        .video-container img {{
            width: 100%;
            height: 100%;
            object-fit: contain;
        }}

        .telemetry-grid {{
            display: grid;
            grid-template-columns: repeat(3, 1fr);
            gap: 12px;
            padding: 16px;
            background: #0d1117;
            border-bottom: 1px solid var(--border-color);
        }}

        .metric-box {{
            background: #161b22;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            padding: 10px;
            text-align: center;
        }}

        .metric-label {{
            font-size: 0.72rem;
            color: var(--text-muted);
            margin-bottom: 4px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .metric-val {{
            font-size: 1.3rem;
            font-weight: 700;
            color: #58a6ff;
            font-family: 'Outfit', sans-serif;
        }}

        .log-section {{
            padding: 16px;
            flex-grow: 1;
            display: flex;
            flex-direction: column;
        }}

        .log-title {{
            font-size: 0.8rem;
            font-weight: 600;
            color: var(--text-heading);
            margin-bottom: 10px;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }}

        .table-wrapper {{
            overflow-y: auto;
            max-height: 230px;
            border: 1px solid var(--border-color);
            border-radius: 6px;
        }}

        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 0.8rem;
            text-align: left;
        }}

        th {{
            background: #161b22;
            color: var(--text-muted);
            padding: 8px 12px;
            font-weight: 600;
            position: sticky;
            top: 0;
            border-bottom: 1px solid var(--border-color);
        }}

        td {{
            padding: 7px 12px;
            border-bottom: 1px solid #1f242c;
            color: var(--text-primary);
        }}

        tr:nth-child(even) td {{
            background: #0f1319;
        }}

        .tag-zone {{
            background: #1f2a37;
            color: #79c0ff;
            padding: 2px 6px;
            border-radius: 4px;
            font-size: 0.72rem;
            font-weight: 600;
        }}

        .controls-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
            gap: 18px;
        }}

        .form-group {{
            padding: 16px;
            display: flex;
            flex-direction: column;
            gap: 10px;
        }}

        label {{
            font-size: 0.8rem;
            color: var(--text-muted);
            font-weight: 500;
        }}

        select, input[type="text"], input[type="file"] {{
            background: #0d1117;
            border: 1px solid var(--border-color);
            border-radius: 6px;
            color: var(--text-primary);
            padding: 8px 12px;
            font-size: 0.85rem;
            outline: none;
            width: 100%;
        }}

        select:focus, input:focus {{
            border-color: #58a6ff;
        }}

        .btn-group {{
            display: flex;
            gap: 8px;
            margin-top: 4px;
        }}

        .btn {{
            cursor: pointer;
            padding: 8px 14px;
            border-radius: 6px;
            font-size: 0.82rem;
            font-weight: 600;
            display: inline-flex;
            align-items: center;
            gap: 6px;
            border: 1px solid transparent;
            transition: all 0.15s ease;
        }}

        .btn:disabled {{
            opacity: 0.5;
            cursor: not-allowed;
        }}

        .btn-success {{
            background-color: var(--accent-green);
            color: #fff;
        }}
        .btn-success:hover:not(:disabled) {{
            background-color: var(--accent-green-hover);
        }}

        .btn-danger {{
            background-color: var(--accent-red);
            color: #fff;
        }}
        .btn-danger:hover:not(:disabled) {{
            background-color: var(--accent-red-hover);
        }}

        .btn-secondary {{
            background-color: #21262d;
            color: var(--text-primary);
            border: 1px solid var(--border-color);
        }}
        .btn-secondary:hover:not(:disabled) {{
            background-color: #30363d;
        }}

        .btn-deploy {{
            background: linear-gradient(135deg, #1f6feb 0%, #388bfd 100%);
            color: #fff;
            border: none;
        }}
        .btn-deploy:hover:not(:disabled) {{
            background: linear-gradient(135deg, #388bfd 0%, #58a6ff 100%);
            transform: translateY(-1px);
        }}

        #clientVideo, #clientCanvas {{
            display: none;
        }}
    </style>
</head>
<body>

    <header>
        <div class="header-title">
            <h1>Crowd Analytics & Flow Tracker</h1>
            <span class="badge-module">AI ENGINE</span>
        </div>
        <div class="status-pills">
            <button class="btn btn-danger" style="padding: 5px 12px; font-size: 0.78rem;" onclick="resetDatasets()">🗑️ Delete Records</button>
            <button class="btn btn-deploy" id="deployCSVBtn" style="padding: 5px 12px; font-size: 0.78rem;" onclick="deployDataset('csv')">⬇️ CSV</button>
            <button class="btn btn-deploy" id="deployJSONBtn" style="padding: 5px 12px; font-size: 0.78rem;" onclick="deployDataset('json')">📦 JSON</button>
            <div class="pill">
                <span class="dot dot-green" id="streamDot"></span>
                <span id="streamStatusText">Standby</span>
            </div>
            <div class="pill">
                <span class="dot dot-green"></span>
                <span>SQLite Active</span>
            </div>
        </div>
    </header>

    <div class="network-banner">
        <span>📱 <strong>Zero-Lag Phone Streaming:</strong> Open <code>http://{local_ip}:8000</code> on your phone</span>
        <span style="color:#79c0ff; font-size:0.8rem;">⚡ High-FPS Non-Blocking Engine</span>
    </div>

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
                <h2>Live Telemetry &amp; Detection Breakdown</h2>
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
            </div>

            <div class="log-section">
                <div class="log-title">Incoming Detections Log</div>
                <div class="table-wrapper">
                    <table>
                        <thead>
                            <tr>
                                <th>#</th>
                                <th>Track ID</th>
                                <th>Detected At</th>
                            </tr>
                        </thead>
                        <tbody id="logTableBody">
                            <tr>
                                <td colspan="3" style="text-align: center; color: var(--text-muted); padding: 20px;">
                                    No detections recorded yet
                                </td>
                            </tr>
                        </tbody>
                    </table>
                </div>
            </div>
        </div>
    </div>

    <!-- Hidden elements for WebRTC client streaming -->
    <video id="clientVideo" playsinline autoplay muted></video>
    <canvas id="clientCanvas"></canvas>

    <div class="controls-grid">
        <!-- 1. Image Upload & Recognition -->
        <div class="card" style="border-color: #1f6feb;">
            <div class="card-header" style="background: #112238;">
                <h2 style="color: #58a6ff;">📸 Upload Image &amp; Recognize</h2>
            </div>
            <div class="form-group">
                <label for="imageFileInput">Select Image (JPG, PNG, WEBP):</label>
                <input type="file" id="imageFileInput" accept="image/*">
                <div class="btn-group">
                    <button class="btn btn-deploy" style="width: 100%; justify-content: center;" id="btnUploadImage" onclick="uploadAndProcessImage()">
                        🔍 Recognize Image &amp; People
                    </button>
                </div>
                <div style="font-size:0.75rem; color:#8b949e; line-height:1.4;">
                    ✔ Detects &amp; recognizes people and faces instantly with full bounding boxes.
                </div>
            </div>
        </div>

        <!-- 2. Direct Phone / Browser Camera (Zero App) -->
        <div class="card">
            <div class="card-header">
                <h2>📱 Instant Phone / Browser Camera</h2>
            </div>
            <div class="form-group">
                <label>Direct browser camera stream (Recommended for Phone):</label>
                <div class="btn-group">
                    <button class="btn btn-deploy" style="width:100%; justify-content:center;" id="btnWebCam" onclick="toggleWebCamStream()">
                        🎥 Start Phone / Browser Camera
                    </button>
                </div>
                <div class="btn-group">
                    <button class="btn btn-secondary" style="width:100%; justify-content:center;" id="btnFlipCam" onclick="flipCameraFacing()">
                        🔄 Switch Front / Back Camera
                    </button>
                </div>
                <div style="font-size:0.75rem; color:#8b949e; line-height:1.5;">
                    ✔ Open <code>http://{local_ip}:8000</code> on phone &amp; tap Start!
                </div>
            </div>
        </div>

        <!-- 3. PC DirectShow Webcam -->
        <div class="card">
            <div class="card-header">
                <h2>💻 PC / USB Webcam</h2>
            </div>
            <div class="form-group">
                <label for="cameraSource">Source Index:</label>
                <select id="cameraSource">
                    <option value="0">Camera 1 (Default Webcam / Index 0)</option>
                    <option value="1">Camera 2 (External / Index 1)</option>
                    <option value="2">Camera 3 (Index 2)</option>
                </select>
                <div class="btn-group">
                    <button class="btn btn-success" id="btnStartCam" onclick="startCamera()">Start PC Camera</button>
                    <button class="btn btn-danger" onclick="stopCamera()">Stop</button>
                </div>
            </div>
        </div>

        <!-- 4. Mobile IP Camera (DroidCam / IP Webcam) -->
        <div class="card">
            <div class="card-header">
                <h2>📡 Mobile App Stream (DroidCam / RTSP)</h2>
            </div>
            <div class="form-group">
                <label for="ipCamUrl">Stream URL (Zero-Lag Draining Enabled):</label>
                <input type="text" id="ipCamUrl" placeholder="e.g. http://192.168.1.5:4747/video" style="font-family: monospace; font-size:0.82rem;">
                <div class="btn-group">
                    <button class="btn btn-secondary" style="width:100%; justify-content:center;" id="btnIPCam" onclick="startIPCamera()">Connect Stream</button>
                </div>
            </div>
        </div>

        <!-- 5. Video Files -->
        <div class="card">
            <div class="card-header">
                <h2>🎬 Sample &amp; Uploaded Videos</h2>
            </div>
            <div class="form-group">
                <button class="btn btn-success" style="width: 100%; justify-content: center;" onclick="startSampleVideo()">▶ Play Sample Video</button>
                <label for="videoFileInput" style="margin-top:4px;">Upload MP4/AVI:</label>
                <input type="file" id="videoFileInput" accept="video/*">
                <div class="btn-group">
                    <button class="btn btn-secondary" style="width: 100%; justify-content: center;" onclick="uploadAndProcessVideo()">Upload &amp; Process</button>
                </div>
            </div>
        </div>
    </div>

    <script>
        let clientStream = null;
        let clientInterval = null;
        let currentFacingMode = 'environment';
        let isPushing = false;

        async function fetchTelemetry() {{
            try {{
                const res = await fetch('/api/telemetry');
                const data = await res.json();

                document.getElementById('valEntries').innerText = data.total_entered;
                document.getElementById('fpsBadge').innerText = data.fps + ' FPS';
                document.getElementById('valFaces').innerText = data.unique_faces;
                document.getElementById('countBadge').innerText = 'PEOPLE DETECTED: ' + data.people_detected;
                
                const flowRate = (data.total_entered * 4.0).toFixed(1);
                document.getElementById('valFlowRate').innerText = flowRate + ' /min';

                const statusText = document.getElementById('streamStatusText');
                const streamDot = document.getElementById('streamDot');
                if (data.status === 'LIVE') {{
                    statusText.innerText = 'Live (' + data.mode + ')';
                    streamDot.className = 'dot dot-green';
                }} else {{
                    statusText.innerText = 'Standby';
                    streamDot.className = 'dot dot-yellow';
                }}

                const tbody = document.getElementById('logTableBody');
                if (data.recent_events && data.recent_events.length > 0) {{
                    tbody.innerHTML = data.recent_events.map(ev => `
                        <tr>
                            <td>${{ev.series}}</td>
                            <td><strong>Person #${{ev.track_id}}</strong></td>
                            <td>${{ev.time}}</td>
                        </tr>
                    `).join('');
                }} else {{
                    tbody.innerHTML = '<tr><td colspan="3" style="text-align: center; color: var(--text-muted); padding: 20px;">No detections recorded yet</td></tr>';
                }}
            }} catch (err) {{
                console.error("Telemetry error:", err);
            }}
        }}

        setInterval(fetchTelemetry, 400);

        function reloadVideoFeed() {{
            const feed = document.getElementById('videoFeed');
            if (feed) {{
                feed.src = '/video_feed?t=' + Date.now();
            }}
        }}

        // --- 1. DIRECT IN-BROWSER / PHONE WEBCAM STREAMING ---
        async function toggleWebCamStream() {{
            const btn = document.getElementById('btnWebCam');
            if (clientStream) {{
                stopClientStream();
                await stopCamera();
                btn.innerText = '🎥 Start Phone / Browser Camera';
                btn.classList.remove('btn-danger');
                btn.classList.add('btn-deploy');
                return;
            }}

            try {{
                btn.innerText = '⏳ Accessing Camera...';
                const constraints = {{
                    video: {{
                        width: {{ ideal: 480 }},
                        height: {{ ideal: 360 }},
                        facingMode: currentFacingMode
                    }},
                    audio: false
                }};
                clientStream = await navigator.mediaDevices.getUserMedia(constraints);
                const video = document.getElementById('clientVideo');
                video.srcObject = clientStream;
                await video.play();

                btn.innerText = '⏹ Stop Phone / Browser Camera';
                btn.classList.remove('btn-deploy');
                btn.classList.add('btn-danger');

                startFramePushLoop();
                reloadVideoFeed();
            }} catch (err) {{
                console.error("Camera access failed:", err);
                alert("Could not access camera: " + err.message + "\\nMake sure camera permissions are allowed in browser.");
                btn.innerText = '🎥 Start Phone / Browser Camera';
            }}
        }}

        function stopClientStream() {{
            if (clientInterval) {{
                clearInterval(clientInterval);
                clientInterval = null;
            }}
            if (clientStream) {{
                clientStream.getTracks().forEach(track => track.stop());
                clientStream = null;
            }}
            const video = document.getElementById('clientVideo');
            if (video) video.srcObject = null;
            isPushing = false;
        }}

        async function flipCameraFacing() {{
            currentFacingMode = currentFacingMode === 'user' ? 'environment' : 'user';
            if (clientStream) {{
                stopClientStream();
                await toggleWebCamStream();
            }}
        }}

        function startFramePushLoop() {{
            const video = document.getElementById('clientVideo');
            const canvas = document.getElementById('clientCanvas');
            const ctx = canvas.getContext('2d');

            if (clientInterval) clearInterval(clientInterval);

            // Scale to 480x360 for ultra-fast zero-latency streaming
            canvas.width = 480;
            canvas.height = 360;

            clientInterval = setInterval(() => {{
                if (!clientStream || isPushing || video.readyState < 2) return;
                isPushing = true;
                try {{
                    ctx.drawImage(video, 0, 0, 480, 360);
                    canvas.toBlob((blob) => {{
                        if (!blob) {{
                            isPushing = false;
                            return;
                        }}
                        const fd = new FormData();
                        fd.append('file', blob, 'frame.jpg');
                        fetch('/api/push_frame', {{ method: 'POST', body: fd }})
                            .catch(e => console.error(e))
                            .finally(() => {{ isPushing = false; }});
                    }}, 'image/jpeg', 0.55);
                }} catch (e) {{
                    isPushing = false;
                }}
            }}, 40); // 25 FPS push rate
        }}

        // --- 2. BACKEND PC WEBCAM ---
        async function startCamera() {{
            stopClientStream();
            const btn = document.getElementById('btnStartCam');
            btn.innerText = 'Starting...';
            const src = document.getElementById('cameraSource').value;
            const formData = new FormData();
            formData.append('source', src);
            try {{
                const res = await fetch('/api/start_camera', {{ method: 'POST', body: formData }});
                const json = await res.json();
                if (json.status !== 'success') {{
                    alert(json.message || 'Camera could not be opened.');
                }}
                reloadVideoFeed();
            }} catch (e) {{
                console.error(e);
            }}
            btn.innerText = 'Start PC Camera';
        }}

        async function stopCamera() {{
            stopClientStream();
            await fetch('/api/stop_camera', {{ method: 'POST' }});
            reloadVideoFeed();
        }}

        // --- 3. SAMPLE VIDEO ---
        async function startSampleVideo() {{
            stopClientStream();
            await fetch('/api/start_sample_video', {{ method: 'POST' }});
            reloadVideoFeed();
        }}

        // --- 4. MOBILE IP CAMERA STREAM (DroidCam / IP Webcam) ---
        async function startIPCamera() {{
            stopClientStream();
            const url = document.getElementById('ipCamUrl').value.trim();
            if (!url) {{
                alert('Please enter a stream URL first.\\nExample: http://192.168.1.193:4747/video');
                return;
            }}
            const btn = document.getElementById('btnIPCam');
            btn.disabled = true;
            btn.innerText = '⏳ Connecting...';

            try {{
                const formData = new FormData();
                formData.append('url', url);
                const res = await fetch('/api/start_ip_camera', {{ method: 'POST', body: formData }});
                const data = await res.json();
                if (data.status === 'success') {{
                    btn.innerText = '✅ Connected!';
                    reloadVideoFeed();
                    setTimeout(() => {{ btn.innerText = 'Connect Stream'; btn.disabled = false; }}, 2500);
                }} else {{
                    alert('Connection failed:\\n' + (data.message || 'Unknown error'));
                    btn.innerText = 'Connect Stream';
                    btn.disabled = false;
                }}
            }} catch (err) {{
                alert('Error: ' + err);
                btn.innerText = 'Connect Stream';
                btn.disabled = false;
            }}
        }}

        async function uploadAndProcessImage() {{
            stopClientStream();
            const input = document.getElementById('imageFileInput');
            if (!input.files || input.files.length === 0) {{
                alert('Please select an image file first (JPG, PNG, WEBP).');
                return;
            }}
            const btn = document.getElementById('btnUploadImage');
            const origText = btn.innerText;
            btn.innerText = '⏳ Recognizing...';
            btn.disabled = true;

            try {{
                const formData = new FormData();
                formData.append('file', input.files[0]);
                const res = await fetch('/api/upload_image', {{ method: 'POST', body: formData }});
                const data = await res.json();
                if (data.status === 'success') {{
                    btn.innerText = '✅ Recognized!';
                    reloadVideoFeed();
                    await fetchTelemetry();
                    setTimeout(() => {{ btn.innerText = origText; btn.disabled = false; }}, 2000);
                }} else {{
                    alert('Image Recognition Failed: ' + (data.message || 'Unknown error'));
                    btn.innerText = origText;
                    btn.disabled = false;
                }}
            }} catch (err) {{
                alert('Upload error: ' + err);
                btn.innerText = origText;
                btn.disabled = false;
            }}
        }}

        async function uploadAndProcessVideo() {{
            stopClientStream();
            const input = document.getElementById('videoFileInput');
            if (!input.files || input.files.length === 0) {{
                alert('Please select a video file first!');
                return;
            }}
            const formData = new FormData();
            formData.append('file', input.files[0]);
            await fetch('/api/upload_video', {{ method: 'POST', body: formData }});
            reloadVideoFeed();
        }}

        async function resetDatasets() {{
            if (!confirm('Are you sure you want to permanently DELETE all dataset records, CSV logs, and detection history?')) {{
                return;
            }}
            try {{
                const res = await fetch('/api/reset_data', {{ method: 'POST' }});
                const data = await res.json();
                alert(data.message || 'Datasets cleared successfully!');
                await fetchTelemetry();
            }} catch (err) {{
                alert('Error resetting datasets: ' + err);
            }}
        }}

        async function deployDataset(format) {{
            const btnId = format === 'csv' ? 'deployCSVBtn' : 'deployJSONBtn';
            const btn = document.getElementById(btnId);
            const origText = btn.innerText;
            btn.innerText = '⏳ Exporting...';
            btn.disabled = true;
            try {{
                const url = format === 'csv' ? '/api/download_csv' : '/api/download_json';
                const res = await fetch(url);
                if (!res.ok) throw new Error('Server returned ' + res.status);
                const blob = await res.blob();
                const filename = format === 'csv' ? 'crowd_flow_dataset.csv' : 'crowd_flow_dataset.json';
                const a = document.createElement('a');
                a.href = URL.createObjectURL(blob);
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                document.body.removeChild(a);
                URL.revokeObjectURL(a.href);
                btn.innerText = '✅ Exported!';
                setTimeout(() => {{ btn.innerText = origText; btn.disabled = false; }}, 2000);
            }} catch (err) {{
                alert('Export failed: ' + err);
                btn.innerText = origText;
                btn.disabled = false;
            }}
        }}
    </script>

</body>
</html>
"""
    return HTMLResponse(content=html_content)


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
