"""
Detection & Video Processing Pipeline Module
Integrates:
- OpenCV frame acquisition
- Ultralytics YOLO vehicle detection
- ByteTrack unique vehicle tracking
- Dedicated License Plate Detection (models/license_plate_detector.pt with fallback)
- Precise plate crop extraction with padding
- Asynchronous OCR processing queue using OCREngine (PaddleOCR with EasyOCR fallback)
- Multi-frame temporal consensus voting
- Periodic OCR intervals per tracked vehicle (OCR_FRAME_INTERVAL)
- Immediate 1:1 database record creation & synchronization
- Comprehensive diagnostic logging and debug visualization
"""

import os
import sys
import time
import queue
import logging
import threading
from collections import defaultdict, Counter
import cv2
import numpy as np
from dotenv import load_dotenv

# Database and OCR integrations
from database import (
    check_booking,
    create_incoming_vehicle,
    update_incoming_vehicle
)
from ocr import OCREngine, IndianPlateValidator

load_dotenv()

# Logger setup
logger = logging.getLogger("parking_detection")
if not logger.handlers:
    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("[%(asctime)s] [%(levelname)s] [CV]: %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)

# ==============================================================================
# Model & Pipeline Configuration (Easy to tune)
# ==============================================================================
VEHICLE_MODEL_PATH = os.getenv("VEHICLE_MODEL_PATH", "yolov8n.pt")
PLATE_MODEL_PATH = os.getenv("PLATE_MODEL_PATH", "models/license_plate_detector.pt")
VEHICLE_CONF_THRESH = float(os.getenv("VEHICLE_CONFIDENCE_THRESHOLD", "0.40"))
PLATE_CONF_THRESH = float(os.getenv("PLATE_CONFIDENCE_THRESHOLD", "0.35"))
OCR_CONF_THRESH = float(os.getenv("OCR_CONFIDENCE_THRESHOLD", "0.55"))
MAX_OCR_ATTEMPTS = int(os.getenv("MAX_OCR_ATTEMPTS", "10"))
OCR_FRAME_INTERVAL = int(os.getenv("OCR_FRAME_INTERVAL", "3"))
PLATE_DEBUG = os.getenv("PLATE_DEBUG", "True").lower() in ("true", "1", "yes")

# COCO Vehicle Classes: 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}


class DetectionPipeline:
    """
    Manages YOLO Vehicle Model, Dedicated License Plate Detector,
    and the OCREngine.
    """
    def __init__(self):
        self.vehicle_model = None
        self.plate_model = None
        self.ocr_engine = OCREngine()
        self._init_models()

    def _init_models(self):
        """Loads YOLO vehicle detection and license plate models."""
        from ultralytics import YOLO

        # 1. Vehicle Detection Model
        try:
            logger.info(f"Loading YOLO vehicle detector from: {VEHICLE_MODEL_PATH}")
            self.vehicle_model = YOLO(VEHICLE_MODEL_PATH)
            logger.info("YOLO vehicle detector loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load YOLO vehicle model: {e}")
            self.vehicle_model = None

        # 2. Dedicated License Plate Detector Model
        try:
            if os.path.exists(PLATE_MODEL_PATH):
                logger.info(f"Loading dedicated license plate detector from: {PLATE_MODEL_PATH}")
                self.plate_model = YOLO(PLATE_MODEL_PATH)
                logger.info(f"PLATE DETECTOR: Dedicated YOLO model loaded from '{PLATE_MODEL_PATH}'.")
            else:
                logger.info(f"PLATE DETECTOR: No custom model found at '{PLATE_MODEL_PATH}'. "
                            f"Using selective vehicle ROI & aspect-ratio plate candidate extractor as fallback.")
                self.plate_model = None
        except Exception as e:
            logger.warning(f"Could not load custom plate model ({e}). Using selective ROI fallback.")
            self.plate_model = None

    def detect_plate_in_vehicle(self, frame: np.ndarray, vehicle_bbox: tuple) -> list[dict]:
        """
        Runs dedicated plate detection on the vehicle bounding box crop and frame.
        Falls back to selective lower-center ROI if no dedicated model or no plate detected.
        
        Returns:
            list[dict]: List of candidate dicts with:
                - 'crop': cropped plate image (np.ndarray)
                - 'bbox': (px1, py1, px2, py2) in global frame coords
                - 'confidence': float
                - 'is_dedicated_detector': bool
        """
        vx1, vy1, vx2, vy2 = vehicle_bbox
        h_frame, w_frame = frame.shape[:2]
        vx1, vy1 = max(0, int(vx1)), max(0, int(vy1))
        vx2, vy2 = min(w_frame, int(vx2)), min(h_frame, int(vy2))

        vw, vh = vx2 - vx1, vy2 - vy1
        if vw < 30 or vh < 30:
            return []

        vehicle_crop = frame[vy1:vy2, vx1:vx2]
        candidates = []

        # A. Dedicated License Plate YOLO Detector on Vehicle Crop
        if self.plate_model is not None:
            try:
                results = self.plate_model(vehicle_crop, conf=PLATE_CONF_THRESH, verbose=False)
                for r in results:
                    for box in r.boxes:
                        bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                        conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.80

                        # Add small 5% padding around plate for cleaner OCR margins
                        bw, bh = bx2 - bx1, by2 - by1
                        pad_x = int(bw * 0.05)
                        pad_y = int(bh * 0.05)

                        px1 = max(0, int(vx1 + bx1 - pad_x))
                        py1 = max(0, int(vy1 + by1 - pad_y))
                        px2 = min(w_frame, int(vx1 + bx2 + pad_x))
                        py2 = min(h_frame, int(vy1 + by2 + pad_y))

                        plate_crop = frame[py1:py2, px1:px2]
                        if plate_crop.size > 0:
                            candidates.append({
                                "crop": plate_crop,
                                "bbox": (px1, py1, px2, py2),
                                "confidence": conf,
                                "is_dedicated_detector": True
                            })
            except Exception as e:
                logger.debug(f"Plate model inference error on crop: {e}")

            # Also check full frame if vehicle crop returned 0 candidates
            if not candidates:
                try:
                    results_full = self.plate_model(frame, conf=PLATE_CONF_THRESH, verbose=False)
                    for r in results_full:
                        for box in r.boxes:
                            bx1, by1, bx2, by2 = box.xyxy[0].cpu().numpy()
                            conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.80
                            cx, cy = (bx1 + bx2) / 2.0, (by1 + by2) / 2.0
                            if (vx1 - 10 <= cx <= vx2 + 10) and (vy1 - 10 <= cy <= vy2 + 10):
                                bw, bh = bx2 - bx1, by2 - by1
                                pad_x = int(bw * 0.05)
                                pad_y = int(bh * 0.05)
                                px1 = max(0, int(bx1 - pad_x))
                                py1 = max(0, int(by1 - pad_y))
                                px2 = min(w_frame, int(bx2 + pad_x))
                                py2 = min(h_frame, int(by2 + pad_y))
                                plate_crop = frame[py1:py2, px1:px2]
                                if plate_crop.size > 0:
                                    candidates.append({
                                        "crop": plate_crop,
                                        "bbox": (px1, py1, px2, py2),
                                        "confidence": conf,
                                        "is_dedicated_detector": True
                                    })
                except Exception as e:
                    logger.debug(f"Plate model inference error on frame: {e}")

        # B. Selective Fallback ROI Extractor (Lower 42%, Center 76% horizontally)
        if not candidates:
            py1 = int(vy1 + vh * 0.56)
            py2 = int(vy2 - vh * 0.02)
            px1 = int(vx1 + vw * 0.12)
            px2 = int(vx1 + vw * 0.88)

            py1, py2 = max(0, py1), min(h_frame, py2)
            px1, px2 = max(0, px1), min(w_frame, px2)

            if (px2 - px1) > 25 and (py2 - py1) > 12:
                roi_crop = frame[py1:py2, px1:px2]
                if roi_crop.size > 0:
                    candidates.append({
                        "crop": roi_crop,
                        "bbox": (px1, py1, px2, py2),
                        "confidence": 0.50,
                        "is_dedicated_detector": False
                    })

        return candidates


class VideoProcessingWorker:
    """
    Background worker managing real-time detection/tracking and the asynchronous OCR queue.
    """
    def __init__(self, pipeline: DetectionPipeline):
        self.pipeline = pipeline
        self.thread = None
        self.ocr_thread = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()

        # Asynchronous OCR Queue
        # Items: (track_id, series_number, plate_crop, class_name, vehicle_conf, plate_conf, plate_bbox)
        self.ocr_queue = queue.Queue(maxsize=300)

        # Stream & Capture State
        self.source_type = "NONE"
        self.source_path = None
        self.camera_index = 0
        self.status = "IDLE"
        self.error_message = None
        self.progress_percent = 0
        self.fps = 0.0

        # Session Metrics & Counting (Persists until user starts a new session)
        self.seen_vehicle_ids = set()
        self.vehicle_count = 0
        self.track_to_series = {}

        # Per-track vehicle tracking data & consensus OCR
        self.tracked_vehicles = defaultdict(lambda: {
            "series_number": None,
            "class_name": "CAR",
            "vehicle_conf": 0.0,
            "plate_bbox": None,
            "plate_conf": 0.0,
            "frame_counter": 0,
            "ocr_attempts": 0,
            "valid_candidates": Counter(),       # raw_plate -> count
            "candidate_confidences": {},         # raw_plate -> max_conf
            "candidate_displays": {},            # raw_plate -> formatted_display
            "raw_candidate_logs": [],            # list of all raw OCR readings
            "locked_plate": None,
            "locked_display": "Not clear",
            "locked_conf": 0.0,
            "ocr_status": "PROCESSING",          # "PROCESSING" | "CONFIRMED" | "NOT_CLEAR"
            "parking_allocation": "NO",
            "parking_slot": "-",
            "db_finalized": False,
            "last_seen": time.time()
        })

        # Latest global detection event for dashboard card
        self.current_vehicle = {
            "vehicle_type": "-",
            "vehicle_number": "Not clear",
            "vehicle_confidence": 0.0,
            "plate_confidence": 0.0,
            "parking_allocation": "-",
            "parking_slot": "-",
            "ocr_status": "IDLE"
        }

        # Latest processed frame for MJPEG stream
        self.current_frame = None

    def reset_session(self):
        """Resets counters and session state for a new camera, video, or image run."""
        with self.lock:
            self.seen_vehicle_ids.clear()
            self.vehicle_count = 0
            self.track_to_series.clear()
            self.tracked_vehicles.clear()
            self.progress_percent = 0
            self.error_message = None
            self.current_vehicle = {
                "vehicle_type": "-",
                "vehicle_number": "Not clear",
                "vehicle_confidence": 0.0,
                "plate_confidence": 0.0,
                "parking_allocation": "-",
                "parking_slot": "-",
                "ocr_status": "IDLE"
            }
            while not self.ocr_queue.empty():
                try:
                    self.ocr_queue.get_nowait()
                    self.ocr_queue.task_done()
                except Exception:
                    break
        logger.info("Session state & unique vehicle count reset.")

    def start_camera(self, camera_index: int = 0):
        """Starts continuous video acquisition and detection from a camera device."""
        self.stop()
        self.reset_session()
        self.source_type = "CAMERA"
        self.camera_index = camera_index
        self.source_path = camera_index
        self.status = "RUNNING_CAMERA"
        self.stop_event.clear()
        
        self.ocr_thread = threading.Thread(target=self._ocr_worker_loop, daemon=True)
        self.ocr_thread.start()

        self.thread = threading.Thread(target=self._run_processing_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started camera processing thread on index: {camera_index}")

    def start_video(self, video_file_path: str):
        """Starts continuous frame-by-frame processing for an uploaded video file."""
        self.stop()
        self.reset_session()
        self.source_type = "VIDEO"
        self.source_path = video_file_path
        self.status = "RUNNING_VIDEO"
        self.stop_event.clear()

        self.ocr_thread = threading.Thread(target=self._ocr_worker_loop, daemon=True)
        self.ocr_thread.start()

        self.thread = threading.Thread(target=self._run_processing_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started video processing thread for: {video_file_path}")

    def start_image(self, image_file_path: str):
        """Processes a static image file for vehicle and plate detection."""
        self.stop()
        self.reset_session()
        self.source_type = "IMAGE"
        self.source_path = image_file_path
        self.status = "RUNNING_IMAGE"
        self.stop_event.clear()

        self.ocr_thread = threading.Thread(target=self._ocr_worker_loop, daemon=True)
        self.ocr_thread.start()

        self.thread = threading.Thread(target=self._process_single_image_task, daemon=True)
        self.thread.start()
        logger.info(f"Started image processing thread for: {image_file_path}")

    def stop(self):
        """Signals background processing threads to stop and releases resources."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self._drain_ocr_queue(timeout_seconds=2.0)
        self.status = "IDLE" if self.status not in ("COMPLETED", "ERROR") else self.status
        logger.info("Processing loops stopped.")

    def _drain_ocr_queue(self, timeout_seconds: float = 4.0):
        """Waits for remaining OCR tasks to finish processing."""
        start_t = time.time()
        while not self.ocr_queue.empty() and (time.time() - start_t < timeout_seconds):
            time.sleep(0.1)

    def _run_processing_loop(self):
        """Main frame acquisition loop for live camera or uploaded video."""
        cap = None
        try:
            if self.source_type == "CAMERA":
                cap = cv2.VideoCapture(int(self.source_path), cv2.CAP_DSHOW if os.name == 'nt' else cv2.CAP_ANY)
            else:
                cap = cv2.VideoCapture(str(self.source_path))

            if not cap.isOpened():
                err = f"Failed to open video source: {self.source_path}"
                logger.error(err)
                with self.lock:
                    self.status = "ERROR"
                    self.error_message = err
                return

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if self.source_type == "VIDEO" else 0
            frame_idx = 0
            fps_start_time = time.time()
            fps_counter = 0

            while not self.stop_event.is_set():
                ret, frame = cap.read()
                if not ret:
                    if self.source_type == "VIDEO":
                        logger.info("Reached final video frame. Draining pending OCR queue...")
                        self._drain_ocr_queue(timeout_seconds=4.5)
                        self._finalize_unconfirmed_vehicles()
                        with self.lock:
                            self.status = "COMPLETED"
                            self.progress_percent = 100
                        break
                    else:
                        time.sleep(0.1)
                        continue

                frame_idx += 1
                fps_counter += 1
                if time.time() - fps_start_time >= 1.0:
                    self.fps = round(fps_counter / (time.time() - fps_start_time), 1)
                    fps_counter = 0
                    fps_start_time = time.time()

                if self.source_type == "VIDEO" and total_frames > 0:
                    self.progress_percent = min(99, int((frame_idx / total_frames) * 100))

                annotated_frame = self._process_single_frame(frame, is_image=False)
                with self.lock:
                    self.current_frame = annotated_frame

                time.sleep(0.005 if self.source_type == "CAMERA" else 0.008)

        except Exception as e:
            logger.error(f"Exception in video processing loop: {e}", exc_info=True)
            with self.lock:
                self.status = "ERROR"
                self.error_message = str(e)
        finally:
            if cap is not None:
                cap.release()
            logger.info("Video capture released.")

    def _process_single_image_task(self):
        """Processes a single static image file."""
        try:
            frame = cv2.imread(str(self.source_path))
            if frame is None:
                err = f"Failed to read image file: {self.source_path}"
                logger.error(err)
                with self.lock:
                    self.status = "ERROR"
                    self.error_message = err
                return

            annotated_frame = self._process_single_frame(frame, is_image=True)
            self._drain_ocr_queue(timeout_seconds=4.0)
            self._finalize_unconfirmed_vehicles()
            with self.lock:
                self.current_frame = annotated_frame
                self.progress_percent = 100
                self.status = "COMPLETED"
            logger.info(f"Image processing completed for: {self.source_path}")
        except Exception as e:
            logger.error(f"Exception during image processing: {e}", exc_info=True)
            with self.lock:
                self.status = "ERROR"
                self.error_message = str(e)

    def _process_single_frame(self, frame: np.ndarray, is_image: bool = False) -> np.ndarray:
        """
        Executes YOLO vehicle detection & ByteTrack tracking, registers new vehicles immediately,
        submits plate crops to the async OCR queue, and annotates the frame with GREEN plate boxes.
        """
        annotated_frame = frame.copy()
        if self.pipeline.vehicle_model is None:
            cv2.putText(annotated_frame, "Vehicle YOLO Model Not Loaded", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return annotated_frame

        try:
            if is_image:
                results = self.pipeline.vehicle_model(
                    source=frame,
                    classes=list(VEHICLE_CLASSES.keys()),
                    conf=VEHICLE_CONF_THRESH,
                    verbose=False
                )
            else:
                results = self.pipeline.vehicle_model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=list(VEHICLE_CLASSES.keys()),
                    conf=VEHICLE_CONF_THRESH,
                    verbose=False
                )

            if results and len(results) > 0:
                boxes = results[0].boxes
                if boxes is not None and len(boxes) > 0:
                    for idx, box in enumerate(boxes):
                        xyxy = box.xyxy[0].cpu().numpy()
                        vx1, vy1, vx2, vy2 = map(int, xyxy)
                        conf = float(box.conf[0].cpu().numpy()) if box.conf is not None else 0.0
                        cls_id = int(box.cls[0].cpu().numpy()) if box.cls is not None else 2
                        class_name = VEHICLE_CLASSES.get(cls_id, "car").upper()

                        track_id = int(box.id[0].cpu().numpy()) if box.id is not None else (idx + 1)

                        # 1. NEW UNIQUE VEHICLE DETECTION -> Immediate 1:1 MySQL Record Creation
                        if track_id not in self.seen_vehicle_ids:
                            self.seen_vehicle_ids.add(track_id)
                            self.vehicle_count = len(self.seen_vehicle_ids)
                            
                            series_num = create_incoming_vehicle(track_id, vehicle_type=class_name)
                            self.track_to_series[track_id] = series_num

                            if PLATE_DEBUG:
                                print("\n" + "=" * 50)
                                print("VEHICLE DETECTED")
                                print(f"Tracking ID:           {track_id}")
                                print(f"Vehicle Type:          {class_name}")
                                print(f"Vehicle Confidence:    {round(conf * 100, 1)}%")
                                print(f"Current Vehicle Count: {self.vehicle_count}")
                                print(f"Database Series:       {series_num}")
                                print("OCR Status:            PROCESSING")
                                print("=" * 50 + "\n", flush=True)

                        series_num = self.track_to_series.get(track_id)
                        v_data = self.tracked_vehicles[track_id]
                        v_data["series_number"] = series_num
                        v_data["class_name"] = class_name
                        v_data["vehicle_conf"] = conf
                        v_data["frame_counter"] += 1
                        v_data["last_seen"] = time.time()

                        # 2. Plate Detection for this vehicle (run on every frame to tightly track physical plate)
                        plate_candidates = self.pipeline.detect_plate_in_vehicle(frame, (vx1, vy1, vx2, vy2))
                        if plate_candidates:
                            p_info = plate_candidates[0]
                            p_crop = p_info["crop"]
                            px1, py1, px2, py2 = p_info["bbox"]
                            p_conf = p_info["confidence"]

                            v_data["plate_bbox"] = (px1, py1, px2, py2)
                            v_data["plate_conf"] = p_conf

                            # Async OCR Submission
                            should_ocr = is_image or (
                                v_data["ocr_status"] == "PROCESSING"
                                and v_data["ocr_attempts"] < MAX_OCR_ATTEMPTS
                                and (v_data["frame_counter"] % OCR_FRAME_INTERVAL == 0)
                            )

                            if should_ocr and not self.ocr_queue.full():
                                self.ocr_queue.put((
                                    track_id, series_num, p_crop, class_name,
                                    conf, p_conf, (px1, py1, px2, py2)
                                ))

                        # 3. Draw Vehicle Annotations (BLUE) & GREEN Plate Box on Video Stream
                        self._draw_vehicle_annotation(annotated_frame, track_id, (vx1, vy1, vx2, vy2), v_data)

        except Exception as e:
            logger.debug(f"Frame processing error: {e}")

        self._draw_hud_banner(annotated_frame)
        return annotated_frame

    def _ocr_worker_loop(self):
        """
        Dedicated background worker thread consuming plate crops from ocr_queue,
        running multi-pass OCR (PaddleOCR/EasyOCR), validating plates, and updating MySQL.
        """
        logger.info("Asynchronous OCR worker thread active.")
        while not self.stop_event.is_set():
            try:
                task = self.ocr_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            track_id, series_num, plate_crop, class_name, vehicle_conf, plate_conf, plate_bbox = task
            v_data = self.tracked_vehicles[track_id]

            if v_data["ocr_status"] != "PROCESSING":
                self.ocr_queue.task_done()
                continue

            v_data["ocr_attempts"] += 1

            # Diagnostic Terminal Log for Plate Detection
            if PLATE_DEBUG and v_data["ocr_attempts"] == 1:
                px1, py1, px2, py2 = plate_bbox
                pw, ph = px2 - px1, py2 - py1
                print("\n" + "=" * 50)
                print("PLATE DETECTED")
                print(f"Tracking ID:        {track_id}")
                print(f"Plate BBox:         x1={px1} y1={py1} x2={px2} y2={py2}")
                print(f"Plate Size:         {pw}x{ph}")
                print(f"Plate Confidence:   {round(plate_conf * 100, 1)}%")
                print(f"OCR Engine:         {self.pipeline.ocr_engine.engine_name}")
                print("OCR Status:         PROCESSING")
                print("=" * 50 + "\n", flush=True)

            is_valid, raw_plate, fmt_display, ocr_conf = self.pipeline.ocr_engine.recognize_plate(
                plate_crop, tracking_id=track_id
            )

            if raw_plate and raw_plate != "Not clear":
                v_data["raw_candidate_logs"].append((raw_plate, ocr_conf))

            if is_valid:
                v_data["valid_candidates"][raw_plate] += 1
                v_data["candidate_confidences"][raw_plate] = max(
                    v_data["candidate_confidences"].get(raw_plate, 0.0), ocr_conf
                )
                v_data["candidate_displays"][raw_plate] = fmt_display

                top_raw, top_count = v_data["valid_candidates"].most_common(1)[0]
                top_conf = v_data["candidate_confidences"][top_raw]
                top_display = v_data["candidate_displays"][top_raw]

                # Consensus condition: >= 2 consistent frames, high confidence >= 0.75, or image run
                if top_count >= 2 or top_conf >= 0.75 or self.source_type == "IMAGE" or v_data["ocr_attempts"] >= MAX_OCR_ATTEMPTS:
                    self._confirm_vehicle_plate(track_id, series_num, top_raw, top_display, top_conf, class_name, vehicle_conf)

            elif v_data["ocr_attempts"] >= MAX_OCR_ATTEMPTS and not v_data["locked_plate"]:
                self._finalize_single_vehicle_unclear(track_id, series_num, class_name, vehicle_conf)

            self.ocr_queue.task_done()

        logger.info("Asynchronous OCR worker thread stopped.")

    def _confirm_vehicle_plate(self, track_id, series_num, raw_plate, fmt_display, plate_conf, class_name, vehicle_conf):
        """Confirms a validated plate, performs MySQL booking check, and logs result."""
        v_data = self.tracked_vehicles[track_id]
        if v_data["db_finalized"]:
            return

        booking = check_booking(raw_plate)
        alloc = booking.get("parking_allocation", "NO")
        slot = booking.get("parking_slot", "-")

        v_data["locked_plate"] = raw_plate
        v_data["locked_display"] = fmt_display
        v_data["locked_conf"] = plate_conf
        v_data["ocr_status"] = "CONFIRMED"
        v_data["parking_allocation"] = alloc
        v_data["parking_slot"] = slot
        v_data["db_finalized"] = True

        if series_num:
            update_incoming_vehicle(
                series_number=series_num,
                vehicle_number=raw_plate,
                parking_allocation=alloc,
                parking_slot=slot,
                ocr_status="CONFIRMED",
                plate_confidence=plate_conf
            )

        with self.lock:
            self.current_vehicle = {
                "vehicle_type": class_name,
                "vehicle_number": fmt_display,
                "vehicle_confidence": round(vehicle_conf * 100, 1),
                "plate_confidence": round(plate_conf * 100, 1),
                "parking_allocation": alloc,
                "parking_slot": slot,
                "ocr_status": "CONFIRMED"
            }

        # Terminal Output for Confirmed OCR
        print("\n" + "=" * 50)
        print("OCR RESULT")
        print(f"Tracking ID:        {track_id}")
        print(f"Database Series:    {series_num}")
        print(f"Raw OCR:            {raw_plate}")
        print(f"Normalized:         {raw_plate}")
        print(f"Display:            {fmt_display}")
        print(f"OCR Confidence:     {round(plate_conf * 100, 1)}%")
        print("Status:             CONFIRMED")
        print(f"Parking Allocation: {alloc}")
        print(f"Parking Slot:       {slot}")
        print("=" * 50 + "\n", flush=True)

    def _finalize_single_vehicle_unclear(self, track_id, series_num, class_name, vehicle_conf):
        """Finalizes an unreadable plate as Not clear without creating duplicate records."""
        v_data = self.tracked_vehicles[track_id]
        if v_data["db_finalized"]:
            return

        v_data["locked_plate"] = "Not clear"
        v_data["locked_display"] = "Not clear"
        v_data["ocr_status"] = "NOT_CLEAR"
        v_data["parking_allocation"] = "NO"
        v_data["parking_slot"] = "-"
        v_data["db_finalized"] = True

        if series_num:
            update_incoming_vehicle(
                series_number=series_num,
                vehicle_number="Not clear",
                parking_allocation="NO",
                parking_slot="-",
                ocr_status="NOT_CLEAR",
                plate_confidence=0.0
            )

        with self.lock:
            self.current_vehicle = {
                "vehicle_type": class_name,
                "vehicle_number": "Not clear",
                "vehicle_confidence": round(vehicle_conf * 100, 1),
                "plate_confidence": 0.0,
                "parking_allocation": "NO",
                "parking_slot": "-",
                "ocr_status": "NOT_CLEAR"
            }

        last_raw = v_data["raw_candidate_logs"][-1][0] if v_data["raw_candidate_logs"] else "None"
        print("\n" + "=" * 50)
        print("OCR RESULT")
        print(f"Tracking ID:        {track_id}")
        print(f"Database Series:    {series_num}")
        print(f"Raw OCR:            {last_raw}")
        print("Status:             NOT_CLEAR")
        print("Final Plate:        Not clear")
        print("Parking Allocation: NO")
        print("=" * 50 + "\n", flush=True)

    def _finalize_unconfirmed_vehicles(self):
        """At session completion, finalizes any remaining pending vehicles."""
        for track_id, v_data in list(self.tracked_vehicles.items()):
            if not v_data["db_finalized"]:
                series_num = v_data.get("series_number")
                class_name = v_data.get("class_name", "CAR")
                v_conf = v_data.get("vehicle_conf", 0.0)

                if v_data["valid_candidates"]:
                    top_raw, top_count = v_data["valid_candidates"].most_common(1)[0]
                    top_conf = v_data["candidate_confidences"][top_raw]
                    top_display = v_data["candidate_displays"][top_raw]
                    self._confirm_vehicle_plate(track_id, series_num, top_raw, top_display, top_conf, class_name, v_conf)
                else:
                    self._finalize_single_vehicle_unclear(track_id, series_num, class_name, v_conf)

    def _draw_vehicle_annotation(self, frame: np.ndarray, track_id: int, bbox: tuple, v_data: dict):
        """
        Draws:
        - BLUE bounding box around the vehicle.
        - BRIGHT GREEN bounding box tightly around the license plate.
        """
        vx1, vy1, vx2, vy2 = bbox
        
        # 1. BLUE Vehicle Bounding Box & Badge (BGR: 235, 140, 20)
        vehicle_blue = (235, 140, 20)
        cv2.rectangle(frame, (vx1, vy1), (vx2, vy2), vehicle_blue, 2)

        lines = [f"ID:{track_id} {v_data['class_name']} ({int(v_data['vehicle_conf'] * 100)}%)"]
        if v_data["ocr_status"] == "CONFIRMED":
            lines.append(f"{v_data['locked_display']} [Alloc:{v_data['parking_allocation']}]")
        elif v_data["ocr_status"] == "NOT_CLEAR":
            lines.append("Plate: Not clear")
        else:
            lines.append("Plate: Reading...")

        badge_y = max(24, vy1 - 8)
        for idx, line in enumerate(lines):
            y_pos = badge_y - (len(lines) - 1 - idx) * 20
            (lw, lh), _ = cv2.getTextSize(line, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 1)
            cv2.rectangle(frame, (vx1, y_pos - lh - 4), (vx1 + lw + 6, y_pos + 4), vehicle_blue, -1)
            cv2.putText(frame, line, (vx1 + 3, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)

        # 2. BRIGHT GREEN bounding box tightly around the physical License Plate (BGR: 0, 255, 0)
        if v_data.get("plate_bbox"):
            px1, py1, px2, py2 = v_data["plate_bbox"]
            cv2.rectangle(frame, (px1, py1), (px2, py2), (0, 255, 0), 2)
            
            plate_tag = "PLATE"
            if v_data["ocr_status"] == "CONFIRMED":
                plate_tag = v_data["locked_display"]
            elif v_data["ocr_status"] == "NOT_CLEAR":
                plate_tag = "PLATE: Not clear"

            (ptw, pth), _ = cv2.getTextSize(plate_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
            tag_y = max(14, py1 - 4)
            cv2.rectangle(frame, (px1, tag_y - pth - 3), (px1 + ptw + 4, tag_y + 2), (0, 255, 0), -1)
            cv2.putText(frame, plate_tag, (px1 + 2, tag_y - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    def _draw_hud_banner(self, frame: np.ndarray):
        """Draws a clean professional HUD overlay banner."""
        cv2.rectangle(frame, (12, 12), (300, 78), (255, 255, 255), -1)
        cv2.rectangle(frame, (12, 12), (300, 78), (200, 205, 215), 1)

        count_text = f"VEHICLES DETECTED: {self.vehicle_count}"
        cv2.putText(frame, count_text, (22, 42), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (15, 23, 42), 2)

        status_text = f"Source: {self.source_type} | FPS: {self.fps}"
        cv2.putText(frame, status_text, (22, 66), cv2.FONT_HERSHEY_SIMPLEX, 0.44, (100, 116, 139), 1)

    def get_jpeg_frame(self):
        """Encodes current frame as JPEG for MJPEG stream."""
        with self.lock:
            if self.current_frame is None:
                blank = np.zeros((480, 640, 3), dtype=np.uint8)
                blank[:] = (248, 250, 252)
                cv2.putText(blank, "No Active Video Stream", (180, 240),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.75, (100, 116, 139), 2)
                _, buffer = cv2.imencode('.jpg', blank)
                return buffer.tobytes()

            ret, buffer = cv2.imencode('.jpg', self.current_frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
            if not ret:
                return None
            return buffer.tobytes()

    def get_status_summary(self):
        """Returns structured status telemetry."""
        with self.lock:
            return {
                "status": self.status,
                "source_type": self.source_type,
                "progress_percent": self.progress_percent,
                "vehicle_count": self.vehicle_count,
                "fps": self.fps,
                "error_message": self.error_message,
                "current_vehicle": self.current_vehicle
            }
