#!/usr/bin/env python3
"""
Event Crowd Flow Tracker — CLI Tool.

Real-time person detection, tracking, multi-gate line crossing counts,
lightweight face detection, and probabilistic zone scattering simulation
with live terminal dashboard rendering.
"""

from __future__ import annotations

# Fix OpenMP "already initialized" crash (OMP Error #15) that occurs on Windows
# when Anaconda's numpy and PyTorch each ship their own libiomp5md.dll.
# Must be set BEFORE any torch/cv2 import.
import os
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import sys
import time
from pathlib import Path
from typing import Optional

from src.db import DatabaseManager
from src.detector import FaceDetector, PersonDetector
from src.dashboard import TerminalDashboard
from src.tracker import LineCrossingTracker
from src.video import VideoProcessor
from src.zones import ZoneManager


def parse_arguments() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(
        prog="event-tracker",
        description="Event Crowd Flow Tracker: Detection, Tracking, Face Detection & Zone Scattering Simulation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Required argument
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file (.mp4, .mov, .avi, etc.).",
    )

    # Simulation & Gate Configuration
    parser.add_argument(
        "--gates",
        type=int,
        default=1,
        help="Number of simulated entry gates/lines splitting the frame width.",
    )
    parser.add_argument(
        "--zones",
        type=str,
        default="A,B,C,D",
        help="Comma-separated zone names for crowd scattering (e.g. 'A,B,C,D' or 'MainStage,Bar,VIP,Expo').",
    )
    parser.add_argument(
        "--line-pos",
        type=float,
        default=0.65,
        help="Y-axis position of virtual counting line as fraction of frame height (0.1 to 0.9).",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default="both",
        choices=["both", "bottom-to-top", "top-to-bottom"],
        help="Valid crossing direction considered an entry.",
    )

    # Simulation Dynamics
    parser.add_argument(
        "--repulsion",
        type=float,
        default=0.08,
        help="Soft-repulsion factor preventing zone overfilling.",
    )
    parser.add_argument(
        "--drift-interval",
        type=float,
        default=3.0,
        help="Video time interval (in seconds) between crowd re-distribution passes.",
    )
    parser.add_argument(
        "--drift-rate",
        type=float,
        default=0.08,
        help="Fraction of attendees redistributed during each drift pass.",
    )

    # Output options
    parser.add_argument(
        "--output-video",
        type=str,
        default=None,
        help="Path to export annotated video with bounding boxes, gates, trails, and HUD.",
    )
    parser.add_argument(
        "--save-log",
        type=str,
        default=None,
        help="Path to save CSV entry log (person_id, entry_timestamp, gate, assigned_zone).",
    )
    parser.add_argument(
        "--db",
        type=str,
        default=None,
        help="Path to SQLite database to persist structured runs, entries, transitions, snapshots, and faces.",
    )

    # Performance & Model options
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Detection confidence threshold for YOLOv8 person detector.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="yolov8n.pt",
        help="YOLO model checkpoint name or path.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Computation device ('cpu', 'cuda', '0', etc.).",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Optional maximum number of frames to process before stopping.",
    )
    parser.add_argument(
        "--ui-fps",
        type=float,
        default=8.0,
        help="Target refresh rate (Hz) for the live terminal UI.",
    )

    return parser.parse_args()


def main() -> None:
    """Main execution pipeline."""
    args = parse_arguments()

    video_path = Path(args.video)
    if not video_path.exists():
        print(f"\n[ERROR] Input video not found: {args.video}", file=sys.stderr)
        sys.exit(1)

    # Parse zones
    zones = [z.strip() for z in args.zones.split(",") if z.strip()]
    if not zones:
        zones = ["A", "B", "C", "D"]

    # 1. Initialize Video Pipeline
    video_proc = VideoProcessor(video_path=str(video_path), output_path=args.output_video)
    metadata = video_proc.metadata

    # 2. Initialize Database (if requested)
    db_mgr: Optional[DatabaseManager] = None
    run_id: Optional[int] = None
    if args.db:
        db_mgr = DatabaseManager(args.db)
        run_id = db_mgr.start_run(str(video_path))

    # 3. Initialize CSV Log (if requested)
    csv_file = None
    csv_writer = None
    if args.save_log:
        log_path = Path(args.save_log)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        csv_file = open(log_path, mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["person_id", "entry_timestamp", "gate", "assigned_zone"])

    # 4. Initialize AI Models & Tracker
    person_detector = PersonDetector(model_name=args.model, device=args.device)
    face_detector = FaceDetector()
    tracker = LineCrossingTracker(
        num_gates=args.gates,
        line_pos_ratio=args.line_pos,
        direction=args.direction,
    )
    tracker.set_frame_dimensions(metadata.width, metadata.height)

    zone_mgr = ZoneManager(
        zone_names=zones,
        repulsion_factor=args.repulsion,
        drift_interval=args.drift_interval,
        drift_rate=args.drift_rate,
    )

    # 5. Initialize Live Terminal Dashboard
    total_frames_target = (
        min(metadata.total_frames, args.max_frames)
        if args.max_frames and metadata.total_frames > 0
        else (args.max_frames or metadata.total_frames or 1000)
    )
    dashboard = TerminalDashboard(
        total_frames=total_frames_target,
        fps=metadata.fps,
        duration_s=metadata.duration_s,
        update_hz=args.ui_fps,
    )

    video_proc.open()
    dashboard.start()

    # Runtime tracking
    frame_count = 0
    total_drift_transitions = 0
    start_wall_time = time.time()
    last_fps_time = start_wall_time
    fps_frames = 0
    current_fps = 0.0

    try:
        while True:
            ret, frame, frame_idx, video_time_s = video_proc.read_frame()
            if not ret or frame is None:
                break

            frame_count += 1
            fps_frames += 1
            now = time.time()
            if now - last_fps_time >= 0.5:
                current_fps = fps_frames / (now - last_fps_time)
                fps_frames = 0
                last_fps_time = now

            # Step 1: Person Detection & ByteTrack Tracking
            tracks = person_detector.track(frame, conf_threshold=args.conf)

            # Step 2: Face Detection & Person Matching
            faces = face_detector.detect_faces(
                frame, frame_idx, person_tracks=tracks
            )

            # Record faces to SQLite
            if db_mgr and run_id is not None:
                for f in faces:
                    db_mgr.log_face(
                        run_id=run_id,
                        frame_number=frame_idx,
                        video_time_s=video_time_s,
                        matched_track_id=f.matched_track_id,
                    )

            # Step 3: Line Crossing & Multi-Gate Entry Verification
            new_entries = tracker.update_tracks(tracks, frame_idx, video_time_s)

            # Step 4: Zone Assignment with Soft Repulsion
            for entry in new_entries:
                assigned_zone = zone_mgr.assign_zone(entry.track_id)
                entry.assigned_zone = assigned_zone

                # Save to CSV
                if csv_writer and csv_file:
                    csv_writer.writerow(
                        [entry.track_id, round(entry.video_time_s, 3), entry.gate_name, assigned_zone]
                    )
                    csv_file.flush()

                # Save to SQLite
                if db_mgr and run_id is not None:
                    db_mgr.log_entry(
                        run_id=run_id,
                        track_id=entry.track_id,
                        gate=entry.gate_name,
                        video_time_s=entry.video_time_s,
                        assigned_zone=assigned_zone,
                    )

                # Broadcast to Activity Feed
                dashboard.log_event(
                    f"Person [bold green]#{entry.track_id}[/bold green] entered via [bold cyan]{entry.gate_name}[/bold cyan] "
                    f"-> Assigned [bold magenta]Zone {assigned_zone}[/bold magenta] ({entry.video_time_s:.1f}s)"
                )

            # Step 5: Zone Scattering Crowd Drift Simulation
            drift_events = zone_mgr.update_drift(video_time_s)
            if drift_events:
                total_drift_transitions += len(drift_events)
                for pid, from_z, to_z in drift_events:
                    if db_mgr and run_id is not None:
                        db_mgr.log_transition(
                            run_id=run_id,
                            track_id=pid,
                            from_zone=from_z,
                            to_zone=to_z,
                            video_time_s=video_time_s,
                        )
                    dashboard.log_event(
                        f"Crowd Drift: Person #{pid} moved [dim]{from_z}[/dim] -> [bold yellow]Zone {to_z}[/bold yellow]"
                    )

            # Step 6: Periodic 1-Second Zone Snapshots
            if db_mgr and run_id is not None and zone_mgr.check_snapshot_due(video_time_s):
                for z, cnt in zone_mgr.occupancy.items():
                    db_mgr.log_snapshot(
                        run_id=run_id,
                        video_time_s=video_time_s,
                        zone=z,
                        occupancy=cnt,
                    )

            # Step 7: Video Annotation Export (if requested)
            if args.output_video:
                annotated = video_proc.annotate_frame(
                    frame=frame,
                    person_tracks=tracks,
                    faces=faces,
                    tracker=tracker,
                    zone_mgr=zone_mgr,
                    frame_number=frame_idx,
                    video_time_s=video_time_s,
                    fps_current=current_fps,
                )
                video_proc.write_frame(annotated)

            # Step 8: Terminal Dashboard Refresh
            dashboard.update(
                frame_idx=frame_count,
                video_time_s=video_time_s,
                fps_realtime=current_fps,
                tracker=tracker,
                zone_mgr=zone_mgr,
                current_faces=faces,
                unique_faces_seen=face_detector.unique_faces_seen,
            )

            if args.max_frames and frame_count >= args.max_frames:
                break

    except KeyboardInterrupt:
        dashboard.log_event("[bold red]Processing interrupted by user (Ctrl+C). Finalizing...[/bold red]")
    finally:
        # Final render flush
        dashboard.stop()
        video_proc.close()

        if csv_file:
            csv_file.close()

        if db_mgr and run_id is not None:
            db_mgr.finish_run(run_id=run_id, total_entered=tracker.total_entered)
            db_mgr.close()

        # Display full summary
        duration = metadata.duration_s if metadata.duration_s > 0 else (frame_count / max(1.0, metadata.fps))
        dashboard.print_final_summary(
            tracker=tracker,
            zone_mgr=zone_mgr,
            total_transitions=total_drift_transitions,
            unique_faces_seen=face_detector.unique_faces_seen,
            duration_s=duration,
            output_video=args.output_video,
            save_log=args.save_log,
            save_db=args.db,
        )


if __name__ == "__main__":
    main()
