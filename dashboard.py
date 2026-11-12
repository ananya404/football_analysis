import os
import sys
import gc
import tempfile

# Matplotlib config: use a cross-platform temp dir (not a hardcoded Windows path)
os.environ["MPLCONFIGDIR"] = os.path.join(tempfile.gettempdir(), "mpl_cache")

# Always run relative to project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
os.chdir(PROJECT_ROOT)
sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
import cv2
import numpy as np
import pandas as pd

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚽ Football Analysis Dashboard",
    page_icon="⚽",
    layout="wide",
)

st.title("⚽ Football Analysis Dashboard")
st.caption("Player tracking · Team assignment · Ball possession · Speed & Distance")

# ── Auto-download model weights if missing (Streamlit Cloud / fresh clone) ─────
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "best.pt")
MODEL_GDRIVE_ID = "1DC2kCygbBWUKheQ_9cFziCsYVSRw6axK"

if not os.path.exists(MODEL_PATH) or os.path.getsize(MODEL_PATH) < 1_000_000:
    with st.spinner("⬇️ Downloading model weights (best.pt, ~172 MB) — first run only…"):
        import gdown
        os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
        gdown.download(id=MODEL_GDRIVE_ID, output=MODEL_PATH, quiet=False)
    st.success("Model downloaded ✅")

# ── Lazy project imports (after path setup) ────────────────────────────────────
from utils import read_video
from trackers import Tracker
from team_assigner import TeamAssigner
from player_ball_assigner import PlayerBallAssigner
from camera_movement_estimator import CameraMovementEstimator
from view_transformer import ViewTransformer
from speed_and_distance_estimator import SpeedAndDistance_Estimator

# ── Helpers ────────────────────────────────────────────────────────────────────

def bgr_to_hex(bgr):
    """Convert OpenCV BGR tuple/array to a CSS hex string."""
    b, g, r = int(bgr[0]), int(bgr[1]), int(bgr[2])
    return f"#{r:02x}{g:02x}{b:02x}"


def draw_camera_movement_inplace(frames, camera_movement_per_frame):
    """Memory-efficient in-place version: modifies frames without creating a new list.
    The original creates TWO full frame lists; this reuses the existing one.
    """
    for frame_num, frame in enumerate(frames):
        overlay = frame.copy()          # one copy for blending — unavoidable
        cv2.rectangle(overlay, (0, 0), (500, 100), (255, 255, 255), -1)
        alpha = 0.6
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)  # write result back into frame
        cv2.putText(frame, f"Camera Movement X: {camera_movement_per_frame[frame_num][0]:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Camera Movement Y: {camera_movement_per_frame[frame_num][1]:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
    return frames


def save_as_mp4(frames, output_path):
    """Save annotated frames as H.264 MP4 (browser-compatible) via imageio-ffmpeg."""
    import imageio_ffmpeg as ioff
    ffmpeg_exe = ioff.get_ffmpeg_exe()
    h, w = frames[0].shape[:2]
    writer = ioff.write_frames(
        output_path,
        size=(w, h),
        fps=24,
        codec="libx264",
        quality=None,
        bitrate=None,
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "23", "-preset", "fast"],
    )
    writer.send(None)  # prime the generator
    for frame in frames:
        writer.send(frame.tobytes())
    writer.close()


def convert_to_h264(src_path, dst_path):
    """Re-encode an existing video file to H.264 MP4 using imageio-ffmpeg."""
    import imageio_ffmpeg as ioff
    import subprocess
    ffmpeg_exe = ioff.get_ffmpeg_exe()
    subprocess.run(
        [ffmpeg_exe, "-y", "-i", src_path,
         "-c:v", "libx264", "-crf", "23", "-preset", "fast",
         "-pix_fmt", "yuv420p", dst_path],
        check=True, capture_output=True,
    )


def collect_player_stats(tracks):
    """Extract per-player speed / distance / team from tracks."""
    stats = {}
    for frame_num, player_track in enumerate(tracks["players"]):
        for pid, info in player_track.items():
            if pid not in stats:
                stats[pid] = {
                    "Player ID": pid,
                    "Team": info.get("team", "?"),
                    "Max Speed (km/h)": 0.0,
                    "Distance (m)": 0.0,
                }
            spd = info.get("speed", 0) or 0
            dist = info.get("distance_covered", 0) or 0
            stats[pid]["Max Speed (km/h)"] = max(stats[pid]["Max Speed (km/h)"], spd)
            if dist:
                stats[pid]["Distance (m)"] = dist
    return pd.DataFrame(stats.values()).sort_values("Player ID").reset_index(drop=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("⚙️ Settings")

    # ── Video source ───────────────────────────────────────────────────────────
    st.subheader("📂 Video source")
    source_mode = st.radio("Choose input", ["Upload a video", "Select from input_videos/"],
                           horizontal=False)

    video_path = None

    if source_mode == "Upload a video":
        uploaded = st.file_uploader("Upload MP4", type=["mp4", "avi", "mov", "mkv"])
        if uploaded is not None:
            # Save to a temp file so OpenCV can read it by path
            upload_dir = os.path.join(PROJECT_ROOT, "input_videos", "_uploads")
            os.makedirs(upload_dir, exist_ok=True)
            video_path = os.path.join(upload_dir, uploaded.name)
            with open(video_path, "wb") as f:
                f.write(uploaded.getbuffer())
            st.success(f"Uploaded: **{uploaded.name}**")
        else:
            st.info("Upload a video file to continue.")
    else:
        input_dir = os.path.join(PROJECT_ROOT, "input_videos")
        mp4_files = [f for f in os.listdir(input_dir) if f.lower().endswith(".mp4")]
        if not mp4_files:
            st.error("No .mp4 files found in input_videos/")
        else:
            selected_video = st.selectbox("Select video", mp4_files)
            video_path = os.path.join(input_dir, selected_video)

    use_stubs = st.checkbox(
        "Use cached stubs (faster)",
        value=True,
        help="Loads pre-computed tracking & camera data instead of re-running YOLO inference.\n"
             "⚠️ Stubs only match the original sample video — disable for a new upload.",
    )

    st.divider()
    run_btn = st.button(
        "▶ Run Analysis",
        type="primary",
        use_container_width=True,
        disabled=(video_path is None),
    )

OUTPUT_MP4 = os.path.join(PROJECT_ROOT, "output_videos", "output_video.mp4")

# ── Analysis pipeline ──────────────────────────────────────────────────────────
if run_btn:
    progress = st.progress(0, text="Starting…")
    status   = st.empty()

    def update(pct, msg):
        progress.progress(pct, text=msg)
        status.info(f"🔄 {msg}")

    update(0.05, "Reading video frames…")
    video_frames = read_video(video_path)
    n_frames = len(video_frames)

    update(0.12, f"Loaded {n_frames} frames — initialising tracker…")
    tracker = Tracker(os.path.join(PROJECT_ROOT, "models", "best.pt"))

    update(0.20, "Tracking players, referees & ball…")
    tracks = tracker.get_object_tracks(
        video_frames,
        read_from_stub=use_stubs,
        stub_path=os.path.join(PROJECT_ROOT, "stubs", "track_stubs.pkl"),
    )
    tracker.add_position_to_tracks(tracks)

    update(0.35, "Estimating camera movement…")
    cam_est = CameraMovementEstimator(video_frames[0])
    cam_movement = cam_est.get_camera_movement(
        video_frames,
        read_from_stub=use_stubs,
        stub_path=os.path.join(PROJECT_ROOT, "stubs", "camera_movement_stub.pkl"),
    )
    cam_est.add_adjust_positions_to_tracks(tracks, cam_movement)

    update(0.45, "Applying perspective transform…")
    ViewTransformer().add_transformed_position_to_tracks(tracks)

    update(0.50, "Interpolating ball positions…")
    tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])

    update(0.55, "Calculating speed & distance…")
    speed_est = SpeedAndDistance_Estimator()
    speed_est.add_speed_and_distance_to_tracks(tracks)

    update(0.65, "Assigning player teams by shirt colour…")
    team_assigner = TeamAssigner()
    team_assigner.assign_team_color(video_frames[0], tracks["players"][0])
    for frame_num, player_track in enumerate(tracks["players"]):
        for pid, info in player_track.items():
            team = team_assigner.get_player_team(video_frames[frame_num], info["bbox"], pid)
            tracks["players"][frame_num][pid]["team"]       = team
            tracks["players"][frame_num][pid]["team_color"] = team_assigner.team_colors[team]

    update(0.75, "Assigning ball possession…")
    ball_assigner   = PlayerBallAssigner()
    team_ball_ctrl  = []
    for frame_num, player_track in enumerate(tracks["players"]):
        ball_bbox       = tracks["ball"][frame_num][1]["bbox"]
        assigned_player = ball_assigner.assign_ball_to_player(player_track, ball_bbox)
        if assigned_player != -1:
            tracks["players"][frame_num][assigned_player]["has_ball"] = True
            team_ball_ctrl.append(tracks["players"][frame_num][assigned_player]["team"])
        else:
            team_ball_ctrl.append(team_ball_ctrl[-1] if team_ball_ctrl else 1)
    team_ball_ctrl = np.array(team_ball_ctrl)

    # ── Lazy import of foot-position helper ───────────────────────────────────
    from utils import get_foot_position

    update(0.85, "Drawing annotations & saving video (streaming — one frame at a time)…")

    # Streaming H.264 writer via imageio-ffmpeg (browser-compatible)
    import imageio_ffmpeg as ioff
    h_vid, w_vid = video_frames[0].shape[:2]
    vw = ioff.write_frames(
        OUTPUT_MP4,
        size=(w_vid, h_vid),
        fps=24,
        codec="libx264",
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        output_params=["-crf", "23", "-preset", "fast"],
    )
    vw.send(None)  # prime generator

    n_frames_total = len(video_frames)
    for frame_num in range(n_frames_total):
        frame = video_frames[frame_num].copy()
        video_frames[frame_num] = None      # free raw frame the moment we've copied it

        player_dict  = tracks["players"][frame_num]
        ball_dict    = tracks["ball"][frame_num]
        referee_dict = tracks["referees"][frame_num]

        # players
        for track_id, player in player_dict.items():
            color = player.get("team_color", (0, 0, 255))
            frame = tracker.draw_ellipse(frame, player["bbox"], color, track_id)
            if player.get("has_ball", False):
                frame = tracker.draw_traingle(frame, player["bbox"], (0, 0, 255))

        # referees
        for _, referee in referee_dict.items():
            frame = tracker.draw_ellipse(frame, referee["bbox"], (0, 255, 255))

        # ball
        for track_id, ball in ball_dict.items():
            frame = tracker.draw_traingle(frame, ball["bbox"], (0, 255, 0))

        # team ball control panel (sub-region blend — no full-frame copy)
        team_ball_ctrl_till = team_ball_ctrl[:frame_num + 1]
        t1f = int((team_ball_ctrl_till == 1).sum())
        t2f = int((team_ball_ctrl_till == 2).sum())
        tot = (t1f + t2f) or 1
        roi = frame[850:970, 1350:1900]
        white = np.ones_like(roi) * 255
        frame[850:970, 1350:1900] = cv2.addWeighted(white, 0.4, roi, 0.6, 0)
        cv2.putText(frame, f"Team 1 Ball Control: {t1f/tot*100:.2f}%",
                    (1400, 900), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Team 2 Ball Control: {t2f/tot*100:.2f}%",
                    (1400, 950), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        # camera movement panel (sub-region blend — no full-frame copy)
        roi2 = frame[0:100, 0:500]
        white2 = np.ones_like(roi2) * 255
        frame[0:100, 0:500] = cv2.addWeighted(white2, 0.6, roi2, 0.4, 0)
        cx, cy = cam_movement[frame_num]
        cv2.putText(frame, f"Camera Movement X: {cx:.2f}",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)
        cv2.putText(frame, f"Camera Movement Y: {cy:.2f}",
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 3)

        # speed / distance labels
        for obj, obj_tracks in tracks.items():
            if obj == "ball":
                continue
            for _, track in obj_tracks[frame_num].items():
                speed = track.get("speed")
                dist  = track.get("distance_covered")
                if speed is None or dist is None:
                    continue
                pos = list(get_foot_position(track["bbox"]))
                pos[1] += 40
                pos = tuple(map(int, pos))
                cv2.putText(frame, f"{speed:.2f}km/h", pos,
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                cv2.putText(frame, f"{dist:.2f}m", (pos[0], pos[1] + 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        vw.send(frame.tobytes())

        # Periodic GC to reclaim None slots
        if frame_num % 100 == 0:
            gc.collect()

    vw.close()
    del video_frames
    gc.collect()

    # Stash results so the UI section below can read them
    st.session_state["done"]           = True
    st.session_state["team_ball_ctrl"] = team_ball_ctrl
    st.session_state["team_colors"]    = team_assigner.team_colors
    st.session_state["tracks"]         = tracks
    st.session_state["n_frames"]       = n_frames

    progress.progress(1.0, text="✅ Analysis complete!")
    status.success("Done! Scroll down to see results.")

# ── Results ────────────────────────────────────────────────────────────────────
if st.session_state.get("done") and os.path.exists(OUTPUT_MP4):
    st.divider()
    st.header("📊 Results")

    team_ball_ctrl = st.session_state["team_ball_ctrl"]
    team_colors    = st.session_state["team_colors"]
    tracks         = st.session_state["tracks"]
    n_frames       = st.session_state["n_frames"]

    # ── KPI row ──────────────────────────────────────────────────────────────
    t1 = int((team_ball_ctrl == 1).sum())
    t2 = int((team_ball_ctrl == 2).sum())
    total = t1 + t2 or 1
    t1_pct = t1 / total * 100
    t2_pct = t2 / total * 100

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("🎬 Total Frames",    f"{n_frames:,}")
    col2.metric("⏱️ Duration (s)",    f"{n_frames / 24:.1f}")
    col3.metric("🔵 Team 1 Possession", f"{t1_pct:.1f}%")
    col4.metric("🔴 Team 2 Possession", f"{t2_pct:.1f}%")

    st.divider()

    # ── Possession bar chart ──────────────────────────────────────────────────
    left, right = st.columns([1, 2])

    with left:
        st.subheader("Ball Possession")
        hex1 = bgr_to_hex(team_colors.get(1, [0, 0, 255]))
        hex2 = bgr_to_hex(team_colors.get(2, [255, 0, 0]))
        # color list must match number of columns, not rows — one column per team
        poss_df = pd.DataFrame(
            {"Team 1": [t1_pct], "Team 2": [t2_pct]},
            index=["Possession %"],
        )
        st.bar_chart(poss_df, color=[hex1, hex2], height=250)

    # ── Player stats table ────────────────────────────────────────────────────
    with right:
        st.subheader("Player Stats")
        df = collect_player_stats(tracks)
        df["Team"] = df["Team"].apply(lambda t: f"Team {t}")
        df["Max Speed (km/h)"] = df["Max Speed (km/h)"].round(1)
        df["Distance (m)"]     = df["Distance (m)"].round(1)
        st.dataframe(df, use_container_width=True, height=260)

    st.divider()

    # ── Output video ─────────────────────────────────────────────────────────
    st.subheader("🎥 Annotated Output Video")
    with open(OUTPUT_MP4, "rb") as f:
        video_bytes = f.read()
    st.video(video_bytes)

    col_path, col_dl = st.columns([3, 1])
    col_path.caption(f"Saved to: `{OUTPUT_MP4}`")
    col_dl.download_button(
        label="⬇️ Download",
        data=video_bytes,
        file_name="football_analysis_output.mp4",
        mime="video/mp4",
        use_container_width=True,
    )

elif os.path.exists(OUTPUT_MP4) and not st.session_state.get("done"):
    # Previous run exists — show it without stats
    st.divider()
    st.subheader("🎥 Last Output Video")
    with open(OUTPUT_MP4, "rb") as f:
        video_bytes = f.read()
    st.video(video_bytes)

    col_path, col_dl = st.columns([3, 1])
    col_path.info("Hit **▶ Run Analysis** in the sidebar to generate fresh stats.")
    col_dl.download_button(
        label="⬇️ Download",
        data=video_bytes,
        file_name="football_analysis_output.mp4",
        mime="video/mp4",
        use_container_width=True,
    )
