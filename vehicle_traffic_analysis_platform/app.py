"""
Vehicle Traffic Analysis Platform
---------------------------------
A Streamlit application for detecting and tracking vehicles in uploaded traffic videos using YOLOv8 and ByteTrack.

Run with:
    streamlit run app.py
"""

from __future__ import annotations

import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Tuple

import cv2
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from ultralytics import YOLO

from utils.detection_utils import (
    DEFAULT_VEHICLE_CLASS_IDS,
    ensure_project_folders,
    process_video,
)


BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / "uploads"
OUTPUT_DIR = BASE_DIR / "outputs"
MODEL_DIR = BASE_DIR / "models"
CUSTOM_TRACKER_PATH = BASE_DIR / "custom_bytetrack.yaml"

def get_tracker_config() -> str:
    """Use custom ByteTrack settings if available, otherwise use Ultralytics default."""
    return str(CUSTOM_TRACKER_PATH) if CUSTOM_TRACKER_PATH.exists() else "bytetrack.yaml"

@st.cache_resource(show_spinner=False)
def load_yolo_model(model_choice: str) -> YOLO:
    """Load a YOLO model once and reuse it during the Streamlit session."""
    local_model_path = MODEL_DIR / model_choice

    # If the model exists locally, use it. Otherwise, Ultralytics will download it.
    model_source = str(local_model_path) if local_model_path.exists() else model_choice
    return YOLO(model_source)


def save_uploaded_video(uploaded_file) -> Path:
    """Save uploaded video to the uploads folder and return the saved path."""
    suffix = Path(uploaded_file.name).suffix.lower() or ".mp4"
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = UPLOAD_DIR / f"traffic_video_{timestamp}{suffix}"

    with open(output_path, "wb") as file:
        file.write(uploaded_file.getbuffer())

    return output_path


def get_output_path(input_path: Path) -> Path:
    """Create an output path for the processed video."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return OUTPUT_DIR / f"processed_{input_path.stem}_{timestamp}.mp4"


def apply_custom_style() -> None:
    """Apply custom Streamlit page styling for a colorful platform look."""
    st.markdown(
        """
        <style>
        :root {
            color-scheme: dark;
        }
        .stApp {
            background: radial-gradient(circle at top left, rgba(30, 136, 229, 0.25), transparent 25%),
                        radial-gradient(circle at bottom right, rgba(16, 185, 129, 0.20), transparent 20%),
                        linear-gradient(135deg, #0b1120 0%, #111827 45%, #1e3a8a 100%);
            color: #f8fafc;
        }
        [data-testid="stSidebar"] {
            background: rgba(8, 17, 42, 0.95);
            border-right: 1px solid rgba(255, 255, 255, 0.08);
            color: #e2e8f0;
        }
        .css-1d391kg, .css-1kyxreq, .css-1v0mbdj {
            background: rgba(255, 255, 255, 0.04) !important;
            border-radius: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.08);
        }
        .app-card {
            background: rgba(15, 23, 42, 0.88);
            border: 1px solid rgba(255, 255, 255, 0.10);
            border-radius: 24px;
            padding: 1.5rem;
            box-shadow: 0 24px 60px rgba(0, 0, 0, 0.25);
            margin-bottom: 1.5rem;
        }
        .app-card h2, .app-card h3, .app-card h4 {
            color: #f8fafc;
            margin-top: 0;
        }
        .app-card .section-label {
            color: #93c5fd;
            font-size: 0.95rem;
            margin-bottom: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.12em;
        }
        .dashboard-metrics {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 1rem;
            margin-bottom: 1.5rem;
        }
        .metric-card {
            background: rgba(15, 23, 42, 0.95);
            border: 1px solid rgba(56, 189, 248, 0.18);
            border-radius: 1.5rem;
            padding: 1.25rem;
            box-shadow: 0 20px 40px rgba(8, 15, 34, 0.22);
        }
        .metric-card h4 {
            color: #cbd5e1;
            margin-bottom: 0.5rem;
        }
        .metric-card .metric-value {
            font-size: 2rem;
            font-weight: 700;
            color: #38bdf8;
        }
        .metric-card.small {
            border-top: 4px solid #38bdf8;
        }
        .metric-card.accent-blue {
            border-top-color: #38bdf8;
        }
        .metric-card.accent-teal {
            border-top-color: #22d3ee;
        }
        .metric-card.accent-purple {
            border-top-color: #c084fc;
        }
        .metric-card.accent-orange {
            border-top-color: #fb923c;
        }
        .settings-card {
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
        }
        .settings-card h3 {
            margin-bottom: 1rem;
        }
        .settings-card .subtext {
            color: #cbd5e1;
            margin-bottom: 1rem;
        }
        .app-top-bar {
            background: rgba(15, 23, 42, 0.65);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 24px;
            padding: 1.5rem;
            margin-bottom: 1.5rem;
            box-shadow: 0 18px 50px rgba(0, 0, 0, 0.20);
        }
        .app-top-bar h1 {
            margin-bottom: 0.25rem;
        }
        .app-top-bar p {
            color: #cbd5e1;
            margin-top: 0;
        }
        .app-summary {
            color: #cbd5e1;
        }
        .stButton>button {
            background: linear-gradient(90deg, #60a5fa, #7dd3fc);
            color: #0f172a;
            border: none;
            box-shadow: 0 8px 16px rgba(96, 165, 250, 0.24);
        }
        .stButton>button:hover {
            background: linear-gradient(90deg, #7dd3fc, #a5f3fc);
            color: #0f172a;
        }
        .stMetric > div {
            background: rgba(255, 255, 255, 0.06);
            border-radius: 1rem;
            border: 1px solid rgba(255, 255, 255, 0.10);
        }
        .sidebar-panel {
            background: rgba(15, 23, 42, 0.92);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 1rem;
            padding: 1rem;
            margin-bottom: 1rem;
        }
        .sidebar-section-title {
            color: #93c5fd;
            font-size: 0.9rem;
            margin-bottom: 0.75rem;
            text-transform: uppercase;
            letter-spacing: 0.08em;
        }
        .sidebar-list {
            color: #cbd5e1;
            margin-bottom: 0.8rem;
        }
        .sidebar-list li {
            margin-bottom: 0.4rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_settings() -> Tuple[str, float, float, int, bool, bool]:
    """Render processing settings in a dedicated tab and return selected options."""
    st.write(
        "Configure model selection, detection thresholds, and tracking settings here. "
        "These settings are used when processing traffic videos."
    )

    model_choice = st.selectbox(
        "YOLOv8 model",
        options=["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"],
        index=0,
        help="yolov8n.pt is fastest. yolov8s.pt and yolov8m.pt can be more accurate but slower.",
    )

    confidence = st.slider(
        "Confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.35,
        step=0.05,
        help="Lower values detect more objects but may produce more false detections.",
    )

    iou = st.slider(
        "IoU threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.45,
        step=0.05,
        help="Controls how overlapping boxes are filtered by non-maximum suppression.",
    )

    frame_skip = st.slider(
        "Process every Nth frame",
        min_value=1,
        max_value=5,
        value=1,
        step=1,
        help="Use 1 for best results. Higher values are faster but may miss detections.",
    )

    show_preview = st.checkbox(
        "Show live preview while processing",
        value=True,
        help="Turn this off if processing is slow.",
    )

    use_tracking = st.checkbox(
        "Enable ByteTrack vehicle tracking",
        value=True,
        help="Use ByteTrack to assign stable track IDs and compute tracking-grade analytics.",
    )

    #st.markdown("---")
    #st.info(
     #   "Vehicle classes used: bicycle, car, motorcycle, bus, and truck."
   # )

    return model_choice, confidence, iou, frame_skip, show_preview, use_tracking

def render_vehicle_type_pie_chart(class_counts: dict) -> None:
    """Display vehicle type summary as a simple pie chart."""
    chart_data = {
        vehicle_type: count
        for vehicle_type, count in class_counts.items()
        if count > 0
    }

    if not chart_data:
        st.info("No vehicle type data is available for this analysis.")
        return

    labels = list(chart_data.keys())
    values = list(chart_data.values())
    total = sum(values)
    
    def show_percentage(percent: float) -> str:
        """Hide very small percentage labels to prevent overlap."""
        return f"{percent:.1f}%" if percent >= 5 else ""

    fig, ax = plt.subplots(figsize=(3, 3))
    wedges, _, autotexts = ax.pie(
        values,
        labels=None,
        autopct=show_percentage,
        startangle=90,
        pctdistance=0.72,
        textprops={"fontsize": 8},
    )

    for autotext in autotexts:
        autotext.set_fontsize(8)
        autotext.set_weight("bold")

    legend_labels = [
        f"{label}: {value} ({(value / total) * 100:.1f}%)"
        for label, value in zip(labels, values)
    ]

    ax.legend(
        wedges,
        legend_labels,
        title="Vehicle types",
        loc="center left",
        bbox_to_anchor=(1.0, 0.5),
        fontsize=8,
        title_fontsize=9,
        frameon=False,
    )

    ax.axis("equal")
    ax.set_title("Vehicle Type Distribution", fontsize=11)
    fig.tight_layout()

    chart_col, _ = st.columns([1.2, 1])
    with chart_col:
        st.pyplot(fig, use_container_width=False)

    plt.close(fig)

    st.caption(
        "The pie chart shows the proportion of unique tracked vehicles by type."
    )

def main() -> None:
    ensure_project_folders(BASE_DIR)

    st.set_page_config(
        page_title="TrafficAnalyzer",
        #page_icon="🚗",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_style()

    if "analysis_history" not in st.session_state:
        st.session_state.analysis_history = []
    if "current_stats" not in st.session_state:
        st.session_state.current_stats = None
    if "last_output_path" not in st.session_state:
        st.session_state.last_output_path = ""
    if "last_uploaded_name" not in st.session_state:
        st.session_state.last_uploaded_name = ""

    with st.sidebar:
        st.markdown(
            "<div class='sidebar-header'>VehicleTrafficAnalyzer</div>",
            #"<div class='sidebar-subtitle'>Video Traffic Dashboard</div>",
            unsafe_allow_html=True,
        )
        nav = st.radio(
            "",
            ["Dashboard", "Analysis History", "About"],
            index=0,
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown("<div class='sidebar-panel'><div class='sidebar-section-title'>Navigation</div></div>", unsafe_allow_html=True)
        st.markdown(
            "<p class='sidebar-list'>Upload a vehicle traffic video and start analysis to detect vehicles and track them across frames.</p>"
            "<p class='sidebar-list'>Analyze traffic patterns and review historical data.</p>",
            unsafe_allow_html=True,
        )

    if nav == "Analysis History":
        st.markdown(
            "<div class='app-card'><h2>Analysis History</h2>"
            "<p class='card-subtitle'>View previous vehicle traffic analysis and stored metrics.</p></div>",
            unsafe_allow_html=True,
        )

        if not st.session_state.analysis_history:
            st.warning("No previous analysis runs have been recorded yet.")
            st.write("Upload a video and run analysis.")
            return

        history_df = pd.DataFrame(st.session_state.analysis_history)
        st.dataframe(history_df, use_container_width=True)
        return

    if nav == "About":
        st.markdown(
            "<div class='app-card'><h2>About VehicleTrafficAnalysis</h2>"
            "<p class='card-subtitle'>Obtain tracking-grade traffic analytics from uploaded videos.</p></div>",
            unsafe_allow_html=True,
        )
        st.write("Upload a video on the Dashboard tab, run analysis, and inspect results instantly.")
        return

   

    st.markdown(
        "<div class='app-top-bar'>"
        "<div style='display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap;'>"
        "<div><h1>Vehicle Traffic Analysis Platform</h1>"
        "<p>Upload a traffic video to analyze it, and view the results below.</p></div>"
        #f"<div class='topbar-meta'>{datetime.now().strftime('%b %d, %Y | %I:%M %p')}</div>"
        "</div>"
        "</div>",
        unsafe_allow_html=True,
    )
    
    current_stats = st.session_state.current_stats
    output_path = Path(st.session_state.last_output_path) if st.session_state.last_output_path else None
    left_col, right_col = st.columns([2.5, 1])

    with left_col:

        uploaded_file = st.file_uploader(
        "Upload Traffic Video",
        type=["mp4", "avi", "mov", "mkv"],
        help="Upload a traffic video to analyze.",
    )

        if uploaded_file is not None:
            st.session_state.last_uploaded_name = uploaded_file.name


        if uploaded_file is None and not current_stats:
            st.info("Upload a traffic video and configure settings to begin analysis.")
        if uploaded_file is not None:
            st.session_state.last_uploaded_name = uploaded_file.name
            st.success(f"Selected file: {uploaded_file.name}")
            st.caption("The video is ready. Configure the options below, then click Start Analysis.")
        #elif current_stats is None:
         #   st.info("Upload a traffic video and configure settings to begin analysis.")
            
        st.markdown(
            "<div class='app-card'><div class='section-label'>Analysis Settings</div><h3>Dashboard</h3></div>",
            unsafe_allow_html=True,
        )

    

        model_choice, confidence, iou, frame_skip, show_preview, use_tracking = render_settings()

        
        start_processing = st.button("Start Analysis", type="primary")

        if start_processing:
            if uploaded_file is None:
                st.warning("Please upload a video before starting analysis.")
            else:
                processing_mode = "YOLOv8 + ByteTrack" if use_tracking else "YOLOv8 detection only"
                with st.spinner("Processing uploaded video..."):
                    input_path = save_uploaded_video(uploaded_file)
                    output_path = get_output_path(input_path)

                    try:
                        model = load_yolo_model(model_choice)
                    except Exception as exc:
                        st.error("The YOLOv8 model could not be loaded.")
                        st.exception(exc)
                        return

                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    preview_area = st.empty()

                    def progress_callback(progress: float, message: str, preview_frame=None) -> None:
                        progress_bar.progress(min(max(progress, 0.0), 1.0))
                        status_text.markdown(message)
                        if show_preview and preview_frame is not None:
                            preview_rgb = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
                            preview_area.image(preview_rgb, caption="Processing preview", width=640)

                    try:
                        stats = process_video(
                            input_path=input_path,
                            output_path=output_path,
                            model=model,
                            vehicle_class_ids=DEFAULT_VEHICLE_CLASS_IDS,
                            confidence_threshold=confidence,
                            iou_threshold=iou,
                            frame_skip=frame_skip,
                            use_tracker=use_tracking,
                            progress_callback=progress_callback,
                        )
                    except Exception as exc:
                        st.error("An error occurred while processing the video.")
                        st.exception(exc)
                        return

                    progress_bar.progress(1.0)
                    status_text.success("Analysis completed successfully.")
                    st.session_state.current_stats = stats
                    st.session_state.last_output_path = str(output_path)

                     # Update local variables so results display immediately after processing.
                    current_stats = stats
                    output_path = output_path
                    
                    st.session_state.analysis_history.insert(0, {
                        "Timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        "Video": uploaded_file.name,
                        "Model": model_choice,
                        "Tracking": "ByteTrack" if use_tracking else "Disabled",
                        "Frames": stats["frames_processed"],
                        "Total Detections": stats["total_vehicle_detections"],
                        "Unique Vehicles": stats["unique_vehicle_detections"],
                        "Down Crossings": stats.get("direction_counts", {}).get("down", 0),
                        "Up Crossings": stats.get("direction_counts", {}).get("up", 0),
                        "Congestion": stats["congestion_level"],
                    })

    with right_col:
        st.markdown(
            "<div class='settings-card'><h3>Configuration Summary</h3><p class='subtext'>Current configuration and last analysis details.</p></div>",
            unsafe_allow_html=True,
        )
        #if current_stats is not None:
         #   st.metric("Frames Processed", current_stats["frames_processed"])
          #  st.metric("Tracked Vehicles", current_stats.get("tracked_vehicles", current_stats["unique_vehicle_detections"]))
        #else:
         
         #   st.metric("Status", "Ready")

        #st.markdown("---")
        st.write(f"**Last Uploaded:** {st.session_state.last_uploaded_name or 'None'}")
        st.write(f"**Model:** {model_choice}")
        st.write(f"**Tracking:** {'ByteTrack' if use_tracking else 'Disabled'}")
        st.write(f"**Confidence:** {confidence:.2f}")
        st.write(f"**IoU Threshold:** {iou:.2f}")
        st.write(f"**Frame Skip:** {frame_skip}")
        st.write(f"**Preview:** {'Enabled' if show_preview else 'Disabled'}")

    if current_stats is not None:
        st.markdown("<div class='app-card'><h3>Processed Output</h3></div>", unsafe_allow_html=True)
        if output_path and output_path.exists():
            st.video(str(output_path))
            with open(output_path, "rb") as file:
                st.download_button(label="Download Processed Video", data=file, file_name=output_path.name, mime="video/mp4")
        else:
            st.warning("Processed video is not available yet.")

        st.markdown("<div class='dashboard-metrics'>"
                    f"<div class='metric-card accent-blue'><h4>Total Detections</h4><div class='metric-value'>{current_stats['total_vehicle_detections']}</div></div>"
                    f"<div class='metric-card accent-teal'><h4>Unique Vehicles</h4><div class='metric-value'>{current_stats['unique_vehicle_detections']}</div></div>"
                    f"<div class='metric-card accent-purple'><h4>Down Crossings</h4><div class='metric-value'>{current_stats.get('direction_counts', {}).get('down', 0)}</div></div>"
                    f"<div class='metric-card accent-orange'><h4>Up Crossings</h4><div class='metric-value'>{current_stats.get('direction_counts', {}).get('up', 0)}</div></div>"
                    "</div>",
                    unsafe_allow_html=True)

        st.markdown("<div class='app-card'><div class='section-label'>Traffic Flow</div><h3>Line Crossing Summary</h3></div>", unsafe_allow_html=True)
        st.write(
            f"**Down crossings:** {current_stats.get('direction_counts', {}).get('down', 0)}  \n"
            f"**Up crossings:** {current_stats.get('direction_counts', {}).get('up', 0)}  \n"
            f"**Congestion level:** {current_stats['congestion_level']}  \n"
            f"**Avg vehicles/frame:** {current_stats['avg_vehicles_per_frame']:.2f}"
        )
        if current_stats.get("class_counts"):
            st.markdown("### Vehicle Type Summary")
            render_vehicle_type_pie_chart(current_stats["class_counts"])

if __name__ == "__main__":
    main()
