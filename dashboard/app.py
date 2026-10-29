import json
import sys
from pathlib import Path

# Resolve the project root before importing project modules.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import requests
import streamlit as st

from src.database.review_db import get_review, save_review


# --------------------------------------------------
# Configuration
# --------------------------------------------------

ROOT = PROJECT_ROOT
REPORT_FILE = ROOT / "outputs" / "incidents" / "incident_report.json"
FRAME_DIR = ROOT / "outputs" / "incidents" / "frame_inspection"
DETECTION_DIR = ROOT / "outputs" / "detections"
API_URL = "http://127.0.0.1:8000"

STATUS_OPTIONS = ["Pending Review", "Confirmed", "False Positive"]
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov", ".mkv"}


# --------------------------------------------------
# Streamlit page
# --------------------------------------------------

st.set_page_config(
    page_title="RoadGuard-AI | Incident Dashboard",
    page_icon="🚦",
    layout="wide",
)

st.title("🚦 RoadGuard-AI")
st.caption("AI-assisted road incident detection and review")

st.sidebar.header("Dashboard Settings")
source = st.sidebar.radio("Data source", ["Incident report", "FastAPI"])
st.sidebar.markdown("---")
st.sidebar.caption("Experimental prototype")
st.sidebar.caption("Human review is required.")
st.sidebar.caption("Visual-risk scores are not accident probabilities.")


# --------------------------------------------------
# Load incidents
# --------------------------------------------------

def load_incidents():
    """Load incident data from the JSON report or FastAPI."""
    if source == "FastAPI":
        response = requests.get(f"{API_URL}/incidents", timeout=5)
        response.raise_for_status()
        data = response.json()
    else:
        if not REPORT_FILE.exists():
            raise FileNotFoundError(
                f"Incident report not found: {REPORT_FILE}\n"
                "Run scripts\\generate_incident_report.py first."
            )
        data = json.loads(REPORT_FILE.read_text(encoding="utf-8"))

    # Support either {"incidents": [...]} or a direct list response.
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        incidents = data.get("incidents", [])
        if isinstance(incidents, list):
            return incidents
    raise ValueError("Incident data must be a list or an object containing an 'incidents' list.")


try:
    incidents = load_incidents()
except requests.RequestException as exc:
    st.error(
        "Cannot connect to FastAPI. Start it with "
        "`uvicorn api.main:app --reload` or select Incident report."
    )
    st.caption(f"Connection details: {exc}")
    st.stop()
except (FileNotFoundError, json.JSONDecodeError, ValueError) as exc:
    st.error(str(exc))
    st.stop()


# --------------------------------------------------
# Review database helpers
# --------------------------------------------------

def load_review_safely(incident_id):
    """Load one review and display an error if the database is unavailable."""
    try:
        return get_review(incident_id)
    except Exception as exc:
        st.error(f"Could not load review for {incident_id}: {exc}")
        return {
            "incident_id": incident_id,
            "review_status": "Pending Review",
            "reviewer_notes": "",
            "updated_at": None,
        }


def get_assessment(incident):
    return incident.get("visual_risk_assessment", {}) or {}


# --------------------------------------------------
# Discover processed videos
# Prefer converted MP4 videos over AVI files.
# --------------------------------------------------

if DETECTION_DIR.exists():
    all_videos = [
        path
        for path in DETECTION_DIR.rglob("*")
        if path.is_file()
        and path.suffix.lower() in {".mp4", ".avi", ".mov", ".mkv"}
    ]

    mp4_videos = [
        path for path in all_videos
        if path.suffix.lower() == ".mp4"
    ]

    # Prefer MP4 files; fall back to other formats if necessary.
    video_files = sorted(
        mp4_videos if mp4_videos else all_videos,
        key=lambda path: (
            0 if "tracking" in str(path).lower() else 1,
            str(path).lower(),
        ),
    )
else:
    video_files = []

# --------------------------------------------------
# Calculate dashboard metrics
# --------------------------------------------------

risk_counts = {}
review_counts = {status: 0 for status in STATUS_OPTIONS}

for incident in incidents:
    assessment = get_assessment(incident)
    level = assessment.get("level", "UNKNOWN")
    risk_counts[level] = risk_counts.get(level, 0) + 1

    incident_id = incident.get("incident_id", "Unknown incident")
    review = load_review_safely(incident_id)
    status = review.get("review_status", "Pending Review")
    if status in review_counts:
        review_counts[status] += 1

col1, col2, col3, col4 = st.columns(4)
col1.metric("Candidate Incidents", len(incidents))
col2.metric("Pending Review", review_counts["Pending Review"])
col3.metric("Human-Confirmed", review_counts["Confirmed"])
col4.metric("False Positives", review_counts["False Positive"])


# --------------------------------------------------
# Visual-risk overview
# --------------------------------------------------

st.divider()
st.subheader("Visual-Risk Overview")

if risk_counts:
    chart_data = pd.DataFrame(
        [{"Risk Level": level, "Candidates": count}
         for level, count in risk_counts.items()]
    ).set_index("Risk Level")
    st.bar_chart(chart_data)
else:
    st.info("No candidate incidents are available.")


# --------------------------------------------------
# Incident filters
# --------------------------------------------------

st.divider()
st.subheader("Incident Candidates")
filter_col1, filter_col2 = st.columns(2)

with filter_col1:
    selected_risk = st.selectbox(
        "Filter by visual-risk level",
        ["All"] + sorted(risk_counts.keys()),
    )

with filter_col2:
    selected_review = st.selectbox(
        "Filter by review status",
        ["All"] + STATUS_OPTIONS,
    )


# --------------------------------------------------
# Render incident cards
# --------------------------------------------------

visible_incidents = 0

for incident in incidents:
    assessment = get_assessment(incident)
    level = assessment.get("level", "UNKNOWN")
    score = assessment.get("score", 0)
    frame = incident.get("frame")
    incident_id = incident.get("incident_id", "Unknown incident")

    review = load_review_safely(incident_id)
    current_status = review.get("review_status", "Pending Review")

    if selected_risk != "All" and level != selected_risk:
        continue
    if selected_review != "All" and current_status != selected_review:
        continue

    visible_incidents += 1
    frame_label = str(frame) if frame is not None else "Unknown"

    with st.expander(
        f"{incident_id} | Frame {frame_label} | {level} | {current_status}",
        expanded=(visible_incidents == 1),
    ):
        st.subheader("Incident Overview")
        left, right = st.columns(2)

        with left:
            st.write("**Incident ID:**", incident_id)
            st.write("**Source video:**", incident.get("source_video", "Unknown"))
            st.write("**Frame:**", frame_label)
            st.write("**Visual-risk level:**", level)
            st.write("**Visual-risk score:**", score)
            st.write(
                "**Model confidence:**",
                assessment.get("confidence", "unknown"),
            )

        with right:
            st.write("**Objects involved**")
            objects = incident.get("objects", [])
            if objects:
                st.dataframe(
                    pd.DataFrame(objects),
                    hide_index=True,
                    use_container_width=True,
                )
            else:
                st.info("No object details available.")

            st.write("**Collision evidence**")
            st.json(incident.get("collision_evidence", {}))

        st.subheader("Assessment Reasons")
        reasons = assessment.get("reasons", [])
        if reasons:
            for reason in reasons:
                st.write(f"- {reason}")
        else:
            st.write("No assessment reasons recorded.")

        with st.expander("Assessment Limitations"):
            limitations = assessment.get("limitations", [])
            if limitations:
                for limitation in limitations:
                    st.write(f"- {limitation}")
            else:
                st.write("No limitations recorded in the report.")

        # Visual evidence frame
        st.divider()
        st.subheader("Visual Evidence")
        frame_path = None

        if frame is not None:
            try:
                frame_number = int(frame)
                frame_path = FRAME_DIR / f"frame_{frame_number:04d}.jpg"
            except (TypeError, ValueError):
                frame_path = None

        if frame_path and frame_path.exists():
            st.image(
                str(frame_path),
                caption=f"Source video frame {frame_label}",
                use_container_width=True,
            )
        else:
            st.info(
                "No extracted image is available for this incident. "
                "Run scripts\\inspect_collision_frames.py if that script is configured."
            )

        # Annotated video: search recursively and include .avi files.
        st.divider()
        st.subheader("Annotated Detection Video")

        if video_files:
            tracking_videos = [
                path for path in video_files
                if "tracking" in str(path).lower()
            ]
            choices = tracking_videos if tracking_videos else video_files

            # Prefer a filename related to the incident source video when possible.
            source_video = str(incident.get("source_video", "")).lower()
            source_stem = Path(source_video).stem if source_video else ""
            matching = [
                path for path in choices
                if source_stem and source_stem in path.stem.lower()
            ]
            default_video = matching[0] if matching else choices[0]

            selected_video = st.selectbox(
                "Select an annotated video",
                options=choices,
                index=choices.index(default_video),
                format_func=lambda path: str(path.relative_to(ROOT)),
                key=f"video_select_{incident_id}",
            )
            st.caption(f"Video path: {selected_video.relative_to(ROOT)}")

            try:
                # Pass the file path so Streamlit can infer the media type.
                st.video(str(selected_video))
                if selected_video.suffix.lower() == ".avi":
                    st.caption(
                        "If this AVI does not play in your browser, convert it "
                        "to MP4 (H.264) and select the converted file."
                    )
            except Exception as exc:
                st.warning(f"Could not display this video: {exc}")
        else:
            st.warning(
                "No processed video was found under outputs/detections. "
                "Run python scripts\\run_tracking.py first."
            )

        # Human review form
        st.divider()
        st.subheader("Human Review")
        st.write("Review the available evidence before choosing a status.")

        with st.form(key=f"review_form_{incident_id}"):
            status_index = (
                STATUS_OPTIONS.index(current_status)
                if current_status in STATUS_OPTIONS else 0
            )
            selected_status = st.selectbox(
                "Review status",
                options=STATUS_OPTIONS,
                index=status_index,
            )
            reviewer_notes = st.text_area(
                "Reviewer notes",
                value=review.get("reviewer_notes", ""),
                placeholder="Explain the evidence supporting your decision...",
            )
            submitted = st.form_submit_button("Save Review")

            if submitted:
                try:
                    save_review(
                        incident_id=incident_id,
                        review_status=selected_status,
                        reviewer_notes=reviewer_notes,
                    )
                    st.success(f"Review saved: {selected_status}")
                    st.rerun()
                except (ValueError, OSError) as exc:
                    st.error(f"Could not save review: {exc}")
                except Exception as exc:
                    st.error(f"Unexpected database error: {exc}")

        st.caption(
            "Last updated: " + str(review.get("updated_at") or "Not reviewed yet")
        )
        st.caption(
            "A human review decision is stored separately from "
            "the model-generated visual-risk assessment."
        )

if visible_incidents == 0:
    st.info("No incidents match the selected filters.")


# --------------------------------------------------
# Footer
# --------------------------------------------------

st.divider()
st.caption(
    "RoadGuard-AI is an experimental computer-vision prototype. "
    "It does not independently confirm accidents, estimate injuries, "
    "or dispatch emergency services. Human review is required."
)
