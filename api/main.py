
from pathlib import Path
import json

from fastapi import FastAPI, HTTPException

ROOT = Path(__file__).resolve().parents[1]
REPORT_FILE = ROOT / "outputs" / "incidents" / "incident_report.json"

app = FastAPI(
    title="RoadGuard-AI",
    description="AI-assisted road incident analysis API",
    version="0.1.0",
)


def load_report():
    """Load the latest generated incident report."""
    if not REPORT_FILE.exists():
        raise HTTPException(
            status_code=404,
            detail="Incident report not found. Generate a report first.",
        )

    try:
        return json.loads(REPORT_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not read the incident report.",
        ) from exc


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "service": "RoadGuard-AI",
    }


@app.get("/incidents")
def get_incidents():
    """Return all potential incident candidates."""
    report = load_report()

    return {
        "candidate_count": report.get("candidate_count", 0),
        "incidents": report.get("incidents", []),
        "report_type": report.get("report_type"),
    }


@app.get("/incidents/summary")
def get_incident_summary():
    """Return a count of incidents by visual-risk level."""
    report = load_report()
    incidents = report.get("incidents", [])

    risk_counts = {}

    for incident in incidents:
        assessment = incident.get("visual_risk_assessment", {})
        level = assessment.get("level", "UNKNOWN")
        risk_counts[level] = risk_counts.get(level, 0) + 1

    return {
        "candidate_count": len(incidents),
        "risk_breakdown": risk_counts,
        "human_review_required": sum(
            1 for incident in incidents
            if incident.get("human_review_required", True)
        ),
        "confirmed_accidents": sum(
            1 for incident in incidents
            if incident.get("confirmed_accident", False)
        ),
    }


@app.get("/incidents/{incident_id}")
def get_incident(incident_id: str):
    """Retrieve one incident using its incident ID."""
    report = load_report()

    for incident in report.get("incidents", []):
        if incident.get("incident_id") == incident_id:
            return incident

    raise HTTPException(
        status_code=404,
        detail=f"Incident '{incident_id}' not found.",
    )
