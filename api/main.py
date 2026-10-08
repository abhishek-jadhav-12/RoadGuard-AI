from fastapi import FastAPI


app = FastAPI(
    title="RoadGuard AI API",
    description=(
        "AI-powered road accident detection "
        "and emergency intelligence API."
    ),
    version="0.1.0",
)


@app.get("/")
def root():
    return {
        "project": "RoadGuard AI",
        "message": "Road Accident & Emergency Intelligence System",
        "status": "running",
        "version": "0.1.0",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "roadguard-api",
        "version": "0.1.0",
    }