# RoadGuard-AI

**AI-Powered Road Accident Detection & Emergency Intelligence System**

RoadGuard-AI is a portfolio project exploring how computer vision,
object tracking, temporal deep learning, and incident intelligence can
be combined to analyze road-scene videos. The current implementation
establishes the dataset-inspection workflow and a working YOLO-based
object-detection and ByteTrack tracking prototype.

> **Current status:** Dataset inspection and initial EDA are complete.
> YOLO object detection, detection-statistics export, and a ByteTrack
> tracking prototype have been run successfully on a sample CCD video.
> Accident classification, severity estimation, RAG assistance, and
> real-time alerting are planned work---not yet implemented as complete
> features.

## Project goals

-   Detect road users in video frames.
-   Track detected road users across consecutive frames.
-   Extract object trajectories and motion features.
-   Build a temporal accident-detection model.
-   Estimate incident severity as an experimental model output.
-   Capture incident evidence and generate structured reports.
-   Provide an API and dashboard for reviewing incidents.
-   Evaluate performance, document limitations, and package the
    application for reproducible use.

## Current implementation status

  -----------------------------------------------------------------------
  Component               Status                  Notes
  ----------------------- ----------------------- -----------------------
  Python project          Complete                Python 3.12 virtual
  environment                                     environment;
                                                  dependencies installed
                                                  as needed

  Project structure and   Complete                Source modules,
  configuration                                   scripts, configs,
                                                  tests, and
                                                  documentation folders
                                                  created

  FastAPI foundation      Complete                Root and health
                                                  endpoints are available

  Streamlit dashboard     Complete                Initial status
  foundation                                      dashboard scaffold

  CCD dataset inspection  Complete                Verified 1,500 crash
                                                  videos and 3,000 normal
                                                  videos in the expected
                                                  folders

  Sample video validation Complete                Sample checked at 50
                                                  frames, 10 FPS, 1280 ×
                                                  720, approximately 5
                                                  seconds

  CCD annotation parsing  Complete                Metadata-generation
  and metadata CSV                                script and EDA workflow
                                                  prepared

  Initial dataset EDA     Complete                Notebook created and
                                                  run; metadata parsing
                                                  was corrected during
                                                  setup

  YOLO11n object          Working prototype       Processes a sample
  detection                                       video and saves
                                                  annotated output

  Detection statistics    Working prototype       Exports per-frame CSV
                                                  and JSON summary

  ByteTrack multi-object  Working prototype       Tracks objects and
  tracking                                        exports road-object
                                                  detections with frame,
                                                  ID, class, confidence,
                                                  and bounding-box
                                                  coordinates

  Trajectory and motion   Planned                 Next development task
  feature extraction                              

  CNN + LSTM accident     Planned                 Not yet trained or
  classification                                  evaluated

  Collision reasoning and Planned                 Requires motion
  severity estimation                             features and validation

  OCR and incident report Planned                 Not yet integrated
  generation                                      

  RAG emergency-response  Planned                 Not yet integrated
  assistant                                       

  Incident database and   Planned                 Database
  evidence management                             schema/integration work
                                                  remains

  Complete FastAPI +      In progress             Foundations exist; AI
  Streamlit workflow                              pipeline integration
                                                  remains

  ONNX optimization,      Planned                 Validate after the
  Docker, CI/CD hardening                         pipeline is integrated
  -----------------------------------------------------------------------

## Architecture roadmap

``` text
Road video / webcam / future RTSP source
                  |
                  v
          OpenCV video input
                  |
                  v
           YOLO object detection
                  |
                  v
          ByteTrack object tracking
                  |
                  v
       Trajectories and motion features
                  |
                  v
       Temporal accident classification
                  |
                  v
   Collision reasoning and severity estimate
                  |
                  v
       Evidence and incident metadata
                  |
          +-------+--------+
          |                |
          v                v
    NLP incident report   RAG assistant
          |                |
          +-------+--------+
                  |
                  v
        Database / FastAPI / Dashboard
                  |
                  v
       Simulated alert and incident review
```

The diagram represents the intended architecture. Only the components
explicitly marked as implemented in the status table should be
considered currently available.

## Technology stack

-   **Language:** Python 3.12
-   **Computer vision:** OpenCV
-   **Object detection:** Ultralytics YOLO11n pretrained model
-   **Object tracking:** ByteTrack through Ultralytics tracking
-   **Data processing:** Pandas, NumPy
-   **Visualization / dashboard:** Matplotlib, Streamlit, Plotly
-   **API:** FastAPI, Uvicorn, Pydantic
-   **Database:** SQLAlchemy (integration work pending)
-   **Deep learning:** PyTorch, torchvision (temporal model planned)
-   **NLP / RAG / OCR:** Transformers, Sentence Transformers, ChromaDB,
    EasyOCR (planned integration)
-   **Testing / quality:** pytest, Black, Ruff
-   **Packaging / automation:** Docker and GitHub Actions (initial CI
    workflow scaffolded)

## Dataset

The initial dataset is the **Car Crash Dataset (CCD)** from the
[official repository](https://github.com/Cogito2012/CarCrashDataset).

The local setup currently expects this layout:

``` text
data/
├── raw/
│   └── CCD/
│       ├── videos/
│       │   ├── Normal/
│       │   └── Crash-1500/
│       └── Crash-1500.txt
├── processed/
│   └── ccd_metadata.csv
├── annotations/
└── samples/
```

Dataset notes:

-   The dataset is used for accident-video research and experimentation.
-   The metadata CSV currently describes the crash-video annotations; it
    does **not** by itself represent every normal video.
-   Keep downloaded datasets out of Git. The repository `.gitignore`
    should exclude raw/processed dataset files and generated model
    weights.
-   Before training, create train/validation/test splits **by video**,
    never by randomly distributing frames from the same video across
    splits.
-   Review the official dataset license/terms and attribution
    requirements before redistribution or publication.

## Repository structure

``` text
RoadGuard-AI/
├── .github/workflows/ci.yml
├── api/
├── configs/
├── dashboard/
├── data/
├── docs/
├── models/
├── notebooks/
├── outputs/
├── scripts/
├── src/
│   ├── accident/
│   ├── database/
│   ├── detection/
│   ├── nlp/
│   ├── ocr/
│   ├── pipeline/
│   ├── tracking/
│   ├── utils/
│   └── vision/
├── tests/
├── .env.example
├── .gitignore
├── Dockerfile
├── LICENSE
├── README.md
├── requirements.txt
└── pyproject.toml
```

## Setup

These instructions assume Windows PowerShell and that the repository is
located at `D:\Nooral\RoadGuard-AI`. Adjust the path for your machine.

``` powershell
cd D:\Nooral\RoadGuard-AI

py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
pip install -r requirements.txt

# Notebook support, if needed
python -m pip install ipykernel jupyter
```

If the environment already exists and works, do not recreate it
unnecessarily.

## Run the application foundations

### FastAPI

``` powershell
uvicorn api.main:app --reload
```

-   API root: `http://127.0.0.1:8000/`
-   Health check: `http://127.0.0.1:8000/health`
-   Interactive API docs: `http://127.0.0.1:8000/docs`

### Streamlit dashboard

``` powershell
streamlit run dashboard/app.py
```

## Run the current computer-vision pipeline

Activate the project virtual environment first.

### 1. Inspect CCD folders and video counts

``` powershell
python scripts\inspect_ccd.py
```

### 2. Inspect a sample video

``` powershell
python scripts\check_video.py
```

### 3. Build CCD annotation metadata

``` powershell
python scripts\build_ccd_metadata.py
```

Expected output:

``` text
data/processed/ccd_metadata.csv
```

### 4. Run the EDA notebook

Open `notebooks/01_dataset_analysis.ipynb` in VS Code and select the
project `.venv` kernel.

### 5. Run YOLO detection

``` powershell
python scripts\run_detection.py
```

The script uses the pretrained `yolo11n.pt` model and saves annotated
video output under:

``` text
outputs/detections/ccd_sample/
```

The first run may download the model weights.

### 6. Evaluate detection statistics

``` powershell
python scripts\evaluate_detection.py
```

Outputs include:

``` text
outputs/detections/evaluation/
├── per_frame_detections.csv
├── detection_summary.json
└── annotated_video/
```

These are inference statistics, not ground-truth detection accuracy or
mAP.

### 7. Run ByteTrack

``` powershell
python scripts\run_tracking.py
```

Outputs include:

``` text
outputs/detections/tracking/
├── tracking_results.csv
├── tracking_summary.json
└── tracked_video/
```

The CSV stores frame numbers, track IDs where available, class labels,
confidence scores, bounding boxes, and box centers. Track IDs can be
lost or switched; these initial results are not a validated
tracking-accuracy benchmark.

## Next development milestones

1.  **Trajectory and motion features:** compute per-track center
    movement, speed proxies, direction changes, and track continuity.
2.  **Collision reasoning:** explore trajectory intersections, relative
    motion, bounding-box overlap, and deceleration cues.
3.  **Temporal accident model:** prepare video-level splits and
    train/evaluate a lightweight CNN + LSTM baseline.
4.  **Evaluation:** report precision, recall, F1, PR-AUC/ROC-AUC where
    appropriate, and false-positive examples.
5.  **Incident workflow:** combine model output, evidence frames,
    incident metadata, and an experimental severity estimate.
6.  **NLP / OCR / RAG:** add grounded incident summaries and a
    retrieval-based emergency information assistant.
7.  **Application integration:** connect the pipeline to the database,
    FastAPI, and Streamlit.
8.  **Production readiness:** tests, logs, Docker, ONNX benchmarks, CI,
    and deployment documentation.

## Evaluation principles

-   Split datasets at the **video level** to avoid frame leakage.
-   Do not treat repeated detections across frames as unique vehicles.
-   Do not equate model confidence with prediction correctness.
-   Evaluate on held-out videos and inspect false positives and false
    negatives.
-   Report limitations and benchmark settings alongside metrics.
-   Test across varied lighting, weather, camera angles, and road
    conditions before making reliability claims.

## Safety, privacy, and limitations

RoadGuard-AI is an educational and research prototype. It is **not a
certified road-safety or emergency-dispatch system**. Model predictions
may be wrong, and the current object-detection and tracking pipeline
does not itself establish that an accident occurred.

Any future severity estimate or emergency recommendation must be clearly
presented as experimental and reviewed by a human. The project should
use simulated alerts unless a separately validated, authorized
integration is developed. Handle video evidence and any identifiable
information lawfully, securely, and with appropriate privacy safeguards.

## Contributing

Issues and pull requests are welcome. Keep datasets, secrets, local
databases, model weights, and generated outputs out of Git unless there
is a deliberate, documented reason to include a small sample artifact.

## Author

**Abhishek Jadhav**

-   GitHub: [abhishek-jadhav-12](https://github.com/abhishek-jadhav-12)
-   LinkedIn: add your preferred public profile URL here

------------------------------------------------------------------------

*Project status reflects the current development checkpoint and should
be updated as features are implemented and validated.*
