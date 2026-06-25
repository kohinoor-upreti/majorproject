Pothole Detection & Repair-Tracking Analytics Platform
An end-to-end computer vision platform designed to detect, track, and manage road damage. Driven by an attention-centric YOLOv12s deep learning model at its core, the platform allows field operations and municipal admins to upload data, analyze streams in real-time, and preserve an uncompressed history log for maintenance workflows.

 Key Features
 Vision Architecture (YOLOv12s)
-Attention-Centric Engine: Leverages area-attention mechanisms and FlashAttention optimization to capture structural pavement boundaries far more reliably than legacy CNNs.

-Multi-Modal Pipelines: Built-in decoupled runtime processing for three distinct detection entryways: static images, asynchronous sequential video files, and raw webcam hardware inputs.

 Scalable Flask Back-End
-Asynchronous Job Workers: Offloads heavy processing arrays into standalone back-end worker contexts to prevent standard web request-response blocking timeouts.

-Concurrency Guard: Uses optimized SQLite configuration structures (with multi-second busy-timeouts) to prevent concurrent write collisions (database is locked) during heavy video frame-batch insertions.

-Role-Based Security: Custom validation middleware ensuring general users can only track their individual tasks, while system administrators maintain sweeping global pipeline oversight.
Reactive Front-End Dashboard
-State-Guarded Views: Explicit React Router DOM authentication gates (<ProtectedRoute>) ensuring that operational visual pipelines remain strictly shielded behind secure credentials.

-Clean Component Layout: Modular styling with standard layout containers, integrated performance metrics, and a clean dark-mode presentation theme.
Technology Stack
-Front-End: React 18, React Router DOM, Web Vitals, Jest, Testing Library

-Back-End: Flask, Python 3, PyTorch, Ultralytics YOLOv12

-Database: SQLite3 (Persistent audit layers)
── backend/
- ── app.py                # Main Flask entrypoint & configuration settings
- ── models/               # YOLOv12 weights (.pt) and network graph configs
│   ── pothole_app.db        # Core SQLite database schema tracking state logs
  frontend/
    ├── public/
    └── src/
        ├── components/
        │   --- Home.js             # General Landing Dashboard View
        │   --- Navbar.js           # Navigation layout matrix
        │   --- Login.js            # User authorization portal
        │   --- Register.js         # New operative registration system
        │   --- Prediction.js       # Static Image submission point
        │   --- VideoPrediction.js  # Async Video worker control deck
        │   --- WebcamPrediction.js # Live local stream capture interceptor
        │   --- History.js          # Persistent audit database log viewer
        ├── config/
        │   --- auth.js             # Client local storage storage interceptors
        --- App.js                  # Frontend Router mapping & guard matrices
        --- App.css                 # Dark interface variables & global styling
        --- index.js                # Core React Client initialization hook
        --- index.css               # Global boilerplate CSS resets
  Installation & Setup
1. Back-End Configuration
Navigate into your back-end workspace directory, construct a clean virtual environment, and install your dependencies:
cd backend
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate

# Install core packages (Ensure Torch is compiled with your local CUDA architecture)
pip install flask python-dotenv ultralytics sqlite3
Initialize your target application database (pothole_app.db) and run the Flask runtime:
python app.py
Note: The Flask application defaults to hosting locally on [http://127.0.0.1:5000](http://127.0.0.1:5000).

2. Front-End Setup
Open a secondary terminal workspace, step into your front-end repository root, and clear down package layers:
cd frontend
npm install
To run the local developer web server:
npm start
The React bundle compiles and launches automatically in your local browser environment at http://localhost:3000.
🔒 API Specifications & Access Control Guardrails
The system separates standard operations (available to all logged-in workers) from high-level database monitoring (restricted to administrators). All protected endpoints require a valid token passed inside your network request headers (Authorization: Bearer <your_token>).

Public Authentication Routes
These endpoints are fully open to the public so users can safely manage their credentials.

User Registration (POST /register)

What it does: Registers a new user or field worker account.

Access level: Open to anyone.

User Login (POST /login)

What it does: Validates user credentials and issues a secure access token.

Access level: Open to anyone.

Core Detection Pipelines
These endpoints process road imagery. They run synchronously, processing your files in real-time before returning the final image or video back to your screen.

Instant Image Detection (POST /predict)

What it does: Accepts an image file, runs it through the YOLOv12s model, saves a copy to your history database, and returns the newly annotated image.

Access level: Any logged-in user or admin.

Instant Video Detection (POST /predict_video)

What it does: Accepts a short video file, runs frame-by-frame inference, streams the processed output back to the UI, and logs individual detections.

Access level: Any logged-in user or admin.

Asynchronous Background Job Engine
For larger files that take time to process, these endpoints create a background worker thread. They return a unique job_id right away so your application doesn't freeze or time out while waiting.

Queue Image Task (POST /predict_image_job)

What it does: Starts an isolated background worker thread to process an image file. Returns a job_id.

Access level: Any logged-in user or admin.

Queue Video Task (POST /predict_video_job)

What it does: Starts an isolated background worker thread to process a video file frame-by-frame. Returns a job_id.

Access level: Any logged-in user or admin.

Check Task Progress (GET /progress/<job_id>)

What it does: Returns the current status (queued, processing, completed, or failed) and percentage completion of a background task.

Access Guardrail: Standard users can only check the progress of their own jobs. Administrators can view any active task id.

Download Completed Task File (GET /job_result/<job_id>)

What it does: Serves the raw, annotated final image or video file once a background job hits 100%. Safely clears out temporary system storage after delivery.

Access Guardrail: Standard users can only fetch results from their own jobs. Administrators can download any completed file.
Running Unit Tests
To verify component rendering lifecycles alongside the Jest testing suite hooks:
cd frontend
npm test
  
