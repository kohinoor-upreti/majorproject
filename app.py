from flask import Flask, request, jsonify, send_file, after_this_request, g
from flask_cors import CORS
import io
from PIL import Image
import cv2
import os
import tempfile
import shutil
import threading
import uuid
import time
import sqlite3
import json
import mimetypes
from functools import wraps
from ultralytics import YOLO
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)
CORS(
    app,
    expose_headers=[
        'X-Detection-Count',
        'X-Confidence-Threshold',
        'X-Progress-Percent',
        'X-Processed-Frames',
    ],
)
app.config['SECRET_KEY'] = os.environ.get('APP_SECRET_KEY', 'change-this-secret-key')


# Load YOLOv12s model
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'trained_yolov12s_pothole_detector.pt')
model = YOLO(MODEL_PATH)
job_store = {}
job_lock = threading.Lock()
DB_PATH = os.path.join(BASE_DIR, 'pothole_app.db')
MEDIA_DIR = os.path.join(BASE_DIR, 'history_media')
TOKEN_SALT = 'pothole-auth-token'
TOKEN_MAX_AGE_SECONDS = 60 * 60 * 24
token_serializer = URLSafeTimedSerializer(app.config['SECRET_KEY'])
webcam_snapshot_lock = threading.Lock()
last_webcam_snapshot_ts = {}
WEBCAM_SNAPSHOT_COOLDOWN_SECONDS = 8


def parse_conf_threshold(raw_value, default=0.5):
    try:
        value = float(raw_value)
        return max(0.05, min(0.95, value))
    except (TypeError, ValueError):
        return default


def get_db_connection():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    os.makedirs(MEDIA_DIR, exist_ok=True)
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('admin', 'user')),
            created_at REAL NOT NULL
        )
        '''
    )
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS audit_logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            username TEXT,
            role TEXT,
            action TEXT NOT NULL,
            endpoint TEXT,
            file_name TEXT,
            status TEXT,
            detections_count INTEGER,
            confidence REAL,
            message TEXT,
            created_at REAL NOT NULL
        )
        '''
    )
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            file_name TEXT,
            source_type TEXT NOT NULL,
            job_id TEXT,
            frame_index INTEGER,
            class_id INTEGER,
            bbox TEXT NOT NULL,
            confidence REAL NOT NULL,
            created_at REAL NOT NULL
        )
        '''
    )
    cur.execute(
        '''
        CREATE TABLE IF NOT EXISTS detection_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            username TEXT NOT NULL,
            detection_type TEXT NOT NULL,
            source_original TEXT,
            result_detected TEXT,
            confidence REAL,
            created_at REAL NOT NULL
        )
        '''
    )
    conn.commit()
    cur.execute('SELECT id FROM users WHERE username = ?', ('admin',))
    admin = cur.fetchone()
    if not admin:
        cur.execute(
            'INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)',
            ('admin', generate_password_hash('admin'), 'admin', time.time()),
        )
        conn.commit()
        print('Created default admin account: username=admin password=admin')
    conn.close()


def create_access_token(user):
    payload = {'uid': user['id'], 'username': user['username'], 'role': user['role']}
    return token_serializer.dumps(payload, salt=TOKEN_SALT)


def verify_access_token(token):
    payload = token_serializer.loads(token, salt=TOKEN_SALT, max_age=TOKEN_MAX_AGE_SECONDS)
    return payload


def fetch_user_by_id(user_id):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, username, role FROM users WHERE id = ?', (user_id,))
    row = cur.fetchone()
    conn.close()
    return dict(row) if row else None


def create_audit_log(user, action, endpoint, status, file_name=None, detections_count=None, confidence=None, message=None):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        '''
        INSERT INTO audit_logs (user_id, username, role, action, endpoint, file_name, status, detections_count, confidence, message, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        (
            user.get('id') if user else None,
            user.get('username') if user else None,
            user.get('role') if user else None,
            action,
            endpoint,
            file_name,
            status,
            detections_count,
            confidence,
            message,
            time.time(),
        ),
    )
    conn.commit()
    conn.close()


def save_detections(user_id, file_name, detections, source_type, job_id=None, frame_index=None):
    if not detections or not user_id:
        return 0
    now_ts = time.time()
    rows = []
    for det in detections:
        bbox = det.get('bbox') or []
        rows.append(
            (
                int(user_id),
                file_name,
                source_type,
                job_id,
                frame_index,
                int(det.get('class', 0)),
                json.dumps(bbox),
                float(det.get('confidence', 0.0)),
                now_ts,
            )
        )
    conn = get_db_connection()
    cur = conn.cursor()
    cur.executemany(
        '''
        INSERT INTO detections (user_id, file_name, source_type, job_id, frame_index, class_id, bbox, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        rows,
    )
    conn.commit()
    conn.close()
    return len(rows)


def save_detection_rows(rows):
    if not rows:
        return 0
    conn = get_db_connection()
    cur = conn.cursor()
    cur.executemany(
        '''
        INSERT INTO detections (user_id, file_name, source_type, job_id, frame_index, class_id, bbox, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''',
        rows,
    )
    conn.commit()
    conn.close()
    return len(rows)


def _safe_ext(file_name, default_ext):
    _, ext = os.path.splitext(file_name or '')
    if not ext:
        return default_ext
    return ext.lower()


def save_bytes_to_media(file_name, file_bytes, default_ext, prefix):
    ext = _safe_ext(file_name, default_ext)
    stored_name = f'{prefix}_{uuid.uuid4().hex}{ext}'
    full_path = os.path.join(MEDIA_DIR, stored_name)
    with open(full_path, 'wb') as f:
        f.write(file_bytes)
    return stored_name


def copy_file_to_media(src_path, file_name, default_ext, prefix):
    ext = _safe_ext(file_name, default_ext)
    stored_name = f'{prefix}_{uuid.uuid4().hex}{ext}'
    full_path = os.path.join(MEDIA_DIR, stored_name)
    shutil.copy2(src_path, full_path)
    return stored_name


def create_detection_history(user_id, username, detection_type, source_original, result_detected, confidence):
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        '''
        INSERT INTO detection_history (user_id, username, detection_type, source_original, result_detected, confidence, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''',
        (int(user_id), username, detection_type, source_original, result_detected, confidence, time.time()),
    )
    conn.commit()
    conn.close()


def should_store_webcam_snapshot(user_id, detections_count):
    if not user_id or detections_count <= 0:
        return False
    now = time.time()
    with webcam_snapshot_lock:
        last_ts = last_webcam_snapshot_ts.get(user_id, 0.0)
        if now - last_ts < WEBCAM_SNAPSHOT_COOLDOWN_SECONDS:
            return False
        last_webcam_snapshot_ts[user_id] = now
        return True


def user_can_access_media(user, file_name):
    conn = get_db_connection()
    cur = conn.cursor()
    if user['role'] == 'admin':
        cur.execute(
            '''
            SELECT 1 FROM detection_history
            WHERE source_original = ? OR result_detected = ?
            LIMIT 1
            ''',
            (file_name, file_name),
        )
    else:
        cur.execute(
            '''
            SELECT 1 FROM detection_history
            WHERE user_id = ? AND (source_original = ? OR result_detected = ?)
            LIMIT 1
            ''',
            (user['id'], file_name, file_name),
        )
    row = cur.fetchone()
    conn.close()
    return row is not None


def auth_required(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        auth_header = request.headers.get('Authorization', '')
        if not auth_header.startswith('Bearer '):
            return jsonify({'error': 'Missing bearer token'}), 401
        token = auth_header.split(' ', 1)[1].strip()
        try:
            payload = verify_access_token(token)
        except SignatureExpired:
            return jsonify({'error': 'Token expired'}), 401
        except BadSignature:
            return jsonify({'error': 'Invalid token'}), 401
        user = fetch_user_by_id(payload.get('uid'))
        if not user:
            return jsonify({'error': 'User not found'}), 401
        g.current_user = user
        return fn(*args, **kwargs)

    return wrapper


def admin_required(fn):
    @wraps(fn)
    @auth_required
    def wrapper(*args, **kwargs):
        user = g.current_user
        if user.get('role') != 'admin':
            return jsonify({'error': 'Admin role required'}), 403
        return fn(*args, **kwargs)

    return wrapper


def create_job(kind, **extra):
    job_id = str(uuid.uuid4())
    with job_lock:
        job_store[job_id] = {
            'id': job_id,
            'kind': kind,
            'status': 'queued',
            'progress': 0,
            'error': None,
            'result_type': None,
            'result_bytes': None,
            'result_path': None,
            'cleanup_path': None,
            'created_at': time.time(),
            'updated_at': time.time(),
            **extra,
        }
    return job_id


def update_job(job_id, **kwargs):
    with job_lock:
        job = job_store.get(job_id)
        if not job:
            return
        for key, value in kwargs.items():
            job[key] = value
        job['updated_at'] = time.time()


def get_job(job_id):
    with job_lock:
        job = job_store.get(job_id)
        return dict(job) if job else None


def process_image_job(job_id, image_bytes, conf_threshold):
    job = get_job(job_id) or {}
    audit_user = {'id': job.get('owner_user_id'), 'username': job.get('owner_username'), 'role': job.get('owner_role')}
    temp_dir = tempfile.mkdtemp(prefix='pothole_image_')
    input_path = os.path.join(temp_dir, 'input.jpg')
    try:
        update_job(job_id, status='processing', progress=10)
        with open(input_path, 'wb') as f:
            f.write(image_bytes)
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

        update_job(job_id, progress=50)
        results = model.predict(source=image, conf=conf_threshold, verbose=False)
        if not results:
            raise RuntimeError('Prediction returned no results')

        annotated_bgr = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_rgb)
        image_io = io.BytesIO()
        annotated_pil.save(image_io, format='JPEG', quality=95)
        image_bytes_out = image_io.getvalue()
        detections = parse_detections(results[0])
        detections_count = len(detections)
        top_conf = max((float(d.get('confidence', 0.0)) for d in detections), default=0.0)
        source_media = save_bytes_to_media(
            file_name=job.get('file_name') or 'source.jpg',
            file_bytes=image_bytes,
            default_ext='.jpg',
            prefix='src',
        )
        result_media = save_bytes_to_media(
            file_name='detected.jpg',
            file_bytes=image_bytes_out,
            default_ext='.jpg',
            prefix='res',
        )
        create_detection_history(
            user_id=job.get('owner_user_id'),
            username=job.get('owner_username') or '',
            detection_type='image',
            source_original=source_media,
            result_detected=result_media,
            confidence=top_conf,
        )
        save_detections(
            user_id=job.get('owner_user_id'),
            file_name=job.get('file_name'),
            detections=detections,
            source_type='image_job',
            job_id=job_id,
        )

        update_job(
            job_id,
            status='completed',
            progress=100,
            result_type='image/jpeg',
            result_bytes=image_bytes_out,
        )
        create_audit_log(
            user=audit_user,
            action='image_job_completed',
            endpoint='/predict_image_job',
            status='completed',
            file_name=job.get('file_name'),
            detections_count=detections_count,
            confidence=conf_threshold,
        )
    except Exception as e:
        update_job(job_id, status='failed', progress=100, error=str(e))
        create_audit_log(
            user=audit_user,
            action='image_job_failed',
            endpoint='/predict_image_job',
            status='failed',
            file_name=job.get('file_name'),
            confidence=conf_threshold,
            message=str(e),
        )
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def process_video_job(job_id, temp_dir, input_path, conf_threshold):
    job = get_job(job_id) or {}
    audit_user = {'id': job.get('owner_user_id'), 'username': job.get('owner_username'), 'role': job.get('owner_role')}
    temp_output_path = os.path.join(temp_dir, 'result.mp4')
    out = None
    try:
        update_job(job_id, status='processing', progress=5)
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError('Could not open uploaded video')

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        if not input_fps or input_fps <= 0:
            input_fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()

        vid_stride = 2
        output_fps = max(1.0, input_fps / vid_stride)
        expected_frames = max(1, total_frames // vid_stride) if total_frames > 0 else 1
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, output_fps, (width, height))
        if not out.isOpened():
            raise RuntimeError('Could not create output video file')

        results_stream = model.predict(
            source=input_path,
            conf=conf_threshold,
            imgsz=640,
            stream=True,
            vid_stride=vid_stride,
            verbose=False,
        )
        processed = 0
        detection_rows = []
        for result in results_stream:
            frame = result.plot()
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
            frame_detections = parse_detections(result)
            if frame_detections:
                ts = time.time()
                for det in frame_detections:
                    detection_rows.append(
                        (
                            int(job.get('owner_user_id') or 0),
                            job.get('file_name'),
                            'video_job',
                            job_id,
                            processed,
                            int(det.get('class', 0)),
                            json.dumps(det.get('bbox') or []),
                            float(det.get('confidence', 0.0)),
                            ts,
                        )
                    )
            processed += 1
            progress = min(99, int((processed / expected_frames) * 100))
            update_job(job_id, progress=progress)

        out.release()
        out = None

        if detection_rows:
            save_detection_rows(detection_rows)

        if processed == 0:
            raise RuntimeError('No frames were processed from video')
        if not os.path.exists(temp_output_path):
            raise RuntimeError('Output video was not generated')

        source_media = copy_file_to_media(
            src_path=input_path,
            file_name=job.get('file_name') or 'source.mp4',
            default_ext='.mp4',
            prefix='src',
        )
        result_media = copy_file_to_media(
            src_path=temp_output_path,
            file_name='detected.mp4',
            default_ext='.mp4',
            prefix='res',
        )
        top_video_conf = max((float(r[7]) for r in detection_rows), default=0.0) if detection_rows else 0.0
        create_detection_history(
            user_id=job.get('owner_user_id'),
            username=job.get('owner_username') or '',
            detection_type='video',
            source_original=source_media,
            result_detected=result_media,
            confidence=top_video_conf,
        )

        update_job(
            job_id,
            status='completed',
            progress=100,
            result_type='video/mp4',
            result_path=temp_output_path,
            cleanup_path=temp_dir,
        )
        create_audit_log(
            user=audit_user,
            action='video_job_completed',
            endpoint='/predict_video_job',
            status='completed',
            file_name=job.get('file_name'),
            confidence=conf_threshold,
            message=f'Processed frames: {processed}',
        )
    except Exception as e:
        if out is not None:
            out.release()
        shutil.rmtree(temp_dir, ignore_errors=True)
        update_job(job_id, status='failed', progress=100, error=str(e))
        create_audit_log(
            user=audit_user,
            action='video_job_failed',
            endpoint='/predict_video_job',
            status='failed',
            file_name=job.get('file_name'),
            confidence=conf_threshold,
            message=str(e),
        )


def parse_detections(result):
    detections = []
    boxes = getattr(result, 'boxes', None)
    if boxes is None or boxes.xyxy is None:
        return detections

    xyxy = boxes.xyxy.cpu().numpy()
    conf = boxes.conf.cpu().numpy()
    cls = boxes.cls.cpu().numpy()

    for i in range(len(xyxy)):
        x1, y1, x2, y2 = xyxy[i]
        detections.append(
            {
                'bbox': [int(x1), int(y1), int(x2), int(y2)],
                'confidence': float(conf[i]),
                'class': int(cls[i]),
            }
        )
    return detections


@app.route('/predict', methods=['POST'])
@auth_required
def predict():
    user = g.current_user
    print('Received /predict POST request')
    if 'image' not in request.files:
        print('No image part in request.files:', request.files)
        return jsonify({'error': 'No image uploaded'}), 400
    image_file = request.files['image']
    print('Image file received:', image_file.filename)
    try:
        image_bytes = image_file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception as e:
        print('Error reading image:', e)
        return jsonify({'error': 'Invalid image file'}), 400
    try:
        conf_threshold = parse_conf_threshold(request.form.get('conf'), default=0.5)
        create_audit_log(
            user=user,
            action='predict_image_upload',
            endpoint='/predict',
            status='processing',
            file_name=image_file.filename,
            confidence=conf_threshold,
        )
        # Ultralytics YOLO accepts PIL images directly.
        results = model.predict(source=image, conf=conf_threshold, verbose=False)
        if not results:
            return jsonify({'error': 'Prediction returned no results'}), 500

        # Render image with bounding boxes + labels.
        annotated_bgr = results[0].plot()
        annotated_rgb = cv2.cvtColor(annotated_bgr, cv2.COLOR_BGR2RGB)
        annotated_pil = Image.fromarray(annotated_rgb)

        image_io = io.BytesIO()
        annotated_pil.save(image_io, format='JPEG', quality=95)
        image_io.seek(0)

        detections = parse_detections(results[0])
        print('Detections:', detections)
        top_conf = max((float(d.get('confidence', 0.0)) for d in detections), default=0.0)
        source_type = 'webcam' if (image_file.filename or '').startswith('webcam-frame') else 'image'
        store_history = True
        if source_type == 'webcam':
            store_history = should_store_webcam_snapshot(user['id'], len(detections))
        if store_history:
            source_media = save_bytes_to_media(
                file_name=image_file.filename or 'source.jpg',
                file_bytes=image_bytes,
                default_ext='.jpg',
                prefix='src',
            )
            result_media = save_bytes_to_media(
                file_name='detected.jpg',
                file_bytes=image_io.getvalue(),
                default_ext='.jpg',
                prefix='res',
            )
            create_detection_history(
                user_id=user['id'],
                username=user['username'],
                detection_type=source_type,
                source_original=source_media,
                result_detected=result_media,
                confidence=top_conf,
            )
        save_detections(
            user_id=user['id'],
            file_name=image_file.filename,
            detections=detections,
            source_type=source_type,
            job_id=None,
        )

        # Keep detections in headers for debugging while returning image output.
        response = send_file(image_io, mimetype='image/jpeg')
        response.headers['X-Detection-Count'] = str(len(detections))
        response.headers['X-Confidence-Threshold'] = f'{conf_threshold:.2f}'
        response.headers['X-Progress-Percent'] = '100'
        create_audit_log(
            user=user,
            action='predict_image_complete',
            endpoint='/predict',
            status='completed',
            file_name=image_file.filename,
            detections_count=len(detections),
            confidence=conf_threshold,
        )
        return response
    except Exception as e:
        print('Error during prediction:', e)
        create_audit_log(
            user=user,
            action='predict_image_failed',
            endpoint='/predict',
            status='failed',
            file_name=image_file.filename,
            message=str(e),
        )
        return jsonify({'error': 'Prediction failed', 'details': str(e)}), 500


# Video prediction endpoint
@app.route('/predict_video', methods=['POST'])
@auth_required
def predict_video():
    user = g.current_user
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    video_file = request.files['video']
    temp_dir = tempfile.mkdtemp(prefix='pothole_video_')
    original_name = video_file.filename or 'input.mp4'
    _, ext = os.path.splitext(original_name)
    if not ext:
        ext = '.mp4'
    temp_input_path = os.path.join(temp_dir, f'input{ext}')
    temp_output_path = os.path.join(temp_dir, 'result.mp4')
    response_ready = False
    out = None

    try:
        video_file.save(temp_input_path)
        conf_threshold = parse_conf_threshold(request.form.get('conf'), default=0.5)
        create_audit_log(
            user=user,
            action='predict_video_upload',
            endpoint='/predict_video',
            status='processing',
            file_name=video_file.filename,
            confidence=conf_threshold,
        )

        cap = cv2.VideoCapture(temp_input_path)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open uploaded video'}), 400

        input_fps = cap.get(cv2.CAP_PROP_FPS)
        if not input_fps or input_fps <= 0:
            input_fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()

        vid_stride = 2  # Process every 2nd frame for faster inference.
        output_fps = max(1.0, input_fps / vid_stride)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, output_fps, (width, height))
        if not out.isOpened():
            return jsonify({'error': 'Could not create output video file'}), 500

        results_stream = model.predict(
            source=temp_input_path,
            conf=conf_threshold,
            imgsz=640,
            stream=True,
            vid_stride=vid_stride,
            verbose=False,
        )
        processed = 0
        detection_rows = []
        for result in results_stream:
            frame = result.plot()
            if frame.shape[1] != width or frame.shape[0] != height:
                frame = cv2.resize(frame, (width, height))
            out.write(frame)
            frame_detections = parse_detections(result)
            if frame_detections:
                ts = time.time()
                for det in frame_detections:
                    detection_rows.append(
                        (
                            user['id'],
                            video_file.filename,
                            'video',
                            None,
                            processed,
                            int(det.get('class', 0)),
                            json.dumps(det.get('bbox') or []),
                            float(det.get('confidence', 0.0)),
                            ts,
                        )
                    )
            processed += 1
            if processed % 60 == 0:
                print(f'Processed {processed} frames...')
        out.release()
        out = None

        if detection_rows:
            save_detection_rows(detection_rows)

        if processed == 0:
            return jsonify({'error': 'No frames were processed from video'}), 400
        if not os.path.exists(temp_output_path):
            return jsonify({'error': 'Output video was not generated'}), 500

        source_media = copy_file_to_media(
            src_path=temp_input_path,
            file_name=video_file.filename or 'source.mp4',
            default_ext='.mp4',
            prefix='src',
        )
        result_media = copy_file_to_media(
            src_path=temp_output_path,
            file_name='detected.mp4',
            default_ext='.mp4',
            prefix='res',
        )
        top_video_conf = 0.0
        if detection_rows:
            top_video_conf = max((float(r[7]) for r in detection_rows), default=0.0)
        create_detection_history(
            user_id=user['id'],
            username=user['username'],
            detection_type='video',
            source_original=source_media,
            result_detected=result_media,
            confidence=top_video_conf,
        )

        @after_this_request
        def cleanup_temp_dir(response):
            try:
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir, ignore_errors=True)
            except Exception:
                pass
            return response

        response_ready = True
        response = send_file(temp_output_path, mimetype='video/mp4', as_attachment=False, download_name='result.mp4')
        response.headers['X-Confidence-Threshold'] = f'{conf_threshold:.2f}'
        response.headers['X-Processed-Frames'] = str(processed)
        response.headers['X-Progress-Percent'] = '100'
        create_audit_log(
            user=user,
            action='predict_video_complete',
            endpoint='/predict_video',
            status='completed',
            file_name=video_file.filename,
            confidence=conf_threshold,
            message=f'Processed frames: {processed}',
        )
        return response
    except Exception as e:
        print('Error during video prediction:', e)
        create_audit_log(
            user=user,
            action='predict_video_failed',
            endpoint='/predict_video',
            status='failed',
            file_name=video_file.filename,
            message=str(e),
        )
        return jsonify({'error': 'Video prediction failed', 'details': str(e)}), 500
    finally:
        if out is not None:
            out.release()
        # If request fails before after_this_request runs, clean temp files now.
        if not response_ready and os.path.exists(temp_dir):
            shutil.rmtree(temp_dir, ignore_errors=True)


@app.route('/predict_image_job', methods=['POST'])
@auth_required
def predict_image_job():
    user = g.current_user
    if 'image' not in request.files:
        return jsonify({'error': 'No image uploaded'}), 400
    image_file = request.files['image']
    try:
        image_bytes = image_file.read()
        Image.open(io.BytesIO(image_bytes)).convert('RGB')
    except Exception:
        return jsonify({'error': 'Invalid image file'}), 400

    conf_threshold = parse_conf_threshold(request.form.get('conf'), default=0.5)
    job_id = create_job(
        'image',
        owner_user_id=user['id'],
        owner_username=user['username'],
        owner_role=user['role'],
        file_name=image_file.filename,
    )
    create_audit_log(
        user=user,
        action='image_job_created',
        endpoint='/predict_image_job',
        status='queued',
        file_name=image_file.filename,
        confidence=conf_threshold,
    )
    t = threading.Thread(target=process_image_job, args=(job_id, image_bytes, conf_threshold), daemon=True)
    t.start()
    return jsonify({'job_id': job_id}), 202


@app.route('/predict_video_job', methods=['POST'])
@auth_required
def predict_video_job():
    user = g.current_user
    if 'video' not in request.files:
        return jsonify({'error': 'No video uploaded'}), 400
    video_file = request.files['video']
    conf_threshold = parse_conf_threshold(request.form.get('conf'), default=0.5)

    temp_dir = tempfile.mkdtemp(prefix='pothole_video_job_')
    original_name = video_file.filename or 'input.mp4'
    _, ext = os.path.splitext(original_name)
    if not ext:
        ext = '.mp4'
    input_path = os.path.join(temp_dir, f'input{ext}')
    try:
        video_file.save(input_path)
    except Exception as e:
        shutil.rmtree(temp_dir, ignore_errors=True)
        return jsonify({'error': f'Failed to save uploaded video: {e}'}), 500

    job_id = create_job(
        'video',
        owner_user_id=user['id'],
        owner_username=user['username'],
        owner_role=user['role'],
        file_name=video_file.filename,
    )
    create_audit_log(
        user=user,
        action='video_job_created',
        endpoint='/predict_video_job',
        status='queued',
        file_name=video_file.filename,
        confidence=conf_threshold,
    )
    t = threading.Thread(target=process_video_job, args=(job_id, temp_dir, input_path, conf_threshold), daemon=True)
    t.start()
    return jsonify({'job_id': job_id}), 202


@app.route('/progress/<job_id>', methods=['GET'])
@auth_required
def progress(job_id):
    user = g.current_user
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if user['role'] != 'admin' and job.get('owner_user_id') != user['id']:
        return jsonify({'error': 'Forbidden'}), 403
    return jsonify(
        {
            'job_id': job['id'],
            'kind': job['kind'],
            'status': job['status'],
            'progress': job['progress'],
            'error': job['error'],
        }
    )


@app.route('/result/<job_id>', methods=['GET'])
@auth_required
def result(job_id):
    user = g.current_user
    job = get_job(job_id)
    if not job:
        return jsonify({'error': 'Job not found'}), 404
    if user['role'] != 'admin' and job.get('owner_user_id') != user['id']:
        return jsonify({'error': 'Forbidden'}), 403
    if job['status'] != 'completed':
        return jsonify({'error': f"Job is not completed (status: {job['status']})"}), 409

    if job['result_type'] == 'image/jpeg' and job['result_bytes'] is not None:
        return send_file(io.BytesIO(job['result_bytes']), mimetype='image/jpeg')

    if job['result_type'] == 'video/mp4' and job['result_path']:
        cleanup_path = job.get('cleanup_path')

        @after_this_request
        def cleanup_video_artifacts(response):
            try:
                if cleanup_path and os.path.exists(cleanup_path):
                    shutil.rmtree(cleanup_path, ignore_errors=True)
            except Exception:
                pass
            return response

        return send_file(job['result_path'], mimetype='video/mp4', as_attachment=False, download_name='result.mp4')

    return jsonify({'error': 'Job result is not available'}), 500


@app.route('/auth/login', methods=['POST'])
def auth_login():
    payload = request.get_json(silent=True) or {}
    username = (payload.get('username') or '').strip()
    password = payload.get('password') or ''
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, username, password_hash, role FROM users WHERE username = ?', (username,))
    user_row = cur.fetchone()
    conn.close()
    if not user_row or not check_password_hash(user_row['password_hash'], password):
        return jsonify({'error': 'Invalid credentials'}), 401

    user = {'id': user_row['id'], 'username': user_row['username'], 'role': user_row['role']}
    token = create_access_token(user)
    create_audit_log(user=user, action='auth_login', endpoint='/auth/login', status='completed')
    return jsonify({'token': token, 'user': user})


@app.route('/auth/register', methods=['POST'])
def auth_register():
    payload = request.get_json(silent=True) or {}
    username = (payload.get('username') or '').strip()
    password = payload.get('password') or ''
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    if len(username) < 3:
        return jsonify({'error': 'Username must be at least 3 characters'}), 400
    if len(password) < 4:
        return jsonify({'error': 'Password must be at least 4 characters'}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            'INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)',
            (username, generate_password_hash(password), 'user', time.time()),
        )
        conn.commit()
        user_id = cur.lastrowid
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Username already exists'}), 409
    conn.close()

    created_user = {'id': user_id, 'username': username, 'role': 'user'}
    create_audit_log(
        user=created_user,
        action='auth_register',
        endpoint='/auth/register',
        status='completed',
        message='Self-registration completed',
    )
    return jsonify({'message': 'Registration successful. You can now login.'}), 201


@app.route('/auth/me', methods=['GET'])
@auth_required
def auth_me():
    return jsonify({'user': g.current_user})


@app.route('/detections/history', methods=['GET'])
@auth_required
def detections_history():
    user = g.current_user
    limit_raw = request.args.get('limit', '200')
    source_type = (request.args.get('source_type') or '').strip()
    target_user_id = request.args.get('user_id')
    try:
        limit = max(1, min(2000, int(limit_raw)))
    except ValueError:
        limit = 200

    where = []
    params = []
    if user['role'] != 'admin':
        where.append('user_id = ?')
        params.append(user['id'])
    elif target_user_id:
        try:
            target_user_id_int = int(target_user_id)
        except ValueError:
            return jsonify({'error': 'Invalid user_id'}), 400
        where.append('user_id = ?')
        params.append(target_user_id_int)
    if source_type:
        where.append('detection_type = ?')
        params.append(source_type)

    where_sql = f"WHERE {' AND '.join(where)}" if where else ''
    conn = get_db_connection()
    cur = conn.cursor()
    query = f'''
        SELECT id, user_id, username, detection_type, source_original, result_detected, confidence, created_at
        FROM detection_history
        {where_sql}
        ORDER BY id DESC
        LIMIT ?
    '''
    params.append(limit)
    cur.execute(query, tuple(params))
    rows = cur.fetchall()
    conn.close()

    detections = [dict(row) for row in rows]
    return jsonify({'detections': detections, 'count': len(detections)})


@app.route('/media/<path:file_name>', methods=['GET'])
@auth_required
def media_file(file_name):
    user = g.current_user
    safe_name = os.path.basename(file_name)
    if not safe_name:
        return jsonify({'error': 'Invalid media name'}), 400
    if not user_can_access_media(user, safe_name):
        return jsonify({'error': 'Forbidden'}), 403
    full_path = os.path.join(MEDIA_DIR, safe_name)
    if not os.path.exists(full_path):
        return jsonify({'error': 'Media not found'}), 404
    mime_type, _ = mimetypes.guess_type(full_path)
    return send_file(full_path, mimetype=mime_type or 'application/octet-stream', as_attachment=False)


@app.route('/admin/users', methods=['GET'])
@admin_required
def admin_users():
    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute('SELECT id, username, role, created_at FROM users ORDER BY id ASC')
    rows = cur.fetchall()
    conn.close()
    users = [dict(row) for row in rows]
    return jsonify({'users': users})


@app.route('/admin/users', methods=['POST'])
@admin_required
def admin_create_user():
    payload = request.get_json(silent=True) or {}
    username = (payload.get('username') or '').strip()
    password = payload.get('password') or ''
    role = (payload.get('role') or 'user').strip().lower()
    if not username or not password:
        return jsonify({'error': 'Username and password are required'}), 400
    if role not in ('admin', 'user'):
        return jsonify({'error': "Role must be 'admin' or 'user'"}), 400

    conn = get_db_connection()
    cur = conn.cursor()
    try:
        cur.execute(
            'INSERT INTO users (username, password_hash, role, created_at) VALUES (?, ?, ?, ?)',
            (username, generate_password_hash(password), role, time.time()),
        )
        conn.commit()
    except sqlite3.IntegrityError:
        conn.close()
        return jsonify({'error': 'Username already exists'}), 409

    user_id = cur.lastrowid
    conn.close()
    actor = g.current_user
    create_audit_log(
        user=actor,
        action='admin_create_user',
        endpoint='/admin/users',
        status='completed',
        message=f'Created user {username} ({role})',
    )
    return jsonify({'user': {'id': user_id, 'username': username, 'role': role}}), 201


@app.route('/admin/audit-logs', methods=['GET'])
@admin_required
def admin_audit_logs():
    limit_raw = request.args.get('limit', '100')
    try:
        limit = max(1, min(1000, int(limit_raw)))
    except ValueError:
        limit = 100

    conn = get_db_connection()
    cur = conn.cursor()
    cur.execute(
        '''
        SELECT id, user_id, username, role, action, endpoint, file_name, status, detections_count, confidence, message, created_at
        FROM audit_logs
        ORDER BY id DESC
        LIMIT ?
        ''',
        (limit,),
    )
    rows = cur.fetchall()
    conn.close()
    logs = [dict(row) for row in rows]
    return jsonify({'logs': logs, 'count': len(logs)})

@app.route('/')
def home():
    return 'Flask backend for pothole detection is running.'

init_db()

if __name__ == '__main__':
    app.run(debug=True)
