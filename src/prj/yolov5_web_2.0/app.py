import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
from database import db, DetectionRecord
import os
import zipfile
from io import BytesIO
from flask import send_file
import json
import subprocess

app = Flask(__name__)
app.config.update({
    'UPLOAD_FOLDER': 'static/uploads',
    'RESULT_FOLDER': 'static/results',
    'MAX_CONTENT_LENGTH': 100 * 1024 * 1024,
    'SQLALCHEMY_DATABASE_URI': 'sqlite:///detections.db',
    'SQLALCHEMY_TRACK_MODIFICATIONS': False
})

classMapping = {
    0: '针孔',
    1: '擦伤',
    2: '赃污',
    3: '褶皱'
}

db.init_app(app)
with app.app_context():
    db.create_all()

Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['RESULT_FOLDER']).mkdir(parents=True, exist_ok=True)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/export')
def export_data():
    try:
        memory_file = BytesIO()

        with zipfile.ZipFile(memory_file, 'w', zipfile.ZIP_DEFLATED) as zf:
            db_path = 'instance/detections.db' if os.path.exists('instance/detections.db') else 'detections.db'
            zf.write(db_path, 'database.db')

            for root, _, files in os.walk(app.config['UPLOAD_FOLDER']):
                for file in files:
                    path = os.path.join(root, file)
                    zf.write(path, os.path.relpath(path, app.config['UPLOAD_FOLDER']))

            for root, _, files in os.walk(app.config['RESULT_FOLDER']):
                for file in files:
                    path = os.path.join(root, file)
                    zf.write(path, os.path.relpath(path, app.config['RESULT_FOLDER']))

        memory_file.seek(0)

        return send_file(
            memory_file,
            mimetype='application/zip',
            as_attachment=True,
            download_name='system_export.zip'
        )

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_history')
def get_history():
    records = DetectionRecord.query.order_by(DetectionRecord.upload_time.desc()).all()
    return jsonify([{
        'file_id': r.id,
        'timestamp': r.upload_time.strftime("%Y-%m-%d %H:%M:%S"),
        'detections': json.loads(r.detections),
        'thumbnail': r.thumbnail_path,
        'result_img': r.result_path,
        'defect_types': json.loads(r.defect_types)
    } for r in records])

@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type'}), 415

    file_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    upload_dir = Path(app.config['UPLOAD_FOLDER']) / file_id
    upload_dir.mkdir(parents=True)
    source_path = upload_dir / 'source.jpg'
    file.save(source_path)

    try:
        result_dir = Path(app.config['RESULT_FOLDER']) / file_id
        cmd = [
            'python', 'yolov5/detect.py',
            '--weights', 'pts/yolov5_large_full_best_raw_hyp.pt',
            '--source', str(source_path),
            '--project', str(result_dir),
            '--name', '',
            '--exist-ok',
            '--save-txt',
            '--save-conf'
        ]
        subprocess.run(cmd, check=True)

        result_img = next((result_dir).glob('*.jpg'))
        detections = parse_labels(result_dir / 'labels' / f'{result_img.stem}.txt')

        defect_types = list({classMapping[d['class']] for d in detections})

        record = DetectionRecord(
            id=file_id,
            filename=file.filename,
            detections=json.dumps(detections),
            thumbnail_path=f'/uploads/{file_id}/source.jpg',
            result_path=f'/results/{file_id}/{result_img.name}',
            defect_types=json.dumps(defect_types),
        )
        db.session.add(record)
        db.session.commit()

        return jsonify({
            'status': 'success',
            'file_id': file_id,
            'timestamp': timestamp,
            'detections': detections,
            'thumbnail': f'/uploads/{file_id}/source.jpg',
            'result_img': f'/results/{file_id}/{result_img.name}',
            'defect_types': defect_types
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def parse_labels(label_path):
    if not label_path.exists():
        return []

    detections = []
    with open(label_path) as f:
        for line in f:
            parts = line.strip().split()
            detections.append({
                'class': int(parts[0]),
                'confidence': float(parts[5]),
                'bbox': list(map(float, parts[1:5]))
            })
    return detections

@app.route('/uploads/<path:filename>')
def serve_upload(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/results/<path:filename>')
def serve_result(filename):
    return send_from_directory(app.config['RESULT_FOLDER'], filename)

@app.route('/get_result/<file_id>')
def get_result(file_id):
    result_dir = Path(app.config['RESULT_FOLDER']) / file_id
    result_img = next((result_dir).glob('*.jpg'))

    return jsonify({
        'image_url': f'/results/{file_id}/{result_img.name}',
        'detections': parse_labels(result_dir / 'labels' / f'{result_img.stem}.txt')
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8888, debug=True)