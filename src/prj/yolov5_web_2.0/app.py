import uuid
from datetime import datetime
from pathlib import Path
from flask import Flask, render_template, request, jsonify, send_from_directory
import subprocess

app = Flask(__name__)
app.config.update({
    'UPLOAD_FOLDER': 'static/uploads',
    'RESULT_FOLDER': 'static/results',
    'MAX_CONTENT_LENGTH': 100 * 1024 * 1024  # 100MB
})

# 初始化目录
Path(app.config['UPLOAD_FOLDER']).mkdir(parents=True, exist_ok=True)
Path(app.config['RESULT_FOLDER']).mkdir(parents=True, exist_ok=True)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400

    file = request.files['file']
    if not file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        return jsonify({'error': 'Invalid file type'}), 415

    # 生成唯一ID和时间戳
    file_id = str(uuid.uuid4())
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 保存原始文件
    upload_dir = Path(app.config['UPLOAD_FOLDER']) / file_id
    upload_dir.mkdir(parents=True)
    source_path = upload_dir / 'source.jpg'
    file.save(source_path)

    try:
        # 运行检测
        result_dir = Path(app.config['RESULT_FOLDER']) / file_id
        cmd = [
            'python', 'yolov5/detect.py',
            '--weights', 'best.pt',
            '--source', str(source_path),
            '--project', str(result_dir),
            '--name', '',
            '--exist-ok',
            '--save-txt',
            '--save-conf'
        ]
        subprocess.run(cmd, check=True)

        # 获取结果文件
        result_img = next((result_dir).glob('*.jpg'))
        detections = parse_labels(result_dir / 'labels' / f'{result_img.stem}.txt')

        return jsonify({
            'status': 'success',
            'file_id': file_id,
            'timestamp': timestamp,
            'detections': detections,
            'thumbnail': f'/uploads/{file_id}/source.jpg',
            'result_img': f'/results/{file_id}/{result_img.name}'
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
    app.run(host='0.0.0.0', debug=True)
