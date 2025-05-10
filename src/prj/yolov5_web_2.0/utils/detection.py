def parse_yolo_results(result_dir):
    # 获取检测结果图片
    image_path = next((result_dir / 'exp').glob('*.jpg'))

    # 解析标签文件
    label_file = (result_dir / 'exp' / 'labels' / image_path.stem).with_suffix('.txt')
    detections = []
    class_counts = {}

    if label_file.exists():
        with open(label_file) as f:
            for line in f:
                parts = line.strip().split()
                class_id = int(parts[0])
                confidence = float(parts[5])

                # 统计类别
                class_counts[class_id] = class_counts.get(class_id, 0) + 1

                # 转换坐标格式
                detections.append({
                    'class_id': class_id,
                    'confidence': confidence,
                    'bbox': list(map(float, parts[1:5]))
                })

    # 获取颜色映射
    colors = get_class_colors()

    return {
        'image_path': str(image_path),
        'stats': {
            'total': len(detections),
            'class_distribution': class_counts
        },
        'detections': [{
            **d,
            'color': colors[d['class_id']]
        } for d in detections]
    }


def get_class_colors():
    """与YOLOv5颜色配置保持一致"""
    return {
        0: '#FF3838',  # zhen_kong
        1: '#FF701F',  # ca_shang
        2: '#48F90A',  # zang_wu
        3: '#00C2FF'   # zhe_zhou
    }
