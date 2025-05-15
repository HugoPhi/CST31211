# database.py
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()


class DetectionRecord(db.Model):
    id = db.Column(db.String(36), primary_key=True)  # UUID
    filename = db.Column(db.String(255))
    upload_time = db.Column(db.DateTime, default=datetime.now)
    defect_types = db.Column(db.String(255))  # 存储JSON数组
    thumbnail_path = db.Column(db.String(255))
    result_path = db.Column(db.String(255))
    detections = db.Column(db.String(255))

    def to_dict(self):
        return {
            "file_id": self.id,
            "timestamp": self.upload_time.strftime("%Y-%m-%d %H:%M:%S"),
            "detections": json.loads(self.detections),
            "thumbnail": self.thumbnail_path,
            "result_img": self.result_path,
            "defect_types": json.loads(self.defect_types)
        }
