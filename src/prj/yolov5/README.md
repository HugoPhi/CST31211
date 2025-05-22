# YOLOv5 测试命令

## 1. 测试原始模型
```bash
python train.py --img 640 --batch 16 --epochs 50 --data /home/byh/b_llama_2_7b/yolov5/clean_data_complete/clean_data/data.yaml --weights yolov5s.pt --name yolo_with_attention --cfg yolov5s_original.yaml
```

## 2. 测试增加了单层SAM注意力层的模型
```bash
python train.py --img 640 --batch 16 --epochs 50 --data /home/byh/b_llama_2_7b/yolov5/clean_data_complete/clean_data/data.yaml --weights yolov5s.pt --name yolo_with_attention --cfg yolov5s.yaml
```

## 3. 测试增加了多层SAM注意力机制的模型
```bash
python train.py --img 640 --batch 16 --epochs 50 --data /home/byh/b_llama_2_7b/yolov5/clean_data_complete/clean_data/data.yaml --weights yolov5s.pt --name yolo_with_attention --cfg yolov5muls.yaml
```

## 4. 增加了一层SE注意力机制的模型
```bash
python train.py --img 640 --batch 16 --epochs 50 --data /home/byh/b_llama_2_7b/yolov5/clean_data_complete/clean_data/data.yaml --weights yolov5s.pt --name yolo_with_attention --cfg yolov5s_SE.yaml
```

## 5. 增加了多层SE注意力机制的模型
```bash
python train.py --img 640 --batch 16 --epochs 50 --data /home/byh/b_llama_2_7b/yolov5/clean_data_complete/clean_data/data.yaml --weights yolov5s.pt --name yolo_with_attention --cfg yolov5s_mulSE.yaml
```

尝试其他类型的注意力机制可以创建新的yaml文件并且修改相应位置

## 6. 尝试不同类型的损失函数
可以对代码文件里的`utils/loss.py`这个文件的第164行进行修改，参数的`Focal=True`可以修改为其他类型的`CLOU=True`等，从而尝试不同类型的损失函数
