# CST31211

本项目为 CST31211 课程的实验、作业及课程项目的主要源码与文档所在位置。各文件夹采用分级结构，方便查阅与管理。请根据下方结构与说明查找、理解和使用相关内容。

## 1. 作业（Homework）

### 1.1 [`hw1`](./hw1/)
- **说明文件**
  - [`README.md`](./hw1/README.md)：作业 1 说明文档
  - [`discription.txt`](./hw1/discription.txt)：作业详细说明
  - [`chapter2.pdf`](./hw1/chapter2.pdf)：第二章参考资料或教材
- **代码/模块**
  - [`knn/`](./hw1/knn/)：K-近邻相关实现代码
  - [`loss_functions/`](./hw1/loss_functions/)：损失函数相关代码

### 1.2 [`hw2`](./hw2/)
- **说明文件**
  - [`README.md`](./hw2/README.md)：作业 2 说明文档
  - [`chapter3.pdf`](./hw2/chapter3.pdf)：第三章参考资料或教材


## 2. 实验（Lab）

### 2.1 [`lab1`](./lab1/)
- [`README.md`](./lab1/README.md)：实验 1 说明
- [`AlexNet.ipynb`](./lab1/AlexNet.ipynb)：AlexNet 深度学习网络实验 Jupyter 笔记本

### 2.2 [`lab2`](./lab2/)
- **配置与资源**
  - [`config.yml`](./lab2/config.yml)：实验配置文件
  - [`assets/`](./lab2/assets/)：实验用资源文件夹
  - [`log/`](./lab2/log/)：实验日志文件夹
- **主要源码**
  - [`data_process.py`](./lab2/data_process.py)：数据处理脚本
  - [`huanjing`](./lab2/huanjing)：环境相关内容
  - [`main.py`](./lab2/main.py)：实验主入口
  - [`models.py`](./lab2/models.py)：模型结构定义
  - [`plot.py`](./lab2/plot.py)：数据可视化
  - [`plot_main.py`](./lab2/plot_main.py)：可视化主脚本
  - [`summary_main.py`](./lab2/summary_main.py)：结果汇总
  - [`training_script.py`](./lab2/training_script.py)：训练脚本

  > 由于接口限制，`lab2` 内文件列表可能不完整，详见 [lab2 目录](https://github.com/HugoPhi/CST31211/tree/main/src/lab2) 获取全部内容。

### 2.3 [`lab3`](./lab3/)
- [`mindspore/`](./lab3/mindspore/)：基于 MindSpore 的实验代码
- [`torch/`](./lab3/torch/)：基于 PyTorch 的实验代码


## 3. 课程项目（Project）

### 3.1 [`prj`](./prj/)

#### 3.1.1 [`yolov5/`](./prj/yolov5/)
- 目标检测相关项目（YOLOv5）

#### 3.1.2 [`yolov5_web_2.0/`](./prj/yolov5_web_2.0/)
- YOLOv5 Web 可视化/部署相关内容


## 4. 使用与贡献建议

- 各作业/实验/项目均建议先阅读其对应的 `README.md` 或说明文档。
- 代码运行及环境配置请见各子目录内的详细说明。
- 若发现缺失文件或列表不全，请访问 [src 目录](https://github.com/HugoPhi/CST31211/tree/main/src) 在线浏览。

## 5. 反馈与交流

如有问题欢迎在本仓库 [issues](https://github.com/HugoPhi/CST31211/issues) 区留言反馈。