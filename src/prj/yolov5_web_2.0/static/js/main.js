const classMapping = {
    0: '针孔',
    1: '擦伤',
    2: '赃污',
    3: '褶皱'
};


class DetectionSystem {
    constructor() {
        this.cardContainer = document.getElementById('cardContainer');
        this.modal = document.getElementById('resultModal');
        this.initEventListeners();
    }

    initEventListeners() {
        // 文件上传事件
        document.getElementById('fileInput').addEventListener('change', e => {
            this.handleFiles(e.target.files);
        });

        // 模态框关闭
        document.querySelector('.close-btn').addEventListener('click', () => {
            this.modal.style.display = 'none';
        });

        // 卡片点击事件（事件委托）
        this.cardContainer.addEventListener('click', e => {
            const card = e.target.closest('.card');
            if (card) this.showResultModal(card.dataset.id);
        });
    }

    async handleFiles(files) {
        Array.from(files).forEach(async file => {
            if (!file.type.startsWith('image/')) return;

            // 显示加载状态
            this.showLoading(true);

            try {
                const formData = new FormData();
                formData.append('file', file);

                const response = await fetch('/detect', {
                    method: 'POST',
                    body: formData
                });

                const result = await response.json();
                if (result.status === 'success') {
                    this.addResultCard(result);
                }
            } catch (error) {
                console.error('上传失败:', error);
            } finally {
                this.showLoading(false);
            }
        });
    }

    addResultCard(data) {
        const card = document.createElement('div');
        card.className = 'card';
        card.dataset.id = data.file_id;
        card.innerHTML = `
            <img src="${data.thumbnail}">
            <div class="card-info">
                <div class="card-status ${this.getStatusClass(data.detections)}">
                    ${this.getStatusText(data.detections)}
                </div>
                <div class="card-time">${data.timestamp}</div>
            </div>
        `;
        this.cardContainer.prepend(card);
    }

    async showResultModal(fileId) {
        this.showLoading(true);
        try {
            const response = await fetch(`/get_result/${fileId}`);
            const result = await response.json();

            document.getElementById('modalImage').src = result.image_url;
            document.getElementById('resultList').innerHTML = result.detections
                .map(det => `
                    <div class="result-item">
                        <div>类型: ${classMapping[det.class]}</div>
                        <div>置信度: ${(det.confidence * 100).toFixed(1)}%</div>
                        <div>位置: ${det.bbox.map(v => v.toFixed(2)).join(', ')}</div>
                    </div>
                `).join('');

            this.modal.style.display = 'block';
        } catch (error) {
            console.error('获取结果失败:', error);
        } finally {
            this.showLoading(false);
        }
    }

    getStatusClass(detections) {
        return detections.length > 0 ? 'status-danger' : 'status-success';
    }

    getStatusText(detections) {
        return detections.length > 0 ? `发现 ${detections.length} 处缺陷` : '无缺陷';
    }

    showLoading(show) {
        document.getElementById('loading').style.display = show ? 'block' : 'none';
    }
}

// 初始化系统
new DetectionSystem();
