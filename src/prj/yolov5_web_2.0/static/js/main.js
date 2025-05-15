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

        this.selectedTags = new Set();
        this.availableTags = new Set(Object.values(classMapping));

        this.initSearch();

        this.loadHistory();

        document.getElementById('exportBtn').addEventListener('click', () => {
            this.exportSystemData();
        });

        this.selectedTags = new Set(JSON.parse(localStorage.getItem('selectedTags') || '[]'));
        this.updateSelectedTagsDisplay();

        setTimeout(() => this.filterCards(), 100);
    }

    async exportSystemData() {
        this.showLoading(true);
        try {
            const response = await fetch('/export');
            if (!response.ok) throw new Error('导出失败');

            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `system_backup_${new Date().toISOString().slice(0, 10)}.zip`;
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } catch (error) {
            console.error('导出失败:', error);
            alert('数据导出失败，请检查控制台');
        } finally {
            this.showLoading(false);
        }
    }

    async loadHistory() {
        try {
            const response = await fetch('/get_history');
            const history = await response.json();
            history.forEach(data => this.addResultCard(data));
        } catch (error) {
            console.error('加载历史记录失败:', error);
        }
    }

    initEventListeners() {
        document.getElementById('fileInput').addEventListener('change', e => {
            this.handleFiles(e.target.files);
        });

        document.querySelector('.close-btn').addEventListener('click', () => {
            this.modal.style.display = 'none';
        });

        this.cardContainer.addEventListener('click', e => {
            const card = e.target.closest('.card');
            if (card) this.showResultModal(card.dataset.id);
        });
    }

    initSearch() {
        const searchInput = document.getElementById('searchInput');
        const tagOptions = document.getElementById('tagOptions');

        searchInput.addEventListener('click', () => {
            if (this.selectedTags.size < 4) {
                this.showTagOptions();
            }
        });

        tagOptions.addEventListener('click', e => {
            if (e.target.classList.contains('tag-option')) {
                this.addTag(e.target.dataset.value);
            }
        });

        searchInput.addEventListener('input', () => {
            this.filterOptions(searchInput.value.trim());
        });
    }

    showTagOptions() {
        const options = document.getElementById('tagOptions');
        options.innerHTML = Array.from(this.availableTags)
            .filter(tag => !this.selectedTags.has(tag))
            .map(tag => `
        <div class="tag-option" data-value="${tag}">${tag}</div>
      `).join('');
        options.style.display = 'block';
    }

    addTag(tag) {
        if (!this.availableTags.has(tag)) {
            console.warn(`无效标签: ${tag}`);
            return;
        }

        if (this.selectedTags.size >= 4) {
            this.showToast('最多选择4个标签');
            return;
        }

        this.selectedTags.add(tag);
        this.updateTagSystem();
    }

    updateTagSystem() {
        this.updateSelectedTagsDisplay();
        this.filterCards();
        this.refreshTagOptions();

        document.getElementById('searchInput').value = '';
        document.getElementById('tagOptions').style.display = 'none';
    }

    refreshTagOptions() {
        const options = document.getElementById('tagOptions');
        options.innerHTML = Array.from(this.availableTags)
            .filter(tag => !this.selectedTags.has(tag))
            .map(tag => `<div class="tag-option" data-value="${tag}">${tag}</div>`)
            .join('');
    }

    showToast(message) {
        const toast = document.createElement('div');
        toast.className = 'toast-message';
        toast.textContent = message;
        document.body.appendChild(toast);

        setTimeout(() => toast.remove(), 2000);
    }

    removeTag(tag) {
        this.selectedTags.delete(tag);
        this.updateSelectedTagsDisplay();
        this.filterCards();
    }

    updateSelectedTagsDisplay() {
        const container = document.getElementById('selectedTags');
        container.innerHTML = Array.from(this.selectedTags).map(tag => `
      <div class="tag">
        ${tag}
        <span class="tag-remove" onclick="detectionSystem.removeTag('${tag}')">×</span>
      </div>
    `).join('');
    }

    filterCards() {
        const tags = Array.from(this.selectedTags);
        document.querySelectorAll('.card').forEach(card => {
            try {
                const cardTags = JSON.parse(card.dataset.tags);
                const show = tags.every(tag => cardTags.includes(tag));
                card.style.display = show ? 'block' : 'none';
            } catch (e) {
                console.error('卡片数据解析错误:', e);
                card.style.display = 'block';
            }
        });

        localStorage.setItem('selectedTags', JSON.stringify(Array.from(this.selectedTags)));
    }

    async handleFiles(files) {
        Array.from(files).forEach(async file => {
            if (!file.type.startsWith('image/')) return;

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

        card.dataset.tags = JSON.stringify(data.defect_types || []);

        card.innerHTML = `
          <img src="${data.thumbnail}">
          <div class="card-info">
            <div class="card-status ${this.getStatusClass(data.detections)}">
              ${this.getStatusText(data.detections)}
            </div>
            <div class="card-time">${data.timestamp}</div>
          </div>
        `;

        this.bindCardHover(card);
        this.cardContainer.prepend(card);
    }

    bindCardHover(card) {
        card.addEventListener('mouseenter', () => {
            card.style.transform = 'translateY(-5px)';
        });
        card.addEventListener('mouseleave', () => {
            card.style.transform = 'none';
        });
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

const detectionSystem = new DetectionSystem();