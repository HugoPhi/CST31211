from sklearn.neighbors import KNeighborsClassifier
import torch
import time
import numpy as np

from plugins.lrkit.clfs import Clfs


class _knn_clf_numpy(Clfs):
    '''
    K-nearest-neighbor classifier by numpy.

    Parameters
    ----------
    k : int
        KNN的K值，k >= 1 。
    d : str
        距离度量的名称。有如下几种：
        - 'euclid': 欧几里德距离(l2)。
        - 'manhattan': 曼哈顿距离(l1)。
        - 'cosine': 余弦距离。
        - 'chebyshev': 切比雪夫距离(l∞)，无穷范数。
    batch_size : tuple
        计算距离的批次大小。其中，batch_size[0]是测试集的批次大小，batch_size[1]是训练集的批次大小。
    '''

    def __init__(self,
                 k=1,
                 d='euclid',
                 batch_size=(128, 2048)):

        super(_knn_clf_numpy, self).__init__()

        print('[*] use numpy version.')
        self.k = k
        if k < 1:
            raise ValueError(f'[x] k should be a number >=1, but get {self.k}')

        self.d = d
        self.batch_size = batch_size

        if d == 'euclid':
            self.distance = self.__euclid_distance
        elif d == 'manhattan':
            self.distance = self.__manhattan_distance
        elif d == 'cosine':
            self.distance = self.__cosine_distances
        elif d == 'chebyshev':
            self.distance = self.__chebyshev_distances
        else:
            print('[!] d should be euclid, manhattan, cosine or chebyshev !')
            print('[!] use default p: euclid')
            self.distance = self.__manhattan_distance

    def __euclid_distance(self, x, y):
        '''
        计算两个数据集的欧几里德距离，就是两两向量之间的欧几里德距离。

        Parameters
        ----------
        x, y: np.ndarray
            两个数据集，形状为(Nx, D)和(Ny, D)，Nx, Ny是数据集大小；D是向量的维度。

        Returns
        -------
        distance : np.ndarray
            两个数据集之间的距离，形状为(Ny, Nx)
        '''
        dists = np.sqrt(
            np.sum(y**2, axis=1)[:, np.newaxis] + np.sum(x**2, axis=1) - 2 * np.dot(y, x.T)
        )
        return dists

    def __manhattan_distance(self, x, y):
        '''
        计算两个数据集的曼哈顿距离，就是两两向量之间的曼哈顿距离。

        Parameters
        ----------
        x, y: np.ndarray
            两个数据集，形状为(Nx, D)和(Ny, D），Nx, Ny是数据集大小；D是向量的维度。

        Returns
        -------
        distance : np.ndarray
            两个数据集之间的距离，形状为(Ny, Nx)
        '''

        dists = np.sum(np.abs(y[:, np.newaxis] - x), axis=2)
        return dists

    def __chebyshev_distances(self, x, y):
        '''
        计算两个数据集的切比雪夫距离，就是两两向量之间的切比雪夫距离。

        Parameters
        ----------
        x, y: np.ndarray
            两个数据集，形状为(Nx, D)和(Ny, D），Nx, Ny是数据集大小；D是向量的维度。

        Returns
        -------
        distance : np.ndarray
            两个数据集之间的距离，形状为(Ny, Nx)
        '''

        dists = np.max(np.abs(y[:, np.newaxis] - x), axis=2)
        return dists

    def __cosine_distances(self, x, y):
        '''
        计算两个数据集的余弦距离，就是两两向量之间的余弦距离。

        Parameters
        ----------
        x, y: np.ndarray
            两个数据集，形状为(Nx, D)和(Ny, D），Nx, Ny是数据集大小；D是向量的维度。

        Returns
        -------
        distance : np.ndarray
            两个数据集之间的距离，形状为(Ny, Nx)
        '''

        # 归一化向量以简化余弦距离的计算
        x_normalized = x / np.linalg.norm(x, axis=1, keepdims=True)
        y_normalized = y / np.linalg.norm(y, axis=1, keepdims=True)

        # 计算余弦相似度矩阵，然后转换为余弦距离
        similarity = np.dot(y_normalized, x_normalized.T)
        dists = 1 - similarity
        return dists

    def fit(self, X_train, y_train):
        start = time.time()

        self.x = X_train
        self.y = y_train.reshape(-1)
        classes, self.static = np.unique(self.y, return_counts=True)  # 统计每种类别

        if classes[0] != 0:
            raise ValueError('[x] Make sure y is start form 0 !')  # 确保类别是形如：0, 1, 2 ...

        self.static = self.static / self.static.sum()  # 每种类别可能的概率，用于抵抗数据不均衡

        self.training_time = time.time() - start

    def predict(self, x_test):
        proba = self.predict_proba(x_test)
        diff = proba - self.static

        return np.argmax(diff, axis=1)  # 可能性最大

    def predict_proba(self, x_test):

        def __calculate_proba(batch_x_test, n_k_neighbors):
            batch_size, k = n_k_neighbors.shape
            num_classes = self.static.shape[0]

            class_counts = np.zeros((batch_size, num_classes))  # 批次的概率矩阵：(B_test, C) or (N_test % B_test, C)

            for i in range(batch_size):  # 遍历每个样本和它的最近邻，统计每个类别出现的次数
                for j in range(k):
                    neighbor_index = n_k_neighbors[i, j]
                    class_label = self.y[neighbor_index]
                    class_counts[i, class_label] += 1

            proba_batch = class_counts / k  # 计算概率：将每个样本的类别计数除以最近邻的数量(k)

            return proba_batch

        start = time.time()

        proba = np.zeros((x_test.shape[0], self.static.shape[0]))  # 每种类别的可能性，（N_test, C）

        self.n_k_neighbors = []
        for i in range(0, x_test.shape[0], self.batch_size[0]):  # N_test -> N_test // B_test个(B_test, :)和一个(N_test % B_test, :)
            batch_x_test = x_test[i:i + self.batch_size[0]]  # 不足B_test的batch会自动取不到，所以此处形状可能是(B_test, :)或者(N_test % B_test, :)

            distance = np.zeros((batch_x_test.shape[0], self.x.shape[0]))

            for j in range(0, self.x.shape[0], self.batch_size[1]):  # 这一步会得到完整的distance。
                batch_x_train = self.x[j:j + self.batch_size[1]]
                dist_batch = self.distance(batch_x_train, batch_x_test)  # 计算距离，得到：(x_test.shape[0], x_train.shape[0])
                distance[:, j:j + dist_batch.shape[1]] = dist_batch  # 把距离逐步装到distance第一个维度

            n_k_neighbors = np.argsort(distance, axis=1)[:, :self.k]  # K个邻居
            self.n_k_neighbors.append(n_k_neighbors)

            proba[i:i + batch_x_test.shape[0]] = __calculate_proba(batch_x_test, n_k_neighbors)  # 把计算得到的proba添加到第一个维度。

        self.testing_time = time.time() - start

        self.n_k_neighbors = np.concatenate(self.n_k_neighbors, axis=0)  # 得到x_test的每个样本的k个最近邻居 -> (N_test, k)
        return proba

    def get_k_neighbors(self):
        return self.n_k_neighbors

    def get_testing_time(self):
        return self.testing_time

    def get_training_time(self):
        return self.training_time

    def get_pre_proba(self):
        return self.static


class _knn_clf_torch(Clfs):
    def __init__(self,
                 k=1,
                 d='euclid',
                 cpu=False,  # 强制使用CPU进行计算
                 batch_size=(128, 2048)):

        super(_knn_clf_torch, self).__init__()

        self.k = k
        if k < 1:
            raise ValueError(f'[x] k should be a number >=1, but get {self.k}')
        self.d = d
        self.batch_size = batch_size

        # 检查是否有可用的GPU，并设置设备
        if cpu:
            self.device = torch.device('cpu')
        else:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f'[*] use torch version({self.device.type}).')

        self.k = k
        # 根据距离度量选择相应的函数
        if d == 'euclid':
            self.distance = self.__euclid_distance
        elif d == 'manhattan':
            self.distance = self.__manhattan_distance
        elif d == 'cosine':
            self.distance = self.__cosine_distances
        elif d == 'chebyshev':
            self.distance = self.__chebyshev_distances
        else:
            print('[!] d should be euclid, manhattan, cosine or chebyshev !')
            print('[!] use default p: euclid')
            self.distance = self.__euclid_distance

    def __to_tensor(self, x):
        return torch.tensor(x, dtype=torch.float32).to(self.device)

    def __euclid_distance(self, x, y):
        # x, y = self.__to_tensor(x), self.__to_tensor(y)
        dists = torch.sqrt(
            torch.sum(y**2, dim=1).unsqueeze(1) + torch.sum(x**2, dim=1) - 2 * torch.mm(y, x.t())
        )
        return dists

    def __manhattan_distance(self, x, y):
        # x, y = self.__to_tensor(x), self.__to_tensor(y)
        dists = torch.sum(torch.abs(y.unsqueeze(1) - x), dim=2)
        return dists

    def __chebyshev_distances(self, x, y):
        # x, y = self.__to_tensor(x), self.__to_tensor(y)
        dists = torch.max(torch.abs(y.unsqueeze(1) - x), dim=2)[0]
        return dists

    def __cosine_distances(self, x, y):
        # x, y = self.__to_tensor(x), self.__to_tensor(y)
        x_normalized = x / torch.norm(x, p=2, dim=1, keepdim=True)
        y_normalized = y / torch.norm(y, p=2, dim=1, keepdim=True)
        similarity = torch.mm(y_normalized, x_normalized.t())
        dists = 1 - similarity
        return dists

    def fit(self, X_train, y_train):
        start = time.time()
        self.x = self.__to_tensor(X_train)
        self.y = torch.tensor(y_train.reshape(-1), dtype=torch.long).to(self.device)
        classes, self.static = torch.unique(self.y, return_counts=True)

        if classes[0] != 0:
            raise ValueError('[x] Make sure y is start form 0 !')

        self.static = self.static / self.static.sum()
        self.training_time = time.time() - start

    def predict(self, x_test):
        proba = self.predict_proba(x_test)
        diff = proba - self.static.cpu().numpy()
        return np.argmax(diff, axis=1)

    def predict_proba(self, x_test):

        def __calculate_proba(batch_x_test, n_k_neighbors):
            batch_size, k = n_k_neighbors.size()
            num_classes = self.static.size(0)

            class_counts = torch.zeros((batch_size, num_classes), device=self.device)

            for i in range(batch_size):
                for j in range(k):
                    neighbor_index = n_k_neighbors[i, j]
                    class_label = self.y[neighbor_index]
                    class_counts[i, class_label] += 1

            proba_batch = class_counts / k

            return proba_batch  # 这里不用转成numpy因为后面还要加入到大的proba里面

        start = time.time()
        proba = torch.zeros((x_test.shape[0], self.static.size(0)), device=self.device)

        self.n_k_neighbors = []
        for i in range(0, x_test.shape[0], self.batch_size[0]):
            batch_x_test = self.__to_tensor(x_test[i:i + self.batch_size[0]])
            distance = torch.zeros((batch_x_test.size(0), self.x.size(0)), device=self.device)

            for j in range(0, self.x.size(0), self.batch_size[1]):
                batch_x_train = self.x[j:j + self.batch_size[1]]
                dist_batch = self.distance(batch_x_train, batch_x_test)
                distance[:, j:j + dist_batch.size(1)] = dist_batch

            n_k_neighbors = torch.argsort(distance, dim=1)[:, :self.k]
            self.n_k_neighbors.append(n_k_neighbors.cpu().numpy())
            proba[i:i + batch_x_test.size(0), :] = __calculate_proba(batch_x_test, n_k_neighbors)

        self.testing_time = time.time() - start

        self.n_k_neighbors = np.concatenate(self.n_k_neighbors, axis=0)
        return proba.cpu().numpy()

    def get_k_neighbors(self):
        return self.n_k_neighbors

    def get_testing_time(self):
        return self.testing_time

    def get_training_time(self):
        return self.training_time

    def get_pre_proba(self):
        return self.static.cpu().numpy()


class KNNClf(Clfs):
    '''
    K-nearest-neighbor classifier by numpy.

    Parameters
    ----------
    k : int
        KNN的K值，k >= 1 。
    d : str
        距离度量的名称。有如下几种：
        - 'euclid': 欧几里德距离(l2)。
        - 'manhattan': 曼哈顿距离(l1)。
        - 'cosine': 余弦距离。
        - 'chebyshev': 切比雪夫距离(l∞)，无穷范数。
    batch_size : tuple
        计算距离的批次大小。其中，batch_size[0]是测试集的批次大小，batch_size[1]是训练集的批次大小。
    backend : str
        使用的后端，有三个可选值：'numpy', 'torch', 'torch_cpu'。
    '''

    def __init__(self,
                 k=1,
                 d='euclid',
                 batch_size=(128, 2048),
                 backend='numpy'):

        super(KNNClf, self).__init__()

        if backend.lower() == 'numpy':
            self.knn = _knn_clf_numpy(k=k, d=d, batch_size=batch_size)
        elif backend.lower() == 'torch':
            self.knn = _knn_clf_torch(k=k, d=d, batch_size=batch_size)
        elif backend.lower() == 'torch_cpu':
            self.knn = _knn_clf_torch(k=k, d=d, batch_size=batch_size, cpu=True)
        else:
            raise ValueError("[x] Backend should be either 'numpy' or 'torch'")

    def fit(self, X_train, y_train):
        return self.knn.fit(X_train, y_train)

    def predict(self, x_test):
        return self.knn.predict(x_test)

    def predict_proba(self, x_test):
        return self.knn.predict_proba(x_test)

    def get_k_neighbors(self):
        return self.knn.get_k_neighbors()

    def get_testing_time(self):
        return self.knn.get_testing_time()

    def get_training_time(self):
        return self.knn.get_training_time()

    def get_pre_proba(self):
        return self.knn.get_pre_proba()


class SklearnKNNClf(Clfs):  # by Qwen.
    def __init__(self, n_neighbors=5, weights='uniform', algorithm='auto', metric='manhattan'):
        super(SklearnKNNClf, self).__init__()
        # 初始化KNN分类器，将传入的参数传递给它，并设置距离度量为曼哈顿距离
        self.knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights, algorithm=algorithm, metric=metric)
        self.train_time = None
        self.test_time = None

        print('[*] use sklearn stdlib version.')

    def fit(self, X_train, y_train):
        # 记录训练开始时间
        start_time = time.time()
        # 使用KNN分类器进行训练
        self.knn.fit(X_train, y_train)
        # 记录训练结束时间
        self.train_time = time.time() - start_time

    def predict(self, X_test):
        # 记录预测开始时间
        start_time = time.time()
        # 进行预测
        predictions = self.knn.predict(X_test)
        # 记录预测结束时间
        self.test_time = time.time() - start_time
        return predictions

    def predict_proba(self, X_test):
        # 返回每个测试样本属于各个类别的概率
        return self.knn.predict_proba(X_test)

    def get_training_time(self):
        # 返回训练所花费的时间
        return self.train_time

    def get_testing_time(self):
        # 返回测试（预测）所花费的时间
        return self.test_time
