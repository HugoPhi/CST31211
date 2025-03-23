from plugins.lrkit.executer import Executer

from knn import KNNClf, SklearnKNNClf
from data_process import X_train, X_test, y_test, y_train

n_train = 50000
n_test = 1000
X_train = X_train[:n_train]
y_train = y_train[:n_train]
X_test = X_test[:n_test]
y_test = y_test[:n_test]


clf_dict = {}
k = 10

clf_dict[f'knn_{k}_pf_torch-cpu'] = KNNClf(k=k, d='manhattan', batch_size=(n_test, 1), backend='torch_cpu')
clf_dict[f'knn_{k}_pf_torch-gpu'] = KNNClf(k=k, d='manhattan', batch_size=(n_test, 1), backend='torch')
clf_dict['knn_sklearn'] = SklearnKNNClf(n_neighbors=k, metric='manhattan', algorithm='brute')


exc = Executer(
    X_train, y_train, X_test, y_test,
    metric_list=['accuracy', 'avg_recall'],
    clf_dict=clf_dict,
    log=False,
)

exc.run_all(sort_by='testing time')
