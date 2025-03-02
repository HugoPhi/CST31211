from plugins.lrkit.executer import Executer

from knn import KNNClf, SklearnKNNClf
from data_process import X_train, X_test, y_test, y_train

n_train = 5000
n_test = 1000
X_train = X_train[:n_train]
y_train = y_train[:n_train]
X_test = X_test[:n_test]
y_test = y_test[:n_test]


clf_dict = {}
k = 5

clf_dict[f'knn_{k}_sklearn'] = SklearnKNNClf(n_neighbors=k, metric='manhattan', algorithm='brute')

# f -> for loop; b -> batch parallel; p -> parallel.
clf_dict[f'knn_{k}_bb_numpy'] = KNNClf(k=k, d='manhattan', batch_size=(256, 1024), backend='numpy')
clf_dict[f'knn_{k}_fb_numpy'] = KNNClf(k=k, d='manhattan', batch_size=(n_test, 64), backend='numpy')
clf_dict[f'knn_{k}_bf_numpy'] = KNNClf(k=k, d='manhattan', batch_size=(16, n_train), backend='numpy')
clf_dict[f'knn_{k}_pf_numpy'] = KNNClf(k=k, d='manhattan', batch_size=(n_test, 1), backend='numpy')
clf_dict[f'knn_{k}_fp_numpy'] = KNNClf(k=k, d='manhattan', batch_size=(1, n_train), backend='numpy')
clf_dict[f'knn_{k}_ff_numpy'] = KNNClf(k=k, d='manhattan', batch_size=(1, 1), backend='numpy')

clf_dict[f'knn_{k}_bb_torch_cpu'] = KNNClf(k=k, d='manhattan', batch_size=(256, 1024), backend='torch_cpu')
clf_dict[f'knn_{k}_fb_torch_cpu'] = KNNClf(k=k, d='manhattan', batch_size=(n_test, 64), backend='torch_cpu')
clf_dict[f'knn_{k}_bf_torch_cpu'] = KNNClf(k=k, d='manhattan', batch_size=(16, n_train), backend='torch_cpu')
clf_dict[f'knn_{k}_pf_torch_cpu'] = KNNClf(k=k, d='manhattan', batch_size=(n_test, 1), backend='torch_cpu')
clf_dict[f'knn_{k}_fp_torch_cpu'] = KNNClf(k=k, d='manhattan', batch_size=(1, n_train), backend='torch_cpu')
clf_dict[f'knn_{k}_ff_torch_cpu'] = KNNClf(k=k, d='manhattan', batch_size=(1, 1), backend='torch_cpu')

clf_dict[f'knn_{k}_bb_torch'] = KNNClf(k=k, d='manhattan', batch_size=(256, 1024), backend='torch')
clf_dict[f'knn_{k}_fb_torch'] = KNNClf(k=k, d='manhattan', batch_size=(n_test, 64), backend='torch')
clf_dict[f'knn_{k}_bf_torch'] = KNNClf(k=k, d='manhattan', batch_size=(16, n_train), backend='torch')
clf_dict[f'knn_{k}_fp_torch'] = KNNClf(k=k, d='manhattan', batch_size=(1, n_train), backend='torch')
clf_dict[f'knn_{k}_pf_torch'] = KNNClf(k=k, d='manhattan', batch_size=(n_test, 1), backend='torch')
clf_dict[f'knn_{k}_ff_torch'] = KNNClf(k=k, d='manhattan', batch_size=(1, 1), backend='torch')

exc = Executer(
    X_train, y_train, X_test, y_test,
    metric_list=['accuracy', 'avg_recall'],
    clf_dict=clf_dict,
    log=True,
    log_dir='./task2/'
)

exc.run_all()
