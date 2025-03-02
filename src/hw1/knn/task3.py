from plugins.lrkit.executer import KFlodCrossExecuter

from knn import KNNClf
from data_process import X_train, X_test, y_test, y_train

n_train = 50000
n_test = 10000
X_train = X_train[:n_train]
y_train = y_train[:n_train]
X_test = X_test[:n_test]
y_test = y_test[:n_test]

clf_dict = {}
for k in [x for x in range(1, 20)] + [x for x in range(20, 220, 20)]:
    clf_dict[f'knn_{k}'] = KNNClf(k=k, d='manhattan', batch_size=(1000, 1), backend='torch')

exc = KFlodCrossExecuter(
    X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
    metric_list=['accuracy', 'avg_recall'],
    k=5,
    clf_dict=clf_dict,
    log=True,
    log_dir='./task3/'
)

exc.run_all()
