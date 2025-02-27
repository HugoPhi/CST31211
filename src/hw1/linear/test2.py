# from jax import grad, vamp, jit
import jax.numpy as jnp


def relu_svm(score, y):
    # delta = 1 for convenient here
    margins = jnp.maximum(0, score - score[jnp.arange(y.shape[0]), y][:, None] + 1)
    margins = margins.at[jnp.arange(y.shape[0]), y].set(0)

    return jnp.sum(margins, axis=1)


socre = jnp.array([[3.2, 5.1, -1.7],
                  [1.3, 4.9, 2.0],
                  [2.2, 2.5, -3.1]])

y = jnp.array([0, 1, 2])
print(relu_svm(socre, y))
