import jax.numpy as jnp


def svm(logits, y, delta):
    margins = jnp.maximum(0, logits - logits[jnp.arange(y.shape[0]), y][:, None] + delta)
    margins = margins.at[jnp.arange(y.shape[0]), y].set(0)

    return jnp.sum(margins, axis=1)
