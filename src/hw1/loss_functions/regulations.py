import jax
import jax.numpy as jnp


def l1_regularization(params, lambda_l1=0.01):  # L1正则化项
    return lambda_l1 * sum(jnp.abs(p).sum() for p in jax.tree_util.tree_leaves(params))


def l2_regularization(params, lambda_l2=0.01):  # L2正则化项
    return lambda_l2 * sum((p ** 2).sum() for p in jax.tree_util.tree_leaves(params))


def softmax(logits):
    logits_stable = logits - jnp.max(logits, axis=1, keepdims=True)
    exp_logits = jnp.exp(logits_stable)
    return exp_logits / jnp.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy_loss(y, y_pred):
    epsilon = 1e-9
    y_pred_clipped = jnp.clip(y_pred, epsilon, 1. - epsilon)  # clip here is very important, or you will get Nan when you training.
    loss = -jnp.sum(y * jnp.log(y_pred_clipped), axis=1)
    return loss.mean()
