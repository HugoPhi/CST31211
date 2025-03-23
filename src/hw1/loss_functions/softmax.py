import jax.numpy as jnp


def softmax(logits: jnp.ndarray):
    logits_stable = logits - jnp.max(logits, axis=1, keepdims=True)
    exp_logits = jnp.exp(logits_stable)
    return exp_logits / jnp.sum(exp_logits, axis=1, keepdims=True)


def cross_entropy_loss(y: jnp.ndarray, y_pred: jnp.ndarray):
    '''
    Calculate Cross Entropy Loss of given true label(one-hot) & predict proba.

    Input
    -----
    y: true label(one-hot)
    y_pred: predict proba

    Output
    ------
    jnp.float32, Cross Entropy Loss
    '''

    epsilon = 1e-9
    y_pred_clipped = jnp.clip(y_pred, epsilon, 1. - epsilon)  # clip here is very important, or you will get Nan when you training.
    loss = -jnp.sum(y * jnp.log(y_pred_clipped), axis=1)
    return loss.mean()
