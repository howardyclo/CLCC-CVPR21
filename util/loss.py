import math
import tensorflow as tf

def angular_loss(tensor1, tensor2, return_dot_product=False, average=False):
    """ 
    Args:
        - `tensor1` and `tensor2` should be in the same shape, e.g., (?, c) or (?, h, w, c).
        - `average`: True to average loss over batch.
    Returns:
        - `angle`: (?,) if `average` == False else ()
    """
    assert len(tensor1.shape) == len(tensor2.shape)

    safe_dot_prod = 0.999999
    
    normalized_tensor1 = tf.nn.l2_normalize(tensor1, -1)
    normalized_tensor2 = tf.nn.l2_normalize(tensor2, -1)
    dot_prod = tf.reduce_sum(normalized_tensor1 * normalized_tensor2, -1)
    dot_prod = tf.clip_by_value(dot_prod, -safe_dot_prod, safe_dot_prod)
    
    angle = tf.acos(dot_prod) * (180 / math.pi)
    
    if not return_dot_product:
        return tf.reduce_mean(angle) if average else angle
    else:
        return (tf.reduce_mean(angle), tf.reduce_mean(dot_prod)) if average else angle, dot_prod

def nce_loss(a_features, p_features, n_features, num_negatives, average=True):
    # n_features (N * b, c)
    TEMPERATURE = 0.87
    TRAINING_BATCH_SIZE = tf.shape(a_features)[0]
    
    p_loss, p_scores = angular_loss(a_features, p_features, return_dot_product=True) # (b,)
    
    n_loss = 0
    n_scores_list = []
    
    for n_features_ in tf.split(n_features, num_negatives):
        n_loss_, n_score = angular_loss(a_features, n_features_, return_dot_product=True)
        n_scores_list.append(n_score)
        n_loss += n_loss_
    n_loss = n_loss/num_negatives
      
    scores = tf.stack([p_scores] + n_scores_list, axis=1) # (b, N+1) (N+1)-way classification
    scores = scores / TEMPERATURE
    onehot_labels = tf.concat([tf.ones((TRAINING_BATCH_SIZE, 1)),
                               tf.zeros((TRAINING_BATCH_SIZE, num_negatives))], axis=-1)
    losses = tf.nn.softmax_cross_entropy_with_logits(labels=onehot_labels, logits=scores)
    loss = tf.reduce_mean(losses) if average else losses

    return loss