import tensorflow as tf

def l1_normalize(x, dim, epsilon=1e-12, name=None):
  """Normalizes along dimension `dim` using an L2 norm.
  For a 1-D tensor with `dim = 0`, computes
      output = x / max(sum(abs(x)), epsilon)
  For `x` with more dimensions, independently normalizes each 1-D slice along
  dimension `dim`.
  Args:
    x: A `Tensor`.
    dim: Dimension along which to normalize.  A scalar or a vector of
      integers.
    epsilon: A lower bound value for the norm. Will use `sqrt(epsilon)` as the
      divisor if `norm < sqrt(epsilon)`.
    name: A name for this operation (optional).
  Returns:
    A `Tensor` with the same shape as `x`.
  """
  with tf.name_scope(name, "l1_normalize", [x]) as name:
    x          = tf.convert_to_tensor(x, name            = "x")
    abs_sum    = tf.reduce_sum(tf.abs(x), dim, keep_dims = True)
    x_inv_norm = tf.reciprocal(tf.maximum(abs_sum, epsilon))
    return tf.multiply(x, x_inv_norm, name=name)

