import numpy as np
import tensorflow as tf


def get_angles(pos, i, d_model):
    angle_rates = 1 / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return pos * angle_rates


def positional_encoding(position, d_model):
    angle_rads = get_angles(np.arange(position)[:, np.newaxis],
                            np.arange(d_model)[np.newaxis, :],
                            d_model)

    # apply sin to even indices in the array; 2i
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])

    # apply cos to odd indices in the array; 2i+1
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    pos_encoding = angle_rads[np.newaxis, ...]

    return tf.cast(pos_encoding, dtype=tf.float32)


def scaled_dot_product_attention(q, k, v, mask):
    """Calculate the attention weights.
  q, k, v must have matching leading dimensions.
  k, v must have matching penultimate dimension, i.e.: seq_len_k = seq_len_v.
  The mask has different shapes depending on its type(padding or look ahead)
  but it must be broadcastable for addition.

  Args:
    q: query shape == (..., seq_len_q, depth)
    k: key shape == (..., seq_len_k, depth)
    v: value shape == (..., seq_len_v, depth_v)
    mask: Float tensor with shape broadcastable
          to (..., seq_len_q, seq_len_k). Defaults to None.

  Returns:
    output, attention_weights
  """

    matmul_qk = tf.matmul(q, k, transpose_b=True)  # (..., seq_len_q, seq_len_k)

    # scale matmul_qk
    dk = tf.cast(tf.shape(k)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # add the mask to the scaled tensor.
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax is normalized on the last axis (seq_len_k) so that the scores
    # add up to 1.
    attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)  # (..., seq_len_q, seq_len_k)
    output = tf.matmul(attention_weights, v)  # (..., seq_len_q, depth_v)

    return output, attention_weights


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0

        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)

        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x):
        heads = tf.split(x, self.num_heads, axis=-1)
        x = tf.stack(heads, axis=1)
        return x

    def call(self, v, k, q, mask=None, training=False):
        seq_len_q = tf.shape(q)[1]

        q = self.wq(q)  # (batch_size, seq_len, d_model)
        k = self.wk(k)  # (batch_size, seq_len, d_model)
        v = self.wv(v)  # (batch_size, seq_len, d_model)

        q = self.split_heads(q)  # (batch_size, num_heads, seq_len_q, depth)
        k = self.split_heads(k)  # (batch_size, num_heads, seq_len_k, depth)
        v = self.split_heads(v)  # (batch_size, num_heads, seq_len_v, depth)

        # scaled_attention.shape == (batch_size, num_heads, seq_len_q, depth)
        # attention_weights.shape == (batch_size, num_heads, seq_len_q, seq_len_k)
        scaled_attention, attention_weights = scaled_dot_product_attention(
            q, k, v, mask)

        scaled_attention = tf.transpose(scaled_attention,
                                        perm=[0, 2, 1, 3])  # (batch_size, seq_len_q, num_heads, depth)

        concat_attention = tf.reshape(scaled_attention,
                                      (-1, seq_len_q, self.d_model))  # (batch_size, seq_len_q, d_model)

        output = self.dense(concat_attention)  # (batch_size, seq_len_q, d_model)

        return output, attention_weights


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation=tf.keras.activations.gelu),  # (batch_size, seq_len, dff)
        tf.keras.layers.Dense(d_model)  # (batch_size, seq_len, d_model)
    ])


class ReZero(tf.keras.layers.Layer):
    """
    Implementation of `ReZero` activation function from:
    `ReZero is All You Need: Fast Convergence at Large Depth` (https://arxiv.org/abs/2003.04887)
    """
    def __init__(self, name):
        super(ReZero, self).__init__(name=name)
        a_init = tf.zeros_initializer()
        self.alpha = tf.Variable(name=self.name + '-alpha',
                                 initial_value=a_init(shape=(1,), dtype="float32"), trainable=True
                                 )

    def call(self, inputs):
        return self.alpha * inputs


class EncoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, rate=0.1):
        super(EncoderLayer, self).__init__()

        self.mha = MultiHeadAttention(d_model, num_heads)
        self.ffn0 = point_wise_feed_forward_network(d_model, dff)
        self.ffn1 = point_wise_feed_forward_network(d_model, dff)

        self.rz0 = ReZero(self.name + 'rz0')
        self.rz1 = ReZero(self.name + 'rz1')
        self.rz2 = ReZero(self.name + 'rz2')

    def call(self, x, training=False, mask=None):

        ffn_output = self.ffn0(x)  # (batch_size, input_seq_len, d_model)
        out0 = x + self.rz0(ffn_output)

        attn_output, attention_weights = self.mha(out0, out0, out0, mask,
                                                  training=training)  # (batch_size, input_seq_len, d_model)
        out1 = x + self.rz1(attn_output)

        ffn_output = self.ffn1(out1)  # (batch_size, input_seq_len, d_model)
        out2 = out1 + self.rz2(ffn_output)

        return out2, attention_weights


class TransformerEncoder(tf.keras.layers.Layer):
    def __init__(self, output_dim, x_attr_sizes, t_attr_sizes, y_attr_sizes,
                 num_layers, d_model, num_heads, dff):
        super(TransformerEncoder, self).__init__()
        self.d_model = d_model
        self.attr_sizes = list(x_attr_sizes) + list(t_attr_sizes)
        self.embeddings = [tf.keras.layers.Dense(d_model) for _ in self.attr_sizes]
        self.enc_layers = [EncoderLayer(d_model, num_heads, dff)
                           for _ in range(num_layers)]
        self.output_layer = point_wise_feed_forward_network(output_dim, dff)

    def call(self, x):
        x = tf.split(x, self.attr_sizes, axis=1)
        x = [tf.expand_dims(lay(inp), axis=1)
             for (lay, inp) in zip(self.embeddings, x)]
        x = tf.concat(x, axis=1) / (self.d_model ** .5)

        for lay in self.enc_layers:
            x, _ = lay(x)

        x = tf.reduce_max(x, axis=1)
        x = self.output_layer(x)
        return x
