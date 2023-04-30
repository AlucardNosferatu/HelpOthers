import tensorflow as tf
import tensorflow_addons as tfa


def apply_seq(x, layers):
    for layer in layers:
        x = layer(x)
    return x


def td_dot(a, b):
    aa = tf.reshape(a, (-1, a.shape[2], a.shape[3]))
    bb = tf.reshape(b, (-1, b.shape[2], b.shape[3]))
    cc = tf.keras.backend.batch_dot(aa, bb)
    return tf.reshape(cc, (-1, a.shape[1], cc.shape[1], cc.shape[2]))


def gelu(x):
    tanh_res = tf.keras.activations.tanh(x * 0.7978845608 * (1 + 0.044715 * (x ** 2)))
    return 0.5 * x * (1 + tanh_res)


class PaddedConv2D(tf.keras.layers.Layer):
    def __init__(self, channels, kernel_size, padding=0, stride=1):
        super().__init__()
        self.padding2d = tf.keras.layers.ZeroPadding2D((padding, padding))
        self.conv2d = tf.keras.layers.Conv2D(
            channels, kernel_size, strides=(stride, stride)
        )

    def call(self, x, **kwargs):
        x = self.padding2d(x)
        return self.conv2d(x)


class ResBlock(tf.keras.layers.Layer):
    def __init__(self, channels, out_channels):
        super().__init__()
        self.in_layers = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
            tf.keras.activations.swish,
            PaddedConv2D(out_channels, 3, padding=1),
        ]
        self.emb_layers = [
            tf.keras.activations.swish,
            tf.keras.layers.Dense(out_channels),
        ]
        self.out_layers = [
            tfa.layers.GroupNormalization(epsilon=1e-5),
            tf.keras.activations.swish,
            PaddedConv2D(out_channels, 3, padding=1),
        ]
        self.skip_connection = (
            PaddedConv2D(out_channels, 1) if channels != out_channels else lambda x: x
        )

    def call(self, inputs, **kwargs):
        x, emb = inputs
        h = apply_seq(x, self.in_layers)
        emb_out = apply_seq(emb, self.emb_layers)
        h = h + emb_out[:, None, None]
        h = apply_seq(h, self.out_layers)
        ret = self.skip_connection(x) + h
        return ret


class CrossAttention(tf.keras.layers.Layer):
    def __init__(self, n_heads, d_head):
        super().__init__()
        self.to_q = tf.keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.to_k = tf.keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.to_v = tf.keras.layers.Dense(n_heads * d_head, use_bias=False)
        self.scale = d_head ** -0.5
        self.num_heads = n_heads
        self.head_size = d_head
        self.to_out = [tf.keras.layers.Dense(n_heads * d_head)]

    def call(self, inputs, **kwargs):
        assert type(inputs) is list
        if len(inputs) == 1:
            inputs = inputs + [None]
        x, context = inputs
        context = x if context is None else context
        q, k, v = self.to_q(x), self.to_k(context), self.to_v(context)
        assert len(x.shape) == 3
        q = tf.reshape(q, (-1, x.shape[1], self.num_heads, self.head_size))
        k = tf.reshape(k, (-1, context.shape[1], self.num_heads, self.head_size))
        v = tf.reshape(v, (-1, context.shape[1], self.num_heads, self.head_size))

        q = tf.keras.layers.Permute((2, 1, 3))(q)  # (bs, num_heads, time, head_size)
        k = tf.keras.layers.Permute((2, 3, 1))(k)  # (bs, num_heads, head_size, time)
        v = tf.keras.layers.Permute((2, 1, 3))(v)  # (bs, num_heads, time, head_size)

        score = td_dot(q, k) * self.scale
        weights = tf.keras.activations.softmax(score)  # (bs, num_heads, time, time)
        attention = td_dot(weights, v)
        attention = tf.keras.layers.Permute((2, 1, 3))(
            attention
        )  # (bs, time, num_heads, head_size)
        h_ = tf.reshape(attention, (-1, x.shape[1], self.num_heads * self.head_size))
        return apply_seq(h_, self.to_out)


class GEGLU(tf.keras.layers.Layer):
    def __init__(self, dim_out):
        super().__init__()
        self.proj = tf.keras.layers.Dense(dim_out * 2)
        self.dim_out = dim_out

    def call(self, x, **kwargs):
        xp = self.proj(x)
        x, gate = xp[..., : self.dim_out], xp[..., self.dim_out:]
        return x * gelu(gate)


# todo
class BasicTransformerBlock(tf.keras.layers.Layer):
    def __init__(self, dim, n_heads, d_head):
        super().__init__()
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn1 = CrossAttention(n_heads, d_head)

        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.attn2 = CrossAttention(n_heads, d_head)

        self.norm3 = tf.keras.layers.LayerNormalization(epsilon=1e-5)
        self.geglu = GEGLU(dim * 4)
        self.dense = tf.keras.layers.Dense(dim)

    def call(self, inputs, **kwargs):
        x, context = inputs
        x = self.attn1([self.norm1(x)]) + x
        x = self.attn2([self.norm2(x), context]) + x
        return self.dense(self.geglu(self.norm3(x))) + x


class SpatialTransformer(tf.keras.layers.Layer):
    def __init__(self, channels, n_heads, word_vec_dim):
        super().__init__()
        self.norm = tfa.layers.GroupNormalization(epsilon=1e-5)
        assert channels == n_heads * word_vec_dim
        self.proj_in = PaddedConv2D(n_heads * word_vec_dim, 1)
        self.transformer_blocks = [
            BasicTransformerBlock(channels, n_heads, word_vec_dim)
        ]
        self.proj_out = PaddedConv2D(channels, 1)

    def call(self, inputs, **kwargs):
        x, context = inputs
        b, h, w, c = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = tf.reshape(x, (-1, h * w, c))
        for block in self.transformer_blocks:
            x = block([x, context])
        x = tf.reshape(x, (-1, h, w, c))
        return self.proj_out(x) + x_in
