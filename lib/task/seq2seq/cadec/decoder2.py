from lib.layers.attn import MultiHeadAttn
from lib.layers.basic import *
import math


class DelNetworkDecoder2:
    def __init__(
            self, name,
            out_voc,
            *_args,

            emb_size=None, hid_size=512,
            key_size=None, value_size=None,
            ff_size=None,
            num_heads=8, num_layers=6,
            attn_dropout=0.0, attn_value_dropout=0.0, relu_dropout=0.0, res_dropout=0.1,
            rescale_emb=False,
            dst_reverse=False, dst_rand_offset=False,
            res_steps='nlda', normalize_out=False, multihead_attn_format='v1',
            dec1_attn_mode='rdo_and_emb',
            emb_out_device='',
            emb_out_matrix=None,
            reuse=False,
            max_ctx_sents=5,
            use_dst_ctx=None,
            **_kwargs
    ):

        if isinstance(ff_size, str):
            ff_size = [int(i) for i in ff_size.split(':')]

        if _args:
            raise Exception("Unexpected positional arguments")

        emb_size = emb_size if emb_size else hid_size
        key_size = key_size if key_size else hid_size
        value_size = value_size if value_size else hid_size
        if key_size % num_heads != 0:
            raise Exception("Bad number of heads")
        if value_size % num_heads != 0:
            raise Exception("Bad number of heads")

        self.name = name
        self.out_voc = out_voc
        self.num_layers_dec = num_layers
        self.res_dropout = res_dropout
        self.emb_size = emb_size
        self.hid_size = hid_size
        self.rescale_emb = rescale_emb
        self.dst_reverse = dst_reverse
        self.dst_rand_offset = dst_rand_offset
        self.normalize_out = normalize_out
        self.dec1_attn_mode = dec1_attn_mode
        self.relu_dropout = relu_dropout

        self.max_ctx_sents = max_ctx_sents
        self.use_dst_ctx = use_dst_ctx
        print("DEC2: use_dst_ctx", self.use_dst_ctx)
        assert self.use_dst_ctx is not None

        if "dec2_layers" in _kwargs:
            self.num_layers_dec = _kwargs['dec2_layers']

        with tf.variable_scope(name, reuse=reuse):
            max_voc_size = out_voc.size()

            self.emb_out = Embedding(
                'emb_out', out_voc.size(), emb_size,
                matrix=emb_out_matrix,
                initializer=tf.random_normal_initializer(0, emb_size ** -.5),
                device=emb_out_device)

            # Decoder layers
            self.dec_attn = [ResidualLayerWrapper(
                'dec_attn-%i' % i,
                MultiHeadAttn(
                    'dec_attn-%i' % i,
                    inp_size=emb_size if i == 0 else hid_size,
                    key_depth=key_size,
                    value_depth=value_size,
                    output_depth=hid_size,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    attn_value_dropout=attn_value_dropout),
                inp_size=emb_size if i == 0 else hid_size,
                out_size=emb_size if i == 0 else hid_size,
                steps=res_steps,
                dropout=res_dropout)
                for i in range(self.num_layers_dec)]

            self.dec_enc_attn = [ResidualLayerWrapper(
                'dec_enc_attn-%i' % i,
                MultiHeadAttn(
                    'dec_enc_attn-%i' % i,
                    inp_size=emb_size if i == 0 else hid_size,
                    kv_inp_size=self.emb_size + self.max_ctx_sents + 1,
                    key_depth=key_size,
                    value_depth=value_size,
                    output_depth=hid_size,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    attn_value_dropout=attn_value_dropout,
                    _format='use_kv' if multihead_attn_format == 'v1' else 'combined',
                ),
                inp_size=emb_size if i == 0 else hid_size,
                out_size=emb_size if i == 0 else hid_size,
                steps=res_steps,
                dropout=res_dropout)
                for i in range(self.num_layers_dec)]

            self.dec_ffn = [ResidualLayerWrapper(
                'dec_ffn-%i' % i,
                FFN(
                    'dec_ffn-%i' % i,
                    inp_size=emb_size if i == 0 else hid_size,
                    hid_size=ff_size if ff_size else hid_size,
                    out_size=hid_size,
                    relu_dropout=relu_dropout),
                inp_size=emb_size if i == 0 else hid_size,
                out_size=hid_size,
                steps=res_steps,
                dropout=res_dropout)
                for i in range(self.num_layers_dec)]

            if self.normalize_out:
                self.dec_out_norm = LayerNorm('dec_out_norm',
                                              inp_size=emb_size if self.num_layers_dec == 0 else hid_size)

        with tf.variable_scope(name, reuse=False):
            self.dec_dec_attn = [ResidualLayerWrapper(
                'dec_dec_attn-%i' % i,
                MultiHeadAttn(
                    'dec_dec_attn-%i' % i,
                    inp_size=emb_size if i == 0 else hid_size,
                    key_depth=key_size,
                    value_depth=value_size,
                    output_depth=hid_size,
                    num_heads=num_heads,
                    attn_dropout=attn_dropout,
                    attn_value_dropout=attn_value_dropout,
                    kv_inp_size=emb_size + (
                            self.max_ctx_sents + 1) * self.use_dst_ctx if dec1_attn_mode == 'emb_only' else
                    hid_size + self.max_ctx_sents + 1 if dec1_attn_mode == 'rdo_only' else
                    emb_size + hid_size + (self.max_ctx_sents + 1) * self.use_dst_ctx,
                    _format='use_kv' if multihead_attn_format == 'v1' else 'combined',
                ),
                inp_size=emb_size if i == 0 else hid_size,
                out_size=emb_size if i == 0 else hid_size,
                steps=res_steps,
                dropout=res_dropout)
                for i in range(self.num_layers_dec)]

    def _make_enc_attn_mask(self, inp, inp_len, dtype=tf.float32):
        """
        inp = [batch_size * ninp]
        inp_len = [batch_size]

        attn_mask = [batch_size * 1 * 1 * ninp]
        """
        with tf.variable_scope("make_enc_attn_mask"):
            inp_mask = tf.sequence_mask(inp_len, dtype=dtype, maxlen=tf.shape(inp)[1])

            attn_mask = inp_mask[:, None, None, :]
            return attn_mask

    def _make_dec_attn_mask(self, out, dtype=tf.float32):
        """
        out = [baatch_size * nout]

        attn_mask = [1 * 1 * nout * nout]
        """
        with tf.variable_scope("make_dec_attn_mask"):
            length = tf.shape(out)[1]
            lower_triangle = tf.matrix_band_part(tf.ones([length, length], dtype=dtype), -1, 0)
            attn_mask = tf.reshape(lower_triangle, [1, 1, length, length])
            return attn_mask

    def _add_timing_signal(self, inp, min_timescale=1.0, max_timescale=1.0e4, offset=0, inp_reverse=None):
        """
        inp: (batch_size * ninp * hid_dim)
        :param offset: add this number to all character positions.
            if offset == 'random', picks this number uniformly from [-32000,32000] integers
        :type offset: number, tf.Tensor or 'random'
        """
        with tf.variable_scope("add_timing_signal"):
            ninp = tf.shape(inp)[1]
            hid_size = tf.shape(inp)[2]

            position = tf.to_float(tf.range(ninp))[None, :, None]

            if offset == 'random':
                BIG_LEN = 32000
                offset = tf.random_uniform(tf.shape(position), minval=-BIG_LEN, maxval=BIG_LEN, dtype=tf.int32)

            # force broadcasting over batch axis
            if isinstance(offset * 1, tf.Tensor):  # multiply by 1 to also select variables, special generators, etc.
                assert offset.shape.ndims in (0, 1, 2)
                new_shape = [tf.shape(offset)[i] for i in range(offset.shape.ndims)]
                new_shape += [1] * (3 - len(new_shape))
                offset = tf.reshape(offset, new_shape)

            position += tf.to_float(offset)

            if inp_reverse is not None:
                position = tf.multiply(
                    position,
                    tf.where(
                        tf.equal(inp_reverse, 0),
                        tf.ones_like(inp_reverse, dtype=tf.float32),
                        -1.0 * tf.ones_like(inp_reverse, dtype=tf.float32)
                    )[:, None, None]  # (batch_size * ninp * dim)
                )
            num_timescales = hid_size // 2
            log_timescale_increment = (
                    math.log(float(max_timescale) / float(min_timescale)) /
                    (tf.to_float(num_timescales) - 1))
            inv_timescales = min_timescale * tf.exp(
                tf.to_float(tf.range(num_timescales)) * -log_timescale_increment)

            # scaled_time: [ninp * hid_dim]
            scaled_time = position * inv_timescales[None, None, :]
            signal = tf.concat([tf.sin(scaled_time), tf.cos(scaled_time)], axis=-1)
            signal = tf.pad(signal, [[0, 0], [0, 0], [0, tf.mod(hid_size, 2)]])
            return inp + signal

    def decode(self, out, out_len, enc_out, enc_attn_mask,
               dec1_out, dec1_attn_mask, dec1_rdo, is_train, **kwargs):
        with dropout_scope(is_train), tf.name_scope(self.name + '_dec') as scope:

            # Embeddings
            emb_out = self.emb_out(out)  # [batch_size * nout * emb_dim]
            if self.rescale_emb:
                emb_out *= self.emb_size ** .5

            # Shift right; drop embedding for last word
            emb_out = tf.pad(emb_out, [[0, 0], [1, 0], [0, 0]])[:, :-1, :]

            # Prepare decoder
            dec_attn_mask = self._make_dec_attn_mask(out)  # [1 * 1 * nout * nout]

            offset = 'random' if self.dst_rand_offset else 0
            dec_inp = self._add_timing_signal(emb_out, offset=offset)
            # Apply dropouts
            if is_dropout_enabled():
                dec_inp = tf.nn.dropout(dec_inp, 1.0 - self.res_dropout)

            if self.dec1_attn_mode == 'rdo_only':
                dec1_attn_source = dec1_rdo
            elif self.dec1_attn_mode == 'emb_only':
                dec1_attn_source = self.emb_out(dec1_out)
            elif self.dec1_attn_mode == 'rdo_and_emb':
                dec1_attn_source = tf.concat([self.emb_out(dec1_out), dec1_rdo], axis=-1)
            else:
                raise ValueError("dec1_attn_source must be either of 'rdo_only', 'emb_only', 'rdo_and_emb'")

            # Decoder
            for layer in range(self.num_layers_dec):
                dec_inp = self.dec_attn[layer](dec_inp, dec_attn_mask)
                dec_inp = self.dec_enc_attn[layer](dec_inp, enc_attn_mask, enc_out)
                dec_inp = self.dec_dec_attn[layer](dec_inp, dec1_attn_mask, dec1_attn_source)
                dec_inp = self.dec_ffn[layer](dec_inp)

            if self.normalize_out:
                dec_inp = self.dec_out_norm(dec_inp)

            dec_out = dec_inp

            _out = dec_out
            tf.add_to_collection(lib.meta.ACTIVATIONS, tf.identity(_out, name=scope))

            return _out

