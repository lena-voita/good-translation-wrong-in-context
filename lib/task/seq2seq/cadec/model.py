import tensorflow as tf

import lib.task.seq2seq.data
import lib.task.seq2seq.models.transformer as transformer_module
from lib.task.seq2seq.cadec.decoder2 import DelNetworkDecoder2
from lib.layers.basic import LossXentLm
from lib.ops.basic import infer_length
from collections import namedtuple
from lib.layers.basic import dropout_scope
from lib.task.seq2seq.problems.default import word_dropout

import numpy as np


class CADecModel(lib.task.seq2seq.models.TranslateModelBase):
    DecState = namedtuple("deliberation_state", ["dec2_state", "dec1_out", "dec1_attn_mask", "dec1_rdo"])

    def __init__(self, name, inp_voc, out_voc,
                 model1_name='model1', decoder2_name='decoder2',
                 get_dec1_max_len=lambda inp_len, out_len: tf.to_int32(tf.to_float(out_len) * 1.2) + 3,
                 **hp):
        self.name = name
        self.inp_voc = inp_voc
        self.out_voc = out_voc
        self.hp = hp

        self.max_ctx_sents = hp.get('max_ctx_sents', 5)
        self.get_dec1_max_len = get_dec1_max_len
        self.use_dst_ctx = hp['use_dst_ctx']
        print("MODEL: use_dst_ctx ", self.use_dst_ctx)

        # Base model
        with tf.variable_scope(name, reuse=False):
            self.model1 = transformer_module.Model(model1_name, inp_voc, out_voc, **hp)
            emb_out_matrix = self.model1.transformer.emb_out.mat

        with tf.variable_scope(name):
            print('reusing...')
            self.decoder2 = DelNetworkDecoder2(decoder2_name, out_voc,
                                               emb_out_matrix=emb_out_matrix,
                                               reuse=model1_name == decoder2_name,
                                               **hp)

            projection_matrix = logits_bias = None
            if hp.get('dwwt', False):
                projection_matrix = tf.transpose(self.model1.transformer.emb_out.mat)
            elif hp.get('share_loss', True):
                projection_matrix = self.model1.loss._rdo_to_logits.W
                logits_bias = self.model1.loss._rdo_to_logits.b

            self.loss = LossXentLm(
                hp.get('loss2_name', 'loss2_xent_lm'),
                hp['hid_size'],
                out_voc,
                hp,
                matrix=projection_matrix,
                bias=logits_bias if hp.get("loss_bias", False) else 0)

    def make_feed_dict(self, batch, **kwargs):
        """Data in the format:
        batch - pairs (inp, out)
        inp = src _eos ctx_{n-1} _eos ... ctx_1 _eos_eos dst_ctx_{n-1} _eos ... dst_ctx_1
        out = dst"""

        # Sentinels.
        ibos = self.inp_voc.bos
        ieos = self.inp_voc.eos
        obos = self.out_voc.bos
        oeos = self.out_voc.eos

        splitter = '_eos'
        splitter_dst = '_eos_eos'

        def ids_from_sent(sent, use_inp_voc=True):
            src_words = sent.strip().split(' ')
            inp = []
            if self.hp.get('force_bos', False):
                inp.append(ibos if use_inp_voc else obos)
            if use_inp_voc:
                inp += self.inp_voc.ids(src_words) + [ieos]
            else:
                inp += self.out_voc.ids(src_words) + [oeos]
            return inp

        # Read as-is, without padding.
        inps = []
        outs = []
        ctx_sents = []
        ctx_sents_out = []
        ctx_sent_diff = []
        out_reverse = []
        for inp, out in batch:
            inps.append(ids_from_sent(inp.split(splitter_dst)[0].split(splitter)[0]))
            ctx_sents_ = inp.split(splitter_dst)[0].split(splitter)[1:]
            if self.use_dst_ctx:
                ctx_sents_out_ = inp.split(splitter_dst)[1].split(splitter)
                assert len(ctx_sents_) == len(ctx_sents_out_)
            else:
                ctx_sents_out_ = [''] * len(ctx_sents_)

            ctx_sents.append([])
            ctx_sents_out.append([])
            ctx_sent_diff.append([])
            for i in range(len(ctx_sents_)):
                ctx_sents[-1].append(ids_from_sent(ctx_sents_[i]))
                ctx_sents_out[-1].append(ids_from_sent(ctx_sents_out_[i], use_inp_voc=False))
                ctx_sent_diff[-1].append(len(ctx_sents_) - i)

            dst_words = out.strip().split(' ')

            reverse = 0
            if self.hp.get('dst_reverse', 0) == 1:
                reverse = 1
            elif self.hp.get('dst_reverse', 0) == 'rand':
                reverse = np.random.randint(2)

            if reverse:
                dst_words = list(reversed(dst_words))
            out_reverse.append(reverse)

            out = []
            if self.hp.get('force_bos', False):
                out.append(obos)
            out += self.out_voc.ids(dst_words) + [oeos]

            outs.append(out)

        # Pad and transpose.
        inps, inp_len = self._pad(inps, ieos)
        ctx_sents, ctx_lens, ctx_sent_lens = self._pad_large(ctx_sents, ieos)
        ctx_sents_out, ctx_lens_out, ctx_sent_lens_out = self._pad_large(ctx_sents_out, ieos)
        outs, out_len = self._pad(outs, oeos)
        ctx_sent_diff, _ = self._pad(ctx_sent_diff, 0)

        # Return.
        return {
            'inp': np.array(inps).astype('int32'),
            'out': np.array(outs).astype('int32'),
            'inp_len': np.array(inp_len).astype('int32'),
            'out_len': np.array(out_len).astype('int32'),
            'ctx_sents': np.array(ctx_sents).astype('int32'),
            'ctx_lens': np.array(ctx_lens).astype('int32'),
            'ctx_sent_lens': np.array(ctx_sent_lens).astype('int32'),
            'out_reverse': np.array(out_reverse).astype('int32'),
            'ctx_sent_diff': np.array(ctx_sent_diff).astype('int32'),

            'ctx_out_sents': np.array(ctx_sents_out).astype('int32'),
            'ctx_out_sent_lens': np.array(ctx_sent_lens_out).astype('int32'),
        }

    def _get_batch_sample(self):
        print("new batch sample")
        return [(
                "i fed the cat _eos i saw a cat _eos it was hungry _eos_eos я видел кота _eos он был голодный",
                "я покормил кота")]

    # ==============================================================================================================
    #   DEAL WITH CONTEXT
    # ==============================================================================================================
    def encode_decode_ctx_large(self, ctxs, ctx_sent_lens, ctx_sent_diff,
                                ctxs_out, ctxs_out_len,
                                is_train):
        batch, nsents, ninp = tf.shape(ctxs)[0], tf.shape(ctxs)[1], tf.shape(ctxs)[2]
        nout = tf.shape(ctxs_out)[2]

        # ctxs: [batch, nsents, ninp]

        # -------- ENCODE SEPARATELY -------------------------------------------
        ctxs = tf.reshape(ctxs, [-1, ninp])  # [batch * nsents, ninp]
        ctx_sent_lens = tf.reshape(ctx_sent_lens, [-1])  # lens: [batch * nsents]

        sents, ctx_attn_mask = self.model1.transformer.encode(ctxs, ctx_sent_lens, is_train=is_train)

        #  sents:           [batch * nsents, ninp, vec_size]
        #  ctx_attn_mask :  [batch * nsents, 1, 1, ninp]

        # -------- DECODE SEPARATELY -------------------------------------------

        if self.use_dst_ctx:
            ctxs_out = tf.reshape(ctxs_out, [-1, nout])  # [batch * nsents, nout]
            ctxs_out_to_out = tf.reshape(ctxs_out, [batch, -1])  # [batch, nsents * nout]
            ctxs_out_len = tf.reshape(ctxs_out_len, [-1])  # lens: [batch * nsents]

            dec_out = self.model1.transformer.decode(ctxs_out, ctxs_out_len,
                                                     None,  # out_reverse - DUMMY
                                                     sents, ctx_attn_mask,  # enc_out, enc_attn_mask
                                                     is_train=is_train)

            dec_out_attn_mask = self.model1.transformer._make_enc_attn_mask(ctxs_out, ctxs_out_len)

        # ------ RESHAPE ALL FROM ENCODER --------------------------------------------------

        sents = tf.reshape(sents, [batch, nsents, ninp, -1])  # [batch, nsents, ninp, vec_size]

        # ctx_sent_diff: batch, nsents
        sent_diff_embs = tf.one_hot(ctx_sent_diff, depth=self.max_ctx_sents + 1)  # [batch, nsents, max_sents + 1]
        sent_diff_embs = tf.expand_dims(sent_diff_embs, 2)  # [batch, nsents, 1, max_sents + 1]
        sent_diff_embs_enc = tf.tile(sent_diff_embs,
                                     [1, 1, tf.shape(sents)[2], 1])  # [batch, nsents, ninp, max_sents + 1]

        sents = tf.concat([sents, sent_diff_embs_enc], axis=-1)  # [batch, nsents, ninp, vec_size + max_sents + 1]

        ctx_attn_mask = tf.reshape(ctx_attn_mask, [batch, nsents, 1, 1, -1])  # [batch, nsents, 1, 1, ninp]
        sents = tf.reshape(sents, [batch, -1, tf.shape(sents)[3]])  # [batch, nsents * ninp, vec_size + max_sents + 1]

        ctx_attn_mask = tf.transpose(ctx_attn_mask, [0, 2, 3, 1, 4])  # [batch, 1, 1, nsents, ninp]
        ctx_attn_mask = tf.reshape(ctx_attn_mask, [batch, 1, 1, -1])  # [batch, 1, 1, nsents * ninp]

        # ------ RESHAPE ALL FROM DECODER --------------------------------------------------

        if self.use_dst_ctx:
            dec_out = tf.reshape(dec_out, [batch, nsents, nout, -1])  # [batch, nsents, nout, vec_size]

            sent_diff_embs_dec = tf.tile(sent_diff_embs,
                                         [1, 1, tf.shape(dec_out)[2], 1])  # [batch, nsents, nout, max_sents + 1]

            dec_out = tf.concat([dec_out, sent_diff_embs_dec],
                                axis=-1)  # [batch, nsents, ninp, vec_size + max_sents + 1]

            dec_out_attn_mask = tf.reshape(dec_out_attn_mask, [batch, nsents, 1, 1, -1])  # [batch, nsents, 1, 1, nout]
            dec_out = tf.reshape(dec_out,
                                 [batch, -1, tf.shape(dec_out)[3]])  # [batch, nsents * nout, vec_size + max_sents + 1]

            dec_out_attn_mask = tf.transpose(dec_out_attn_mask, [0, 2, 3, 1, 4])  # [batch, 1, 1, nsents, nout]
            dec_out_attn_mask = tf.reshape(dec_out_attn_mask, [batch, 1, 1, -1])  # [batch, 1, 1, nsents * nout]
        else:
            dec_out = None
            dec_out_attn_mask = None
            ctxs_out_to_out = None

        return sents, ctx_attn_mask, dec_out, dec_out_attn_mask, ctxs_out_to_out

    # ==============================================================================================================

    def encode(self, batch, is_train=False,
               decoder1_flags={},
               inference_flags={},
               dec1_word_dropout_params={},
               denoising=0,
               denoising_word_dropout_params={},
               **kwargs):
        model1, decoder2 = self.model1, self.decoder2
        inp_voc, out_voc = self.inp_voc, self.out_voc
        inp, out = batch['inp'], batch['out']
        inp_len = batch.get('inp_len', infer_length(inp, inp_voc.eos))

        # ================= ENCODE_DECODE CONTEXT =========================================================
        ctx_enc_out, ctx_enc_attn_mask, ctx_dec_out, ctx_dec_out_attn_mask, ctxs_out_to_out = \
            self.encode_decode_ctx_large(batch['ctx_sents'],
                                         batch['ctx_sent_lens'],
                                         batch['ctx_sent_diff'],
                                         batch['ctx_out_sents'],
                                         batch['ctx_out_sent_lens'],
                                         is_train=False)

        # =================================================================================================

        def sampled_translation():
            dec1_out = model1.symbolic_translate(batch,
                                                 **inference_flags,
                                                 ).best_out
            dec1_out_len = infer_length(dec1_out, out_voc.eos)
            if is_train:
                dec1_out = word_dropout(dec1_out, dec1_out_len,
                                        dec1_word_dropout_params.get("dropout", 0),
                                        dec1_word_dropout_params.get("method", "random_word"),
                                        out_voc)
            return dec1_out

        with dropout_scope(is_train), tf.name_scope(self.model1.transformer.name):
            dec1_out_sampled = sampled_translation()
            dec1_denoising = word_dropout(batch['out'], batch['out_len'],
                                          denoising_word_dropout_params.get("dropout", 0),
                                          denoising_word_dropout_params.get("method", "random_word"),
                                          out_voc)

            if is_train and denoising > 0:

                cond = tf.less_equal(tf.random_uniform([]), denoising)

                dec1_out = tf.cond(cond,
                                   lambda: dec1_denoising,
                                   lambda: dec1_out_sampled,
                                   )
            else:
                dec1_out = dec1_out_sampled


            dec1_out_len = infer_length(dec1_out, out_voc.eos)

            enc_out, enc_attn_mask = model1.transformer.encode(inp, inp_len, is_train)
            dec1_rdo = model1.transformer.decode(dec1_out, dec1_out_len, None, enc_out, enc_attn_mask, is_train)
            # [batch, nout, vec_size]

            # ======================= CONTEXT in DECODER ==================================================================
            sent_diff_embs = tf.one_hot(tf.zeros_like(batch['ctx_sent_diff'][:, 0]),
                                        depth=self.max_ctx_sents + 1)  # [batch, max_sents + 1]
            sent_diff_embs = tf.expand_dims(sent_diff_embs, 1)  # [batch, 1, max_sents + 1]
            # add sent_diff emb
            if self.use_dst_ctx:
                sent_diff_embs_dec1 = tf.tile(sent_diff_embs,
                                              [1, tf.shape(dec1_out)[1], 1])  # [batch, nout, max_sents + 1]

                dec1_rdo = tf.concat([dec1_rdo, sent_diff_embs_dec1],
                                     axis=-1)  # [batch, nout, vec_size + max_sents + 1]

            # concat with ctx dec
            dec1_attn_mask = self.model1.transformer._make_enc_attn_mask(dec1_out,
                                                                         dec1_out_len)  # [batch * 1 * 1 * nout] (???)

            if self.use_dst_ctx:
                dec1_rdo = tf.concat([ctx_dec_out, dec1_rdo],
                                     axis=1)  # [batch, nsents * nout + nout, vec_size + max_sents + 1]
                dec1_attn_mask = tf.concat([ctx_dec_out_attn_mask, dec1_attn_mask],
                                           axis=-1)  # [batch, 1, 1, nsents * nout + nout]

                dec1_out = tf.concat([ctxs_out_to_out, dec1_out], axis=-1)  # [batch, nsents * nout + nout]

            # ======================= CONTEXT in ENCODER ==============================================================
            # add sent diff emb
            sent_diff_embs_enc = tf.tile(sent_diff_embs, [1, tf.shape(inp)[1], 1])  # [batch, ninp, max_sents + 1]
            enc_out = tf.concat([enc_out, sent_diff_embs_enc], axis=-1)  # [batch, ninp, vec_size + max_sents + 1]

            # concat with ctx enc
            enc_out = tf.concat([ctx_enc_out, enc_out],
                                axis=1)  # [batch, nsents * ninp + ninp, vec_size + max_sents + 1]
            enc_attn_mask = tf.concat([ctx_enc_attn_mask, enc_attn_mask],
                                      axis=-1)  # [batch, 1, 1, nsents * ninp + ninp]
            # =========================================================================================================

            # Decoder dummy input/output
            ninp = tf.shape(inp)[1]
            batch_size = tf.shape(inp)[0]
            hid_size = tf.shape(enc_out)[-1]
            out_seq = tf.zeros([batch_size, 0], dtype=inp.dtype)
            rdo = tf.zeros([batch_size, hid_size], dtype=enc_out.dtype)

            attnP = tf.ones([batch_size, ninp]) / tf.to_float(inp_len)[:, None]

            return self._decode_impl((enc_out, enc_attn_mask, dec1_out, dec1_rdo, dec1_attn_mask,
                                      attnP, out_seq, rdo), **kwargs)

    def decode(self, dec_state, words, **kwargs):
        """
        Performs decoding step given words

        words: [batch_size]
        """
        with tf.name_scope(self.decoder2.name):
            (enc_out, enc_attn_mask, dec1_out, dec1_rdo, dec1_attn_mask,
             attnP, prev_out_seq, rdo) = dec_state

            out_seq = tf.concat([prev_out_seq, tf.expand_dims(words, 1)], 1)
            return self._decode_impl((enc_out, enc_attn_mask, dec1_out, dec1_rdo, dec1_attn_mask,
                                      attnP, out_seq, rdo), **kwargs)

    def _decode_impl(self, dec_state, is_train=False, **kwargs):
        (enc_out, enc_attn_mask, dec1_out, dec1_rdo, dec1_attn_mask, attnP, out_seq, rdo) = dec_state

        with dropout_scope(is_train):
            out = tf.pad(out_seq, [(0, 0), (0, 1)])
            out_len = tf.fill(dims=(tf.shape(out)[0],), value=tf.shape(out_seq)[1])

            dec_out = self.decoder2.decode(out, out_len, enc_out, enc_attn_mask,
                                           dec1_out, dec1_attn_mask, dec1_rdo,
                                           is_train=is_train, **kwargs)

            rdo = dec_out[:, -1, :]  # [batch_size * hid_dim]

            attnP = enc_attn_mask[:, 0, 0, :]  # [batch_size * ninp ]
            attnP /= tf.reduce_sum(attnP, axis=1, keep_dims=True)

            return (enc_out, enc_attn_mask, dec1_out, dec1_rdo, dec1_attn_mask,
                    attnP, out_seq, rdo)

    def get_rdo(self, dec_state, **kwargs):
        rdo = dec_state[-1]
        out = dec_state[-2]
        return rdo, out

    def get_attnP(self, dec_state, **kwargs):
        return dec_state[-3]

    def _pad(self, array, sentinel, max_len=None):
        """
        Add padding, compose lengths
        """
        # Compute max length.
        maxlen = 0
        for seq in array:
            maxlen = max(maxlen, len(seq))

        if max_len is not None:
            maxlen = max(maxlen, max_len)

        # Pad.
        padded = []
        lens = []
        for seq in array:
            padding = maxlen - len(seq)
            padded.append(seq + [sentinel] * padding)
            lens.append(len(seq))

        return padded, lens

    def _pad_large(self, arrays, sentinel):
        """
        Add padding, compose lengths
        For array of arrays (contexts of different lens)
        arrays - list of ctx, ctx - list of sents, sent - list of tokens
        dummy_sent - list of tokens
        sentinel - pad for each sent
        """
        # Compute max length.
        maxlen_ctx = 0
        maxlen_sent = 0
        for array in arrays:
            maxlen_ctx = max(maxlen_ctx, len(array))
            for seq in array:
                maxlen_sent = max(maxlen_sent, len(seq))

        # Pad contexts
        ctx_lens = []
        ctx_sent_lens = []
        padded_ctxs = []
        for array in arrays:
            ctx_lens.append(len(array))
            padding = maxlen_ctx - len(array)
            padded_ctx = array + [[sentinel]] * padding
            # Pad sents
            padded = []
            lens = []
            for i, seq in enumerate(padded_ctx):
                padding = maxlen_sent - len(seq)
                padded.append(seq + [sentinel] * padding)
                lens.append(len(seq) if i < ctx_lens[-1] else 0)

            padded_ctxs.append(padded)
            ctx_sent_lens.append(lens)

        return padded_ctxs, ctx_lens, ctx_sent_lens
