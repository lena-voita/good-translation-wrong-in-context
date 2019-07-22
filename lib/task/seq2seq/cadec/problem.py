
import lib
from lib.task.seq2seq.cadec.model import CADecModel
import lib.task.seq2seq.problems.default as default
from lib.task.seq2seq.summary import *
from lib.ops import infer_mask
from lib.util import merge_dicts


def get_sequence_logprobas(logits, tokens, eos=1, mean=False):
    """ takes logits[batch,time,voc_size] and tokens[batch,time], returns logp of full sequences"""
    mask = infer_mask(tokens, eos, dtype=tf.float32)
    logp_next = - tf.nn.sparse_softmax_cross_entropy_with_logits(labels=tokens, logits=logits)
    logp_seq = tf.reduce_sum(logp_next * mask, axis=-1)
    if mean:
        logp_seq /= tf.reduce_sum(mask, axis=-1, keep_dims=True)
    return logp_seq


class CADecProblem(default.DefaultProblem):

    def __init__(self, models, inference_flags={},
                 stop_gradient_over_model1=False,
                 get_dec1_max_len=lambda inp_len, out_len: tf.to_int32(tf.to_float(out_len) * 1.2) + 3,
                 dec1_word_dropout_params={},
                 denoising=0,
                 denoising_word_dropout_params={},
                 **kwargs):
        assert len(models) == 1
        assert isinstance(list(models.values())[0], CADecModel)
        super(self.__class__, self).__init__(models, **kwargs)

        inference_flags['mode'] = inference_flags.get("mode", "sample")
        inference_flags['back_prop'] = inference_flags.get("back_prop", False)
        inference_flags['swap_memory'] = inference_flags.get("swap_memory", True)
        self.inference_flags = inference_flags
        self.stop_gradient_over_model1 = stop_gradient_over_model1
        self.get_dec1_max_len = get_dec1_max_len

        word_dropout_params = dict(
            dropout=0,
            method='random_word',
            keep_first=self.model.hp.get('force_bos', False)
        )
        self.dec1_word_dropout_params = merge_dicts(word_dropout_params, dec1_word_dropout_params or {})
        self.denoising = denoising
        self.denoising_word_dropout_params = merge_dicts(word_dropout_params,
                                                        denoising_word_dropout_params or {})

        print("PROBLEM INIT: word dropout for denoising", self.denoising_word_dropout_params)

    def batch_counters(self, batch, is_train):
        model1, decoder2 = self.model.model1, self.model.decoder2

        enc_out, enc_attn_mask, dec1_out, dec1_rdo, dec1_attn_mask, \
        attnP, out_seq, rdo = self.model.encode(batch,
                                                is_train=is_train,
                                                decoder1_flags={},
                                                inference_flags=self.inference_flags,
                                                dec1_word_dropout_params=self.dec1_word_dropout_params,
                                                denoising=self.denoising,
                                                denoising_word_dropout_params=self.denoising_word_dropout_params)

        if self.stop_gradient_over_model1:
            dec1_rdo = tf.stop_gradient(dec1_rdo)
            enc_out = tf.stop_gradient(enc_out)

        # feed everything to dec2
        dec2_batch = {
            'inp': batch['inp'], 'inp_len': batch['inp_len'],
            'out': batch['out'], 'out_len': batch['out_len'],
            'enc_out': enc_out,
            'enc_attn_mask': enc_attn_mask,

            'dec1_out': dec1_out,
            'dec1_attn_mask': dec1_attn_mask,
            'dec1_rdo': dec1_rdo,
        }

        dec2_rdo = decoder2.decode(**dec2_batch, is_train=is_train)

        with lib.layers.basic.dropout_scope(is_train):
            logits = self.model.loss.rdo_to_logits(dec2_rdo, batch['out'], batch['out_len'])
            loss_xent_values = self.model.loss.logits2loss(logits, batch['out'], batch['out_len'])
            assert loss_xent_values.shape.ndims == 1

            loss_values = loss_xent_values

        counters = dict(
            loss=tf.reduce_sum(loss_values),
            out_len=tf.to_float(tf.reduce_sum(batch['out_len'])),
        )
        append_counters_common_metrics(counters, logits, batch['out'], batch['out_len'], is_train)
        append_counters_xent(counters, loss_xent_values, batch['out_len'])
        append_counters_io(counters, batch['inp'], batch['out'], batch['inp_len'], batch['out_len'])
        return counters

    def loss_values(self, batch, is_train):
        model1, decoder2 = self.model.model1, self.model.decoder2

        enc_out, enc_attn_mask, dec1_out, dec1_rdo, dec1_attn_mask, \
        attnP, out_seq, rdo = self.model.encode(batch,
                                                is_train=is_train,
                                                decoder1_flags={},
                                                inference_flags=self.inference_flags,
                                                dec1_word_dropout_params=self.dec1_word_dropout_params,
                                                denoising=self.denoising,
                                                denoising_word_dropout_params=self.denoising_word_dropout_params)

        if self.stop_gradient_over_model1:
            dec1_rdo = tf.stop_gradient(dec1_rdo)
            enc_out = tf.stop_gradient(enc_out)

        # feed everything to dec2
        dec2_batch = {
            'inp': batch['inp'], 'inp_len': batch['inp_len'],
            'out': batch['out'], 'out_len': batch['out_len'],
            'enc_out': enc_out,
            'enc_attn_mask': enc_attn_mask,

            'dec1_out': dec1_out,
            'dec1_attn_mask': dec1_attn_mask,
            'dec1_rdo': dec1_rdo,
        }

        dec2_rdo = decoder2.decode(**dec2_batch, is_train=is_train)

        with lib.layers.basic.dropout_scope(is_train):
            logits = self.model.loss.rdo_to_logits(dec2_rdo, batch['out'], batch['out_len'])
            loss_xent_values = self.model.loss.logits2loss(logits, batch['out'], batch['out_len'])
            assert loss_xent_values.shape.ndims == 1

            loss_values = loss_xent_values

        return loss_values






