#!/bin/bash

REPO_DIR="../" # insert the dir to the good-translation-wrong-in-context repo
DATA_DIR="../" # insert your datadir

NMT="${REPO_DIR}/scripts/nmt.py"

# path to preprocessed data (tokenized, bpe-ized) and base model vocabularies
# data format:
#              train.src: <src_sent_1> _eos <src_sent_2> _eos ... _eos <src_sent_n>
#              train.dst: <dst_sent_1> _eos <dst_sent_2> _eos ... _eos <dst_sent_n>
# example:
#  (src)        i saw a cat . _eos the cat was hungry . _eos i fed the cat .
#  (dst)        я видел кота . _eos кот был голодный . _eos я покормил кота .
train_src="${DATA_DIR}/train.src"
train_dst="${DATA_DIR}/train.dst"
dev_src="${DATA_DIR}/dev.src"
dev_dst="${DATA_DIR}/dev.dst"
voc_src="${DATA_DIR}/src.voc"
voc_dst="${DATA_DIR}/dst.voc"

# path to the trained baseline checkpoint (base model inside CADec)
base_ckpt="path_to_base_checkpoint"  # insert the path to your base checkpoint

# path where results will be stored
model_path="./build"
mkdir -p $model_path

# convert baseline checkpoint to init checkpoint for CADec
if [ ! -f $model_path/cadec_init.npz ]; then
  echo "Make initial checkpoint for CADec"
  /usr/bin/env python3 ${REPO_DIR}/lib/task/seq2seq/cadec/convert_checkpoint.py --base-ckpt $base_ckpt -o $model_path/cadec_init.npz
fi

# make data in CADec format
if [ ! -f $model_path/train.src.cadec ]; then
  echo "Make train data in CADec format"
  /usr/bin/env python3 ${REPO_DIR}/lib/task/seq2seq/cadec/data_to_cadec.py --src-inp $train_src --dst-inp $train_dst --src-out $model_path/train.src.cadec --dst-out $model_path/train.dst.cadec
fi
if [ ! -f $model_path/dev.src.cadec ]; then
  echo "Make dev data in CADec format"
  /usr/bin/env python3 ${REPO_DIR}/lib/task/seq2seq/cadec/data_to_cadec.py --src-inp $dev_src --dst-inp $dev_dst --src-out $model_path/dev.src.cadec --dst-out $model_path/dev.dst.cadec
fi


# shuffle data
shuffle_seed=42

get_random_source()
{
  openssl enc -aes-256-ctr -pass pass:"$1" -nosalt </dev/zero 2>/dev/null
}


if [ ! -f $model_path/train.src.shuf ]; then
  echo "Shuffling train src"
  shuf -o $model_path/train.src.shuf --random-source=<(get_random_source $shuffle_seed) $model_path/train.src.cadec
fi
if [ ! -f $model_path/train.dst.shuf ]; then
  echo "Shuffling train dst"
  shuf -o $model_path/train.dst.shuf --random-source=<(get_random_source $shuffle_seed) $model_path/train.dst.cadec
fi


# maybe add openmpi wrapper
RUN_NMT=$(/usr/bin/env python3 -c "
import os, sys, tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
num_gpus = sum(x.device_type == 'GPU' for x in tf.Session().list_devices())
if num_gpus > 1:
    sys.stdout.write('mpirun --allow-run-as-root --host {} {}'.format(','.join(['localhost'] * num_gpus), '$NMT'))
else:
    sys.stdout.write('$NMT')
")


# Model hp (the same as in the original Transformer-base)
# hp are split into groups:
#      main model hp,
#      minor model hp (probably you do not want to change them)
#      regularization and label smoothing
#      inference params (beam search with a beam of 4)
#      CADec-specific parameters (below are the ones you may want to change)
#           dec1_attn_mode - one of these: {"rdo", "emb", "rdo_and_emb"} - which base decoder states are used
#                            We use both top-layer representation and token embedding - "rdo_and_emb"
#           max_ctx_sents - maximum number of context sentences for an example present in your data
#           use_dst_ctx - whether to use dst side of context (as we do) or only use source representations
MODEL_HP=$(/usr/bin/env python3 -c '

hp = {
     "num_layers": 6,
     "num_heads": 8,
     "ff_size": 2048,
     "ffn_type": "conv_relu",
     "hid_size": 512,
     "emb_size": 512,
     "res_steps": "nlda",

     "rescale_emb": True,
     "inp_emb_bias": True,
     "normalize_out": True,
     "share_emb": False,
     "replace": 0,

     "relu_dropout": 0.1,
     "res_dropout": 0.1,
     "attn_dropout": 0.1,
     "label_smoothing": 0.1,

     "translator": "ingraph",
     "beam_size": 4,
     "beam_spread": 3,
     "len_alpha": 0.6,
     "attn_beta": 0,
     
     "dec1_attn_mode": "rdo_and_emb",
     "share_loss": False,
     "decoder2_name": "decoder2",
     "max_ctx_sents": 3,
     "use_dst_ctx": True,
    }

print(end=repr(hp))
')

# Problem options (how to train your model)
# inference_flags:
#      how to produce first-pass translation (we sample with temperature 0.5)
#      at test time, this is always beam search with the beam of 4
# stop_gradient_over_model1:
#      whether to freeze the first-pass model or not (we train only CADec, keeping it fixed)
# dec1_word_dropout_params:
#      which kind of dropout to apply to first-pass translation
# denoising:
#      the probability of using noise version of target sentence as first-pass translation
#      (the value p introduced in Section 5.1 of the paper)
# denoising_word_dropout_params:
#      which kind of dropout to apply to target sentence when it's used as first-pass translation
PROBLEM_OPTS=$(/usr/bin/env python3 -c '

hp = {
     "inference_flags": {"mode": "sample", "sampling_temperature":0.5},

     "stop_gradient_over_model1": True,

     "dec1_word_dropout_params": {"dropout": 0.1, "method": "random_word",},

     "denoising": 0.5,
     "denoising_word_dropout_params": {"dropout": 0.2, "method": "random_word",},
    }

print(end=repr(hp))
')

params=(
    --folder $model_path
    --seed 42

    --train-src $model_path/train.src.shuf
    --train-dst $model_path/train.dst.shuf
    --dev-src $model_path/dev.src.cadec
    --dev-dst $model_path/dev.dst.cadec

    --ivoc $voc_src
    --ovoc $voc_dst

    # Model you want to train
    --model lib.task.seq2seq.cadec.model.CADecModel
    # Model hp (the same as in the original Transformer-base)
    # Specified above
    --hp "`echo $MODEL_HP`"

    # Problem, i.e. how to train your model (loss function).
    # For the baseline, it is the standard cross-entropy loss, which is implemented in the Default problem.
    --problem lib.task.seq2seq.cadec.problem.CADecProblem
    # Problem options.
    # For CADec, see the options and their description in PROBLEM_OPTS above
    --problem-opts "`echo $PROBLEM_OPTS`"

    # Starting checkpoint.
    # If you start from a trained model, you have to specify a starting checkpoint.
    # For CADec, you always have an initial checkpoint converted from a trained baseline
    --pre-init-model-checkpoint $model_path/cadec_init.npz
    #                             ^---YOU MAY WANT TO CHANGE THIS

    # Maximum number of tokens in a sentence.
    # Sentences longer than this will not participate in training.
    --max-srclen 500
    --max-dstlen 500

    # How to form batches.
    # Note that for CADec we use different batch-maker: multi-ctx-with-dst
    # This means that the length of an example is calculated differently from the baseline
    # (where it's max of lengths of source and target sentences).
    # For CADec it is sum of (1) maximum of lengths of source sides of context sentences multiplied by their number
    #                        (2) maximum of lengths of target sides of context sentences multiplied by their number
    #                        (3) length of current source
    #                        (4) length of current target
    # This means that values of "batch-len" in baseline and CADec configs are not comparable
    # (because of different batch-makers)
    # To approximately match the baseline batch in the number of translation instances,
    # you need batch size of approximately 150000 in total.
    # Here is 7500 for 4 gpus and sync_every_steps=5: 5 * 4 * 7500 in total.
    # (sync_every_steps description is below)
    --batch-len 4000    # YOU MAY WANT TO CHANGE THIS
    --batch-maker multi-ctx-with-dst
    --shuffle-len 100000
    --batch-shuffle-len 10000
    --split-len 200000
    --maxlen-quant 1
    --maxlen-min 8

    # Optimization.
    # This is the optimizer used in the original Transformer.
    --optimizer lazy_adam
    #--optimizer-opts '{'"'"'beta1'"'"': 0.9, '"'"'beta2'"'"': 0.998,
    #                   '"'"'variables'"'"': ['"'"'mod/decoder2*'"'"', '"'"'mod/loss2_xent*'"'"',],}'

    # Alternative optimizer opts in case you do not have several gpus.
    # sync_every_steps=5 means that you accumulate gradients for 5 steps before making an update.
    # This is equivalent to having 'sync_every_steps' times more gpus.
    # The actual batch size will be then batch-len * sync_every_steps * num_gpus
    # (or batch-len * num_gpus if you are using the first version of optimizer-opts)
    --optimizer-opts '{'"'"'beta1'"'"': 0.9, '"'"'beta2'"'"': 0.998,
                       '"'"'variables'"'"': ['"'"'mod/decoder2*'"'"', '"'"'mod/loss2_xent*'"'"',],
                       '"'"'sync_every_steps'"'"': 5, '"'"'average_grads'"'"': True, }'  \

    # Learning rate schedule.
    # For CADec we used lr=1 (it's smaller than for the baseline because of smaller amount of training data).
    --learning-rate 1.0
    --learning-rate-stop-value 1e-08
    --decay-steps 16000
    --decay-policy t2t_noam

    # How long to train.
    # Now it says 8m batches, which basically means that you have to look at your tensorboard and stop training manually
    --num-batches 8000000

    # Checkpoints.
    # How often to make a checkpoint
    --checkpoint-every-steps 2048
    # How many checkpoints you want to keep.
    --keep-checkpoints-max 10
    #                       ^---YOU MAY WANT TO CHANGE THIS

    # How often to score dev set (and put a dot on your tensorboard)
    --score-dev-every 256

    # BLEU on your tensorboard.
    # This says that you want to see BLEU score on your tensorboard.
    --translate-dev
    # How often to translate dev and add this info to your tensorboard.
    --translate-dev-every 2048

    # This argument has to passed last.
    # It controls that nmt.py has received all your arguments
    --end-of-params
)

$RUN_NMT train "${params[@]}"

