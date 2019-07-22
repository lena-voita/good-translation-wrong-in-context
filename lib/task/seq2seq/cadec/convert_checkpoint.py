
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--base-ckpt', type=str, help="Path to the baseline checkpoint to be converted")
parser.add_argument('-o', '--output', type=str, help="Path to output checkpoint")


def base_to_cadec_checkpoint(in_path, out_path):
    base_ckpt = np.load(in_path)

    out_dict = {}
    for name in base_ckpt:
        out_dict['mod/model1/' + '/'.join(name.split('/')[1:])] = base_ckpt[name]
    out_dict['mod/loss2_xent_lm/logits/W:0'] = base_ckpt['loss_xent_lm/logits/W:0']
    out_dict['mod/loss_xent_lm/logits/W:0'] = base_ckpt['loss_xent_lm/logits/W:0']

    np.savez(out_path, **out_dict)


if __name__ == '__main__':
    args = parser.parse_args()
    base_to_cadec_checkpoint(args.base_ckpt, args.output)
