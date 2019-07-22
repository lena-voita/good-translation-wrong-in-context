
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--src-inp', type=str)
parser.add_argument('--dst-inp', type=str)
parser.add_argument('--src-out', type=str)
parser.add_argument('--dst-out', type=str)

def example_to_cadec(src_line, dst_line):
    src_sents = src_line.strip().split(' _eos ')
    dst_sents = dst_line.strip().split(' _eos ')
    assert len(src_sents) == len(dst_sents), "Different number of sentences in src and dst"
    assert len(src_sents) > 1, "An example without context encountered"
    src_cadec_line = ' _eos '.join(src_sents[-1:] + src_sents[:-1]) + ' _eos_eos ' + ' _eos '.join(dst_sents[:-1]) + '\n'
    dst_cadec_line = dst_sents[-1] + '\n'
    return src_cadec_line, dst_cadec_line

def data_to_cadec(args):
    with open(args.src_out, 'w') as f_src, open(args.dst_out, 'w') as f_dst:
        for example in zip(open(args.src_inp), open(args.dst_inp)):
            src_cadec_line, dst_cadec_line = example_to_cadec(*example)
            f_src.write(src_cadec_line)
            f_dst.write(dst_cadec_line)

if __name__=='__main__':
    data_to_cadec(parser.parse_args())
