import src.utils.arg_parser as arg_parser
import sys

def test_get_parser():
    sys.argv=['']
    args = arg_parser.get_parser()
    isinstance(args.rng_seed,int)
    isinstance(args.num_evo,int)
    isinstance(args.pop_size,int)
    isinstance(args.num_devo,int)
    isinstance(args.out_dir_prefix,str)
    isinstance(args.make_stl,bool)
    isinstance(args.save_analysis,bool)
    isinstance(args.algorithm, str)
    isinstance(args.evodevo, int)
    isinstance(args.load_grn_gen_id, int)
    isinstance(args.load_grn_pop_id, int)

    assert args.rng_seed >= 0
    assert args.num_evo > 0
    assert args.pop_size > 0
    assert args.num_devo > 0
    assert args.load_grn_gen_id >= 0
    assert args.load_grn_pop_id >= 0
    assert args.evodevo >= 0

    print(args)

if __name__ == '__main__':
    test_get_parser()
