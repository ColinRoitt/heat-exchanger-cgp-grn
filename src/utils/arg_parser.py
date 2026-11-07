import argparse


def get_parser(args = None):
    """ Gets the arguments from command line, if any. If not, default values are set
    """
    parser = argparse.ArgumentParser('RIED_Foss', add_help=True)
    # version = '%(prog)s ' + __version__

    parser.add_argument('--rng_seed', '-r',
                        type=int,
                        default = "1",
                        help='Set the random number seed')

    parser.add_argument('--num_evo', '-e',
                        type=int,
                        default = "10",
                        help='Set the number of evo generations')

    parser.add_argument('--pop_size', '-p',
                        type=int,
                        default = "20",
                        help='Set the size of the population')

    parser.add_argument('--num_devo', '-d',
                        type=int,
                        default = "10",
                        help='Set the number of devo generations. Default is 4')

    parser.add_argument('--out_dir_prefix',
                        type=str,
                        default=".",
                        help='Prefix the auto-generated data output directory name')

    parser.add_argument('--make_stl', '-m',
                        action='store_true',
                        help='Creates and saves organism stls in cad_models directory')

    parser.add_argument('--save_analysis',
                         '-a',
                        action='store_true',
                        help='Saves all files associated with the analysis in the calculix directory.' )

    parser.add_argument('--algorithm', '-algo',
                        type=str,
                        default="RNEAT",
                        help='Set the algorithm/GRN Type. Default is RNEAT')

    parser.add_argument('--evodevo', '-ed',
                        type=int,
                        default=1,
                        help='Set True as default for evodevo')

    parser.add_argument('--load_grn_gen_id', '-gid',
                        type=int,
                        default=0,
                        help='Set the default generation id to zero')

    parser.add_argument('--load_grn_pop_id', '-pid',
                        type=int,
                        default=0,
                        help='Set the default population id to zero')
    
    parser.add_argument('--self_stopping', '-ss',
                        action='store_true',
                        default=False,
                        help='Set False as default for self stopping')

    # parser.add_argument('--version', '-v', action='version', version=version)
    args = parser.parse_args(args)

    return args

if __name__ == '__main__':
    parser = get_parser()
    print(parser)
