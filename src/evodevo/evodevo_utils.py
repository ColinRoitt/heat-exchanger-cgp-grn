from src.evodevo.mocgp.cgp_algo import CGP


def get_algorithm(args):
    """
    Build the CGP evo algorithm from the command line arguments.
    Return:
        evo object
    """
    return CGP(args.num_evo, args.pop_size, args.num_devo, args.self_stopping)
