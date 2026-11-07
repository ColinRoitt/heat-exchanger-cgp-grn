import math

from src.evodevo.grn_parent import GRNParent


class CGPGraph(GRNParent):
    """
    The GRN child class that inherits GRNParent
    """
    def __init__(self, gen_id, pop_id):
        super().__init__(gen_id, pop_id)
        self.gen_id = gen_id
        self.grn = None
        self.graph = None
        self.pretty = None

    def activate(self, cell_state):
        """
        Passes the cell state into the GRN outputting some growth mechanism.

        Args:
            cell_state: An array describing the cell state based on interactions with the environment.
        Returns:
            out: delta output from GRN
        """
        out = self.grn(*cell_state)
        out = [0 if math.isnan(o) else o for o in out]
        out = [math.tanh(o) for o in out]
        return out
