from .node import OperatorNode

class Exp(OperatorNode):
    """
    Exp of input
    """

    _arity = 1
    _def_output = "math.exp(x_0)"
    _def_numpy_output = "np.exp(x_0)"
    _def_torch_output = "torch.exp(x_0)"
    _def_sympy_output = "exp(x_0)"

class SineFunction(OperatorNode):
    """
    Sin of input
    """

    _arity = 1
    _def_output = "math.sin(x_0+0.1)"
    _def_numpy_output = "np.sin(x_0+0.1)"
    _def_torch_output = "torch.sin(x_0+0.1)"
    _def_sympy_output = "sin(x_0+0.1)"

class CosineFunction(OperatorNode):
    """
    Cosine of input
    """

    _arity = 1
    _def_output = "math.cos(x_0+0.1)"
    _def_numpy_output = "np.cos(x_0+0.1)"
    _def_torch_output = "torch.cos(x_0+0.1)"
    _def_sympy_output = "cos(x_0+0.1)"

class TanFunction(OperatorNode):
    """
    Tan of input
    """

    _arity = 1
    _def_output = "math.tan(x_0+0.1)"
    _def_numpy_output = "np.tan(x_0+0.1)"
    _def_torch_output = "torch.tan(x_0+0.1)"
    _def_sympy_output = "tan(x_0+0.1)"

class SigmoidFunction(OperatorNode):
    """
    Sigmoid of input
    """

    _arity = 1
    _def_output = "1/(1+math.exp(-x_0))"
    _def_numpy_output = "1/(1+np.exp(-x_0))"
    _def_torch_output = "torch.sigmoid(x_0)"
    _def_sympy_output = "1/(1+exp(-x_0))"

class ReluFunction(OperatorNode):
    """
    ReLU of input
    """

    _arity = 1
    _def_output = "max(0, x_0)"
    _def_numpy_output = "np.maximum(0, x_0)"
    _def_torch_output = "torch.relu(x_0)"
    _def_sympy_output = "Max(0, x_0)"

# class MaxFunction(OperatorNode):
#
#     _arity = 2
#     _def_output = "x_0 if x_0 >= x_1 else x_1"
#
#
# class MinFunction(OperatorNode):
#
#     _arity = 2
#     _def_output = "x_0 if x_0 <= x_1 else x_1"