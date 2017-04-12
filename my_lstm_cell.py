import tensorflow as tf
import tensorflow.contrib.layers as layers
from tensorflow.contrib.rnn.python.ops.core_rnn_cell_impl import LSTMStateTuple
from tensorflow.python.ops import array_ops
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops import nn_ops
from tensorflow.python.util import nest


def _linear(args, output_size, bias, bias_start=0.0):
    """Linear map: sum_i(args[i] * W[i]), where W[i] is a variable.
    Args:
      args: a 2D Tensor or a list of 2D, batch x n, Tensors.
      output_size: int, second dimension of W[i].
      bias: boolean, whether to add a bias term or not.
      bias_start: starting value to initialize the bias; 0 by default.
    Returns:
      A 2D Tensor with shape [batch x output_size] equal to
      sum_i(args[i] * W[i]), where W[i]s are newly created matrices.
    Raises:
      ValueError: if some of the arguments has unspecified or wrong shape.
    """
    if args is None or (nest.is_sequence(args) and not args):
        raise ValueError("`args` must be specified")
    if not nest.is_sequence(args):
        args = [args]

    # Calculate the total size of arguments on dimension 1.
    total_arg_size = 0
    shapes = [a.get_shape() for a in args]
    for shape in shapes:
        if shape.ndims != 2:
            raise ValueError("linear is expecting 2D arguments: %s" % shapes)
        if shape[1].value is None:
            raise ValueError("linear expects shape[1] to be provided for shape %s, "
                             "but saw %s" % (shape, shape[1]))
        else:
            total_arg_size += shape[1].value

    dtype = [a.dtype for a in args][0]

    # Now the computation.
    scope = vs.get_variable_scope()
    with vs.variable_scope(scope) as outer_scope:
        weights = vs.get_variable(
            'weights', [total_arg_size, output_size], dtype=dtype)
        if len(args) == 1:
            res = math_ops.matmul(args[0], weights)
        else:
            res = math_ops.matmul(array_ops.concat(args, 1), weights)
        if not bias:
            return res
        with vs.variable_scope(outer_scope) as inner_scope:
            inner_scope.set_partitioner(None)
            biases = vs.get_variable(
                'biases', [output_size],
                dtype=dtype,
                initializer=init_ops.constant_initializer(bias_start, dtype=dtype))

    return nn_ops.bias_add(res, biases)


class MyLSTMCell(tf.contrib.rnn.LSTMCell):
    """Simplified Version rnn_cell.BasicLSTMCell"""

    def __init__(self, num_units, state_is_tuple=True):
        super(MyLSTMCell, self).__init__(num_units)
        self._num_units = num_units

    def __call__(self, inputs, state, scope="LSTM"):
        """Long short-term memory cell (LSTM)."""
        # Parameters of gates are concatenated into one multiply for efficiency.
        if self._state_is_tuple:
            s, p = state
        else:
            s, p = array_ops.split(value=state, num_or_size_splits=2, axis=1)

        with vs.variable_scope("s"):
            A = (_linear([inputs, s], 4*self._num_units, bias=True))
            Ws = tf.Variable(0, name="output_Weighting", dtype="float32")

        with vs.variable_scope("p"):
            B = (_linear([inputs, p], self._num_units, bias=True))
            Wp = tf.Variable(0, name="output_Weighting", dtype="float32")


        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = array_ops.split(value=A, num_or_size_splits=4, axis=1)

        new_s = (s * sigmoid(f + self._forget_bias) + sigmoid(i)*self._activation(j))

        new_p = sigmoid(o) * self._activation(new_s)

        if self._state_is_tuple:
            new_state = LSTMStateTuple(new_s, new_p)
        else:
            new_state = array_ops.concat([new_s, new_p], 1)

        return new_p, new_state

