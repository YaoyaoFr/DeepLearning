import numpy as np
import scipy.io as sio
import tensorflow as tf

from abc import ABCMeta, abstractmethod
from Model.utils_model import load_initial_value


class LayerObject(object, metaclass=ABCMeta):
    """

    """

    def __init__(self):
        self.initializer = None
        self.optional_pa = {'load_weight': False,
                            'load_bias': False,
                            'L2_lambda': 5e-3,
                            }
        pass

    @abstractmethod
    def build(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        """
        :param args:
        :param kwargs: The arguments should be fed into this function consist of
                        {'placeholders': placeholders of the neural network, which at least should include 'input',
                         'tensors': the default tensor of input,
                        }
        :return:
        """
        tensors = kwargs['tensors']
        placeholders = kwargs['placeholders']

        new_kwargs = {}

        # Data load from tensors of previous layer
        if 'tensors' in self.pa:
            tensor_names = self.pa['tensors']
            for tensor_name in tensor_names:
                new_kwargs[tensor_name] = tensors[tensor_name]

        # Data load from placeholders
        if 'placeholders' in self.pa:
            for placeholder_name in self.pa['placeholders']:
                new_kwargs[placeholder_name] = placeholders[placeholder_name]

        if len(new_kwargs) == 0:
            new_kwargs['output'] = tensors['output']

        new_kwargs['training'] = placeholders['training']

        return self.build(*args, **new_kwargs)

    def set_parameters(self, arguments: dict,
                       parameters: dict = None):
        pa = {}

        if parameters is not None:
            arguments.update(parameters)

        self.optional_pa.update(arguments)

        for arg in self.required_pa:
            assert arg in arguments, 'The required argument {:s} missed.'.format(arg)

            pa[arg] = arguments[arg]

        for arg in self.optional_pa:
            pa[arg] = self.optional_pa[arg]

        return pa

    def get_initial_weight(self,
                           kernel_shape: list,
                           mode: str = 'fan_in',
                           distribution: str = 'norm',
                           ):
        rank = len(kernel_shape)
        assert rank in {2, 3, 4, 5}, 'The rank of kernel expected in {2, 3, 4, 5} but go {:d}'.format(rank)
        assert 'activation' in self.pa, 'The activation function must be given. '

        if rank == 2:
            fan_in = kernel_shape[0] + kernel_shape[1]
            fan_out = kernel_shape[1]
        else:
            receptive_field_size = np.prod(kernel_shape[:-2])
            fan_in = kernel_shape[-2] * receptive_field_size
            fan_out = kernel_shape[-1] * receptive_field_size

        if mode == 'fan_in':
            scale = 1 / max(1., fan_in)
        elif mode == 'fan_out':
            scale = 1 / max(1., fan_out)
        else:
            scale = 1 / max(1., float(fan_in + fan_out) / 2)

        if distribution == 'norm':
            stddev = np.sqrt(2. * scale)
            initial_value = tf.truncated_normal(shape=kernel_shape,
                                                stddev=stddev)
        elif distribution == 'uniform':
            limit = np.sqrt(3. * scale)
            initial_value = tf.random_uniform(shape=kernel_shape,
                                              minval=-limit,
                                              maxval=limit)

        return initial_value

    @staticmethod
    def batch_normalization(tensor: tf.Tensor,
                            training: tf.Tensor = True,
                            scope: str = 'bn',
                            axis: int = -1):
        """
        :param tensor:
        :param training:
        :param scope:
        :param axis:
        :return:
        """
        shape = tensor.get_shape()
        with tf.variable_scope(name_or_scope=scope, reuse=tf.AUTO_REUSE):
            output = tf.layers.batch_normalization(inputs=tensor,
                                                   center=True,
                                                   axis=axis,
                                                   scale=False,
                                                   momentum=0.99,
                                                   training=training,
                                                   epsilon=1e-300,
                                                   )
            mean, var = tf.nn.moments(output, axes=list(np.arange(len(shape) - 1)))

        return {
            'output_bn': output,
            'aft_mean': mean,
            'aft_var': var,
        }

    @staticmethod
    def normalization(tensor: tf.Tensor,
                      norm: bool = False,
                      off_diagonal: bool = False,
                      axis: int or tuple or list = -1,
                      min: float = -1,
                      max: float = 1):
        if off_diagonal:
            [_, _, in_channels, out_channels] = tensor.shape.as_list()
            tensor_diags = []
            for out_ch in range(out_channels):
                tensor_diag_out = []
                for in_ch in range(in_channels):
                    diag_part = tf.diag_part(tensor[..., in_ch, out_ch])
                    tensor_diag_in = tf.expand_dims(tf.expand_dims(tf.diag(diag_part), axis=-1), axis=- 1)
                    tensor_diag_out.append(tensor_diag_in)
                tensor_diag_out = tf.concat(tensor_diag_out, axis=-2)
                tensor_diags.append(tensor_diag_out)
            tensor_diags = tf.concat(tensor_diags, axis=-1)
            tensor = tensor - tensor_diags

        if norm:
            data_min = tf.reduce_min(tensor, axis=axis)
            data_max = tf.reduce_max(tensor, axis=axis)
            tensor = tensor / (data_max - data_min) * (max - min)

        return tensor