import tensorflow as tf
from Structure.Layer.Convolution import Convolution, Convolutions, DeConvolution, DepthwiseConvolution, \
    DepthwiseDeConvolution
from Structure.Layer.BasicLayer import Placeholder, Placeholders, FullyConnected, Unfolds, Fold, Unfold, Softmax, \
    MaxPooling, MaxPoolings, SpatialPyramidPool3D, UnPooling, UnPooling3D
from Structure.Layer.BrainNetCNN import EdgeToEdge, EdgeToNode, EdgeToNodeElementWise, NodeToGraph, EdgeToNodeWithGLasso
from Structure.Layer.GraphNN import GraphConnected
from Structure.Layer.GraphCNN import GraphCNN


def get_layer_by_arguments(arguments, parameters=None):
    type = arguments['type']

    # Set the convolution function corresponding to the layer type
    conv_fun_dict = {
        'Convolution3D': tf.nn.conv3d,
        'Convolutions3D': tf.nn.conv3d,
        'Deconvolution2D': tf.nn.conv2d_transpose,
        'Deconvolution3D': tf.nn.conv3d_transpose,
        'DepthwiseDeConvolution2D': tf.nn.conv2d_transpose,
        'DepthwiseDeConvolution3D': tf.nn.conv3d_transpose,
    }
    if type in conv_fun_dict:
        arguments['conv_fun'] = conv_fun_dict[type]

    # Set the pooling function corresponding to the layer type
    pool_fun_dict = {
        'MaxPooling2D': tf.nn.max_pool,
        'MaxPooling3D': tf.nn.max_pool3d,
        'MaxPoolings3D': tf.nn.max_pool3d,
    }
    if type in pool_fun_dict:
        arguments['pool_fun'] = pool_fun_dict[type]

    # Get the layer by its type
    layer_without_parameters = {
        'Placeholder':
            Placeholder,
        'EdgeToEdge':
            EdgeToEdge,
        'Convolution2D':
            Convolution,
        'Convolution3D':
            Convolution,
        'DeConvolution2D':
            DeConvolution,
        'DeConvolution3D':
            DeConvolution,
        'DepthwiseConvolution2D':
            Convolution,
        'DepthwiseDeConvolution2D':
            DepthwiseDeConvolution,
        'DepthwiseConvolution3D':
            DepthwiseConvolution,
        'DepthwiseDeConvolution3D':
            DepthwiseDeConvolution,
        'GraphCNN':
            GraphCNN,
        'MaxPooling2D':
            MaxPooling,
        'MaxPooling3D':
            MaxPooling,
        'MaxPoolings3D':
            MaxPoolings,
        'SpatialPyramidPool3D':
            SpatialPyramidPool3D,
        'UnPooling':
            UnPooling,
        'UnPooling3D':
            UnPooling3D,
        'Fold':
            Fold,
        'Unfold':
            Unfold,
        'Unfolds':
            Unfolds,
        'Softmax':
            Softmax,
    }
    layer_with_parameters = {
        'Placeholders':
            Placeholders,
        'GraphNN':
            GraphConnected,
        'EdgeToNode':
            EdgeToNode,
        'EdgeToNodeElementWise':
            EdgeToNodeElementWise,
        'EdgeToNodeWithGLasso':
            EdgeToNodeWithGLasso,
        'NodeToGraph':
            NodeToGraph,
        'Convolutions3D':
            Convolutions,
        'FullyConnected':
            FullyConnected,
    }

    if type in layer_without_parameters:
        layer = layer_without_parameters[type](arguments=arguments)
    elif type in layer_with_parameters:
        layer = layer_with_parameters[type](arguments=arguments, parameters=parameters)
    else:
        raise TypeError('Cannot build layer with type of {:s}'.format(type))

    return layer
