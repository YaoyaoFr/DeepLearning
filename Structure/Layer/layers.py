import tensorflow as tf
from Structure.Layer.convolution import Convolution, Convolutions, DeConvolution, DepthwiseConvolution, \
    DepthwiseDeConvolution
from Structure.Layer.others import Placeholder, Placeholders, FullyConnected, Unfolds, Fold, Unfold, Softmax
from Structure.Layer.pool import MaxPooling, MaxPoolings, SpatialPyramidPool3D, UnPooling, UnPooling3D
from Structure.Layer.BrainNetCNN import EdgeToEdge, EdgeToNode, NodeToGraph
from Structure.Layer.GraphNN import GraphConnected


def get_layer_by_arguments(arguments, parameters=None):
    type = arguments['type']
    layer = None

    if type == 'Placeholder':
        layer = Placeholder(arguments=arguments)
    elif type == 'Placeholders':
        layer = Placeholders(arguments=arguments, parameters=parameters)
    elif type == 'GraphNN':
        layer = GraphConnected(arguments=arguments, parameters=parameters)
    elif type == 'EdgeToEdge':
        layer = EdgeToEdge(arguments=arguments)
    elif type == 'EdgeToNode':
        layer = EdgeToNode(arguments=arguments, parameters=parameters)
    elif type == 'NodeToGraph':
        layer = NodeToGraph(arguments=arguments, parameters=parameters)
    elif type == 'Convolution2D':
        layer = Convolution(arguments=arguments)
    elif type == 'Convolution3D':
        arguments['conv_fun'] = tf.nn.conv3d
        layer = Convolution(arguments=arguments)
    elif type == 'Convolutions3D':
        arguments['conv_fun'] = tf.nn.conv3d
        layer = Convolutions(arguments=arguments, parameters=parameters)
    elif type == 'DeConvolution2D':
        arguments['conv_fun'] = tf.nn.conv2d_transpose
        layer = DeConvolution(arguments=arguments)
    elif type == 'DeConvolution3D':
        arguments['conv_fun'] = tf.nn.conv3d_transpose
        layer = DeConvolution(arguments=arguments)
    elif type == 'DepthwiseConvolution2D':
        arguments['conv_fun'] = tf.nn.depthwise_conv2d
        layer = Convolution(arguments=arguments)
    elif type == 'DepthwiseDeConvolution2D':
        arguments['conv_fun'] = tf.nn.conv2d_transpose
        layer = DepthwiseDeConvolution(arguments=arguments)
    elif type == 'DepthwiseConvolution3D':
        layer = DepthwiseConvolution(arguments=arguments)
    elif type == 'DepthwiseDeConvolution3D':
        arguments['convolution'] = tf.nn.conv3d_transpose
        layer = DepthwiseDeConvolution(arguments=arguments)
    elif type == 'MaxPooling2D':
        arguments['pool_fun'] = tf.nn.max_pool
        layer = MaxPooling(arguments=arguments)
    elif type == 'MaxPooling3D':
        arguments['pool_fun'] = tf.nn.max_pool3d
        layer = MaxPooling(arguments=arguments)
    elif type == 'MaxPoolings3D':
        arguments['pool_fun'] = tf.nn.max_pool3d
        layer = MaxPoolings(arguments=arguments)
    elif type == 'SpatialPyramidPool3D':
        layer = SpatialPyramidPool3D(arguments=arguments)
    elif type == 'UnPooling':
        layer = UnPooling(arguments=arguments)
    elif type == 'UnPooling3D':
        layer = UnPooling3D(arguments=arguments)
    elif type == 'Fold':
        layer = Fold(arguments=arguments)
    elif type == 'Unfold':
        layer = Unfold(arguments=arguments)
    elif type == 'Unfolds':
        layer = Unfolds(arguments=arguments)
    elif type == 'FullyConnected':
        layer = FullyConnected(arguments=arguments, parameters=parameters)
    elif type == 'Softmax':
        layer = Softmax(arguments=arguments)

    return layer
