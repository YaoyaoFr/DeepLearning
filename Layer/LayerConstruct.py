import tensorflow as tf
from Layer.Convolution import Convolution, Convolutions, DeConvolution, DepthwiseConvolution, \
    DepthwiseDeConvolution
from Layer.BasicLayer import Placeholder, Placeholders, FullyConnected, Fold, Unfold, Softmax, \
    MaxPooling, MaxPoolings, SpatialPyramidPool3D, UnPooling, UnPooling3D
from Layer.BrainNetCNN import EdgeToEdge, EdgeToNode, NodeToGraph
from Layer.CNNSmallWorld import EdgeToCluster, SelfAttentionGraphPooling
from Layer.CNNElementWise import EdgeToNodeElementWise
from Layer.CNNGLasso import EdgeToEdgeGLasso, EdgeToNodeGLasso
from Layer.GraphNN import GraphConnected
from Layer.GraphCNN import GraphCNN
from Layer.AutoEncoder import AutoEncoder


def build_layer(arguments, parameters=None):
    type = arguments['type']

    # Get the layer by its type
    layer_dict = {
        'Placeholder': Placeholder,
        'EdgeToEdge': EdgeToEdge,
        'Convolution2D': Convolution,
        'Convolution3D': Convolution,
        'DeConvolution2D': DeConvolution,
        'DeConvolution3D': DeConvolution,
        'DepthwiseConvolution2D': Convolution,
        'DepthwiseDeConvolution2D': DepthwiseDeConvolution,
        'DepthwiseConvolution3D': DepthwiseConvolution,
        'DepthwiseDeConvolution3D': DepthwiseDeConvolution,
        'GraphCNN': GraphCNN,
        'MaxPooling2D': MaxPooling,
        'MaxPooling3D': MaxPooling,
        'MaxPoolings3D': MaxPoolings,
        'SpatialPyramidPool3D': SpatialPyramidPool3D,
        'UnPooling': UnPooling,
        'UnPooling3D': UnPooling3D,
        'Fold': Fold,
        'Unfold': Unfold,
        'Softmax': Softmax,
        'Placeholders': Placeholders,
        'GraphNN': GraphConnected,
        'EdgeToNode': EdgeToNode,
        'EdgeToNodeElementWise': EdgeToNodeElementWise,
        'EdgeToEdgeGLasso': EdgeToEdgeGLasso,
        'EdgeToNodeGLasso': EdgeToNodeGLasso,
        'EdgeToNodeCrossSlide': EdgeToCluster,
        'SelfAttentionGraphPooling': SelfAttentionGraphPooling,
        'NodeToGraph': NodeToGraph,
        'Convolutions3D': Convolutions,
        'FullyConnected': FullyConnected,
        'AutoEncoder': AutoEncoder,
    }

    if type in layer_dict:
        layer = layer_dict[type](arguments=arguments, parameters=parameters)
    else:
        raise TypeError('Cannot build layer with type of {:s}'.format(type))

    return layer
