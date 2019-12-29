import h5py
import numpy as np
import tensorflow as tf

from Dataset.utils import vector2onehot
from Log.log import Log
from Model.NN import NeuralNetwork


class CNNSmallWorld(NeuralNetwork):

    def __init__(self,
                 dir_path: str,
                 log: Log = None,
                 scheme: int or str = 1,
                 graph: tf.Graph = None,
                 spe_pas: dict = None, 
                 ):
        self.log = None
        self.graph = None
        self.sess = None
        self.op_layers = []
        self.input_placeholders = {}

        NeuralNetwork.__init__(self,
                               log=log,
                               graph=graph,
                               scheme=scheme,
                               dir_path=dir_path,
                               spe_pas=spe_pas, 
                               )

        self.data_placeholder = {'data': 'input_tensor',
                                 'label': 'output_tensor'}

        self.optimizer = {}
        self.NN_type = 'CNNSW'

    def build_structure(self):
        parameters = list()

        input_tensors = None
        for layer in self.op_layers:
            layer(tensors=input_tensors,
                  placeholders=self.input_placeholders)
            layer_tensor = layer.tensors
            self.tensors[layer.pa['scope']] = layer_tensor
            if 'weight' in layer_tensor:
                parameters.append(layer_tensor['weight'])
            if 'bias' in layer_tensor:
                parameters.append(layer_tensor['bias'])
            input_tensors = layer_tensor

        output_tensor = layer_tensor['output']

        print('Build Convolutional Neural Network With Graphical LASSO.')
        self.build_optimizer(output_tensor=output_tensor,
                             penalties=['Mapping'])

        self.initialization()
        self.log.saver = tf.train.Saver(tf.global_variables(), max_to_keep=1000, name='saver')
