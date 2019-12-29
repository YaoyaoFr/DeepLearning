import os
import h5py
import tensorflow as tf

from Log.log import Log
from Schemes.xml_parse import parse_xml_file
from Model.utils_model import EarlyStop, get_metrics, upper_triangle
from Model.NN import NeuralNetwork


class FullyConnectedNeuralNetwork(NeuralNetwork):

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
        self.tensors = {}
        self.op_layers = []
        self.input_placeholders = {}

        NeuralNetwork.__init__(self,
                               log=log,
                               dir_path=dir_path,
                               scheme=scheme,
                               graph=graph,
                               spe_pas=spe_pas,
                               )
        self.NN_type = 'FCNN'

    def training(self,
                 data: h5py.Group or dict,
                 run_time: int = 1,
                 fold_index: int = None,
                 restored_path: str = None,
                 show_info: bool = True):

        data = self.load_data(data)
        data = upper_triangle(data)

        with self.graph.as_default():
            self.build_structure()
            
        early_stop = self.backpropagation(data=data)
        early_stop.show_results(run_time=run_time, fold_index=fold_index)
        early_stop.clear_models()

        return early_stop.results
