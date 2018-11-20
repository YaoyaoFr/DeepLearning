import os
import sys
import numpy as np
import tensorflow as tf
import xml.etree.ElementTree as ET


def parse_training_parameters(xml_file_path=None):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    parameters = dict()
    for structure in root:
        structure_pa = dict()
        for type in structure:
            type_pa = dict()
            for parameter in type:
                parse_training_parameter(type_pa, parameter.tag, parameter.text)
            structure_pa[type.tag] = type_pa
        parameters[structure.tag] = structure_pa
    return parameters


def parse_training_parameter(training_parameters, tag, text):
    if tag == 'learning_rate' or tag == 'decay_rate':
        training_parameters[tag] = float(text)
    else:
        training_parameters[tag] = int(text)


def parse_structure_parameters(xml_file_path=None):
    '''
    The xml file should be organized as follows:
        <structure>
            <layers>
                <input>
                    <layer>...</layer>
                </input>
                <autoendoer>
                    <encoder>...</encoder>
                    <decoder>...</decoder>
                </autoencoder>
            </layers>
            <ann>
                <layer>...</layer>
            </ann>
        </structure>
    :param xml_file_path: The path of xml file
    :return: A Dictionary of the tensors
    '''

    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    autoencoder_parameters = dict()
    classifier_parameters = dict()
    for structure in root:
        if structure.tag == 'autoencoders':
            autoencoders = list()
            inputs = list()
            for layer in structure:
                if layer.tag == 'input':
                    arguments = dict()
                    for argument in layer:
                        parse_structure_parameter(arguments, argument.tag, argument.text)
                    inputs.append(arguments)
                if layer.tag == 'autoencoder':
                    autoencoder = dict()
                    for coder in layer:
                        arguments = dict()
                        for argument in coder:
                            parse_structure_parameter(arguments, argument.tag, argument.text)

                        if coder.tag == 'encoder':
                            autoencoder['encoder'] = arguments
                        elif coder.tag == 'decoder':
                            autoencoder['decoder'] = arguments
                    autoencoders.append(autoencoder)
            autoencoder_parameters['autoencoder'] = autoencoders
            autoencoder_parameters['input'] = inputs
        elif structure.tag == 'classifier':
            layers = list()
            inputs = list()
            for layer in structure:
                if layer.tag == 'input':
                    arguments = dict()
                    for argument in layer:
                        parse_structure_parameter(arguments, argument.tag, argument.text)
                    inputs.append(arguments)
                if layer.tag == 'layer':
                    arguments = dict()
                    for argument in layer:
                        parse_structure_parameter(arguments, argument.tag, argument.text)
                    layers.append(arguments)
            classifier_parameters['layers'] = layers
            classifier_parameters['input'] = inputs
    return {'autoencoders': autoencoder_parameters,
            'classifier': classifier_parameters}


def parse_structure_parameter(arguments, tag, text):
    if tag in ['kernel_shape', 'strides', 'fold_shape', 'input_shape', 'output_shape']:
        arguments[tag] = list()
        if text:
            for txt in text.split(','):
                shape = None
                try:
                    shape = int(txt)
                except:
                    pass
                arguments[tag].append(shape)
    elif tag == 'output_dim':
        arguments[tag] = np.prod([int(i) for i in text.split('*')])
    elif tag in ['activation', 'dtype', 'batch_normalization', 'bias']:
        if text == 'relu':
            arguments[tag] = tf.nn.relu
        elif text == 'sigmoid':
            arguments[tag] = tf.nn.sigmoid
        elif text == 'tanh':
            arguments[tag] = tf.nn.tanh
        elif text == 'float32':
            arguments[tag] = tf.float32
        elif text == 'True':
            arguments[tag] = True
        elif text == 'False':
            arguments[tag] = False
    elif tag in ['view_num', 'channel']:
        arguments[tag] = int(text)
    else:
        arguments[tag] = text
    return arguments


def parse_log_parameters(xml_file_path=None):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()

    log_parameters = dict()
    for log_parameter in root:
        parse_log_parameter(log_parameters, log_parameter.tag, log_parameter.text)
    return log_parameters


def parse_log_parameter(log_parameters, tag, text):
    if text == '':
        text = None

    if tag == 'restored_epoch':
        text = int(text) if text else None
    log_parameters[tag] = text

def parse_aal_regions(xml_file_path=None):
    tree = ET.parse(xml_file_path)
    root = tree.getroot()
    data = root[1]
    region_index = 0
    region = []
    for label in data:
        name = label[1]
        region.append(name.text)
        region_index = region_index + 1
    return region

