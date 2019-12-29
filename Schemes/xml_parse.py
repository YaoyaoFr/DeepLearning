import os
import sys
import numpy as np
import tensorflow as tf
import warnings
import xml.etree.ElementTree as ET


def parse_str(configuration: str):
    tree = ET.ElementTree()
    parser = ET.XMLParser()
    parser.feed(configuration)
    tree._root = parser.close()
    root = tree.getroot()

    xml_structure = parse_xml_structure(root)
    return xml_structure


def parse_xml_file(file_path):
    tree = ET.parse(file_path)
    root = tree.getroot()

    xml_structure = parse_xml_structure(root)
    return xml_structure


def parse_xml_structure(root=None):
    if len(root) == 0:
        return parse_content(root)

    structure = {}
    for branch in root:
        s = parse_xml_structure(branch)
        if branch.tag in ['input', 'layer']:
            if branch.tag not in structure:
                structure[branch.tag] = []
            structure[branch.tag].append(s)
        else:
            structure[branch.tag] = parse_xml_structure(branch)

    return structure


def parse_content(root):
    tag = root.tag
    text = root.text
    if text is None:
        text = ''
        warnings.warn('The text of {:s} is empty.'.format(tag))

    # Case of List
    if text.startswith('[') and text.endswith(']'):
        text = text[1:-1]
        if ',' in text:
            content = [parse_word(word) for word in text.split(',')]
        else:
            # The shape with []
            if text == '':
                content = []
            else:
                content = [text]
    else:
        content = parse_word(text)

    return content


def parse_word(word: str):
    base_dict = {'float32': tf.float32,
                 'bool': tf.bool,
                 'True': True,
                 'False': False,

                 # Convolution function
                 'conv3d': tf.nn.conv3d,
                 'conv2d_transpose': tf.nn.conv2d_transpose,
                 'conv3d_transpose': tf.nn.conv3d_transpose,

                 # Pooling function
                 'max_pool': tf.nn.max_pool,
                 'max_pool3d': tf.nn.max_pool3d,

                 # Activation function
                 'relu': tf.nn.relu,
                 'sigmoid': tf.nn.sigmoid,
                 'tanh': tf.nn.tanh,
                 'lrelu': tf.nn.leaky_relu,
                 'None': None,
                 }

    # The word is a basic word in predefined dictionary
    if word in base_dict:
        return base_dict[word]

    # Int, float
    try:
        num = float(word)
        if num.is_integer():
            num = int(num)
        return num
    except Exception:
        pass

    # String
    return word



