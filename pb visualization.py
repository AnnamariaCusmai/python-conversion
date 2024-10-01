# -*- coding: utf-8 -*-
"""
Created on Tue Oct  1 09:12:38 2024

@author: anncu
"""

import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.platform import gfile

tf.compat.v1.disable_eager_execution()

def load_pb_file(pb_path):
    with tf.compat.v1.Session() as sess:
        with gfile.FastGFile(pb_path, 'rb') as f:
            graph_def = tf.compat.v1.GraphDef()
            graph_def.ParseFromString(f.read())
            sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
    return sess.graph

def visualize_graph(graph):
    # Write the graph to a log file
    writer = tf.compat.v1.summary.FileWriter('./logs', graph)
    writer.close()
    print("TensorBoard log files written to ./logs")
    print("Run 'tensorboard --logdir=./logs' to visualize the graph")

# Path to your .pb file
pb_file_path = 'D:/6_Inputs/saved_model.pb'

# Load the graph
graph = load_pb_file(pb_file_path)

# Get input and output tensors
input_tensor = graph.get_tensor_by_name('input_tensor_name:0')
output_tensor = graph.get_tensor_by_name('output_tensor_name:0')

print(f"Input tensor: {input_tensor}")
print(f"Output tensor: {output_tensor}")

# Visualize the graph
visualize_graph(graph)