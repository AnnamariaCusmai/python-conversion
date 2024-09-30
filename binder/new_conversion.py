# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:47:43 2024

@author: anncu
"""
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf

def convert_onnx_to_tf_pb(onnx_path, output_path):
    # Load the ONNX model
    onnx_model = onnx.load(onnx_path)
    
    # Convert ONNX model to TensorFlow
    tf_rep = prepare(onnx_model)
    
    # Get input and output tensor names
    input_tensor = tf_rep.inputs[0]
    output_tensor = tf_rep.outputs[0]
    
    # Create a TensorFlow graph
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(tf_rep.graph.as_graph_def(), name='')
        
        # Freeze the graph
        with tf.compat.v1.Session() as sess:
            frozen_graph = tf.compat.v1.graph_util.convert_variables_to_constants(
                sess,
                graph.as_graph_def(),
                [output_tensor]
            )
    
    # Save the frozen graph
    with tf.io.gfile.GFile(output_path, "wb") as f:
        f.write(frozen_graph.SerializeToString())
    
    print(f"Converted and saved frozen TensorFlow model to {output_path}")

# Usage
onnx_model_path = "D:/6_Inputs/nutANN.onnx"
output_pb_path = "D:/6_Inputs/nutANN_model.pb"
convert_onnx_to_tf_pb(onnx_model_path, output_pb_path)