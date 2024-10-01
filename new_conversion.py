# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:47:43 2024

@author: anncu
"""
import torch
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
import numpy as np

# Load the network
class nutANN(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super(nutANN, self).__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(H, H)
        self.linear3 = torch.nn.Linear(H, H)
        self.linear4 = torch.nn.Linear(H, H)
        self.linear5 = torch.nn.Linear(H, H)
        self.linear6 = torch.nn.Linear(H, H)
        self.linear7 = torch.nn.Linear(H, H)
        self.linear8 = torch.nn.Linear(H, D_out)
        self.dropout1 = torch.nn.Dropout(0.05)

    def forward(self, x):
        h1 = torch.relu(self.dropout1(self.linear1(x)))
        h2 = torch.relu(self.linear2(h1))
        h3 = torch.relu(self.linear3(h2))
        h4 = torch.relu(self.linear4(h3))
        h5 = torch.relu(self.linear5(h4))
        h6 = torch.relu(self.linear6(h5))
        h7 = torch.tanh(self.linear7(h6))
        alpha_pred = self.linear8(h7)
        return alpha_pred

# Initialize the model
D_in, H, D_out = 6, 100, 1
model = nutANN(D_in, H, D_out)

# Load weights and biases
model.load_state_dict(torch.load('New_Norm_Deep_model_state_dict_epoch378', map_location=torch.device('cpu')))
model.eval()

# Create dummy input
dummy_input = torch.randn(1, 6)

# Define input and output names
input_names = ['input_1', 'input_2', 'input_3', 'input_4', 'input_5', 'input_6']
output_names = ['output']

# Export the model to ONNX
torch.onnx.export(model, dummy_input, 'nutANN.onnx', verbose=True, input_names=input_names, output_names=output_names)

# Load the ONNX model
onnx_model = onnx.load('nutANN.onnx')

# Modify input dimensions
for input in onnx_model.graph.input:
    input.type.tensor_type.shape.dim[0].dim_param = '?'

# Save the modified ONNX model
onnx.save(onnx_model, 'nutANN_modified.onnx')

# Convert ONNX to TensorFlow
tf_rep = prepare(onnx_model)

# Save the TensorFlow model
tf_rep.export_graph('nutANN_tf.pb')

print("Initial conversion complete. TensorFlow model saved as 'nutANN_tf.pb'")

# Freeze session function
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    from tensorflow.python.framework.graph_util import convert_variables_to_constants
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = convert_variables_to_constants(session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph

# Load the TensorFlow graph
with tf.gfile.GFile('nutANN_tf.pb', "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

# Create a session and load the graph
with tf.Session() as sess:
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')
    
    # Get input and output tensors
    input_tensors = [sess.graph.get_tensor_by_name(f'{name}:0') for name in input_names]
    output_tensor = sess.graph.get_tensor_by_name('output:0')
    
    # Freeze the graph
    frozen_graph = freeze_session(sess, output_names=[output_tensor.op.name])

# Save the frozen graph
with tf.gfile.GFile('nutANN_tf_frozen.pb', "wb") as f:
    f.write(frozen_graph.SerializeToString())

print("Conversion complete. Frozen TensorFlow model saved as 'nutANN_tf_frozen.pb'")

# Optional: Verify the frozen TensorFlow model
def load_tf_graph(graph_filename):
    with tf.gfile.GFile(graph_filename, 'rb') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    with tf.Graph().as_default() as graph:
        tf.import_graph_def(graph_def, name='')
    return graph

tf_graph = load_tf_graph('nutANN_tf_frozen.pb')
print("Frozen TensorFlow graph loaded successfully.")

# Print input and output operations
print("Input Operations:")
for op in tf_graph.get_operations():
    if op.type == 'Placeholder':
        print(op.name)
print("Output Operation:", tf_graph.get_operations()[-1].name)
# Print input and output operations
print("Input Operation:", tf_graph.get_operations()[0].name)
print("Output Operation:", tf_graph.get_operations()[-1].name)
