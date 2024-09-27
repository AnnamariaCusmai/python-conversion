# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 17:04:18 2024

@author: anncu
"""

import torch
import torch.onnx
import onnx
import tensorflow as tf
import onnx_tf

# Step 1: Load the PyTorch model
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

D_in, H, D_out = 8, 100, 1
model = nutANN(D_in, H, D_out)
model.load_state_dict(torch.load('D:/6_Inputs/New_Norm_Deep_model_state_dict_epoch378'))
model.eval()

# Step 2: Convert PyTorch model to ONNX
dummy_input = torch.randn(1, D_in)
torch.onnx.export(model, dummy_input, 'D:/6_Inputs/nutANN.onnx', verbose=True)

# Step 3: Load ONNX model
onnx_model = onnx.load('D:/6_Inputs/nutANN.onnx')
onnx.checker.check_model(onnx_model)

# Step 4: Convert ONNX to TensorFlow SavedModel
tf_rep = onnx_tf.backend.prepare(onnx_model)
tf_rep.export_graph('D:/6_Inputs/nutANN_tf')

print("Conversion completed. TensorFlow model saved in 'nutANN_tf' directory.")