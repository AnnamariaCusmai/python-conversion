import torch
import onnx
import os
from onnx_tf.backend import prepare
import numpy as np

# Define the network structure as provided
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

# Define the network structure
D_in, H, D_out = 6, 100, 1
model_pytorch = nutANN(D_in, H, D_out)

# Load the weights and biases from a previously trained network
model_pytorch.load_state_dict(torch.load('New_Norm_Deep_model_state_dict_epoch378', map_location=torch.device('cpu')))

# Set the model to evaluation mode
model_pytorch.eval()

# Load normalization values from .npz files
min_input = np.load('New_minInput.npz')['minInput']
max_input = np.load('New_maxInput.npz')['maxInput']

# Create dummy inputs tuple and normalize it
dummy_input = torch.randn(1, D_in)  # Adjust the shape as needed
dummy_input_np = dummy_input.numpy()

# Normalize the dummy input
normalized_dummy_input = (dummy_input_np - min_input) / (max_input - min_input)
normalized_dummy_input = torch.from_numpy(normalized_dummy_input).float()

# Define input and output names
input_names = ['input_1']
output_names = ['output_1']

# Export the model to ONNX format
torch.onnx.export(model_pytorch, normalized_dummy_input, "model_exported.onnx", verbose=True,
                  input_names=input_names, output_names=output_names, export_params=True)

# Load the exported ONNX model
model_onnx = onnx.load("model_exported.onnx")

# Modify the input dimensions to accept any first dimension
model_onnx.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'

# Save the new ONNX network with inputs dimension changed and reload it
onnx.save(model_onnx, "model_modified.onnx")
onnx_model_2 = onnx.load("model_modified.onnx")

print("Model has been exported and modified successfully.")

# Convert ONNX model to TensorFlow
tf_rep = prepare(onnx_model_2, device='cpu')
tf_rep.export_graph('NutANN.pb')

print("Model has been converted to TensorFlow and saved as Alpha1Noise.pb")

# Check inputs/outputs name and write the min/max norm files
input_names = [input.name for input in model_onnx.graph.input]
output_names = [output.name for output in model_onnx.graph.output]

print("Input names:", input_names)
print("Output names:", output_names)

# Save the min and max normalization values to text files
np.savetxt('min_norm.txt', min_input)
np.savetxt('max_norm.txt', max_input)

print("Min/Max norm files have been written.")