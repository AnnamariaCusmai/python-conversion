# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 10:47:43 2024

@author: anncu
"""
import torch
import tensorflow as tf
import onnx
from onnx import numpy_helper
import numpy as np

def pytorch_to_onnx(pytorch_model, input_shape, onnx_path):
    dummy_input = torch.randn(1, *input_shape)
    torch.onnx.export(pytorch_model, dummy_input, onnx_path, verbose=True)
    print(f"PyTorch model converted to ONNX. Saved as '{onnx_path}'")

def onnx_to_tensorflow(onnx_model_path, tf_saved_model_path):
    # Load the ONNX model
    onnx_model = onnx.load(onnx_model_path)
    
    # Extract input and output information
    input_info = onnx_model.graph.input[0]
    output_info = onnx_model.graph.output[0]
    
    input_shape = [dim.dim_value for dim in input_info.type.tensor_type.shape.dim]
    output_shape = [dim.dim_value for dim in output_info.type.tensor_type.shape.dim]
    
    # Create a TensorFlow model with the same structure
    inputs = tf.keras.Input(shape=input_shape[1:], name=input_info.name)
    x = inputs
    
    for node in onnx_model.graph.node:
        if node.op_type == 'Linear':
            weight = numpy_helper.to_array(next(tensor for tensor in onnx_model.graph.initializer if tensor.name == node.input[1]))
            bias = numpy_helper.to_array(next(tensor for tensor in onnx_model.graph.initializer if tensor.name == node.input[2]))
            x = tf.keras.layers.Dense(weight.shape[1], use_bias=True, 
                                      kernel_initializer=tf.keras.initializers.Constant(weight.T),
                                      bias_initializer=tf.keras.initializers.Constant(bias))(x)
        elif node.op_type == 'Relu':
            x = tf.keras.layers.ReLU()(x)
        elif node.op_type == 'Tanh':
            x = tf.keras.layers.Activation('tanh')(x)
        elif node.op_type == 'Dropout':
            ratio = float(node.attribute[0].f)
            x = tf.keras.layers.Dropout(ratio)(x)
    
    outputs = tf.keras.layers.Dense(output_shape[1], name=output_info.name)(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    
    # Define a function for inference
    @tf.function(input_signature=[tf.TensorSpec(shape=input_shape, dtype=tf.float32)])
    def serve(x):
        return {'output': model(x)}

    # Save the model with the serve function as a signature
    tf.saved_model.save(model, tf_saved_model_path, signatures={'serving_default': serve})
    
    print(f"Model converted and saved to {tf_saved_model_path}")

# Define the PyTorch model
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
        h7 = torch.tanh((self.linear7(h6)))
        alpha_pred = self.linear8(h7)    
        return alpha_pred

# Main conversion process
if __name__ == "__main__":
    # Define model parameters
    D_in, H, D_out = 6, 100, 1

    # Load PyTorch model
    pytorch_model = nutANN(D_in, H, D_out)
    pytorch_model.load_state_dict(torch.load('D:/6_Inputs/New_Norm_Deep_model_state_dict_epoch378'))
    pytorch_model.eval()

    # Define file paths
    onnx_path = 'D:/6_Inputs/ONNX_nutANN.onnx'
    tf_saved_model_path = 'D:/6_Inputs/TF_nutANN_saved_model'

    # Step 1: Convert PyTorch to ONNX
    print("Converting PyTorch model to ONNX...")
    pytorch_to_onnx(pytorch_model, (D_in,), onnx_path)

    # Step 2: Convert ONNX to TensorFlow SavedModel
    print("Converting ONNX model to TensorFlow SavedModel...")
    onnx_to_tensorflow(onnx_path, tf_saved_model_path)

    print("Conversion process completed.")

    # Verify the SavedModel
    print("Verifying the SavedModel...")
    loaded_model = tf.saved_model.load(tf_saved_model_path)
    infer = loaded_model.signatures["serving_default"]
    
    # Create a sample input
    sample_input = tf.constant(np.random.randn(1, D_in).astype(np.float32))
    
    # Run inference
    try:
        output = infer(sample_input)
        print("Sample output shape:", output['output'].shape)
        print("Sample output:", output['output'].numpy())
        print("Verification completed successfully.")
    except Exception as e:
        print(f"Error during verification: {str(e)}")
        print("Model structure:")
        print(loaded_model.signatures["serving_default"].structured_outputs)
        print("\nModel inputs:")
        print(loaded_model.signatures["serving_default"].inputs)
        print("\nModel outputs:")
        print(loaded_model.signatures["serving_default"].outputs)

    print("Conversion and verification process completed.")