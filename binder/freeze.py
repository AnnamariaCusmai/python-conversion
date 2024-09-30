# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 22:01:17 2024

@author: anncu
"""
import tensorflow as tf
from tensorflow import keras
import os

def freeze_model(model, output_path):
    # Convert Keras model to TensorFlow SavedModel format
    tf.saved_model.save(model, 'temp_saved_model')

    # Convert SavedModel to concrete function
    imported = tf.saved_model.load('temp_saved_model')
    concrete_func = imported.signatures[tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]

    # Get frozen ConcreteFunction
    frozen_func = tf.compat.v1.graph_util.convert_variables_to_constants_v2(
        concrete_func,
        [tensor.name for tensor in concrete_func.inputs] + [tensor.name for tensor in concrete_func.outputs]
    )

    # Save frozen graph
    tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
                      logdir=os.path.dirname(output_path),
                      name=os.path.basename(output_path),
                      as_text=False)

    print(f"Frozen model saved to: {output_path}")
    print(f"Input node name: {concrete_func.inputs[0].name}")
    print(f"Output node name: {concrete_func.outputs[0].name}")

    # Clean up temporary SavedModel directory
    import shutil
    shutil.rmtree('temp_saved_model')

# Load the Keras model
model_path = 'D:/6_Inputs/nutANN_tf_model.keras'  # Replace with your actual path
model = keras.models.load_model(model_path)

# Freeze the model
output_dir =  'D:/6_Inputs'  # You can change this to any directory you prefer
output_graph_name = 'frozen_nutANN_model.pb'
output_path = os.path.join(output_dir, output_graph_name)

freeze_model(model, output_path)
# import tensorflow as tf
# from tensorflow import keras
# import os

# def freeze_model(model, output_path):
#     # Convert Keras model to TensorFlow ConcreteFunction
#     full_model = tf.function(lambda x: model(x))
#     full_model = full_model.get_concrete_function(
#         tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))

#     # Get frozen ConcreteFunction
#     frozen_func = tf.python.framework.convert_to_constants.convert_variables_to_constants_v2(full_model)
#     frozen_func.graph.as_graph_def()

#     # Save frozen graph from frozen ConcreteFunction to hard drive
#     tf.io.write_graph(graph_or_graph_def=frozen_func.graph,
#                       logdir=os.path.dirname(output_path),
#                       name=os.path.basename(output_path),
#                       as_text=False)

#     print(f"Frozen model saved to: {output_path}")
#     print(f"Input node name: {frozen_func.inputs[0].name}")
#     print(f"Output node name: {frozen_func.outputs[0].name}")

# # Load the Keras model
# model_path = 'D:/6_Inputs/nutANN_tf_model.keras'  # Replace with your actual path
# model = keras.models.load_model(model_path)

# # Freeze the model
# output_dir =  'D:/6_Inputs'  # You can change this to any directory you prefer
# output_graph_name = 'frozen_nutANN_model.pb'
# output_path = os.path.join(output_dir, output_graph_name)

# freeze_model(model, output_path)











# import tensorflow as tf
# from tensorflow import keras
# import os

# def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
#     graph = session.graph
#     with graph.as_default():
#         freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
#         output_names = output_names or []
#         output_names += [v.op.name for v in tf.global_variables()]
#         input_graph_def = graph.as_graph_def()
#         if clear_devices:
#             for node in input_graph_def.node:
#                 node.device = ""
#         frozen_graph = tf.graph_util.convert_variables_to_constants(
#             session, input_graph_def, output_names, freeze_var_names)
#         return frozen_graph

# # Load the Keras model
# model_path = 'D:/6_Inputs/nutANN_tf_model.keras'  
# model = keras.models.load_model(model_path)

# # Freeze the model
# frozen_graph = freeze_session(keras.backend.get_session(),
#                               output_names=[out.op.name for out in model.outputs])

# # Save the frozen graph
# output_dir =  'D:/6_Inputs'  # You can change this to any directory you prefer
# output_graph_name = 'frozen_nutANN_model.pb'
# tf.train.write_graph(frozen_graph, output_dir, output_graph_name, as_text=False)

# print(f"Model frozen and saved as {os.path.join(output_dir, output_graph_name)}")