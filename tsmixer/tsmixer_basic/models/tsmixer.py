# coding=utf-8
# Copyright 2023 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# """Implementation of TSMixer."""

# import tensorflow as tf
# from tensorflow.keras import layers


# def res_block(inputs, norm_type, activation, dropout, ff_dim):
#   """Residual block of TSMixer."""

#   norm = (
#       layers.LayerNormalization
#       if norm_type == 'L'
#       else layers.BatchNormalization
#   )

#   # Temporal Linear
#   x = norm(axis=[-2, -1])(inputs)
#   x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
#   x = layers.Dense(x.shape[-1], activation=activation)(x)
#   x = layers.Dropout(dropout)(x)
#   x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]

#   res = x + inputs

#   # Feature Linear
#   x = norm(axis=[-2, -1])(res)
#   x = layers.Dense(ff_dim, activation=activation)(x)  # [Batch, Input Length, FF_Dim]
#   x = layers.Dropout(dropout)(x)
#   x = layers.Dense(inputs.shape[-1])(x)  # [Batch, Input Length, Channel]
#   x = layers.Dropout(dropout)(x)
#   return res + x

# def build_model(
#     input_shape,
#     pred_len,
#     norm_type,
#     activation,
#     n_block,
#     dropout,
#     ff_dim,
#     target_slice,
#     n_channel,
# ):
#   """Build TSMixer model."""

#   inputs = tf.keras.Input(shape=input_shape)
#   x = inputs  # [Batch, Input Length, Channel]
#   for _ in range(n_block):
#     x = res_block(x, norm_type, activation, dropout, ff_dim)

#   if target_slice:
#     x = x[:, :, target_slice]

#   x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
#   x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
#   outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])

#   #ADD HEAD FOR FORCASTING
#   x1 = inputs  # [Batch, Input Length, Channel]
#   x1 = tf.keras.layers.Flatten()(x1)
#   x1 = tf.keras.layers.Dense(pred_len * n_channel)(x1)
#   x1 = tf.keras.layers.Reshape((pred_len, n_channel))(x1)
#   outputs1 = tf.transpose(x1, perm=[0, 2, 1])
  
#   return tf.keras.Model(inputs, outputs+outputs1)

# def build_model(
#     input_shape,
#     pred_len,
#     norm_type,
#     activation,
#     n_block,
#     dropout,
#     ff_dim,
#     target_slice,
# ):
#   """Build TSMixer model."""

#   inputs = tf.keras.Input(shape=input_shape)
#   x = inputs  # [Batch, Input Length, Channel]
#   for _ in range(n_block):
#     x = res_block(x, norm_type, activation, dropout, ff_dim)

#   if target_slice:
#     x = x[:, :, target_slice]

#   x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
#   x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
#   outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])

#   return tf.keras.Model(inputs, outputs)
# def build_model(
#     input_shape,
#     pred_len,
#     norm_type,
#     activation,
#     n_block,
#     dropout,
#     ff_dim,
#     target_slice,
#     n_channel=data_loader.n_feature,
# ):
#   """Build TSMixer model."""

#   inputs = tf.keras.Input(shape=(args.seq_len, data_loader.n_feature))
#   # Define the encoder
#     x1 = layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(inputs)
#     # x1 = layers.MaxPooling1D(pool_size=2, padding='same')(x1)
#     # x1 = layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(x1)
#     # x1 = layers.MaxPooling1D(pool_size=2, padding='same')(x1)
#     # x1 = layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(x1)
#     encoded = tf.keras.layers.Flatten()(x1)

#     # encoded = layers.MaxPooling1D(pool_size=2, padding='same')(x1)
    
#   # Define the decoder
#     latent_inputs = tf.keras.Input(shape=(encoded,))
#     x2 = layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(latent_inputs)
#     # x2 = layers.UpSampling1D(size=2)(x2)
#     # x2 = layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(x2)
#     # x2 = layers.UpSampling1D(size=2)(x2)
#     # x2 = layers.Conv1D(filters=16, kernel_size=3, activation='relu')(x2)
#     # x2 = layers.UpSampling1D(size=2)(x2)
#     x2 = tf.keras.layers.Flatten()(x2)
#     x2 = tf.keras.layers.Dense(pred_len * n_channel)(x2)
#     decoded = tf.keras.layers.Reshape((pred_len, n_channel))(x2)
  
   
#   x = inputs  # [Batch, Input Length, Channel]
#   for _ in range(n_block):
#     x = res_block(x, norm_type, activation, dropout, ff_dim)

#   if target_slice:
#     x = x[:, :, target_slice]

#   x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
#   x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
#   outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])

#   return tf.keras.Model(inputs, outputs + decoded)
# def build_model(
#     input_shape,
#     pred_len,
#     norm_type,
#     activation,
#     n_block,
#     dropout,
#     ff_dim,
#     target_slice,
#     n_channel,
#     kernel_size,
# ):
#     """Build TSMixer model."""
    
#     inputs = tf.keras.Input(shape=input_shape)
#     x = inputs  # [Batch, Input Length, Channel]
    
#     # Add CNN layers
#     x = layers.Conv1D(n_channel, kernel_size, padding='same')(x)
#     x = layers.Activation(activation)(x)
    
#     for _ in range(n_block):
#         x = res_block(x, norm_type, activation, dropout, ff_dim)
    
#     if target_slice:
#         x = x[:, :, target_slice]
    
#     x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    
#     # Depth-wise convolution
#     x = layers.DepthwiseConv1D(kernel_size=3, padding='same')(x)
#     x = layers.Activation(activation)(x)
    
#     # Point-wise convolution
#     x = layers.Conv1D(n_channel, kernel_size=1, padding='same')(x)
#     x = layers.Activation(activation)(x)
    
#     x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
#     outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel]
    
#     return tf.keras.Model(inputs, outputs)

# def build_model(
#     input_shape,
#     pred_len,
#     norm_type,
#     activation,
#     n_block,
#     dropout,
#     ff_dim,
#     target_slice,
#     n_channel=data_loader.n_feature,
#     kernel_size=4,
# ):
#     """Build TSMixer model."""
    
#     inputs = tf.keras.Input(shape=input_shape)
#     x = inputs  # [Batch, Input Length, Channel]
    
#     # Add CNN layers
#     # x = layers.Conv1D(n_channel, kernel_size, padding='same')(x)
#     # x = layers.Activation(activation)(x)
    
#     for _ in range(n_block):
#         x = res_block(x, norm_type, activation, dropout, ff_dim)
    
#     if target_slice:
#         x = x[:, :, target_slice]
    
#     x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
    
#     # Depth-wise convolution
#     x = layers.DepthwiseConv1D(kernel_size=3, padding='same')(x)
#     x = layers.Activation(activation)(x)
    
#     # Point-wise convolution
#     x = layers.Conv1D(n_channel, kernel_size=1, padding='same')(x)
#     x = layers.Activation(activation)(x)
    
#     x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
#     outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel]
    
#     return tf.keras.Model(inputs, outputs)

import tensorflow as tf
from tensorflow.keras import layers
# from tsmixer.tsmixer_basic.data_loader import TSFDataLoader
# from tsmixer.tsmixer_basic.run import parse_args
##################################################################
# args = parse_args()
# data_loader = TSFDataLoader(
#   args.data,
#   args.batch_size,
#   args.seq_len,
#   args.pred_len,
#   args.feature_type,
#   args.target,
# )
# train_data = data_loader.get_train()
# val_data = data_loader.get_val()
# test_data = data_loader.get_test()
#######################################################################
def res_block(inputs, norm_type, activation, dropout, ff_dim):
    
  """Residual block of TSMixer."""

  norm = (
      
    layers.LayerNormalization
    if norm_type == 'L'
    else layers.BatchNormalization
  )

  # Temporal Linear
  x = norm(axis=[-2, -1])(inputs)
  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
  x = layers.Dense(x.shape[-1], activation=activation)(x)
  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Input Length, Channel]
  x = layers.Dropout(dropout)(x)
  res = x + inputs

  # Feature Linear
  x = norm(axis=[-2, -1])(res)
  x = layers.Dense(ff_dim, activation=activation)(
      x
  )  # [Batch, Input Length, FF_Dim]
  x = layers.Dropout(dropout)(x)
  x = layers.Dense(inputs.shape[-1])(x)  # [Batch, Input Length, Channel]
  x = layers.Dropout(dropout)(x)
  return x + res


def build_model(
    input_shape,
    pred_len,
    norm_type,
    activation,
    n_block,
    dropout,
    ff_dim,
    target_slice,
):
  
  
  """Build TSMixer model."""

  inputs = tf.keras.Input(shape=input_shape)
  x = inputs  # [Batch, Input Length, Channel]
  for _ in range(n_block):
    x = res_block(x, norm_type, activation, dropout, ff_dim)

  if target_slice:
    x = x[:, :, target_slice]

  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
  x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
  outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])

  return tf.keras.Model(inputs, outputs)
# def build_model(input_shape, pred_len, norm_type, activation, n_block, dropout, ff_dim, target_slice, n_channel):

# #   """Build TSMixer model."""
#     inputs = tf.keras.Input(shape=input_shape)
    
#   # Define the encoder
#     x1 = layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(inputs)
#     x1 = layers.MaxPooling1D(pool_size=2, padding='same')(x1)
#     x1 = layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(x1)
#     x1 = layers.MaxPooling1D(pool_size=2, padding='same')(x1)
#     x1 = layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(x1)
#     encoded = layers.MaxPooling1D(pool_size=2, padding='same')(x1)

#   # Define the decoder
#     latent_inputs = tf.keras.Input(shape=(encoded,))
#     x2 = layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(latent_inputs)
#     x2 = layers.UpSampling1D(size=2)(x2)
#     x2 = layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(x2)
#     x2 = layers.UpSampling1D(size=2)(x2)
#     x2 = layers.Conv1D(filters=16, kernel_size=3, activation='relu')(x2)
#     x2 = layers.UpSampling1D(size=2)(x2)
#     x2 = tf.keras.layers.Flatten()(x2)
#     x2 = tf.keras.layers.Dense(pred_len * n_channel)(x2)
#     decoded = tf.keras.layers.Reshape((pred_len, n_channel))(x2)
  
   
#     x = inputs  # [Batch, Input Length, Channel]
#     for _ in range(n_block):
        
#         x = res_block(x, norm_type, activation, dropout, ff_dim)

#     if target_slice:
#         x = x[:, :, target_slice]

#     x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
#     x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
#     outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])

#     return tf.keras.Model(inputs, outputs + decoded)