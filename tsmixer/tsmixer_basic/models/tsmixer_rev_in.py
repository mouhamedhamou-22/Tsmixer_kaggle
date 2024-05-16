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

"""Implementation of TSMixer with Reversible Instance Normalization."""

# from .rev_in import RevNorm
from .tsmixer import res_block
import tensorflow as tf
from tensorflow.keras import layers
# from data_loader import TSFDataLoader

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
  """Build TSMixer with Reversible Instance Normalization model."""

  inputs = tf.keras.Input(shape=input_shape)
  x = inputs  # [Batch, Input Length, Channel]
  # rev_norm = RevNorm(axis=-2)
  # x = rev_norm(x, 'norm')
  for _ in range(n_block):
    x = res_block(x, norm_type, activation, dropout, ff_dim)

  if target_slice:
    x = x[:, :, target_slice]

  x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
  x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
  outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])
  # outputs = rev_norm(outputs, 'denorm', target_slice)
  return tf.keras.Model(inputs, outputs)


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
#     ):
  # """Build TSMixer with Reversible Instance Normalization model."""

  # inputs = tf.keras.Input(shape=input_shape)
  # x = inputs  # [Batch, Input Length, Channel]
  # # rev_norm = RevNorm(axis=-2)
  # # x = rev_norm(x, 'norm')
  # # x = layers.Conv1D(n_channel, kernel_size, padding='same')(x)

  # for _ in range(n_block):
  #   x = res_block(x, norm_type, activation, dropout, ff_dim)

  # if target_slice:
  #   x = x[:, :, target_slice]

  # x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
  # x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
  # outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])
  # # outputs = rev_norm(outputs, 'denorm', target_slice)
  # return tf.keras.Model(inputs, outputs)

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

def build_model(
    input_shape,
    pred_len,
    norm_type,
    activation,
    n_block,
    dropout,
    ff_dim,
    target_slice,
    n_channel,
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

  #ADD HEAD FOR FORCASTING
  x1 = inputs  # [Batch, Input Length, Channel]
  x1 = tf.keras.layers.Flatten()(x1)
  x1 = tf.keras.layers.Dense(pred_len * n_channel)(x1)
  x1 = tf.keras.layers.Reshape((pred_len, n_channel))(x1)
  outputs1 = tf.transpose(x1, perm=[0, 2, 1])

  return tf.keras.Model(inputs, outputs+outputs1)
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
#   # Define the encoder

#   x1 = layers.Conv1D(filters=16, kernel_size=3, activation='relu', padding='same')(inputs)
#   x1 = layers.MaxPooling1D(pool_size=2, padding='same')(x1)
#   x1 = layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(x1)
#   x1 = layers.MaxPooling1D(pool_size=2, padding='same')(x1)
#   x1 = layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(x1)
#   encoded = layers.MaxPooling1D(pool_size=2, padding='same')(x1)

# # Define the decoder
#   latent_inputs = tf.keras.Input(shape=(encoded))
#   x2 = layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(latent_inputs)
#   x2 = layers.UpSampling1D(size=2)(x2)
#   x2 = layers.Conv1D(filters=8, kernel_size=3, activation='relu', padding='same')(x2)
#   x2 = layers.UpSampling1D(size=2)(x2)
#   x2 = layers.Conv1D(filters=16, kernel_size=3, activation='relu')(x2)
#   x2 = layers.UpSampling1D(size=2)(x2)
#   x2 = tf.keras.layers.Flatten()(x2)
#   x2 = tf.keras.layers.Dense(pred_len * n_channel)(x2)
#   decoded = tf.keras.layers.Reshape((pred_len, n_channel))(x2)

   
#   x = inputs  # [Batch, Input Length, Channel]
#   for _ in range(n_block):
#     x = res_block(x, norm_type, activation, dropout, ff_dim)

#   if target_slice:
#     x = x[:, :, target_slice]

#   x = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Channel, Input Length]
#   x = layers.Dense(pred_len)(x)  # [Batch, Channel, Output Length]
#   outputs = tf.transpose(x, perm=[0, 2, 1])  # [Batch, Output Length, Channel])

#   return tf.keras.Model(inputs, outputs + decoded)