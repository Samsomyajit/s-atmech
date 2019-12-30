import tensorflow as tf
import os
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import backend as K


from s-atmech.AttentionLayer import AttentionLayer

if __name__ == "__main__":
    lyr = AttentionLayer(lyr)
    lyr.compute_output_shape(10)
