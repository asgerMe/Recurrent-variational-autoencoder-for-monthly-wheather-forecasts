# Helper functions
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config import Conf as config
import network as network
import cv2

def Adapt_Input(inputs):
    sy = np.shape(inputs)[1]
    sx = np.shape(inputs)[0]
    f = np.log(sy / 1.0) / np.log(2.0)
    f = np.ceil(f)
    new_size = np.power(2, f)
    a_input = np.zeros([int(sx), int(new_size), int(new_size)])

    for i in range(sx):
        a_input[i][:][:] = cv2.resize(inputs[i][:][:], (int(new_size), int(new_size)))
    time_stacking = [a_input, a_input, a_input]
    return time_stacking