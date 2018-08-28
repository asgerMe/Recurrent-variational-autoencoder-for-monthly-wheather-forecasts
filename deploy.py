import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config import Conf as config
import cv2
from tensorflow.python.tools.inspect_checkpoint import print_tensors_in_checkpoint_file
from tensorflow.examples.tutorials.mnist import input_data
from helper_functions import Adapt_Input
from process_netCDF import fetch_train_batch
import random, os
import xarray as xr
from mpl_toolkits.basemap import Basemap

def load(dataset, data_in, future_roll_out_length = config.prediction_length,  path = config.save_path):
    tf.reset_default_graph()
    vae_name = 'VAE.ckpt.meta'
    if not config.use_pretrained_model:
        path = os.path.join(path,"custom_trained_model")
        vae_name = 'VAE_CUSTOM.ckpt.meta'

    path2 = os.path.join(path,vae_name)
    print(path2)

    saver = tf.train.import_meta_graph(path2)
    lat_f = np.asarray(dataset.variables['lat'])[:]
    gen_frame_y = int(np.shape(lat_f)[1])
    gen_frame_x = int(np.shape(lat_f)[0])

    predictions = np.zeros([future_roll_out_length , gen_frame_x, gen_frame_y])
    with tf.Session() as sess:
        graph = tf.get_default_graph()
        sub_path = "./weights/"
        if config.use_pretrained_model == False:
            sub_path = './weights/custom_trained_model/'
        saver.restore(sess, tf.train.latest_checkpoint(sub_path))
        print('model restored')

        placeholder = graph.get_tensor_by_name("inputframe:0")
        phase = graph.get_tensor_by_name("phase:0")
        op_to_restore = graph.get_tensor_by_name("output:0")

        test_batch = [data_in]
        for i in range(future_roll_out_length):

            feed_dict = {placeholder: test_batch, phase: [0]}
            case = sess.run(op_to_restore, feed_dict=feed_dict)
            case_stack = np.transpose(case,[0,3,1,2])

            test_batch = np.hstack([test_batch, case_stack])
            test_batch = [test_batch[0][1:][:][:]]

            # Displaying for later
            case = np.squeeze(case)

            span_x = 128 - gen_frame_x
            span_y = 128 - gen_frame_y

            pad_xmin = int(np.floor(span_x / 2))
            pad_xmax = int(np.ceil(span_x / 2))

            pad_ymin = int(np.floor(span_y / 2))
            pad_ymax = int(np.ceil(span_y / 2))

            tfb = case[pad_xmin:(pad_xmin + np.shape(lat_f)[0]),pad_ymin:(pad_ymin +np.shape(lat_f)[1])]
            predictions[i][:][:] = tfb

        return predictions



