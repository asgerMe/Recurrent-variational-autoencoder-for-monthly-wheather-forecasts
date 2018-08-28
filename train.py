import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from config import Conf as config
import network as network
from helper_functions import Adapt_Input
from tensorflow.examples.tutorials.mnist import input_data
from process_netCDF import fetch_train_batch
import os

def train_the_model():
    tf.reset_default_graph()
    nw = network.NetWork()
    init = tf.initialize_all_variables()

    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction = 0.75)

    with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
        sess.run(init)
        writer = tf.summary.FileWriter(os.path.join(config.tensorboard_path,"basic"))
        writer1 = tf.summary.FileWriter(os.path.join(config.tensorboard_path, "inference"))
        writer2 = tf.summary.FileWriter(os.path.join(config.tensorboard_path, "inference_random"))
        writer.add_graph(sess.graph)
        saver = tf.train.Saver(tf.global_variables())
        for i in range(config.epocs):

            batch_dict = fetch_train_batch()
            batch = batch_dict['batch_holder']
            labels = batch_dict['labels']
            feed_dict = {'inputframe:0': batch, 'phase:0' : [1], 'labels:0': labels}
            #print(np.sum(sess.run([nw.broadcast], feed_dict=feed_dict)))
            _, loss = sess.run([nw.train, nw.r_loss], feed_dict=feed_dict)
            print('run # ',i,' -Reconstruction Loss', loss)

            if i % config.save_for_tensorboard == 0:
                batch_dict = fetch_train_batch(inference = True)
                batch = batch_dict['batch_holder']
                labels = batch_dict['labels']
                feed_dict1 = {'inputframe:0': batch, 'phase:0': [0], 'labels:0': labels}

                merge_inf = sess.run(nw.merged, feed_dict=feed_dict1)
                writer1.add_summary(merge_inf,i)

            if i %  config.save_for_tensorboard == 0:
                batch_dict = fetch_train_batch(shuffle_target_frame=True, inference=True)
                batch = batch_dict['batch_holder']
                labels = batch_dict['labels']
                feed_dict2 = {'inputframe:0': batch, 'phase:0': [0], 'labels:0': labels}

                merge_inf_r = sess.run(nw.merged, feed_dict=feed_dict2)
                writer2.add_summary(merge_inf_r, i)


            if i % config.save_for_tensorboard == 0:
                merged = sess.run(nw.merged, feed_dict=feed_dict)
                writer.add_summary(merged, i)

            if i % config.save_at_N_epocs == 0:
                saver.save(sess, os.path.join(config.save_path + "/custom_trained_model","VAE_CUSTOM.ckpt"))







