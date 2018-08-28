import tensorflow as tf
import numpy as np
from config import Conf as config

class NetWork:
    def __init__(self, down_sample_layers = config.enc_dec_layers):

        #f = np.log(config.frame_size / 1.0) / np.log(2.0)
        #f = np.ceil(f)
        #new_size = int(np.power(2, f))

        self.input_frame = tf.placeholder(dtype = tf.float32, shape = (None, config.time_roll_out_length, config.frame_size, config.frame_size), name = "inputframe")
        self.input_frame = tf.expand_dims(self.input_frame, 4)
        #self.input_frame = tf.transpose(self.input_frame, perm=[1, 0, 2, 3, 4])
        self.encoder_conv_out = self.input_frame
        self.phase = tf.placeholder(dtype = tf.int32, shape = (1), name = "phase")

        for i in range(down_sample_layers):
            filters_ = int(config.first_filter_layer * pow(2,(i)))
            act = config.hidden_activation
            self.encoder_conv_out = self.convLayer(x= self.encoder_conv_out, activation = act, filters=filters_, name='encode' + str(i), encoder=True)

        self.encoder = self.encoder_conv_out
        if config.use_conv_lstm:
            self.mea = self.dense_downsample(inputs=self.encoder, name='mean')
            self.sig = self.dense_downsample(inputs=self.encoder, name='variance' ,)
            self.post = self.posterior(mean=self.mea, sigma=self.sig, phase = self.phase)

            self.last_hidden = self.LSTM_layer(self.post)
            self.broadcast = self.dense_upsample(self.last_hidden, enc = self.encoder, name = 'upsample')
        else:
            self.mea = self.convLayer(x= self.encoder, activation = act, filters=int(config.first_filter_layer * pow(2,(config.enc_dec_layers))), name='encode' + 'to_mean', encoder=True)
            self.sig = self.convLayer(x= self.encoder, activation = act, filters=int(config.first_filter_layer * pow(2,(config.enc_dec_layers))), name='encode' + 'to_var', encoder=True)
            self.post = self.posterior(mean=self.meaT, sigma=self.sigT, phase=self.phase)

            self.broadcast = self.conv_LSTM_layer(self.postT)



        for i in range(down_sample_layers):
            act = config.hidden_activation
            filters_ = pow(2,(down_sample_layers - i))*config.first_filter_layer
            nor = True
            if i == down_sample_layers-2:
                filters_ = int(filters_/2)
            if i == down_sample_layers-1:
                filters_ = config.features
                act = config.output_activation
                nor = False
            self.broadcast = self.convLayer(self.broadcast, filters=filters_,name='decode' + str(i), encoder=False, activation=act)

        self.out = tf.identity(self.broadcast, name = 'output')

        self.kl_loss = tf.reduce_sum(self.KL_Loss(self.mea, self.sig))

        self.labels = tf.placeholder(shape = (None, config.frame_size, config.frame_size, 1),dtype = tf.float32, name = 'labels')

        self.r_loss = tf.reduce_sum(self.Reconstruction_Loss(self.out, self.labels))

        self.full_loss = self.r_loss + tf.cast(self.phase, dtype = tf.float32)*self.kl_loss

        self.merged = self.Tensor_Board()

        self.train = tf.train.AdamOptimizer(config.learning_rate).minimize(self.full_loss)

    def convLayer(self, x, filters, name, activation = None, encoder = True):
        with tf.variable_scope(name):
            if encoder:
                strides = (1, config.stride_size, config.stride_size)
                kernel_size = (config.temporal_kernel_size, config.kernel_size, config.kernel_size)
                mx = tf.layers.conv3d(inputs=x, filters = filters, strides = strides, kernel_size=kernel_size, padding='SAME', activation=activation)
            else:
                strides = (config.stride_size, config.stride_size)
                kernel_size = (config.kernel_size, config.kernel_size)
                mx = tf.layers.conv2d_transpose(inputs=x, filters=filters, strides=strides, kernel_size=kernel_size,
                                      padding='SAME', activation=activation)

            return mx


    def dense_downsample(self, inputs, name,  latent_space=config.latent_space_dim,  bias=True, activation=None, initializer= config.hidden_layer_initializer):
        with tf.variable_scope(name):
            sx = inputs.get_shape().as_list()[1]
            sy = inputs.get_shape().as_list()[2]
            sz = inputs.get_shape().as_list()[3]
            sw = inputs.get_shape().as_list()[4]

            w = tf.get_variable(name='weights', shape=(sy, sz, sw, latent_space), initializer=initializer)
            bias_val = tf.get_variable(name='weights_bias', shape=latent_space, initializer=initializer)

            wx = tf.tensordot(inputs, w, [[2, 3, 4], [0, 1, 2]])
            if bias:
                wx += bias_val
            output = wx
            if activation is not None:
                output = activation(wx)

            return output

    def dense_upsample(self, z, enc, name, bias=True, activation=tf.nn.elu,
                         initializer=config.hidden_layer_initializer):
        with tf.variable_scope(name):
            z_shape = z.get_shape().as_list()[1]

            sy = enc.get_shape().as_list()[2]
            sz = enc.get_shape().as_list()[3]
            sw = enc.get_shape().as_list()[4]

            w = tf.get_variable(name='weights', shape=(z_shape, sy, sz, sw), initializer=initializer)
            bias_val = tf.get_variable(name='weights_bias', shape=(1, sy, sz, sw), initializer=initializer)

            wx = tf.tensordot(z, w, [[1], [0]])
            if bias:
                wx += bias_val
            output = wx
            if activation is not None:
                output = activation(wx)

            return output

    def posterior(self, mean, sigma, phase):
        shape = tf.shape(mean)
        samples = tf.random_normal(shape, 0, 1)
        gm = tf.cast(phase, dtype = tf.float32)
        out_samples = mean + tf.multiply(tf.exp(sigma / 2) * samples, gm)
        return out_samples

    def KL_Loss(self, mean, sigma):
        KLloss = 0.5*tf.reduce_sum((tf.exp(sigma) + tf.square(mean) - 1.0 - sigma), axis = 1)
        return KLloss

    def Reconstruction_Loss(self, output, inputs):
        R_loss =  0.5*tf.square(output - inputs)
        return R_loss

    def Tensor_Board(self, on = config.use_tensorboard):
        if on == True:
            tf.summary.scalar('KL Loss', self.kl_loss)
            tf.summary.scalar('Reconstruction Loss', self.r_loss)
            merged = tf.summary.merge_all()
            return merged


    def LSTM_layer(self, input, hidden_units = config.lstm_hidden_units):


        lstm_cell = tf.contrib.rnn.BasicLSTMCell(hidden_units, state_is_tuple=True)
        outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, input,time_major=True, dtype=tf.float32)
        outputs = tf.transpose(outputs, [1,0,2])
        out = tf.gather(outputs, int(outputs.get_shape()[0])- 1)

        return out

    def conv_LSTM_layer(self, input, hidden_units=config.lstm_hidden_units):

        sy = input.get_shape().as_list()[1]
        sz = input.get_shape().as_list()[2]
        sw = input.get_shape().as_list()[3]
        sq = input.get_shape().as_list()[4]

        lstm_cell = tf.contrib.rnn.ConvLSTMCell(conv_ndims = 2, input_shape = ([sz,sw,sq]), kernel_shape = ([config.kernel_size,config.kernel_size]), output_channels = sq);
        outputs, lstm_state = tf.nn.dynamic_rnn(lstm_cell, input, dtype=tf.float32, time_major = True)
        outputs = tf.transpose(outputs, [1,0,2,3,4])
        out = tf.gather(outputs, int(outputs.get_shape()[0]) - 1)

        return out
