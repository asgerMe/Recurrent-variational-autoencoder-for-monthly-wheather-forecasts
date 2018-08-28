import tensorflow as tf
import numpy as np

class Conf:

    # Deploy model
    mode = 1 ## 0 for training, 1 for making predictions..... ##


    # INFERENCE
    input_file_name = "tas_EUR-11_MOHC-HadGEM2-ES_historical_r1i1p1_CLMcom-CCLM4-8-17_v1_mon_1986-2005.remap44.nc"  #Input file for inference
    one_shot_prediction = True     # True: Base all predictions solely on model state rollout just before prediction ; False: Run network in recursive mode for multi-state prediction
    use_pretrained_model = True   #Use stored weights and network from report --- Approx: 18 Hours on GPU: Nvidia GTX 1080

    prediction_length =6  #How many states into the future should be predicted
    start_month = 108    #when should the prediction start. Start month marks first predicted month
    pick_random_file = False    #Inference from random file in the inference folder

    inference_path = '/media/asger/RT/Climate/Inference'  #Folder with inference data
    inference_mode = 0


    # TRAINING

    test_path = '/media/asger/RT/Climate/Train' #Folder with test data

    epocs = 20000000          #Number of training runs - should just keep going...
    learning_rate = 0.00002   #How much the network changes the weights for each gradient decent calculation

    use_tensorboard = True     # Monitor training with tensorboard. Use CMD and write:  tensorboard --logdir "path to tensorboard data"
    save_for_tensorboard = 80   #How often should data be written to tensorboard

    save_at_N_epocs = 1500      #Save the network weights for later recovery
    test_example_N_frame = 3000

    save_path = '/media/asger/ce7a4008-6b8f-447d-9acc-614945aef109/variational_autoencoder/weights' #where to save the network weights
    tensorboard_path = '/media/asger/ce7a4008-6b8f-447d-9acc-614945aef109/variational_autoencoder/tensorboard_data/' #where to save data for tensorboard
    image_path = '/media/asger/ce7a4008-6b8f-447d-9acc-614945aef109/variational_autoencoder/Images/'


    #NETWORK PARAMETERS

    use_conv_lstm = False;  #New Version == true from precipitation - article - does not work as well as dense LSTM (false)

    enc_dec_layers = 4      #Layers in the encoding and decoding step. Frame dims are halved for each layer
    latent_space_dim = 512  #Latent state vector size. Number of encoded features. (Larger states seems to make training unstable)
    first_filter_layer = 16 # number of filters on the first conv-layer. Subsequent conv layers get 2^(conv_layers)*first_filter_layer filters ... i.e 32 64 128 etc

    kernel_size = 5     #Filter size for conv-layers - propably don't touch this
    temporal_kernel_size = 3    #The detailed temporal encoding. 3 encodes target states and before and after states .... dont use too many !!
    stride_size = 2  # ALWAYS USE 2 x 2. Ensures consistent broadcasting and downsampling
    time_roll_out_length =5   #Length of temporal roll out. Number of states encoded by the LSTM


    hidden_layer_initializer = tf.contrib.layers.xavier_initializer()  #Init weights ..
    hidden_activation = tf.nn.elu   #Activation function of internal hidden layers. tf.nn.relu may also yield good results
    output_activation = None    #Activation on the last layer. Should be None with the pixel distance reconstruction loss to get linear output on last layer

    lstm_hidden_units =512  #Hidden units in the LSTM. Probably best with same number as latent_space_dim



    frame_size = 128

    features = 1
    batch_size = 40 #Number of data roll out to pass in at once.








