import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import xarray as xr
import os, random
from config import Conf as config


dataset_list_train = []
dataset_list_inf = []
for G in range(2):

    dataset_list = []

    if G == 0:
        path = config.test_path
    else:
        path = config.inference_path

    file = str(random.choice(os.listdir(path)))
    n_files = len(os.listdir(path))

    Q = 0
    for Q in range(n_files):
        next_dataset = xr.open_dataset(path + '/' + os.listdir(path)[Q], decode_cf=False)
        try:
            full_temps = np.squeeze(np.asarray(np.squeeze(next_dataset.variables['tas']))[:][:][:])
        except Exception:
            full_temps = np.squeeze(np.asarray(np.squeeze(next_dataset.variables['t2m']))[:][:][:])

        next_dataset.close()
        gen_frame_y = np.shape(full_temps)[2]
        gen_frame_x = np.shape(full_temps)[1]

        if gen_frame_x > 128 or gen_frame_y > 128:
            print('Warning:: larger frame format than ' + str(config.frame_size) + 'x' + str(
                config.frame_size) + ' detected in dataset')
            print('skipping')
            continue

        test_unit = full_temps[0][20][20]
        if test_unit > 173.0:
            full_temps= (full_temps)
        if test_unit < 173.0:
            full_temps = (273.0 + full_temps)

        full_temps = full_temps / 273
        full_temps[full_temps > 500] = 0
        full_temps[full_temps < -500] = 0

        span_x = 128 - gen_frame_x
        span_y = 128 - gen_frame_y

        pad_xmin = int(np.floor(span_x / 2))
        pad_xmax = int(np.ceil(span_x / 2))

        pad_ymin = int(np.floor(span_y / 2))
        pad_ymax = int(np.ceil(span_y / 2))

        t_pad = np.pad(full_temps, ((0, 0), (pad_xmin, pad_xmax), (pad_ymin, pad_ymax)), 'constant')

        if G == 0:
            dataset_list_train.append(t_pad)
            print('opening training dataset  --' + str(os.listdir(path)[Q]))
            print(np.shape(t_pad))
        if G == 1:
            dataset_list_inf.append(t_pad)
            print('opening inference dataset  --' + str(os.listdir(path)[Q]))
            print(np.shape(t_pad))







def fetch_train_batch(batch_size = config.batch_size, inference = False, shuffle_target_frame = False):

    batch_holder = np.zeros([config.batch_size,config.time_roll_out_length, 128,128])
    label_holder = np.zeros([config.batch_size, 1, 128,128])
    fetches = 0

    while fetches < batch_size:
        if not inference:
            set_number = np.random.randint(0, np.shape(dataset_list_train)[0])
            full_temps = dataset_list_train[set_number]
        if inference:
            set_number = np.random.randint(0, np.shape(dataset_list_inf)[0])
            full_temps = dataset_list_inf[set_number]


        data_time_dim = np.shape(full_temps)[0]
        time_roll_out = config.time_roll_out_length

        start_index = np.random.randint(0, data_time_dim-time_roll_out-3)

        label_slice_xy = np.asarray(full_temps)[start_index + time_roll_out + 1][:][:]
        time_slice_xy = np.asarray(full_temps)[start_index:(start_index+time_roll_out)][:][:]

        if np.sum(label_slice_xy) == 0 or np.sum(time_slice_xy) == 0:
            continue

        batch_holder[fetches][0:(np.shape(time_slice_xy)[0])][:][:] = time_slice_xy

        if shuffle_target_frame:
            label_slice_xy = np.asarray(full_temps)[np.random.randint(0,data_time_dim-1)][:][:]

        label_holder[fetches][0][:][:] = label_slice_xy
        fetches += 1

    label_holder = np.transpose(label_holder, [0,2,3,1])
    feed_dict = {'batch_holder': batch_holder, 'labels':label_holder}

    return feed_dict
