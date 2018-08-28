import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from netCDF4 import Dataset
import xarray as xr
import os, random
from config import Conf as config
from deploy import load
from decimal import Decimal

def fetch_inference_batch(input_file_name=config.input_file_name, start_month=config.start_month, pick_random=config.pick_random_file,
                          rollout=config.time_roll_out_length, future_roll_out_length = config.prediction_length):
    path = config.inference_path
    file = input_file_name
    if not pick_random:
        try:
            dataset = xr.open_dataset(path + '/' + file, decode_cf=False)
        except Exception:
            print("Invalid file name or path")
            return
    else:
        try:
            file = str(random.choice(os.listdir(path)))
            dataset = xr.open_dataset(path + '/' + file, decode_cf=False)
        except Exception:
            print("Invalid file path")
            return

    batch_holder = np.zeros([1, rollout, 128, 128])
    try:
        T = np.asarray(np.squeeze(dataset.variables['tas']))[:][:][:]
    except Exception:
        T = np.asarray(np.squeeze(dataset.variables['t2m']))[:][:][:]

    gen_frame_y = np.shape(T)[2]
    gen_frame_x = np.shape(T)[1]
    gen_time = len(T[:])

    if gen_frame_x > 128 or gen_frame_y > 128:
        print('Warning: Frame format larger than' + str(config.frame_size) + 'x' + str(config.frame_size) + ' detected in dataset')
        print('That just wont do, should be the same as training sets !!!')
        return

    span_x = 128 - gen_frame_x
    span_y = 128 - gen_frame_y

    pad_xmin = int(np.floor(span_x / 2))
    pad_xmax = int(np.ceil(span_x / 2))

    pad_ymin = int(np.floor(span_y / 2))
    pad_ymax = int(np.ceil(span_y / 2))

    if (start_month + rollout) > gen_time- 2:
        print('temporal roll out exceeds time steps stored in data file')
        return

    result = 0
    T_gt = 0

    result_s = np.zeros([config.prediction_length, gen_frame_x, gen_frame_y])
    T_gt_s = np.zeros([config.prediction_length, gen_frame_x, gen_frame_y])


    if config.one_shot_prediction == True:
        future_roll_out_length = 1
    for K in range(config.prediction_length):
        try:
            T_s = T[12*K + start_month-rollout:12*K + (start_month)][:][:]
            T_gt = T[start_month + 1 :(start_month + future_roll_out_length + 1)][:][:]
        except Exception:
            print("Indices exceed data bounds");
            break;
        test_unit = T_s[0][20][20]

        if test_unit < 173.0:
            T_s = (273.0 + T_s)

        T_s = T_s / 273
        T_s[T_s > 500] = 0
        T_s[T_s < -500] = 0

        T_padded = np.pad(T_s, ((0, 0), (pad_xmin, pad_xmax), (pad_ymin, pad_ymax)), 'constant')


        result = load(dataset, T_padded, future_roll_out_length, config.save_path)

        if config.one_shot_prediction == False:
            break
        if config.one_shot_prediction == True:
            result_s[K][:][:] = result
            T_gt_s[K][:][:] = T_gt


    if config.one_shot_prediction == False:
        make_plot(result, T_gt, dataset, future_roll_out_length)

    if config.one_shot_prediction == True:
        make_plot(result_s, T_gt_s, dataset, config.prediction_length)

    dataset.close()


def make_plot(result,ground_truth, dataset, future_roll_out_length):

    months = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OKT', 'NOV', 'DEC']
    msm = []
    co = 0
    st_m = config.start_month%12
    for ms in range(config.prediction_length):
        msm.append(months[(st_m + ms)%12])

    lon = np.asarray(dataset.variables['lon'][:])
    lat = np.asarray(dataset.variables['lat'][:])
    the_means = []
    plt.figure(0)
    f, ax = plt.subplots(3,future_roll_out_length)

    for i in range(future_roll_out_length):

        if future_roll_out_length == 1:
            ax1 = ax[0]
            ax2 = ax[1]
            ax3 = ax[2]
        else:
            ax1 = ax[0, i]
            ax2 = ax[1, i]
            ax3 = ax[2, i]
        result_s = np.squeeze(result[i][:][:])
        result_s = result_s * 273 - 273

        ground_truth_s = ground_truth - 273
        ground_truth_s = np.squeeze(ground_truth_s[i][:][:])

        difference = abs(result_s - ground_truth_s)
        mean_difference = round(Decimal(np.mean(difference)),2)

        ax1.set_title(str(msm[i]) + '/' + ' Mean T.diff (C) ' + str(mean_difference))

        m = Basemap(width=6000000, height=6500000,resolution='l', projection='stere',lat_ts=1, lat_0=np.mean(lat), lon_0=np.mean(lon), ax = ax1)
        xi, yi = m(lon,lat)
        cs1 = m.pcolor(xi, yi, result_s, vmin =0, vmax = 30)
        m.drawparallels(np.arange(-80., 81., 10.), labels=[1, 0, 0, 0], fontsize=10)
        m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1], fontsize=10)

        # Add Coastlines, States, and Country Boundaries
        m.drawcoastlines()
        m.drawstates()
        m.drawcountries()

        m = Basemap(width=6000000, height=6500000, resolution='l', projection='stere', lat_ts=1, lat_0=np.mean(lat),
                    lon_0=np.mean(lon), ax=ax2)
        xi, yi = m(lon, lat)
        cs2 = m.pcolor(xi, yi, ground_truth_s, vmin =0, vmax = 30)
        m.drawparallels(np.arange(-80., 81., 10.), labels=[1, 0, 0, 0], fontsize=10)
        m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1], fontsize=10)

        # Add Coastlines, States, and Country Boundaries
        m.drawcoastlines()
        m.drawstates()
        m.drawcountries()

        m = Basemap(width=6000000, height=6500000, resolution='l', projection='stere', lat_ts=1, lat_0=np.mean(lat),
                    lon_0=np.mean(lon), ax=ax3)
        xi, yi = m(lon, lat)
        cs3 = m.pcolor(xi, yi, difference, vmin =0, vmax = 6)
        m.drawparallels(np.arange(-80., 81., 10.), labels=[1, 0, 0, 0], fontsize=10)
        m.drawmeridians(np.arange(-180., 181., 10.), labels=[0, 0, 0, 1], fontsize=10)

        # Add Coastlines, States, and Country Boundaries
        m.drawcoastlines()
        m.drawstates()
        m.drawcountries()
        if i == future_roll_out_length-1:
            cbar1 = m.colorbar(cs1, location='right', pad="10%", ax = ax1)
            cbar1.set_label('Reconstruction  T (C)')

            cbar2 = m.colorbar(cs2, location='right', pad="10%", ax=ax2)
            cbar2.set_label('Ground Truth   T (C)')

            cbar3 = m.colorbar(cs3, location='right', pad="10%", ax=ax3)
            cbar3.set_label('Error   T (C)')


    plt.show()


