import os

import h5py
# import scipy
import matplotlib as mpl
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt, colors as colors


def matrix_to_image(data):
    data = data * 255
    new_im = Image.fromarray(data.astype(np.uint8))
    return new_im


def save_img(img, save_path):
    img.save(save_path)


def comparison_show(data1, data2):
    fig = plt.figure()
    p1 = plt.subplot(121)
    p2 = plt.subplot(122)
    p1.imshow(data1)
    p2.imshow(data2)

    plt.show()


def comparison_save(data1, data2, save_path):
    try:
        data = np.concatenate((data1, data2), axis=1)
    except:
        raise TypeError('Data shape mismatch.')

    img = matrix_to_image(data)
    save_img(img=img, save_path=save_path)


def show_reconstruction(fold: h5py.Group or dict = None,
                        data: np.ndarray = None,
                        subject_num: int = 1,
                        slice_num: int = 1,
                        title: str = '',
                        model=None,
                        tag='show',
                        ):
    """
    Visualize reconstruction of autoencoder
    :param fold: h5py.Group contains raw data and reconstruction data
    :param data: data to be reconstructed
    :param subject_num: the number of random selected subject
    :param slice_num: the number of random selected slice
    :param title: figure title
    :param model: the autoencoder model used to encode the input data
    :param tag: 'show' or 'save' the reconstruction data
    :return:
    """
    if fold is not None:
        for tvt in ['train', 'valid', 'test']:
            data = np.array(fold['{:s} data'.format(tvt)])
            shape = np.shape(data)
            sub_indexes = np.random.randint(low=0, high=shape[0], size=subject_num)
            slice_indexes = np.random.randint(low=0, high=shape[1], size=slice_num)
            if model is None:
                try:
                    data = data[sub_indexes]
                    reconstruction = np.array(fold['{:s} data output'.format(tvt)])[sub_indexes]
                except Exception as e:
                    print(e)
                    return
            else:
                data, _, reconstruction, mses = model.feedforward(data[sub_indexes], if_print=False)
    if data is not None:
        sub_indexes = np.random.randint(low=0, high=len(data), size=[subject_num, ])
        data, _, reconstruction, mses = model.run_encoder(data=np.expand_dims(data[sub_indexes, :, :, 0], -1),
                                                          is_print=False)

    titles = ['{:s} {:s} slice {:d}'.format(title, tvt, i) for i in sub_indexes]
    if tag == 'show':
        show(data, reconstruction, titles=titles)
    elif tag == 'save':
        save(data, reconstruction, titles=titles)


def show(data: np.ndarray, reconstruction: np.ndarray = None, titles: list = None) -> None:
    vmax = np.max(data)
    fig_num = 1

    if reconstruction:
        if len(data) != len(reconstruction):
            print('Dimension doesn\'t match!')
            return

        fig_num = 2
        vmax = np.max(data, reconstruction)

    if titles:
        if len(data) != len(titles):
            print('Dimension doesn\'t match!')
            return

    for idx in range(len(data)):
        fig = plt.figure()
        if titles:
            plt.title(titles[idx])
        sub_fig = fig.add_subplot(1, fig_num, 1)
        sub_fig.imshow(data[idx], vmin=0, vmax=vmax)

        if reconstruction:
            sub_fig = fig.add_subplot(1, fig_num, 2)
            sub_fig.imshow(reconstruction[idx], vmin=0, vmax=vmax)
        try:
            fig.show()
        except:
            print('show error!')


def save(data, reconstruction, titles):
    for idx in range(len(data)):
        pass


def save_or_exhibit(weight,
                    save_path: str,
                    prefix: str,
                    if_save: bool = True,
                    if_show: bool = False,
                    if_absolute: bool = False,
                    if_diagonal: bool = False,
                    vmax: float = None,
                    vmin: float = None,
                    ):
    """
    Save or show the figure of the given weights
    :param weight: The given weights with shape [row, col, (channels)]
    :param save_path: The absolute path of saving directory
    :param prefix: Prefix of figure
    :param if_save: The flag of whether saving the figure to file
    :param if_show: The flag of whether showing the weights in platform
    :param if_absolute: The flag of whether save or exhibit absolute of the weights
    :param if_diagonal: The flag of whether save or exhibit the diagonal element of the weights
    :param vmax: Deprecated soon. The maximum value of given weights
    :param vmin: Deprecated soon. The minimum value of given weights
    :return:
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path, exist_ok=True)

    weight = np.squeeze(weight)
    if len(np.shape(weight)) <= 2:
        weight = np.expand_dims(weight, axis=-1)

    if if_absolute:
        weight = np.abs(weight)

    out_channels = np.shape(weight)[2]

    if if_show:
        row_num = int(np.sqrt(out_channels))
        plt.figure()
        for channel in range(out_channels):
            w = weight[..., channel]
            if if_diagonal:
                w[np.eye(90) == 1] = 0

            plt.subplot(row_num, row_num, channel + 1)
            extent = (1, 90, 1, 90)
            color_map = customize_colormap(w)
            im = plt.imshow(w, cmap=color_map, extent=extent)
            plt.colorbar(im)
        plt.show()
    plt.close()

    if if_save:
        for channel in range(out_channels):
            plt.figure()
            w = weight[..., channel]
            if if_diagonal:
                w[np.eye(90) == 1] = 0

            extent = (1, 90, 1, 90)
            color_map = customize_colormap(w)
            im = plt.imshow(w, cmap=color_map, extent=extent)
            plt.colorbar(im)
            plt.savefig(os.path.join(save_path, '{:s} channel {:d}.jpg'.format(prefix, channel + 1)))
            plt.close()


def customize_colormap(matrix: np.ndarray = None):
    if matrix is None:
        vmax = 1
        vmin = -1
    else:
        vmax = np.max(matrix)
        vmin = np.min(matrix)
        assert vmin <= 0 <= vmax, 'Data distribution must span the ' \
                                  'positive and negative half of the number axis'

    if vmax > -vmin:
        color_max = 1
        color_min = -vmin / vmax
    else:
        color_max = vmax / -vmin
        color_min = 1
    color_middle = color_min / (color_max + color_min)

    color_dict = {'alpha': [(0.0, 1, 1),
                            (0.5, 1, 1),
                            (1.0, 1, 1)],
                  'red': [(0.0, 1 - color_min, 1 - color_min),
                          (color_middle, 1.0, 1.0),
                          (1.0, 1, 1)],
                  'green': [(0.0, 1 - color_min, 1 - color_min),
                            (color_middle, 1.0, 1.0),
                            (1.0, 1 - color_max, 1 - color_max)],
                  'blue': [(0.0, 1, 1),
                           (color_middle, 1.0, 1.0),
                           (1.0, 1 - color_max, 1 - color_max)],
                  }
    color_map = colors.LinearSegmentedColormap(segmentdata=color_dict, name='test')
    return color_map
