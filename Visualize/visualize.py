import h5py
# import scipy
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import pyplot as plt


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
