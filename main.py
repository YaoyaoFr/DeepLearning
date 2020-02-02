'''
    Main function of deep learning framework
'''
import os

from Model.Framework import Framework

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DIR_PATH = '/'.join(__file__.split('/')[:-4])


def main(dataset_list: list or str,
         schemes: str or list,
         start_time: int = 1,
         stop_time: int = 10,
         spe_pas: dict = None,
         ):
    """Main function

    Arguments:
        schemes {str or list} -- [The scheme names of training model]

    Keyword Arguments:
        start_time {int} -- [The training times ] (default: {1})
        stop_time {int} -- [description] (default: {10})
        spe_pas {dict} -- [description] (default: {None})
    """
    if isinstance(schemes, str):
        schemes = [schemes]
    if isinstance(dataset_list, str):
        dataset_list = [dataset_list]

    for dataset in dataset_list:
        for scheme in schemes:
            frame = Framework(dataset=dataset,
                              scheme=scheme,
                              dir_path=DIR_PATH,
                              spe_pas=spe_pas,
                              )
        # frame.retrain_scheme(source_scheme='CNNWithGLasso/2019-07-03-20-19-32')
        frame.training(start_time=start_time,
                       stop_time=stop_time,
                       )


def gender_analysis(scheme: str):
    """Analyse the influce of imbalanced gender

    Arguments:
        scheme {str} --
    """
    frame = Framework(scheme=scheme,
                      dir_path=DIR_PATH)
    frame.gender_analysis(date='2020-01-06', clock='16-53-34')


if __name__ == '__main__':
    main(dataset_list=['ADHD200'],
         schemes=['SSAE'],
         start_time=12,
         stop_time=20)
    # gender_analysis(scheme='CNNGLasso')

    # frame = Framework(scheme='CNNGLasso')
    # frame.get_trained_parameters(date='2020-01-07', clock='15-41-26')
