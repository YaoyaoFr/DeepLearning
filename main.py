'''
    Main function of deep learning framework
'''
import os

from Model.Framework import Framework

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

DIR_PATH = '/'.join(__file__.split('/')[:-4])

def main(schemes: str or list,
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

    for scheme in schemes:
        frame = Framework(scheme=scheme,
                          dir_path=DIR_PATH,
                          spe_pas=spe_pas,
                          )
        # frame.retrain_scheme(source_scheme='CNNWithGLasso/2019-07-03-20-19-32')
        frame.training(start_time=start_time,
                       stop_time=stop_time,
                       )

def gender(scheme: str):
    frame = Framework(scheme=scheme, 
                      dir_path=dir_path)
    frame.evalution_trained_models(exp_date='2019-12-31', exp_clock='17-15')

if __name__ == '__main__':
    main(schemes='CNNGLasso')
    # gender(scheme='CNNGLasso')
