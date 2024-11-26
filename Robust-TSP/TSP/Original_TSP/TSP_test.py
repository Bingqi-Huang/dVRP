import os
import sys
# Path Config
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

import time
import logging
from TSPTester import TSPTester as Tester
from utils.utils import create_logger, copy_all_src

# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0


# parameters
n = 20
int_max = 100

env_params = {
    'node_cnt': n,
    'pomo_size': n  # same as node_cnt
}

model_params = {
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256**(1/2),
    'encoder_layer_num': 5,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16**(1/2),
    'head_num': 16,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/16)**(1/2),
    'eval_type': 'softmax',
    'one_hot_seed_cnt': n,  # must be >= node_cnt
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': '../Pretrained_Tsp_Model/saved_models/noneuclidean/saved_tsp_{}_{}_model'.format(n,int_max),  # directory path of pre-trained model and log files saved.
        'epoch': 2000,  # epoch version of pre-trained model to load.
    },
    # TODO: set as actual
    'saved_problem_folder': "../../Data/nominal_tsp/NmR-{}-{}".format(n,int_max),
    'saved_problem_filename': 'R-20-1000-{}.txt',

    'file_count': 20,
    'test_batch_size': 1,
    'augmentation_enable': True,
    'aug_factor': 1,
    'aug_batch_size': 1,
}
if tester_params['augmentation_enable']:
    tester_params['test_batch_size'] = tester_params['aug_batch_size']


logger_params = {
    'log_file': {
        'desc': 'tsp_test',
        'filename': 'log.txt'
    }
}


def main():

    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params)

    copy_all_src(tester.result_folder)
    t_start = time.time()
    tester.run()
    t_end = time.time()
    print(t_end-t_start)


def _set_debug_mode():
    tester_params['aug_factor'] = 10
    tester_params['file_count'] = 100


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]

if __name__ == "__main__":
    main()
