# Path Config
import os
import sys
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

import torch
import logging
from utils.utils import create_logger, copy_all_src
from RTSPTrainer import RTSPTrainer as Trainer

# Machine Environment Config
DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
distributed_train = True
ngpus_per_node = torch.cuda.device_count()


os.environ['TF_CPP_MIN_LOG_LEVEL']='3'
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2'

# parameters
n = 20
intmax = 100
env_params = {
    'node_cnt': n,
    'problem_gen_params': {
        'int_min': 0,
        'int_max': intmax,
        'scaler': intmax,
        'gamma': n * (n - 1) // 2,
    },
    'pomo_size': n  # same as node_cnt
}

model_params = {
    'embedding_dim': 256,
    'sqrt_embedding_dim': 256** (1 / 2),
    'encoder_layer_num': 5,
    'qkv_dim': 16,
    'sqrt_qkv_dim': 16**(1/2),
    'head_num': 16,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'ms_hidden_dim': 16,
    'ms_layer1_init': (1/2)**(1/2),
    'ms_layer2_init': (1/16)**(1/2),
    'eval_type': 'argmax',
    'one_hot_seed_cnt': n,  # must be >= node_cnt
}

optimizer_params = {
    'optimizer': {
        'lr': 2*1e-4,
        'weight_decay': 1e-6
    },
    'scheduler': {
        'milestones': [2001, 2101],
        'gamma': 0.1
    }
}

trainer_params = {
    'world_size': ngpus_per_node,
    'local_rank': os.getenv('LOCAL_RANK', -1),
    'use_cuda': USE_CUDA,
    'distributed': distributed_train,
    'epochs': 4000,
    'train_episodes': 4*100,
    'train_batch_size': 32,
    'logging': {
        'model_save_interval': 10,
        'img_save_interval': 20,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss.json'
        },
    },
    'model_load': {
        'enable': False,  # enable loading pre-trained model
        'path': '../RTSP/rtsp_result/saved_rtsp20_model/encode_plus_100',  # directory path of pre-trained model and log files saved.
        'epoch': 3000,  # epoch version of pre-trained model to laod.
    }
}

logger_params = {
    'log_file': {
        'desc': 'train_rtsp_{}_{}_'.format(n,intmax),
        'filename': 'log.txt',
        'local_rank': os.getenv('LOCAL_RANK', -1),
    }
}

def main():
    if DEBUG_MODE:
        _set_debug_mode()

    create_logger(**logger_params)
    _print_config()

    trainer = Trainer(env_params=env_params,
                      model_params=model_params,
                      optimizer_params=optimizer_params,
                      trainer_params=trainer_params)

    copy_all_src(trainer.result_folder)

    trainer.run()


def _set_debug_mode():

    global trainer_params
    trainer_params['epochs'] = 2
    trainer_params['train_episodes'] = 4
    trainer_params['train_batch_size'] = 2
    trainer_params['validate_episodes'] = 4
    trainer_params['validate_batch_size'] = 2


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}'.format(USE_CUDA))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


if __name__ == "__main__":
    main()
