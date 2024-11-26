import torch
import os
from logging import getLogger
from RCVRPTestEnv import RCVRPEnv as Env
from RCVRPModel import RCVRPModel as Model
from utils.utils import *
from RCVRProblemDef import get_single_test_updown


class RCVRPTester:
    def __init__(self,env_params, model_params,tester_params):

        # save arguments
        self.env_params = env_params
        self.model_params = model_params
        self.tester_params = tester_params

        # saved_models folder, logger
        self.logger = getLogger(name='trainer')
        self.result_folder = get_result_folder()

        # cuda
        USE_CUDA = self.tester_params['use_cuda']
        if USE_CUDA:
            cuda_device_num = self.tester_params['cuda_device_num']
            torch.cuda.set_device(cuda_device_num)
            device = torch.device('cuda', cuda_device_num)
            torch.set_default_tensor_type('torch.cuda.FloatTensor')
        else:
            device = torch.device('cpu')
            torch.set_default_tensor_type('torch.FloatTensor')
        self.device = device

        # ENV and MODEL
        self.env = Env(**self.env_params)
        self.model = Model(**self.model_params)

        # Restore
        model_load = tester_params['model_load']
        checkpoint_fullname = '{path}/checkpoint-{epoch}.pt'.format(**model_load)
        checkpoint = torch.load(checkpoint_fullname, map_location=device)

        self.model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()})

        # utility
        self.time_estimator = TimeEstimator()

        # Load all problems into tensor
        self.logger.info(" *** Loading Saved Problems *** ")

        saved_problem_folder = self.tester_params['saved_problem_folder']
        saved_problem_filename = self.tester_params['saved_problem_filename']
        file_count = self.tester_params['file_count']
        node_cnt = self.env_params['problem_size']

        self.all_depot_node_matrix_up = torch.empty(size=(file_count, node_cnt+1, node_cnt+1))
        self.all_depot_node_matrix_down = torch.empty(size=(file_count, node_cnt+1, node_cnt+1))
        self.all_node_demand = torch.empty(size=(file_count, node_cnt))

        for file_idx in range(file_count):
            formatted_filename = saved_problem_filename.format(file_idx + 1)
            full_filename = os.path.join(saved_problem_folder, formatted_filename)
            # problem = load_single_problem_from_file(full_filename, node_cnt, scaler)
            dis_up, dis_down, demand = get_single_test_updown(full_filename)
            self.all_depot_node_matrix_up[file_idx] = dis_up
            self.all_depot_node_matrix_down[file_idx] = dis_down
            self.all_node_demand[file_idx] = demand

        self.logger.info("Done. ")

    def run(self, CUDA_DEVICE_NUM):
        self.time_estimator.reset()

        score_AM = AverageMeter()
        aug_score_AM = AverageMeter()

        test_num_episode = self.tester_params['file_count']
        episode = 0

        while episode < test_num_episode:

            remaining = test_num_episode - episode
            batch_size = min(self.tester_params['test_batch_size'], remaining)

            score, aug_score = self._test_one_batch(episode, episode+batch_size,  CUDA_DEVICE_NUM)

            score_AM.update(score, batch_size)
            aug_score_AM.update(aug_score, batch_size)

            episode += batch_size

            # Logs
            elapsed_time_str, remain_time_str = self.time_estimator.get_est_string(episode, test_num_episode)
            self.logger.info("episode {:3d}/{:3d}, Elapsed[{}], Remain[{}], score:{:.3f}, aug_score:{:.3f}".format(
                episode, test_num_episode, elapsed_time_str, remain_time_str, score, aug_score))

            all_done = (episode == test_num_episode)

            if all_done:
                self.logger.info(" *** Test Done *** ")
                self.logger.info(" NO-AUG SCORE: {:.4f} ".format(score_AM.avg))
                self.logger.info(" AUGMENTATION SCORE: {:.4f} ".format(aug_score_AM.avg))

    def _test_one_batch(self, idx_start, idx_end ,  CUDA_DEVICE_NUM):
        batch_size = idx_end - idx_start

        depot_node_matrix_up_batched = self.all_depot_node_matrix_up[idx_start:idx_end]
        depot_node_matrix_down_batched = self.all_depot_node_matrix_down[idx_start:idx_end]
        node_demand_batched = self.all_node_demand[idx_start:idx_end]

        # Augmentation
        if self.tester_params['augmentation_enable']:
            aug_factor = self.tester_params['aug_factor']
            batch_size = aug_factor * batch_size
            # (batch,problem+1,problem+1)--->(batch*aug_factor,problem+1,problem+1)
            depot_node_matrix_up_batched = depot_node_matrix_up_batched.repeat(aug_factor,1,1)
            depot_node_matrix_down_batched = depot_node_matrix_down_batched.repeat(aug_factor, 1, 1)
            # (batch,problem+1)--->(batch*aug_factor,problem+1)
            node_demand_batched = node_demand_batched.repeat(aug_factor,1)

        else:
            aug_factor = 1

        # Ready
        self.model.eval()
        with torch.no_grad():
            self.env.load_problems_manual( depot_node_matrix_up_batched,depot_node_matrix_down_batched,node_demand_batched)
            reset_state, _, _ = self.env.reset()
            self.model.pre_forward(reset_state)

            # POMO Rollout
            state, selected_node_list, reward, done = self.env.pre_step()
            while not done:
                selected, _ = self.model(state)
                # shape: (batch, pomo)
                state, selected_node_list, reward, done = self.env.step(selected,CUDA_DEVICE_NUM)

            # Return
            batch_size = batch_size // aug_factor
            aug_reward = reward.reshape(aug_factor, batch_size, self.env.pomo_size)
            # shape: (augmentation, batch, pomo)

            max_pomo_reward, max_pomo_reward_idx = aug_reward.max(dim=2)  # get best results from pomo
            # shape: (augmentation, batch)
            no_aug_score = -max_pomo_reward[0, :].float().mean()  # negative sign to make positive value

            #
            selected_node_list = selected_node_list.reshape(aug_factor, batch_size, self.env.pomo_size,
                                                            -1)  # (aug*batch,pomo,n)-->(aug,batch,pomo,n)
            selected_pomo_node = torch.gather(selected_node_list, 2,
                                              max_pomo_reward_idx.unsqueeze(-1).unsqueeze(-1).expand(-1, -1,
                                                                                                     selected_node_list.size(
                                                                                                         2),
                                                                                                     selected_node_list.size(
                                                                                                         -1))).squeeze(
                -2)  # (aug,batch,pomo,n)
            selected_pomo_node = selected_pomo_node[:, :, 0, :]  # (aug,batch,n)

            max_aug_pomo_reward, max_aug_pomo_reward_idx = max_pomo_reward.max(
                dim=0)  # get best results from augmentation
            # shape: (batch,)
            aug_score = -max_aug_pomo_reward.float().mean()  # negative sign to make positive value


            selected_aug_node = torch.gather(selected_pomo_node, 0,
                                             max_aug_pomo_reward_idx.unsqueeze(0).unsqueeze(-1).expand(
                                                 selected_pomo_node.size(0), -1,
                                                 selected_pomo_node.size(-1)))  # (aug,batch,n)

            selected_aug_node = selected_aug_node[0, :, :]  # (,batch,n) eg:(1,20) batch=1 n=20


            ans = selected_aug_node.squeeze(0).tolist()
            # print(ans)

            return no_aug_score.item(), aug_score.item()