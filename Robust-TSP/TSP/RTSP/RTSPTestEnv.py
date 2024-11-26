import torch
import elkai
from dataclasses import dataclass
from RTSProblemDef import get_random_noneuclidian_problems

import os
import sys
# Call the pre-trained model
root_path = os.path.dirname(os.path.abspath(__file__))
sub_path = os.path.join(root_path, "../Pretrained_Tsp_Model")
sys.path.append(sub_path)
from test_import import import_test_main

@dataclass
class Reset_State:
    problems: torch.Tensor
    # shape: (batch, node, node)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor
    POMO_IDX: torch.Tensor
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, node)


class RTSPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        self.env_params = env_params
        self.node_cnt = env_params['node_cnt']
        self.pomo_size = env_params['pomo_size']

        # Const @Load_Problem
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)
        self.dis_up = None
        self.dis_down = None
        # shape: (batch, node, node)

        # Dynamic
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # STEP-State
        self.step_state = None

    def load_problems(self, batch_size):
        self.batch_size = batch_size
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        problem_gen_params = self.env_params['problem_gen_params']
        # shape: (batch, node, node)
        self.dis_up, self.dis_down = get_random_noneuclidian_problems(batch_size, self.node_cnt, problem_gen_params)

    def load_problems_manual(self, dis_up, dis_down):
        # problems.shape: (batch, node, node)
        self.batch_size = dis_up.size(0)
        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)
        self.dis_up = dis_up
        self.dis_down = dis_down

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.empty((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self._create_step_state()

        reward = None
        done = False
        return Reset_State(problems=self.dis_up), Reset_State(problems=self.dis_down), reward, done

    def _create_step_state(self):
        self.step_state = Step_State(BATCH_IDX=self.BATCH_IDX, POMO_IDX=self.POMO_IDX)
        self.step_state.ninf_mask = torch.zeros((self.batch_size, self.pomo_size, self.node_cnt))
        # shape: (batch, pomo, node)

    def pre_step(self):
        reward = None
        done = False
        return self.step_state, self.selected_node_list, reward, done

    def step(self, node_idx, CUDA_DEVICE_NUM):
        # node_idx.shape: (batch, pomo)

        self.selected_count += 1
        self.current_node = node_idx
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~node)

        self._update_step_state()

        # returning values
        done = (self.selected_count == self.node_cnt)
        if done:
            reward = -self.get_maxregret_travel_distance(CUDA_DEVICE_NUM)
            # reward = -self._lkh_get_travel_distance()
            # shape: (batch, pomo)
        else:
            reward = None
        return self.step_state, self.selected_node_list, reward, done

    def _update_step_state(self):
        self.step_state.current_node = self.current_node
        # shape: (batch, pomo)
        self.step_state.ninf_mask[self.BATCH_IDX, self.POMO_IDX, self.current_node] = float('-inf')
        # shape: (batch, pomo, node)

    def _get_total_distance(self):
        node_from = self.selected_node_list
        # shape: (batch, pomo, node)
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)
        # shape: (batch, pomo, node)
        batch_index = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, self.node_cnt)
        # shape: (batch, pomo, node)
        selected_cost = self.dis_up[batch_index, node_from, node_to]
        # shape: (batch, pomo, node)
        total_distance = selected_cost.sum(2)
        # shape: (batch, pomo)

        return total_distance

    def get_link_seq(self, solutions):
        # solutions.shape = [batch_size, pomo_size, problem_ size]
        batch_size, pomo_size, problem_size = solutions.size()
        solutions = solutions.view(batch_size, -1, problem_size)
        rec = torch.zeros((batch_size, pomo_size, problem_size), device=solutions.device).long()
        index = torch.zeros((batch_size, pomo_size, 1), device=solutions.device).long()

        for i in range(problem_size - 1):
            rec.scatter_(2, solutions.gather(2, index + i), solutions.gather(2, index + i + 1))

        rec.scatter_(2, solutions[:, :, -1].view(batch_size, -1, 1), solutions.gather(2, index))
        return rec

    def get_link_seq_forward(self, solutions):
        # solutions.shape = [batch_size,pomo_size,problem_ size]
        batch_size, pomo_size, problem_size = solutions.size()
        solutions = solutions.view(batch_size, -1, problem_size)
        rec = torch.zeros((batch_size, pomo_size, problem_size), device=solutions.device).long()
        index = torch.zeros((batch_size, pomo_size, 1), device=solutions.device).long()

        for i in range(problem_size - 1, 0, -1):
            rec.scatter_(2, solutions.gather(2, index + i), solutions.gather(2, index + i - 1))

        rec.scatter_(2, solutions[:, :, 0].view(batch_size, -1, 1), solutions.gather(2, index + problem_size - 1))
        return rec

    def _lkh_get_travel_distance(self):

        link = self.get_link_seq(self.selected_node_list)
        link_forward = self.get_link_seq_forward(self.selected_node_list)
        link = link.unsqueeze(3)  # [batch,pomo,problem,1]
        link_forward = link_forward.unsqueeze(3)

        # [batch,pomo,problem,problem]
        final_dis = self.dis_down.clone().unsqueeze(1).repeat(1, self.pomo_size, 1, 1)
        dis_up = self.dis_up.unsqueeze(1).repeat(1, self.pomo_size, 1, 1)
        # wcr matrix [batch,pomo,problem,problem]
        final_dis.scatter_(-1, link, dis_up.gather(-1, link))
        final_dis.scatter_(-1, link_forward, dis_up.gather(-1, link_forward))

        cost_x = final_dis.gather(-1, link).sum(2)

        # worst case max-regret TSP
        problem_size = dis_up.size(2)
        wcr_matrix = [[0] * problem_size for i in range(problem_size)]
        route_y = torch.zeros(self.batch_size, self.pomo_size, problem_size)
        for i in range(self.batch_size):
            for j in range(self.pomo_size):
                wcr_matrix = final_dis[i][j].cpu().numpy()
                temp_list_y = elkai.solve_float_matrix(wcr_matrix)
                route_y[i][j] = torch.tensor(temp_list_y)
                del wcr_matrix

        y = route_y.clone().type(torch.int64)
        link_y = self.get_link_seq(y)
        link_y = torch.unsqueeze(link_y, dim=3)

        cost_y = final_dis.gather(-1, link_y).sum(2)

        max_regret = (cost_x - cost_y).squeeze(-1)

        return max_regret

    def replace_elements(self, selected_node_list, up, down):
        # Flatten tensors
        # print(up.shape)
        batch_size = up.size(0)
        pomo_size = up.size(1)
        problem_size = up.size(2)
        # batch_size, pomo_size, problem_size = up.size()[:-1]
        select_size = selected_node_list.size()[-1] - 1
        index_flattened = selected_node_list[:, :, :-1] * problem_size + selected_node_list[:, :, 1:]
        up_flattened = up.view(batch_size, pomo_size,
                               problem_size ** 2)  # batch size, pomo size, problem size ^ 2. actually you can use -1 for last argument
        down_flattened = down.view(batch_size, pomo_size, -1)  # batch size, pomo size, problem size ^ 2
        # print(index_flattened)
        # Create mask tensors
        pivots = torch.arange(0, problem_size ** 2).view(1, 1, 1, -1).expand(batch_size, pomo_size, select_size, -1)
        index = index_flattened.unsqueeze(-1).expand(-1, -1, -1, problem_size ** 2)
        mask_flattened = torch.where(pivots == index, torch.ones_like(pivots), torch.zeros_like(pivots))
        mask_flattened = torch.sum(mask_flattened, dim=-2)
        # print(mask_flattened)

        # Calculate result and reshape
        out = torch.where(mask_flattened == 1, up_flattened,
                          down_flattened)  # get value from up if mask_flattened has value 1

        out = out.view(batch_size, pomo_size, problem_size, problem_size)  # reshape
        return out

    def get_maxregret_travel_distance(self, CUDA_DEVICE_NUM):
        # selected_node_list (batch_size , pomo_size , problem_size)
        # wcr scenario matrix
        dis_up = self.dis_up.unsqueeze(1).repeat(1, self.pomo_size, 1, 1)
        dis_down = self.dis_down.unsqueeze(1).repeat(1, self.pomo_size, 1, 1)

        # put 1st element in the tail (batch,pomo,selected_list_length)--->(batch,pomo,selected_list_length+1)
        first_elements = self.selected_node_list[:, :, 0]
        expanded_elements = first_elements.unsqueeze(-1).expand(self.batch_size, self.pomo_size, 1)
        expanded_node_list = torch.cat((self.selected_node_list, expanded_elements), dim=-1)
        # reverse
        reverse_expand_node_list = expanded_node_list[:, :, torch.arange(expanded_node_list.size(2) - 1, -1, -1)]

        final_dis = self.replace_elements(expanded_node_list, dis_up, dis_down)
        final_dis = self.replace_elements(reverse_expand_node_list, dis_up, final_dis)

        # cost_x
        cost_x = self._get_total_distance()

        # cost_y
        # (batch,pomo,problem+1,problem+1)--->(batch*pomo,problem+1,problem+1)
        node_cnt = final_dis.size(-1)
        final_dis = final_dis.view(self.batch_size * self.pomo_size, node_cnt, node_cnt)
        # (batch*pomo)--->(batch,pomo)
        no_aug_cost_y, aug_cost_y = import_test_main(final_dis, CUDA_DEVICE_NUM)
        aug_cost_y = aug_cost_y.view(self.batch_size, self.pomo_size)

        max_regret = cost_x - aug_cost_y

        return max_regret