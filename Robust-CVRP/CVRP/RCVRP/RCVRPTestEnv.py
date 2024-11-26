from dataclasses import dataclass
import torch

from RCVRProblemDef import get_random_noneuclidian_problems

import os
import sys
root_path = os.path.dirname(os.path.abspath(__file__))
sub_path = os.path.join(root_path, "../Pretrained_Cvrp_Model")
sys.path.append(sub_path)
from test_import import import_test_main

@dataclass
class Reset_State:
    depot_node_matrix_up: torch.Tensor = None
    depot_node_matrix_down: torch.Tensor = None
    # (batch,problem+1,problem+1)
    depot_node_demand: torch.Tensor = None
    # shape: (batch, problem+1)


@dataclass
class Step_State:
    BATCH_IDX: torch.Tensor = None
    POMO_IDX: torch.Tensor = None
    # shape: (batch, pomo)
    selected_count: int = None
    load: torch.Tensor = None
    # shape: (batch, pomo)
    current_node: torch.Tensor = None
    # shape: (batch, pomo)
    ninf_mask: torch.Tensor = None
    # shape: (batch, pomo, problem+1)
    finished: torch.Tensor = None
    # shape: (batch, pomo)


class RCVRPEnv:
    def __init__(self, **env_params):

        # Const @INIT
        self.env_params = env_params
        self.problem_size = env_params['problem_size']
        self.pomo_size = env_params['pomo_size']

        # Const @Load_Problem
        self.batch_size = None
        self.BATCH_IDX = None
        self.POMO_IDX = None
        # IDX.shape: (batch, pomo)

        self.depot_node_matrix_up = None
        self.depot_node_matrix_down = None
        # shape: (batch,problem+1,problem+1)
        self.depot_node_demand = None
        # shape: (batch, problem+1)
        self.node_demand = None
        # shape: (batch,problem)

        # Dynamic-1
        self.selected_count = None
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = None
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        self.at_the_depot = None
        # shape: (batch, pomo)
        self.load = None
        # shape: (batch, pomo)
        self.visited_ninf_flag = None
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = None
        # shape: (batch, pomo, problem+1)
        self.finished = None
        # shape: (batch, pomo)

        # states to return
        self.reset_state = Reset_State()
        self.step_state = Step_State()

    def load_problems(self, batch_size):
        self.batch_size = batch_size

        problem_gen_params = self.env_params['problem_gen_params']
        depot_node_matrix_up,depot_node_matrix_down, node_demand = get_random_noneuclidian_problems(batch_size, self.problem_size, problem_gen_params)

        # 赋值
        self.depot_node_matrix_up = depot_node_matrix_up
        self.depot_node_matrix_down = depot_node_matrix_down
        self.node_demand = node_demand
        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        self.reset_state.depot_node_matrix_up = self.depot_node_matrix_up
        self.reset_state.depot_node_matrix_down = self.depot_node_matrix_down
        self.reset_state.depot_node_demand = self.depot_node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def load_problems_manual(self, depot_node_matrix_up,  depot_node_matrix_down, node_demand):
        self.batch_size = depot_node_matrix_up.size(0)

        self.depot_node_matrix_up = depot_node_matrix_up
        self.depot_node_matrix_down = depot_node_matrix_down
        self.node_demand = node_demand

        self.BATCH_IDX = torch.arange(self.batch_size)[:, None].expand(self.batch_size, self.pomo_size)
        self.POMO_IDX = torch.arange(self.pomo_size)[None, :].expand(self.batch_size, self.pomo_size)

        depot_demand = torch.zeros(size=(self.batch_size, 1))
        # shape: (batch, 1)
        device = depot_demand.device
        node_demand = node_demand.to(device)
        depot_node_matrix_up = depot_node_matrix_up.to(device)
        depot_node_matrix_down = depot_node_matrix_down.to(device)

        self.depot_node_demand = torch.cat((depot_demand, node_demand), dim=1)
        # shape: (batch, problem+1)
        self.reset_state.depot_node_matrix_up = depot_node_matrix_up
        self.reset_state.depot_node_matrix_down = depot_node_matrix_down
        self.reset_state.depot_node_demand = self.depot_node_demand

        self.step_state.BATCH_IDX = self.BATCH_IDX
        self.step_state.POMO_IDX = self.POMO_IDX

    def reset(self):
        self.selected_count = 0
        self.current_node = None
        # shape: (batch, pomo)
        self.selected_node_list = torch.zeros((self.batch_size, self.pomo_size, 0), dtype=torch.long)
        # shape: (batch, pomo, 0~)

        self.at_the_depot = torch.ones(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)
        self.load = torch.ones(size=(self.batch_size, self.pomo_size))
        # shape: (batch, pomo)
        self.visited_ninf_flag = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.ninf_mask = torch.zeros(size=(self.batch_size, self.pomo_size, self.problem_size+1))
        # shape: (batch, pomo, problem+1)
        self.finished = torch.zeros(size=(self.batch_size, self.pomo_size), dtype=torch.bool)
        # shape: (batch, pomo)

        reward = None
        done = False
        return self.reset_state, reward, done

    def pre_step(self):
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        reward = None
        done = False
        return self.step_state,self.selected_node_list, reward, done

    def step(self, selected, CUDA_DEVICE_NUM):
        # selected.shape: (batch, pomo)

        # Dynamic-1
        self.selected_count += 1
        self.current_node = selected
        # shape: (batch, pomo)
        self.selected_node_list = torch.cat((self.selected_node_list, self.current_node[:, :, None]), dim=2)
        # shape: (batch, pomo, 0~)

        # Dynamic-2
        self.at_the_depot = (selected == 0)

        demand_list = self.depot_node_demand[:, None, :].expand(self.batch_size, self.pomo_size, -1)
        # shape: (batch, pomo, problem+1)
        gathering_index = selected[:, :, None]
        # shape: (batch, pomo, 1)
        selected_demand = demand_list.gather(dim=2, index=gathering_index).squeeze(dim=2)
        # shape: (batch, pomo)
        self.load -= selected_demand
        self.load[self.at_the_depot] = 1  # refill loaded at the depot

        self.visited_ninf_flag[self.BATCH_IDX, self.POMO_IDX, selected] = float('-inf')
        # shape: (batch, pomo, problem+1)
        self.visited_ninf_flag[:, :, 0][~self.at_the_depot] = 0  # depot is considered unvisited, unless you are AT the depot

        self.ninf_mask = self.visited_ninf_flag.clone()
        round_error_epsilon = 0.00001
        demand_too_large = self.load[:, :, None] + round_error_epsilon < demand_list
        # shape: (batch, pomo, problem+1)
        self.ninf_mask[demand_too_large] = float('-inf')
        # shape: (batch, pomo, problem+1)

        newly_finished = (self.visited_ninf_flag == float('-inf')).all(dim=2)
        # shape: (batch, pomo)
        self.finished = self.finished + newly_finished
        # shape: (batch, pomo)

        # do not mask depot for finished episode.
        self.ninf_mask[:, :, 0][self.finished] = 0

        # update_step_state
        self.step_state.selected_count = self.selected_count
        self.step_state.load = self.load
        self.step_state.current_node = self.current_node
        self.step_state.ninf_mask = self.ninf_mask
        self.step_state.finished = self.finished

        # returning values
        done = self.finished.all()
        if done:
            # reward = -self._get_travel_distance()  # note the minus sign!
            reward = -self.get_maxregret_travel_distance(CUDA_DEVICE_NUM)
        else:
            reward = None

        return self.step_state, self.selected_node_list, reward, done

    def get_total_distance(self):

        #[100,20,31] (batch,pomo,selected_list_length)
        selected_list_length = self.selected_node_list.size(-1)

        node_from = self.selected_node_list
        # shape: (batch, pomo, node) node=selected_list_length
        node_to = self.selected_node_list.roll(dims=2, shifts=-1)

        # shape: (batch, pomo, node)
        batch_index = self.BATCH_IDX[:, :, None].expand(self.batch_size, self.pomo_size, selected_list_length)
        # shape: (batch, pomo, node)
        selected_cost = self.depot_node_matrix_up[batch_index, node_from, node_to]

        # shape: (batch, pomo, node)
        total_distance = selected_cost.sum(2)
        # shape: (batch, pomo)

        return total_distance

    def replace_elements(self, selected_node_list, up, down):
        # Flatten tensors
        batch_size = up.size(0)
        pomo_size = up.size(1)
        problem_size = up.size(2)
        # batch_size, pomo_size, problem_size = up.size()[:-1]
        select_size = selected_node_list.size()[-1] - 1
        index_flattened = selected_node_list[:, :, :-1] * problem_size + selected_node_list[:, :, 1:]
        up_flattened = up.view(batch_size, pomo_size,problem_size ** 2)  # batch size, pomo size, problem size ^ 2. actually you can use -1 for last argument
        down_flattened = down.view(batch_size, pomo_size, -1)  # batch size, pomo size, problem size ^ 2

        # Create mask tensors
        pivots = torch.arange(0, problem_size ** 2).view(1, 1, 1, -1).expand(batch_size, pomo_size, select_size, -1)
        index = index_flattened.unsqueeze(-1).expand(-1, -1, -1, problem_size ** 2)
        mask_flattened = torch.where(pivots == index, torch.ones_like(pivots), torch.zeros_like(pivots))
        mask_flattened = torch.sum(mask_flattened, dim=-2)

        # Calculate result and reshape
        out = torch.where(mask_flattened == 1, up_flattened, down_flattened)  # get value from up if mask_flattened has value 1

        out = out.view(batch_size, pomo_size, problem_size, problem_size)  # reshape
        return out


    def get_maxregret_travel_distance(self,CUDA_DEVICE_NUM):
        # wcr scenario depot_node_matrix
        dis_up = self.depot_node_matrix_up.unsqueeze(1).repeat(1, self.pomo_size, 1, 1)
        dis_down = self.depot_node_matrix_down.unsqueeze(1).repeat(1, self.pomo_size, 1, 1)

        # wcr scenario
        # put 1st element in the tail (batch,pomo,selected_list_length)--->(batch,pomo,selected_list_length+1)
        first_elements = self.selected_node_list[:, :, 0]
        expanded_elements = first_elements.unsqueeze(-1).expand(self.batch_size, self.pomo_size, 1)
        expanded_node_list = torch.cat((self.selected_node_list, expanded_elements), dim=-1)
        # reverse
        reverse_expand_node_list = expanded_node_list[:, :, torch.arange(expanded_node_list.size(2)-1, -1, -1)]

        final_dis = self.replace_elements(expanded_node_list, dis_up, dis_down)
        final_dis = self.replace_elements(reverse_expand_node_list, dis_up, final_dis)

        # cost_y
        # (batch,pomo,problem+1,problem+1)--->(batch*pomo,problem+1,problem+1)
        node_cnt = final_dis.size(-1)
        final_dis = final_dis.view(self.batch_size*self.pomo_size,node_cnt,node_cnt)
        # (batch,problem+1)--->(batch,pomo,problem+1)---->(batch*pomo,problem+1)
        final_node_demand = self.node_demand.repeat(1,self.pomo_size,1)
        final_node_demand = final_node_demand.view(self.batch_size*self.pomo_size,-1)
        # (batch*pomo)--->(batch,pomo)
        no_aug_cost_y, aug_cost_y = import_test_main(final_dis, final_node_demand, CUDA_DEVICE_NUM)
        aug_cost_y = aug_cost_y.view(self.batch_size,self.pomo_size)

        # cost_x
        # (batch,pomo)
        cost_x = self.get_total_distance()
        max_regret = cost_x - aug_cost_y

        return max_regret


