import sys
sys.path.append("/home/biswas/overcooked_ai/src/")
from arguments import get_arguments
from state_encodings import ENCODING_SCHEMES
from overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState, OvercookedGridworld, Direction, Action
from overcooked_ai_py.planning.planners import MediumLevelActionManager, NO_COUNTERS_PARAMS
import pandas as pd
from copy import deepcopy
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from torch.utils.data import Dataset, DataLoader


class OvercookedDataset(Dataset):
    def __init__(self, env, encoding_fn, args):
        self.env = env
        self.encode_state_fn = encoding_fn
        self.data_path = args.base_dir / args.data_path / args.dataset
        self.main_trials = pd.read_pickle(self.data_path)
        print(f'Number of all trials: {len(self.main_trials)}')
        self.main_trials = self.main_trials[self.main_trials['layout_name'] == args.layout]
        print(f'Number of {args.layout} trials: {len(self.main_trials)}')
        # Remove all transitions where both players noop-ed
        self.main_trials = self.main_trials[self.main_trials['joint_action'] != '[[0, 0], [0, 0]]']
        print(f'Number of {args.layout} trials without double noops: {len(self.main_trials)}')
        # print(self.main_trials['layout_name'])

        self.action_ratios = {k: 0 for k in Action.ALL_ACTIONS}

        def str_to_actions(joint_action):
            """Convert df cell format of a joint action to a joint action as a tuple of indices"""
            try:
                joint_action = json.loads(joint_action)
            except json.decoder.JSONDecodeError:
                # Hacky fix taken from https://github.com/HumanCompatibleAI/human_aware_rl/blob/master/human_aware_rl/human/data_processing_utils.py#L29
                joint_action = eval(joint_action)
            for i in range(2):
                if type(joint_action[i]) is list:
                    joint_action[i] = tuple(joint_action[i])
                if type(joint_action[i]) is str:
                    joint_action[i] = joint_action[i].lower()
                assert joint_action[i] in Action.ALL_ACTIONS
                self.action_ratios[joint_action[i]] += 1
            return np.array([Action.ACTION_TO_INDEX[a] for a in joint_action])

        def add_state_info(df):
            """Convert from a df cell format of a state to an Overcooked State"""
            state = df['state']
            df["current_state"] = pd.NaT
            if type(state) is str:
                state = json.loads(state)
            state = OvercookedState.from_dict(state)
            state_arr = ['c_n', 'c_n']
            for i, player in enumerate(state.players):
                if player.held_object is not None:
                    if player.held_object.name == 'onion':
                        state_arr[i] = 'c_o'
                    if player.held_object.name == 'dish':
                        state_arr[i] = 'c_d'
                    if player.held_object.name == 'soup':
                        state_arr[i] = 'c_s'   
            # print(state_arr)
            df["current_state"] = state_arr
            return df
        
        def add_subtask(df):
            df['index'] = range(1, len(df) + 1)
            df = df.set_index('index')
            df = df.reset_index()
            df["subtask"] = pd.NaT
            for index, row in df.iterrows():
                curr_overstate = df.iloc[index]['state']
                curr_overstate = json.loads(curr_overstate)
                curr_overstate = OvercookedState.from_dict(curr_overstate)
                curr_state = row['current_state']
                subtask_arr = ["nothing", "nothing"]
                for player_index,state in enumerate(curr_state):
                    if state == 'c_o':
                        for i , player in enumerate(curr_overstate.players):
                            if i == player_index:
                                onion_obj = player.held_object
                                if onion_obj!= None:
                                    curr_pos = onion_obj.position
                                    possible_previous_loc = [(curr_pos[0]+1, curr_pos[1]), (curr_pos[0]-1, curr_pos[1]), (curr_pos[0], curr_pos[1]-1), (curr_pos[0], curr_pos[1]+1) ]
                                    for onion_disp_loc in env.mdp.get_onion_dispenser_locations():
                                        if onion_disp_loc in possible_previous_loc:
                                            #subtask is taking onion from dispenser
                                            subtask_arr[i] = "Pick up onion"
                                        else:
                                            #subtask is passing onion from empty counter
                                            subtask_arr[i]  = "Pick up loose onion"
                    if state == 'c_p':
                        for i , player in enumerate(curr_overstate.players):
                            if i == player_index:
                                plate_obj = player.held_object
                                if plate_obj!= None:
                                    curr_pos = plate_obj.position
                                    possible_previous_loc = [(curr_pos[0]+1, curr_pos[1]), (curr_pos[0]-1, curr_pos[1]), (curr_pos[0], curr_pos[1]-1), (curr_pos[0], curr_pos[1]+1) ]
                                    for dish_disp_loc in env.mdp.get_dish_dispenser_locations():
                                        if onion_disp_loc in possible_previous_loc:
                                            #subtask is taking dish from dispenser
                                            subtask_arr[i] = "Pick up plate"
                                        else:
                                            subtask_arr[i] = "Pick up loose plate"
                                            #subtask is taking dish from empty counter
                                            
                    if state == 'c_s':
                        if df.iloc[index-1]["current_state"][player_index] == 'c_n':
                            subtask_arr[player_index] = "Pick up loose soup"
                        if df.iloc[index-1]["current_state"][player_index] == 'c_p':
                            subtask_arr[player_index] = "Get soup from pot"

                    if state == 'c_n':
                        if(df.iloc[index-1]["current_state"][player_index] == 'c_o'):
                            curr_pos = curr_overstate.players[player_index].position
                            neighbour_positions = [(curr_pos[0]+1, curr_pos[1]), (curr_pos[0]-1, curr_pos[1]), (curr_pos[0], curr_pos[1]-1), (curr_pos[0], curr_pos[1]+1) ]
                            for pot_loc in env.mdp.get_pot_locations():
                                if pot_loc in neighbour_positions:
                                    subtask_arr[player_index] = "Place onion in pot" 
                                else:
                                    subtask_arr[player_index] = "Place onion closer"
                        if(df.iloc[index-1]["current_state"][player_index] == 'c_s'):
                            curr_pos = curr_overstate.players[player_index].position
                            neighbour_positions = [(curr_pos[0]+1, curr_pos[1]), (curr_pos[0]-1, curr_pos[1]), (curr_pos[0], curr_pos[1]-1), (curr_pos[0], curr_pos[1]+1) ]
                            for serving_loc in env.mdp.get_serving_locations():
                                if serving_loc in neighbour_positions:
                                    subtask_arr[player_index] = "Serve soup" 
                                else:
                                    subtask_arr[player_index] = "Bring soup closer" 
                        if(df.iloc[index-1]["current_state"][player_index] == 'c_p'):
                            subtask_arr[player_index] = "Bring plate closer"                             
                        

                df.iloc[index]["subtask"] = subtask_arr
            return 0


            
        def str_to_obss(df):
            """Convert from a df cell format of a state to an Overcooked State"""
            state = df['state']
            if type(state) is str:
                state = json.loads(state)
            
            state = OvercookedState.from_dict(state)
            
            visual_obs, agent_obs = self.encode_state_fn(env.mdp, state, args.horizon)
            df['visual_obs'] = visual_obs
            df['agent_obs'] = agent_obs
            return df

        self.main_trials = self.main_trials.apply(add_state_info, axis=1)
        self.main_trials['joint_action'] = self.main_trials['joint_action'].apply(str_to_actions)
        self.main_trials = self.main_trials.apply(str_to_obss, axis=1)
        integer = add_subtask(self.main_trials)
        

        self.class_weights = np.zeros(6)
        for action in Action.ALL_ACTIONS:
            self.class_weights[Action.ACTION_TO_INDEX[action]] = self.action_ratios[action]

        self.class_weights = 1.0 / self.class_weights
        self.class_weights = len(Action.ALL_ACTIONS) * self.class_weights / self.class_weights.sum()

        

    def get_class_weights(self):
        return self.class_weights

    def __len__(self):
        return len(self.main_trials)

    def __getitem__(self, item):
        data_point = self.main_trials.iloc[item]
        a = {
            'visual_obs': data_point['visual_obs'],
            'agent_obs': data_point['agent_obs'],
            'joint_action': data_point['joint_action']
        }
        return a


def main():
    args = get_arguments()
    env = OvercookedEnv.from_mdp(OvercookedGridworld.from_layout_name(args.layout), horizon=400)
    encoding_fn = ENCODING_SCHEMES[args.encoding_fn]
    OD = OvercookedDataset(env, encoding_fn, args)

    dataloader = DataLoader(OD, batch_size=1, shuffle=True, num_workers=0)
    for batch in dataloader:
        # print(batch)
        exit(0)


if __name__ == '__main__':
    main()
