from overcooked_dataset import Subtasks
from copy import deepcopy
import json
import numpy as np
import os
import pandas as pd
from pathlib import Path
import pickle
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class Heuristic():
    def __init__(self, grid, state, mp):
        self.state = state
        self.players_pos_or_list = self.state.players_pos_and_or
        self.players_pos_list = self.state.player_positions
        self.grid = grid
        self.planner = mp
        # initialize all heuristics as None
        self.pos_heur = self.init_heuristic_dict()
        self.history_heur = self.init_heuristic_dict()
        self.layout_heur = self.init_heuristic_dict()

    def init_heuristic_dict(self):
        return dict.fromkeys(Subtasks.SUBTASKS)

    def compute_position_heuristic(self, free_counter_locations):
        ''''
        Position Heuristic calculates the minimum number of 
        steps required to go from current agent location to the 
        start position to end position for each subtask:
            get_onion_from_dispenser: Player position to onion dispenser
            put_onion_in_pot: Closest onion dispenser to pot
            put_onion_closer: Player position to onion dispenser to closest counter available to both players
            get_onion_from_counter: Closest counter to nearest pot
            get_plate_from_dish_rack: Player position to dish dispenser
            get_soup: Closest dish dispenser to pot
            put_plate_closer: Player position to dish dispenser to closest counter available to both players
            get_plate_from_counter: Closest counter to pot
            put_soup_closer: Player position to pot to closest counter available to both players
            get_soup_from_counter: Closest counter to serving location
            serve_soup: Player position to closest pot to serving location
            unknown: Not computed

        '''
        for i,player in enumerate(self.state.players):
            # current player positions and orientations
            player_pos_or = self.players_pos_or_list[i]
            player_pos = self.players_pos_list[i]

            #calculate heuristics for onion class, find onion dispenser locations and pot locations
            onion_disp_loc = self.grid.get_onion_dispenser_locations()
            pot_loc = self.grid.get_pot_locations()
            # closest onion dispenser to current player location
            min_steps_disp, best_disp = self.planner.min_cost_to_feature(player_pos_or, onion_disp_loc, True)
            # closest pot location to the closest onion dispenser
            min_steps_pot = self.planner.min_cost_between_features([best_disp], pot_loc)
            # closest free counter to the closest onion dispenser
            min_steps_disp_to_counter, best_counter = self.planner.min_cost_between_features([best_disp], free_counter_locations, True)
            # closest pot location to closest free counter to put onion in
            min_steps_counter_to_pot = self.planner.min_cost_between_features([best_counter], pot_loc)
            self.pos_heur['get_onion_from_dispenser'] = min_steps_disp
            self.pos_heur['put_onion_in_pot'] = min_steps_pot
            if min_steps_disp_to_counter ==None:
                self.pos_heur['put_onion_closer'] = min_steps_disp + min_steps_disp_to_counter
            self.pos_heur['get_onion_from_counter'] = min_steps_counter_to_pot


            #calculate heuristics for dish class, find dish dispenser locations
            dish_disp_loc = self.grid.get_dish_dispenser_locations()
            # closest dish dispenser to current player location
            min_steps_disp, best_disp = self.planner.min_cost_to_feature(player_pos_or, dish_disp_loc, True)
            # closest pot location to the closest dish dispenser
            min_steps_pot = self.planner.min_cost_between_features([best_disp], pot_loc)
            # closest free counter to the closest dish dispenser
            min_steps_disp_to_counter, best_counter = self.planner.min_cost_between_features([best_disp], free_counter_locations, True)
            # closest pot location to closest free counter to collect soup
            min_steps_counter_to_pot = self.planner.min_cost_between_features([best_counter], pot_loc)
            self.pos_heur['get_plate_from_dish_rack'] = min_steps_disp
            self.pos_heur['get_soup'] = min_steps_pot 
            self.pos_heur['put_plate_closer'] = min_steps_disp + min_steps_disp_to_counter
            self.pos_heur['get_plate_from_counter'] = min_steps_counter_to_pot

            #calculate heuristics for soup class, find serving locations
            serving_locations = self.grid.get_serving_locations()
            # find closest pot to current player location
            min_steps_to_pot, best_pot_pos = self.planner.min_cost_to_feature(player_pos_or, pot_loc, True)
            # distance between closest pot and serving locations
            min_steps_to_serving = self.planner.min_cost_between_features([best_pot_pos], serving_locations)
            # closest free counter to the closest pot
            min_steps_pot_to_counter, best_counter = self.planner.min_cost_between_features([best_pot_pos], free_counter_locations, True)
            # closest seving location to closest free counter to serve soup
            min_steps_counter_to_serving = self.planner.min_cost_between_features([best_counter], serving_locations)
            self.pos_heur['serve_soup'] = min_steps_to_pot + min_steps_to_serving
            self.pos_heur['put_soup_closer'] = min_steps_pot_to_counter
            self.pos_heur['get_soup_from_counter'] = min_steps_counter_to_serving

    def compute_history_heuristic(self, history):
        '''
        Considering history of our agent's actions
        should we consider actions of other agent also?
         '''
        self.history_heur =  history[0]
        

    def compute_layout_heuristic(self,free_counter_locations):
        '''
        Layout Heuristic calculates the number of steps 
        required for independent vs teamwork task the 
        start position to end position by comparing 
        the steps required in the layout for each subtask:
            Independent Yasks:
            get_onion_from_dispenser, put_onion_in_pot: Best path between onion dispensers and pots
            get_plate_from_dish_rack, get_soup: Best path between dish dispensers and pots
            serve_soup: Best path from pots to serving locations
            Teamwork Tasks:
            put_onion_closer: Best path between onion dispensers and counters available to both players 
            get_onion_from_counter: Best path between closest counter and pots                
            put_plate_closer: Best path between dish dispensers and counters available to both players
            get_plate_from_counter: Best path between slosest counter and pots
            put_soup_closer: Best path between pots and counters available to both players
            get_soup_from_counter: Best path between closest counter and serving locations
            unknown: Not computed
        '''
        # calculate heuristics for onion class
        ### for independent task execution ###
        onion_disp_loc = self.grid.get_onion_dispenser_locations()
        pot_loc = self.grid.get_pot_locations()
        # distance between onion dispensers and pot locations
        min_steps_onion_indep = self.planner.min_cost_between_features(onion_disp_loc, pot_loc)
        ### for teamwork task execution  ###
        # distance between onion dispensers and free counters
        min_steps_onion_to_counter, best_counter  = self.planner.min_cost_between_features(onion_disp_loc, free_counter_locations, True)
        # distance between closest counter and pots
        min_steps_counter_to_pot = self.planner.min_cost_between_features([best_counter], pot_loc)

        self.layout_heur['get_onion_from_dispenser'] = min_steps_onion_indep
        self.layout_heur['put_onion_in_pot'] = min_steps_onion_indep
        self.layout_heur['put_onion_closer'] = min_steps_onion_to_counter
        self.layout_heur['get_onion_from_counter'] = min_steps_counter_to_pot

        # calculate heuristics for dish class
        ### for independent task execution ###
        dish_disp_loc = self.grid.get_dish_dispenser_locations()
        pot_loc = self.grid.get_pot_locations()
        # distance between dish dispensers and pots
        min_steps_dish_indep = self.planner.min_cost_between_features(dish_disp_loc, pot_loc)
        ### for teamwork task execution ###
        # distance between dish dispensers and free counters
        min_steps_disp_to_counter, best_counter  = self.planner.min_cost_between_features(dish_disp_loc, free_counter_locations, True)
        # distance between closest counter and pots
        min_steps_counter_to_pot = self.planner.min_cost_between_features([best_counter], pot_loc)

        self.layout_heur['get_plate_from_dish_rack'] = min_steps_dish_indep
        self.layout_heur['get_soup'] = min_steps_dish_indep
        self.layout_heur['put_plate_closer'] = min_steps_disp_to_counter
        self.layout_heur['get_plate_from_counter'] = min_steps_counter_to_pot

        # calculate heuristics for soup class
        ### for independent task execution ###
        serving_locations = self.grid.get_serving_locations()
        pot_loc = self.grid.get_pot_locations()
        # distance between pots and serving locations
        min_steps_soup_indep = self.planner.min_cost_between_features(pot_loc, serving_locations)
        ### for teamwork task execution ###
        # distance between pots and free counters
        min_steps_pot_to_counter, best_counter  = self.planner.min_cost_between_features(pot_loc, free_counter_locations, True)
        # distance between closest counter and serving locations
        min_steps_counter_to_serving = self.planner.min_cost_between_features([best_counter], serving_locations)

        self.layout_heur['serve_soup'] = min_steps_soup_indep
        self.layout_heur['put_soup_closer'] = min_steps_pot_to_counter
        self.layout_heur['get_soup_from_counter'] = min_steps_counter_to_serving
        
    
    def compute_heuristics(self, history, free_counter_locations= []):
        heuristics = []
        self.compute_position_heuristic(free_counter_locations)
        self.compute_history_heuristic(history)
        self.compute_layout_heuristic(free_counter_locations)
        heuristics.append(self.pos_heur)
        heuristics.append(self.history_heur)
        heuristics.append(self.layout_heur)
        return heuristics
        

