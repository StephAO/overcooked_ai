import pickle
import pandas as pd
import sys
sys.path.append("/home/biswas/overcooked_ai/src/")

from overcooked_ai_py.mdp.overcooked_mdp import OvercookedState
obj = pd.read_pickle(r'2019_hh_trials_all.pickle')
index = obj.index
# a_list = list(index)
# print(type(obj.state[[7241]]))

state = obj['state']
if type(state) is str:
    state = json.loads(state)
state = OvercookedState.from_dict(state)