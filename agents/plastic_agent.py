import pandas as pd
import numpy as np
from sklearn import preprocessing
class PlasticAgent():
    '''
    create a Plastic Policy Agent
    -Input required- Models provided in csv format, with two columns- state and action taken by agent in that state, each row is a sample from which we extract transition function
    -param models: List of models provided
    -param num_models: Number of models for creating Behavior Distribution
    -param tf_probs: Extract transition function from list of models
    -param BehaviorDist: Probability Distribution over all models
    -param eta: Value of Eta, default=0.2
    '''
    def __init__(self, num_models, models):
        self.models = models
        self.num_models = num_models
        self.tf_probs = self.extract_trans()
        self.BehaviorDist = self.init_BehaviorDist(self.num_models)
        print("initial distribution:", self.BehaviorDist)
        self.eta = 0.2
    
    def extract_trans(self):
        # Initialize empty transition function for each model
        tf_probs = []
        for model in self.models:
            # Initialize empty transition function for this model
            tf = pd.DataFrame(columns = ['state', 'action', '#instances'])
            # iterate over all model rows
            for index, row in model.iterrows():
                flag = 0
                if(len(tf.index)>0):
                    for id,row_tf in tf.iterrows():
                        # If this state and action already existed, add to instances count
                        if row.at['state'] == row_tf.at['state'] and row.at['p1_curr_subtask'] == row_tf.at['action']:
                            row_tf.at['#instances'] = row_tf.at['#instances'] + 1
                            flag = 1
                            break
                if flag !=1:
                    tf.loc[len(tf.index)] = [row.at['state'], row.at['p1_curr_subtask'], 1]
            tf_prob = pd.DataFrame(columns = ['state', 'action', '#instances'])
            for index,row in tf.iterrows():
                # for each state, sum instances over all actions and calculate probability of taking each action from that state
                state_tf = tf.loc[tf['state'] == row.at['state']]
                tf = tf[tf['state'] != row.at['state']]
                total_instances = state_tf[['#instances']].sum()
                state_tf['#instances'] = state_tf['#instances'].apply(lambda x: x/total_instances.iloc[0])
                tf_prob = tf_prob.append(state_tf)
            tf_probs.append(tf_prob)
        return tf_probs
    
    def init_BehaviorDist(self,n):
        # initialize initial Probability Distribution as uniform
        return [1/n]*n 

    def UpdateBelief(self,state,action):
        for index, model in enumerate(self.models) :
            # Initialize P(action|model,state)
            prob_from_tf = 0
            tf = self.tf_probs[index]
            # Check if there is matching (state,action) pair
            if tf.loc[ (tf['state']== state) & (tf['action']==action ) ].empty is False :
                prob_from_tf = tf.loc[ (tf['state']== state) & (tf['action']==action ) ][['#instances']]
                for id,row in prob_from_tf.iterrows():
                    prob_from_tf = row.at['#instances']
            # calculate loss for model
            loss_model = 1 - prob_from_tf
            # Update Probability Distribution according to loss for that model
            self.BehaviorDist[index] *= (1-self.eta*loss_model)
            # Normalize Probabiity Distribution 
            self.BehaviorDist = [x/sum(self.BehaviorDist) for x in self.BehaviorDist]
    
def main():
    models = []
    # Create list of models
    for i in range(1,3,1):
        model = pd.read_csv (("{}{}{}".format('./data/model_', i, '.csv')))
        models.append(model)
    agent =PlasticAgent(3, models)
    for index,row in models[1].iterrows():
        agent.UpdateBelief(row.at['state'], row.at['p1_curr_subtask'])
    print("final distribution",agent.BehaviorDist)
if __name__ == '__main__':
    main()


        