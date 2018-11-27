# agent_DQN: Trains a Deep Convolutional Network given a CSV file containing
#            multi-feaure timeseries data, with a window of time per feature
#            to allow 

from gym.envs.registration import register
import gym
import sys
import neat
import os

## \class QAgent
## \brief Q-Learning agent that uses an OpenAI gym environment for fx trading 
##  This agent has separate networks (Pre-trained DeepConvNets) for estimating 
##  each action per step of the simulation environment.
class QAgent():
    
    ## init method
    ## Loads the training and validation datasets, loads the pre-trained models
    #  initialize forex environment.
    def __init__(self, key):
        # First argument is the training dataset
        ts_f = sys.argv[1]
        # Second is validation dataset 
        vs_f = sys.argv[2]
        # Third argument is the prefix (including path) for the dcn pre-trained models 
        # for the actions, all modes are files with .model extention and the prefix is
        # concatenated with a number indicating the action:
        # 0 = Buy/CloseSell/nopCloseBuy
        # 1 = Sell/CloseBuy/nopCloseSell
        # 2 = No Open Buy
        # 3 = No Open Sell
        model_prefix = sys.argv[2]

    ## Load  pretrained models
    def load_action_models(self):
        test=0
    
    ## Evaluate all the action models and select the one with most predicted reward
    def decide_next_action(self):
        test=0
    
    ## Evaluate all the steps on the simulation choosing in each step the best 
    ## action, given the observations per tick. 
    ## \returns the final balance and the cummulative reward
    def evaluate(self):
        test=0       

# main function 
if __name__ == '__main__':
    evaluate()