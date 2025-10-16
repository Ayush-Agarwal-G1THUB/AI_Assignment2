#!/usr/bin/env python3
from MajicWumpusWorld import MagicWumpusWorld
import numpy as np
import time

"""
PUBLICLY AVAILABLE APIs
CurrentState() -> (row, col) or None
TakeAction(a: str) -> (reward: int, next_state: (row,col) or None), where a is one of "Up","Down","Left","Right"
CumulativeRewardAndSteps() -> (reward, steps)
reset(seed: Optional[int]) -> (1, 1), i.e State A
"""

ACTIONS = ("Up", "Right", "Down", "Left")
Q = {} # global variable for state,action -> value mapping
validActionIdx =    [[[1, 2], [1, 2, 3], [1, 2, 3], [2, 3]],
                    [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3], [0, 2, 3]],
                    [[0, 1, 2], [0, 1, 2, 3], [0, 1, 2, 3], [0, 2, 3]],
                    [[0, 1], [0, 1, 3], [0, 1, 3], [0, 3]]]
# global dictionary for the valid action indices for each cell

def initialiseQMap () :
    """
    This function initialises the state,action -> value data structure Q
    """
    # Initialise all values to 0
    for row in range(4) :
        for col in range (4) :
            coord = [row, col]
            for action in range(4) :
                Q[coord, action] = 0

    # TODO try other initialisations like random, optimistic initial values or others

    return

def playWithUser(mw):
    # Interact with environment for two episodes
    for ep in range(1,3):
        while mw.CurrentState() is not None:
            a = input("Enter action : ")
            # Use the immediate reward and next_state in your algorithms
            reward, next_state = mw.TakeAction(a)
    
        # Episode has ended;
        # Print cumulative reward and steps for the episode.
        reward, steps = mw.CumulativeRewardAndSteps()
        print(reward,steps)

        # Reset the environment after each episode. See the reset function.
        mw.reset()

def selectAction (s, policy, epsilon=0.1) -> int:
    """
    This function selects an action from a given state and returns the action index, using a particular action selection policy
        0: Up
        1: Right
        2: Down
        3: Left
    """

    match policy :
        case "greedy" :
            greedy_idx = max(range(len(validActionIdx)), key=lambda a: Q[s, a])
            return greedy_idx

        case "epsilon" :
            greedy_idx = selectAction(s, "greedy")
            if np.random.rand() < epsilon: # random with uniform distribution
                action = np.random.randint(len(validActionIdx[s]))
            else: # greedy
                action = greedy_idx
            return action
        
        # TODO implement UCB?
        
        case default :
            return validActionIdx[s][0]

def SARSA(mw, start_time):
    # Define learning rate alpha
    lr = 0.3

    # Play episodes until terminal conditions
    # TODO Define terminal conditions for learning
    while 1 :
        # initialise s,a
        state = mw.CurrentState()
        action_idx = selectAction(state, None) # TODO Try different action selection policies

        while 1 : # TODO define terminal condition for the episode
            # take action a and observe r,s'
            reward, next_state = mw.takeAction(ACTIONS[action_idx])

            # choose action a' from position s'
            next_action_idx = selectAction(next_state, None) # TODO try different action selection policies

            # update state-action value pair
            # Q(s,a) <- Q(s,a) + alpha[r + gammaQ(s',a') - Q(s,a)]
            Q[state, action_idx] = (1-lr) * Q[state,action_idx] + lr*(reward + Q[next_state, next_action_idx])

            #assign s' to s and a' to a
            state = next_state
            action_idx = next_action_idx
        
    return

def QLearning(mw, start_time) :
    behaviour_policy = None # TODO Try different behaviour policies
    target_policy = "greedy"

    lr = 0.3

    # Play episodes until terminal conditions
    # TODO Define terminal condition for learning
    while 1 :
        # initialise s
        state = mw.CurrentState()

        while 1 : # TODO write terminal condition for the episode
            # select action
            action_idx = selectAction(state, behaviour_policy)

            # take action and observe r,s'
            reward, next_state = mw.TakeAction(ACTIONS[action_idx])

            # find greedy action from s'
            next_action_idx = selectAction(next_state, target_policy)

            # update state-action value pair using greedy action from s'
            # Q(s,a) <- Q(s,a) + alpha[r + maxQ(s',a') - Q(s,a)]
            Q[state, action_idx] = (1-lr) * Q[state,action_idx] + lr*(reward + Q[next_state, next_action_idx])

            # assign s' to s
            state = next_state
        
    return

def expectationActionValue (state, policy) :
    """
    This function returns the expected value of actions taken from the given state
    SIGMA(pi(action|state).Q(state, action))

    Used in expected SARSA
    """

    # TODO Implement this function

    return 0 # TODO placeholder value

def ExpectedSARSA(mw, start_time) :
    lr = 0.3

    # Play episodes until terminal conditions
    # TODO Define terminal conditions
    while 1 :
        # initialise s
        state = mw.CurrentState()

        while 1 : # TODO write terminal condition and take steps until the end of the episode
            # select action a
            action_idx = selectAction(state, None) # TODO try action policies
            
            # take action a observe r,s'
            reward, next_state = mw.takeAction(ACTIONS[action_idx])

            # update state-action value pair
            # Q(s,a) <- Q(s,a) + alpha[r + SIGMA(pi(a'|s').Q(s',a')) - Q(s,a)]
            weighted_next_action_value = expectationActionValue(next_state, None) # TODO try different policies
            Q[state, action_idx] = (1-lr)*Q[state, action_idx] + lr*(reward + weighted_next_action_value)

            # assign s' to s
            state = next_state
        
    return

def DQN() :

    return

def main ():
    playUser = False
    mw = MagicWumpusWorld() # read the ./MWW.json

    initialiseQMap()

    if playUser :
        playWithUser(mw)
    else : # perform learning of choice
        start_time = time.time()
        SARSA(mw, start_time)
        QLearning(mw, start_time)
        ExpectedSARSA(mw, start_time)

if __name__ == '__main__':
    main()

# Play n episodes
# Add terminal conditions for loop