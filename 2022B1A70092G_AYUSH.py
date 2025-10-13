#!/usr/bin/env python3
from MajicWumpusWorld import MagicWumpusWorld

ACTIONS = ("Up", "Down", "Left", "Right")

def main():
    mw = MagicWumpusWorld()  # reads ./MWW.json

    #Interact with environment for two episodes
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

if __name__ == '__main__':
    main()
