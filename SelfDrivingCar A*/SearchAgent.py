import random
import math
from environment import Agent, Environment
from simulator import Simulator
import sys
from searchUtils import searchUtils

class SearchAgent(Agent):
    """ An agent that self drives in the environment.
        This is the object you will be modifying. """

    def __init__(self, env,location=None):
        super(SearchAgent, self).__init__(env)     # Set the agent in the evironment
        self.valid_actions = self.env.valid_actions  # The set of valid actions
        self.searchutil = searchUtils(env)
        self.action_sequence=[]

    def choose_action(self):
        """ The choose_action function is called when the agent is asked to choose
            which action to take next"""
        action=None
        if len(self.action_sequence) >=1:
            action = self.action_sequence[0]

        if len(self.action_sequence) >=2:
            self.action_sequence=self.action_sequence[1:]
        else:
            self.action_sequence=[]
        return action

    def drive(self,goalstates,inputs):
        # print(inputs)
        closedList = []
        openList = []

        # initialize openList
        self.state['previous'] = None
        openList.append([self.state, self.searchutil.heuristic(self.state)])

        #If there are more states to be explored, loop through the list.
        while len(openList) > 0:
            # take out location of car from front of openList
            curr_state = openList[0][0]

            # get the action sequence to that state
            self.action_sequence = self.searchutil.retrieveActionSequenceFromState(curr_state)

            # return if goal state reached
            if self.searchutil.isPresentStateInList(curr_state, goalstates):
                return self.action_sequence

            # remove head from openList
            openList = openList[1:]

            #This list stores all the valid states that the car should move to
            nextList = []
            #For each valid action (forward, left, forward 2x, etc, get the state that the car would go to, and the score)
            for a in self.valid_actions:
                nextLocation = self.env.applyAction(self,curr_state,a)

                #A * Search
                score = len(self.action_sequence) + self.searchutil.heuristic(nextLocation)

                #Check whether it is a valid state where the car can move to, if yes, then add into the nextList
                if inputs[nextLocation["location"][0]][nextLocation["location"][1]] == 0:
                    nextList.append([nextLocation, score])

            if len(nextList) > 0:
                stateWithShortestDistance = min(nextList, key=lambda x: x[1])
                thisState = stateWithShortestDistance[0]

                if self.searchutil.isPresentStateInList(thisState,closedList) == 0 and self.searchutil.isPresentStateInPriorityList(thisState,openList) == 0:
                        self.searchutil.insertStateInPriorityQueue(openList, thisState, stateWithShortestDistance)
                elif self.searchutil.isPresentStateInPriorityList(thisState,openList) == 1:
                        self.searchutil.checkAndUpdateStateInPriorityQueue(openList, thisState, stateWithShortestDistance)
                closedList.append(thisState)

    def update(self):
        """ The update function is called when a time step is completed in the
            environment choose an action """
        startstate = self.state
        goalstates = self.env.getGoalStates()#list of goal states
        inputs = self.env.sense(self) #returns the status in front of the car, 1, 0 or -1
        print("inputs: ", inputs)
        action_sequence = self.drive(goalstates,inputs)

        action = self.choose_action()  # Choose the action needed to move the car
        self.state = self.env.act(self,action) #move the car and return the updated state of the car and the updated environment
        return


def run(filename):
    """ Driving function for running the simulation.
        Press ESC to close the simulation, or [SPACE] to pause the simulation. """
    #change fixmovement to True to fix the movement sequence of other cars across runs

    env = Environment(config_file=filename,fixmovement=False)
    agent = env.create_agent(SearchAgent)
    env.set_primary_agent(agent)

    ##############
    # Create the simulation
    # Flags:
    #   update_delay - continuous time (in seconds) between actions, default is 2.0 seconds
    #   display      - set to False to disable the GUI if PyGame is enabled
    sim = Simulator(env, update_delay=2)

    ##############
    # Run the simulator
    ##############
    sim.run()


if __name__ == '__main__':
    run(sys.argv[1])
