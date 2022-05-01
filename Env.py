# Import routines

import numpy as np
import math
import random
from itertools import permutations

m = 5 # number of cities, ranges from 1 ..... m
t = 24 # number of hours, ranges from 0 .... t-1
d = 7  # number of days, ranges from 0 ... d-1
C = 5 # Per hour fuel and other costs
R = 9 # per hour revenue from a passenger


class CabDriver():
    def __init__(self):
        #initialize the total action space
        self.action_space = [(0,0)] + list(permutations([i for i in range(m)] ,2))
        # initialize the stte space also
        self.state_space = [[city, time, day] for city in range(m) for time in range(t) for day in range(d)]
        # get he initial state
        self.state_init = random.choice(self.state_space)
        # and reset the environment at start to return the environment information.
        self.reset()
                                           
    ## Encoding state (or state-action) for NN input

    def state_encod_arch1(self, state):
        """convert the state into a vector so that it can be fed to the NN. This method converts a given state into a vector format. Hint: The vector is of size m + t + d."""
        ## ONE HOT ENCODING FOR EACH STATE, At which city, at what time and at what day will be dipicted in state encoding.
        state_encod = [0 for _ in range(m+t+d)]
        #city
        state_encod[state[0]] = 1
        #time
        state_encod[m+state[1]] = 1
        #day
        state_encod[m+t+state[2]] = 1         
        return state_encod
    

    def get_requests(self, state):
        """Determining the number of requests basis the location. 
        Use the table specified in the MDP and complete for rest of the locations"""
        #randomly initilize the number of reqeusts for each city.
        location = state[0]
        requests = None
        if location == 0:
            requests = np.random.poisson(2)
        if location == 1:
            requests = np.random.poisson(12)
        if location == 2:
            requests = np.random.poisson(4)
        if location == 3:
            requests = np.random.poisson(7)
        if location == 4:
            requests = np.random.poisson(8)
        #max number of requests cannot exceed 15
        if requests > 15:
            requests = 15
        #get all possible actions indexes
        possible_actions_index = random.sample(range(1, (m-1)*m + 1), requests) + [0]
        actions = [self.action_space[i] for i in possible_actions_index]

        return possible_actions_index,actions   
    
    #fucntion will update the current time and date with respect to the ride time and time passed in episode
    def update_time(self, time, day, ride_time):
        ride_time = int(ride_time)
        if (time + ride_time) < 24:
            time = time + ride_time
        else:
            temp = (time + ride_time) % 24
            days = (time + ride_time) // 24
            day = (day + days) % 7
            time = temp
        return time, day

    
    def next_state_func(self, state, action, Time_matrix):
        """Takes state and action as input and returns next state"""
        next_state = []
        
        total_time = 0
        reach_time = 0
        khali_time = 0
        ride_time = 0
        
        #get all current state information
        current_location = state[0]
        current_time = state[1]
        current_day = state[2]
        drop_location = action[1]
        pickup_location = action[0]
        next_location = state[0]
        
        #action selected is to REJECT the RIDE
        if (pickup_location == 0) and (drop_location == 0):
            khali_time = 1
            next_location = current_location
            
        #accepted the ride but at same location.
        elif current_location == pickup_location:
            ride_time = Time_matrix[current_location][drop_location][current_time][current_day]
            next_location = drop_location
            
        #accepted the ride but at different location.
        else:
            reach_time = Time_matrix[current_location][pickup_location][current_time][current_day]
            new_time, new_day = self.update_time(current_time, current_day, reach_time)
            ride_time = Time_matrix[pickup_location][drop_location][new_time][new_day]
            next_location = drop_location
            
        #update the time and return the reward.
        total_time = (khali_time + reach_time + ride_time)
        next_time, next_day = self.update_time(current_time, current_day,total_time)
        next_state = [next_location, next_time, next_day]
        return next_state, khali_time, reach_time, ride_time
    
    #return the reward
    def reward_func(self, khali_time, reach_time, ride_time):
        usefull_time = ride_time
        khali_time += reach_time
        total_time = khali_time + usefull_time;
        #use full time * reward - total time * Cost penalty
        reward = (R * usefull_time) - (C * (total_time)) 
        return reward
    
    
    def reset(self):
        return self.action_space, self.state_space, self.state_init
    
    #take a step in episode.
    def step(self, state, action, Time_matrix):
        next_state, khali_time, reach_time, ride_time = self.next_state_func(state, action, Time_matrix)
        rewards = self.reward_func(khali_time, reach_time, ride_time)

        total_time = khali_time+ reach_time+ ride_time
        return rewards, next_state, total_time
