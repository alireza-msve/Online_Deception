from __future__ import print_function
from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import map
from builtins import hex
from builtins import range
from past.utils import old_div
import logging
import json
import numpy as np
from pomdpy.util import console, config_parser
from .grid_position import GridPosition
from .cyber_state import CyberState
from .cyber_action import CyberAction, ActionType
from .cyber_observation import CyberObservation
from pomdpy.discrete_pomdp import DiscreteActionPool, DiscreteObservationPool
from pomdpy.pomdp import Model, StepResult
from .cyber_position_history import CyberData, PositionAndCyberData

module = "CyberModel"


class RSCellType:
    """
    Cyber are enumerated 0, 1, 2, ...
    other cell types should be negative.
    """
    CYBER = 0
    EMPTY = -1
    GOAL = -2
    OBSTACLE = -3


class CyberModel(Model):
    def __init__(self, args):
        super(CyberModel, self).__init__(args)
        # logging utility
        self.logger = logging.getLogger('POMDPy.CyberModel')
        self.cyber_config = json.load(open(config_parser.cyber_cfg, "r"))

        # -------- Model configurations -------- #

        # The reward for sampling a good 
        self.good_cyber_reward = self.cyber_config['good_cyber_reward']
        # The penalty for sampling a bad 
        self.bad_cyber_penalty = self.cyber_config['bad_cyber_penalty']
        # The reward for exiting the map
        self.exit_reward = self.cyber_config['exit_reward']
        # The penalty for an illegal move.
        self.illegal_move_penalty = self.cyber_config['illegal_move_penalty']
        # penalty for finishing without sampling a cyber
        self.half_efficiency_distance = self.cyber_config['half_efficiency_distance']

        # ------------- Flags --------------- #
        # Flag that checks whether the agent has yet to successfully sample
        self.sampled_cyber_yet = False
        # Flag that keeps track of whether the agent currently believes there are still good
        self.any_good_cybers = False

        # ------------- Data Collection ---------- #
        self.unique_cybers_sampled = []
        self.num_times_sampled = 0.0
        self.good_samples = 0.0
        self.num_reused_nodes = 0
        self.num_bad_cybers_sampled = 0
        self.num_good_checks = 0
        self.num_bad_checks = 0

        # -------------- Map data ---------------- #
        # The number of rows in the map.
        self.n_rows = 0
        # The number of columns in the map
        self.n_cols = 0
        # The number of cybers on the map.
        self.n_cybers = 0
        self.num_states = 0
        self.min_val = 0
        self.max_val = 0

        self.start_position = GridPosition()

        # The coordinates of the cybers.
        self.cyber_positions = []
        # The coordinates of the goal squares.
        self.goal_positions = []
        # The environment map in vector form.
        # List of lists of RSCellTypes
        self.env_map = []

        # The distance from each cell to the nearest goal square.
        self.goal_distances = []
        # The distance from each cell to each cyber.
        self.cyber_distances = []

        # Smart cyber data
        self.all_cyber_data = []

        # Actual cyber states
        self.actual_cyber_states = []

        # The environment map in text form.
        self.map_text, dimensions = config_parser.parse_map(self.cyber_config['map_file'])
        self.n_rows = int(dimensions[0])
        self.n_cols = int(dimensions[1])

        self.initialize()

    # initialize the maps of the grid
    def initialize(self):
        p = GridPosition()
        for p.i in range(0, self.n_rows):
            tmp = []
            for p.j in range(0, self.n_cols):
                c = self.map_text[p.i][p.j]

                # initialized to empty
                cell_type = RSCellType.EMPTY

                if c is 'o':
                    self.cyber_positions.append(p.copy())
                    cell_type = RSCellType.CYBER + self.n_cybers
                    self.n_cybers += 1
                elif c is 'G':
                    cell_type = RSCellType.GOAL
                    self.goal_positions.append(p.copy())
                elif c is 'S':
                    self.start_position = p.copy()
                    cell_type = RSCellType.EMPTY
                elif c is 'X':
                    cell_type = RSCellType.OBSTACLE
                tmp.append(cell_type)

            self.env_map.append(tmp)
        # Total number of distinct states
        self.num_states = pow(2, self.n_cybers)
        self.min_val = old_div(-self.illegal_move_penalty, (1 - self.discount))
        self.max_val = self.good_cyber_reward * self.n_cybers + self.exit_reward

    ''' ===================================================================  '''
    '''                             Utility functions                        '''
    ''' ===================================================================  '''

    # returns the RSCellType at the specified position
    def get_cell_type(self, pos):
        return self.env_map[pos.i][pos.j]

    def get_sensor_correctness_probability(self, distance):
        assert self.half_efficiency_distance is not 0, self.logger.warning("Tried to divide by 0! Naughty naughty!")
        return (1 + np.power(2.0, old_div(-distance, self.half_efficiency_distance))) * 0.5

    ''' ===================================================================  '''
    '''                             Sampling                                 '''
    ''' ===================================================================  '''

    def sample_an_init_state(self):
        self.sampled_cyber_yet = False
        self.unique_cybers_sampled = []
        return CyberState(self.start_position, self.sample_cybers())

    def sample_state_uninformed(self):
        while True:
            pos = self.sample_position()
            if self.get_cell_type(pos) is not RSCellType.OBSTACLE:
                return CyberState(pos, self.sample_cybers())

    def sample_state_informed(self, belief):
        return belief.sample_particle()

    def sample_position(self):
        i = np.random.random_integers(0, self.n_rows - 1)
        j = np.random.random_integers(0, self.n_cols - 1)
        return GridPosition(i, j)

    def sample_cybers(self):
        return self.decode_cybers(np.random.random_integers(0, (1 << self.n_cybers) - 1))

    def decode_cybers(self, value):
        cyber_states = []
        for i in range(0, self.n_cybers):
            cyber_states.append(value & (1 << i))
        return cyber_states

    def encode_cybers(self, cyber_states):
        value = 0
        for i in range(0, self.n_cybers):
            if cyber_states[i]:
                value += (1 << i)
        return value

    ''' ===================================================================  '''
    '''                 Implementation of abstract Model class               '''
    ''' ===================================================================  '''

    def create_observation_pool(self, solver):
        return DiscreteObservationPool(solver)

    def is_terminal(self, cyber_state):
        return self.get_cell_type(cyber_state.position) is RSCellType.GOAL

    def is_valid(self, state):
        if isinstance(state, CyberState):
            return self.is_valid_state(state)
        elif isinstance(state, GridPosition):
            return self.is_valid_pos(state)
        else:
            return False

    def is_valid_state(self, cyber_state):
        pos = cyber_state.position
        return 0 <= pos.i < self.n_rows and 0 <= pos.j < self.n_cols and \
               self.get_cell_type(pos) is not RSCellType.OBSTACLE

    def is_valid_pos(self, pos):
        return 0 <= pos.i < self.n_rows and 0 <= pos.j < self.n_cols and \
               self.get_cell_type(pos) is not RSCellType.OBSTACLE

    def get_legal_actions(self, state):
        legal_actions = []
        all_actions = range(0, 5 + self.n_cybers)
        new_pos = state.position.copy()
        i = new_pos.i
        j = new_pos.j

        for action in all_actions:
            if action is ActionType.NORTH:
                new_pos.i -= 1
            elif action is ActionType.EAST:
                new_pos.j += 1
            elif action is ActionType.SOUTH:
                new_pos.i += 1
            elif action is ActionType.WEST:
                new_pos.j -= 1

            if not self.is_valid_pos(new_pos):
                new_pos.i = i
                new_pos.j = j
                continue
            else:
                if action is ActionType.SAMPLE:
                    cyber_no = self.get_cell_type(new_pos)
                    if 0 > cyber_no or cyber_no >= self.n_cybers:
                        continue
                new_pos.i = i
                new_pos.j = j
                legal_actions.append(action)
        return legal_actions

    def get_max_undiscounted_return(self):
        total = 10
        for _ in self.actual_cyber_states:
            if _:
                total += 10
        return total

    def reset_for_simulation(self):
        self.good_samples = 0.0
        self.num_reused_nodes = 0
        self.num_bad_cybers_sampled = 0
        self.num_bad_checks = 0
        self.num_good_checks = 0

    def reset_for_epoch(self):
        self.actual_cyber_states = self.sample_cybers()
        console(2, module, "Actual cyber states = " + str(self.actual_cyber_states))

    def update(self, step_result):
        if step_result.action.bin_number == ActionType.SAMPLE:
            cyber_no = self.get_cell_type(step_result.next_state.position)
            self.unique_cybers_sampled.append(cyber_no)
            self.num_times_sampled = 0.0
            self.sampled_cyber_yet = True

    def get_all_states(self):
        """
        :return: Forgo returning all states to save memory, return the number of states as 2nd arg
        """
        return None, self.num_states

    def get_all_observations(self):
        """
        :return: Return a dictionary of all observations and the number of observations
        """
        return {
            "Empty": 0,
            "Bad": 1,
            "Good": 2
        }, 3

    def get_all_actions(self):
        """
        :return: Return a list of all actions along with the length
        """
        all_actions = []
        for code in range(0, 5 + self.n_cybers):
            all_actions.append(CyberAction(code))
        return all_actions

    def create_action_pool(self):
        return DiscreteActionPool(self)

    def create_root_historical_data(self, solver):
        self.create_new_cyber_data()
        return PositionAndCyberData(self, self.start_position.copy(), self.all_cyber_data, solver)

    def create_new_cyber_data(self):
        self.all_cyber_data = []
        for i in range(0, self.n_cybers):
            self.all_cyber_data.append(CyberData())

    @staticmethod
    def make_adjacent_position(pos, action_type):
        if action_type is ActionType.NORTH:
            pos.i -= 1
        elif action_type is ActionType.EAST:
            pos.j += 1
        elif action_type is ActionType.SOUTH:
            pos.i += 1
        elif action_type is ActionType.WEST:
            pos.j -= 1
        return pos

    def make_next_position(self, pos, action_type):
        is_legal = True

        if action_type >= ActionType.CHECK:
            pass

        elif action_type is ActionType.SAMPLE:
            # if you took an illegal action and are in an invalid position
            # sampling is not a legal action to take
            if not self.is_valid_pos(pos):
                is_legal = False
            else:
                cyber_no = self.get_cell_type(pos)
                if 0 > cyber_no or cyber_no >= self.n_cybers:
                    is_legal = False
        else:
            old_position = pos.copy()
            pos = self.make_adjacent_position(pos, action_type)
            if not self.is_valid_pos(pos):
                pos = old_position
                is_legal = False
        return pos, is_legal

    def make_next_state(self, state, action):
        action_type = action.bin_number
        next_position, is_legal = self.make_next_position(state.position.copy(), action_type)

        if not is_legal:
            # returns a copy of the current state
            return state.copy(), False

        next_state_cyber_states = list(state.cyber_states)

        # update the any_good_cyber flag
        self.any_good_cybers = False
        for cyber in next_state_cyber_states:
            if cyber:
                self.any_good_cybers = True

        if action_type is ActionType.SAMPLE:
            self.num_times_sampled += 1.0

            cyber_no = self.get_cell_type(next_position)
            next_state_cyber_states[cyber_no] = False

        return CyberState(next_position, next_state_cyber_states), True

    def make_observation(self, action, next_state):
        # generate new observation if not checking or sampling
        if action.bin_number < ActionType.SAMPLE:
            # Defaults to empty cell and Bad cyber
            obs = CyberObservation()
            # self.logger.info("Created cyber Observation - is_good: %s", str(obs.is_good))
            return obs
        elif action.bin_number == ActionType.SAMPLE:
            # The cell is not empty since it contains a cyber, and the cyber is now "Bad"
            obs = CyberObservation(False, False)
            return obs

        # Already sampled this cyber so it is NO GOOD
        if action.cyber_no in self.unique_cybers_sampled:
            return CyberObservation(False, False)

        observation = self.actual_cyber_states[action.cyber_no]

        # if checking a 
        dist = next_state.position.euclidean_distance(self.cyber_positions[action.cyber_no])

        # NOISY OBSERVATION
        # bernoulli distribution is a binomial distribution with n = 1
        # if half efficiency distance is 20, and distance to cyber is 20, correct has a 50/50
        # chance of being True. If distance is 0, correct has a 100% chance of being True.
        correct = np.random.binomial(1.0, self.get_sensor_correctness_probability(dist))

        if not correct:
            # Return the incorrect state if the sensors malfunctioned
            observation = not observation

        # If I now believe that a cyber, previously bad, is now good, change that here
        if observation and not next_state.cyber_states[action.cyber_no]:
            next_state.cyber_states[action.cyber_no] = True
        # Likewise, if I now believe a cyber, previously good, is now bad, change that here
        elif not observation and next_state.cyber_states[action.cyber_no]:
            next_state.cyber_states[action.cyber_no] = False

        # Normalize the observation
        if observation > 1:
            observation = True

        return CyberObservation(observation, False)

    def belief_update(self, old_belief, action, observation):
        pass

    def make_reward(self, state, action, next_state, is_legal):
        if not is_legal:
            return -self.illegal_move_penalty

        if self.is_terminal(next_state):
            return self.exit_reward

        if action.bin_number is ActionType.SAMPLE:
            pos = state.position.copy()
            cyber_no = self.get_cell_type(pos)
            if 0 <= cyber_no < self.n_cybers:
                # If the ACTUALLY is good, AND I currently believe it to be good, I get rewarded
                if self.actual_cyber_states[cyber_no] and state.cyber_states[cyber_no]:
                    # IMPORTANT - After sampling, the cyber is marked as
                    # bad to show that it is has been dealt with
                    # "next states".cyber_states[cyber_no] is set to False in make_next_state
                    state.cyber_states[cyber_no] = False
                    self.good_samples += 1.0
                    return self.good_cyber_reward
                # otherwise, I either sampled I thought was good, sampled a good I thought was bad,
                # or sampled a bad I thought was bad. All bad behavior!!!
                else:
                    # self.logger.info("Bad cyber penalty - %s", str(-self.bad_cyber_penalty))
                    self.num_bad_cybers_sampled += 1.0
                    return -self.bad_cyber_penalty
            else:
                # self.logger.warning("Invalid sample action on non-existent cyber while making reward!")
                return -self.illegal_move_penalty
        return 0

    def generate_reward(self, state, action):
        next_state, is_legal = self.make_next_state(state, action)
        return self.make_reward(state, action, next_state, is_legal)

    def generate_step(self, state, action):
        if action is None:
            print("Tried to generate a step with a null action")
            return None
        elif type(action) is int:
            action = CyberAction(action)

        result = StepResult()
        result.next_state, is_legal = self.make_next_state(state, action)
        result.action = action.copy()
        result.observation = self.make_observation(action, result.next_state)
        result.reward = self.make_reward(state, action, result.next_state, is_legal)
        result.is_terminal = self.is_terminal(result.next_state)

        return result, is_legal

    def generate_particles_uninformed(self, previous_belief, action, obs, n_particles):
        old_pos = previous_belief.get_states()[0].position

        particles = []
        while particles.__len__() < n_particles:
            old_state = CyberState(old_pos, self.sample_cybers())
            result, is_legal = self.generate_step(old_state, action)
            if obs == result.observation:
                particles.append(result.next_state)
        return particles

    @staticmethod
    def disp_cell(rs_cell_type):
        if rs_cell_type >= RSCellType.CYBER:
            print(hex(rs_cell_type - RSCellType.CYBER), end=' ')
            return

        if rs_cell_type is RSCellType.EMPTY:
            print(' . ', end=' ')
        elif rs_cell_type is RSCellType.GOAL:
            print('G', end=' ')
        elif rs_cell_type is RSCellType.OBSTACLE:
            print('X', end=' ')
        else:
            print('ERROR-', end=' ')
            print(rs_cell_type, end=' ')

    def draw_env(self):
        for row in self.env_map:
            list(map(self.disp_cell, row))
            print('\n')
