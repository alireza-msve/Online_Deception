from __future__ import absolute_import
from __future__ import division
from builtins import str
from builtins import range
from past.utils import old_div
from builtins import object
import numpy as np
from pomdpy.pomdp import HistoricalData
from .cyber_action import ActionType
import itertools


# Utility function
class CyberData(object):
    """
    Stores data about each cyber
    """

    def __init__(self):
        # The number of times this cyber has been checked.
        self.check_count = 0
        # The "goodness number"; +1 for each good observation of this cyber, and -1 for each bad
        # observation of this cyber.
        self.goodness_number = 0
        # The calculated probability that this cyber is good.
        self.chance_good = 0.5

    def to_string(self):
        """
        Pretty printing
        """
        data_as_string = " Check count: " + str(self.check_count) + " Goodness number: " + \
                         str(self.goodness_number) + " Probability that cyber is good: " + str(self.chance_good)
        return data_as_string


class PositionAndCyberData(HistoricalData):
    """
    A class to store the robot position associated with a given belief node, as well as
    explicitly calculated probabilities of goodness for each cyber.
    """

    def __init__(self, model, grid_position, all_cyber_data, solver):
        self.model = model
        self.solver = solver
        self.grid_position = grid_position

        # List of CyberData indexed by the cyber number
        self.all_cyber_data = all_cyber_data

        # Holds reference to the function for generating legal actions
        if self.model.preferred_actions:
            self.legal_actions = self.generate_smart_actions
        else:
            self.legal_actions = self.generate_legal_actions

    @staticmethod
    def copy_cyber_data(other_data):
        new_cyber_data = []
        [new_cyber_data.append(CyberData()) for _ in other_data]
        for i, j in zip(other_data, new_cyber_data):
            j.check_count = i.check_count
            j.chance_good = i.chance_good
            j.goodness_number = i.goodness_number
        return new_cyber_data

    def copy(self):
        """
        Default behavior is to return a shallow copy
        """
        return self.shallow_copy()

    def deep_copy(self):
        """
        Passes along a reference to the cyber data to the new copy of CyberPositionHistory
        """
        return PositionAndCyberData(self.model, self.grid_position.copy(), self.all_cyber_data, self.solver)

    def shallow_copy(self):
        """
        Creates a copy of this object's cyber data to pass along to the new copy
        """
        new_cyber_data = self.copy_cyber_data(self.all_cyber_data)
        return PositionAndCyberData(self.model, self.grid_position.copy(), new_cyber_data, self.solver)

    def update(self, other_belief):
        self.all_cyber_data = other_belief.data.all_cyber_data

    def any_good_cybers(self):
        any_good_cybers = False
        for cyber_data in self.all_cyber_data:
            if cyber_data.goodness_number > 0:
                any_good_cybers = True
        return any_good_cybers

    def create_child(self, cyber_action, cyber_observation):
        next_data = self.deep_copy()
        next_position, is_legal = self.model.make_next_position(self.grid_position.copy(), cyber_action.bin_number)
        next_data.grid_position = next_position

        if cyber_action.bin_number is ActionType.SAMPLE:
            cyber_no = self.model.get_cell_type(self.grid_position)
            next_data.all_cyber_data[cyber_no].chance_good = 0.0
            next_data.all_cyber_data[cyber_no].check_count = 10
            next_data.all_cyber_data[cyber_no].goodness_number = -10

        elif cyber_action.bin_number >= ActionType.CHECK:
            cyber_no = cyber_action.cyber_no
            cyber_pos = self.model.cyber_positions[cyber_no]

            dist = self.grid_position.euclidean_distance(cyber_pos)
            probability_correct = self.model.get_sensor_correctness_probability(dist)
            probability_incorrect = 1 - probability_correct

            cyber_data = next_data.all_cyber_data[cyber_no]
            cyber_data.check_count += 1

            likelihood_good = cyber_data.chance_good
            likelihood_bad = 1 - likelihood_good

            if cyber_observation.is_good:
                cyber_data.goodness_number += 1
                likelihood_good *= probability_correct
                likelihood_bad *= probability_incorrect
            else:
                cyber_data.goodness_number -= 1
                likelihood_good *= probability_incorrect
                likelihood_bad *= probability_correct

            if np.abs(likelihood_good) < 0.01 and np.abs(likelihood_bad) < 0.01:
                # No idea whether good or bad. reset data
                # print "Had to reset CYBERData"
                cyber_data = CyberData()
            else:
                cyber_data.chance_good = old_div(likelihood_good, (likelihood_good + likelihood_bad))

        return next_data

    def generate_legal_actions(self):
        legal_actions = []
        all_actions = range(0, 5 + self.model.n_cybers)
        new_pos = self.grid_position.copy()
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

            if not self.model.is_valid_pos(new_pos):
                new_pos.i = i
                new_pos.j = j
                continue
            else:
                if action is ActionType.SAMPLE:
                    cyber_no = self.model.get_cell_type(new_pos)
                    if 0 > cyber_no or cyber_no >= self.model.n_cybers:
                        continue
                new_pos.i = i
                new_pos.j = j
                legal_actions.append(action)
        return legal_actions

    def generate_smart_actions(self):

        smart_actions = []

        n_cybers = self.model.n_cybers

        # check if we are currently on top of a cyber
        cyber_no = self.model.get_cell_type(self.grid_position)

        # if we are on top of a cyber, and it has been checked, sample it
        if 0 <= cyber_no < n_cybers:
            cyber_data = self.all_cyber_data[cyber_no]
            if cyber_data.chance_good == 1.0 or cyber_data.goodness_number > 0:
                smart_actions.append(ActionType.SAMPLE)
                return smart_actions

        worth_while_cyber_found = False
        north_worth_while = False
        south_worth_while = False
        east_worth_while = False
        west_worth_while = False

        # Check to see which cybers are worthwhile

        # Only pursue one worthwhile cyber at a time to prevent the agent from getting confused and
        # doing nothing
        for i in range(0, n_cybers):
            # Once an interesting cyber is found, break out of the for loop

            if worth_while_cyber_found:
                break
            cyber_data = self.all_cyber_data[i]
            if cyber_data.chance_good != 0.0 and cyber_data.goodness_number >= 0:
                worth_while_cyber_found = True
                pos = self.model.cyber_positions[i]
                if pos.i > self.grid_position.i:
                    south_worth_while = True
                elif pos.i < self.grid_position.i:
                    north_worth_while = True
                if pos.j > self.grid_position.j:
                    east_worth_while = True
                elif pos.j < self.grid_position.j:
                    west_worth_while = True

        # If no worth while states were found, just head east
        if not worth_while_cyber_found:
            smart_actions.append(ActionType.EAST)
            return smart_actions

        if north_worth_while:
            smart_actions.append(ActionType.NORTH)
        if south_worth_while:
            smart_actions.append(ActionType.SOUTH)
        if east_worth_while:
            smart_actions.append(ActionType.EAST)
        if west_worth_while:
            smart_actions.append(ActionType.WEST)

        # see which state we might want to check
        for i in range(0, n_cybers):
            cyber_data = self.all_cyber_data[i]
            if cyber_data.chance_good != 0.0 and cyber_data.chance_good != 1.0 and np.abs(cyber_data.goodness_number) < 2:
                smart_actions.append(ActionType.CHECK + i)

        return smart_actions







