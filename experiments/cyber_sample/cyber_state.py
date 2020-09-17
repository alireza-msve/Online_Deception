from __future__ import print_function
from builtins import range
from pomdpy.discrete_pomdp import DiscreteState


class CyberState(DiscreteState):
    """
    The state contains the position of the robot, as well as a boolean value for each cyber
    representing whether it is good (true => good, false => bad).

    This class also implements DiscretizedState in order to allow the state to be easily
    converted to a List
    """

    def __init__(self, grid_position, cyber_states):
        if cyber_states is not None:
            assert cyber_states.__len__() is not 0
        self.position = grid_position
        self.cyber_states = cyber_states  # list

    def distance_to(self, other_cyber_state):
        """
        Distance is measured between beliefs by the sum of the num of different cybers
        """
        assert isinstance(other_cyber_state, CyberState)
        distance = 0
        # distance = self.position.manhattan_distance(other_cyber_state.position)
        for i, j in zip(self.cyber_states, other_cyber_state.cyber_states):
            if i != j:
                distance += 1
        return distance

    def __eq__(self, other_cyber_state):
        return self.position == other_cyber_state.position and self.cyber_states is other_cyber_state.cyber_states

    def copy(self):
        return Cybertate(self.position, self.cyber_states)

    def __hash__(self):
        """
        Returns a decimal value representing the binary state string
        :return:
        """
        return int(self.to_string(), 2)

    def to_string(self):
        state_string = self.position.to_string()
        state_string += " - "

        for i in self.cyber_states:
            if i:
                state_string += "1 "
            else:
                state_string += "0 "
        return state_string

    def print_state(self):
        """
        Pretty printing
        :return:
        """
        self.position.print_position()

        print('Good: {', end=' ')
        good_cybers = []
        bad_cybers = []
        for i in range(0, self.cyber_states.__len__()):
            if self.cyber_states[i]:
                good_cybers.append(i)
            else:
                bad_cybers.append(i)
        for j in good_cybers:
            print(j, end=' ')
        print('}; Bad: {', end=' ')
        for k in bad_cybers:
            print(k, end=' ')
        print('}')

    def as_list(self):
        """
        Returns a list containing the (i,j) grid position boolean values
        representing the boolean cyber states (good, bad)
        :return:
        """
        state_list = [self.position.i, self.position.j]
        for i in range(0, self.cyber_states.__len__()):
            if self.cyber_states[i]:
                state_list.append(True)
            else:
                state_list.append(False)
        return state_list

    def separate_cybers(self):
        """
        Used for the PyGame sim
        :return:
        """
        good_cybers = []
        bad_cybers = []
        for i in range(0, self.cyber_states.__len__()):
            if self.cyber_states[i]:
                good_cybers.append(i)
            else:
                bad_cybers.append(i)
        return good_cybers, bad_cybers