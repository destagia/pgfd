"""
Train the policy with finite-difference policy gradient
"""

import chainer
import chainer.functions as F
import cupy
import numpy as np

from pgfd import const, game

xp = cupy

def unit_vector(k, e):
    unit_vector = xp.zeros(const.POLICY_PARAMETER_SIZE, dtype=xp.float32)
    unit_vector[k] = e
    return unit_vector

class Policy(object):
    def __init__(self, parameter):
        self.__parameter = parameter

    def __call__(self, state):
        """
        Policy evaluate for a state

        devide the parameter into the two blocks.
        The first one is for converting state to hidden layer.
        The second one is for converting hidden layer to action.
        """
        separator = int(const.POLICY_PARAMETER_SIZE / const.STATE_SIZE)
        w_size = int(separator / const.STATE_SIZE)

        w1 = self.__parameter[0:separator].reshape(w_size, const.STATE_SIZE)
        w2 = self.__parameter[separator:const.POLICY_PARAMETER_SIZE].reshape(const.STATE_SIZE, w_size)

        h = F.linear(state, w1)
        return F.linear(h, w2)

class Trainer(object):
    def __init__(self, policy_factory, game_manager):
        self.__rollout = Rollout(policy_factory, game_manager)
        self.__target_parameter = xp.random.rand(const.POLICY_PARAMETER_SIZE) * const.INITIAL_W_SCALE

    def train(self):
        incomes = np.asarray([self.__rollout(self.__target_parameter) for j in range(0, const.TRAIN_ROLLOUT_COUNT)])
        expected_income = incomes.mean()

        d_J = xp.zeros(const.POLICY_PARAMETER_SIZE)
        d_parameter = xp.zeros((const.POLICY_PARAMETER_SIZE, const.POLICY_PARAMETER_SIZE))

        for k in range(0, const.POLICY_PARAMETER_SIZE):
            d_parameter_increment = unit_vector(k, const.PARAMETER_E)
            p = self.__target_parameter + d_parameter_increment
            d_expected_income = (self.__rollout(p) - expected_income) / const.PARAMETER_E

            d_parameter[k] = d_parameter_increment
            d_J[k] = d_expected_income

        # policy gradient in finite-difference
        gfd = xp.linalg.inv(d_parameter.T.dot(d_parameter)).dot(d_parameter.T).dot(d_J)
        print(gfd)

        # update the parameter of policy with gradient
        self.__target_parameter += (const.LEARNING_RATE * gfd)


class Rollout(object):
    def __init__(self, policy_factory, game_manager):
        self.__policy_factory = policy_factory
        self.__game_manager = game_manager

    def __call__(self, policy_parameter):
        policy = self.__policy_factory(policy_parameter)
        game = game_manager.new_game()

        action = policy(xp.asarray([game.current_state()])).data[0]
        game.shoot(action)

        income = 0
        discount_rate = const.DISCOUNT_RATE
        while not game.is_terminal:
            reward = game.update()
            income += discount_rate * reward
            discount_rate *= const.DISCOUNT_RATE

        return income

if __name__ == '__main__':
    game_manager = game.GameManager()
    trainer = Trainer(Policy, game_manager)
    for _ in range(0, const.ITERATION):
        trainer.train()
