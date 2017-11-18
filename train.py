"""
Train the policy with finite-difference policy gradient
"""

import chainer
import chainer.functions as F
import cupy
import numpy as np

from pgfd import const, game

xp = np


def unit_vector(k, e):
    """
    Unit vector

    unit vector is the vector, the only k-th element of which has value e.
    The other elements are all zero.
    """
    unit_vector = xp.zeros(const.POLICY_PARAMETER_SIZE, dtype=xp.float32)
    unit_vector[k] = e
    return unit_vector


class Policy(object):
    """
    policy for the actor

    This policy has the parameter, size of which is const.POLICY_PARAMATER_SIZE.
    The output (action) of this policy is trasnparent for the input (state)
    if the parameter is unchanged.
    """

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
    """
    Trainer of the policy parameter

    Trainer does it by finite-difference policy gradient.
    By doing roll-outs for the enough many times, it gets the expected return under a parameter.
    It increments the paramaeter with small value, and try to get expected return,
    then compares which is better, the old one or the incremented one.
    """

    def __init__(self, policy_factory, game_manager):
        self.__policy_factory = policy_factory
        self.__rollout = Rollout(policy_factory, game_manager)
        self.__target_parameter = xp.random.rand(const.POLICY_PARAMETER_SIZE) * const.INITIAL_W_SCALE

    def policy(self):
        return self.__policy_factory(self.__target_parameter)

    def train(self):
        incomes = np.asarray([self.__rollout(self.__target_parameter) for j in range(0, const.TRAIN_ROLLOUT_COUNT)])
        expected_income = incomes.mean()

        d_J = xp.zeros(const.POLICY_PARAMETER_SIZE)
        d_parameter = xp.zeros((const.POLICY_PARAMETER_SIZE, const.POLICY_PARAMETER_SIZE))

        print('Expected income: {}'.format(expected_income))
        for k in range(0, const.POLICY_PARAMETER_SIZE):
            increment = np.random.uniform(const.PARAMETER_E_MIN, const.PARAMETER_E_MAX)
            d_parameter_increment = unit_vector(k, increment)
            parameter_increment = self.__target_parameter + d_parameter_increment

            incomes_increment = np.asarray([self.__rollout(parameter_increment) for j in range(0, const.TRAIN_ROLLOUT_COUNT)])
            expected_income_increment = incomes_increment.mean()
            print('Incremented expected income: {}'.format(expected_income_increment))

            d_expected_income = (expected_income_increment - expected_income) / increment

            d_parameter[k] = d_parameter_increment
            d_J[k] = d_expected_income

        # policy gradient in finite-difference
        gfd = xp.linalg.inv(d_parameter.T.dot(d_parameter)).dot(d_parameter.T).dot(d_J)
        print('gradient:\n{}'.format(gfd))

        # update the parameter of policy with gradient
        self.__target_parameter += (const.LEARNING_RATE * gfd)
        print('parameter:\n{}'.format(self.__target_parameter))


class Rollout(object):
    """
    Rollout the game with policy

    By doing rollout, we can estimate the expected return of the parameter!
    """

    def __init__(self, policy_factory, game_manager):
        self.__policy_factory = policy_factory
        self.__game_manager = game_manager

    def __call__(self, policy_parameter):
        policy = self.__policy_factory(policy_parameter)
        game = game_manager.new_game(False)

        action = policy(xp.asarray([game.current_state()])).data[0]
        game.shoot(action)

        income = 0
        while not game.is_terminal:
            reward = game.update()
            income += reward

        return income


class Tester(object):
    """
    Test and display the behaviour of the policy parameter
    """

    def __init__(self, game_manager, trainer):
        self.__game_manager = game_manager
        self.__trainer = trainer

    def test(self, enemy_position):
        game = self.__game_manager.new_game(True, enemy_position)
        policy = self.__trainer.policy()
    
        action = policy(xp.asarray([game.current_state()])).data[0]
        game.shoot(action)
        while not game.is_terminal:
            game.update()


if __name__ == '__main__':
    game_manager = game.GameManager()
    trainer = Trainer(Policy, game_manager)
    tester = Tester(game_manager, trainer)

    for _ in range(0, const.ITERATION):
        trainer.train()

        tester.test([100.0, 50.0])
        tester.test([40.0, 80.0])
        tester.test([140.0, 30.0])
        tester.test([140.0, 70.0])

