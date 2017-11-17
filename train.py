"""
Train the policy with finite-difference policy gradient
"""

import chainer
import cupy
import numpy

from pgfd import const
from pgfd import Game, Bullet

xp = cupy

class Policy(object):
    def __init__(self, parameter):
        self.__parameter = parameter

    def __call__(self, state):
        return 0.1234
    
class TestGame(object):
    def __init__(self):
        self.update(0)

    def current_state(self):
        return self.__state

    def update(self, action):
        self.__state = xp.random.rand(const.STATE_SIZE)
        return numpy.random.uniform()

class Trainer(object):
    def __init__(self, policy_factory, game_factory):
        self.__rollout = Rollout(policy_factory, game_factory)
        self.__target_parameter = xp.random.rand(const.POLICY_PARAMETER_SIZE)

    def train(self):
        incomes = numpy.asarray([self.__rollout(self.__target_parameter) for j in range(0, const.TRAIN_ROLLOUT_COUNT)])
        expected_income = incomes.mean()

        d_J = xp.zeros(const.POLICY_PARAMETER_SIZE)
        d_parameter = xp.zeros((const.POLICY_PARAMETER_SIZE, const.POLICY_PARAMETER_SIZE))

        for k in range(0, const.POLICY_PARAMETER_SIZE):
            d_parameter_increment = self.unit_vector(k, const.PARAMETER_E)
            p = self.__target_parameter + d_parameter_increment
            d_expected_income = (self.__rollout(p) - expected_income) / const.PARAMETER_E

            d_parameter[k] = d_parameter_increment
            d_J[k] = d_expected_income

        gfd = xp.linalg.inv(d_parameter.T.dot(d_parameter)).dot(d_parameter.T).dot(d_J)
        print(gfd)

    def unit_vector(self, k, e):
        unit_vector = xp.zeros(const.POLICY_PARAMETER_SIZE, dtype=xp.float32)
        unit_vector[k] = e
        return unit_vector

class Rollout(object):
    def __init__(self, policy_factory, game_factory):
        self.__policy_factory = policy_factory
        self.__game_factory = game_factory
    
    def __call__(self, policy_parameter):
        game = self.__game_factory()
        policy = self.__policy_factory(policy_parameter)

        income = 0
        discount_rate = const.DISCOUNT_RATE
        for k in range(0, const.ROLLOUT_EPOC):
            state = game.current_state()
            action = policy(state)
            reward = game.update(action)
            income += discount_rate * reward
            discount_rate *= const.DISCOUNT_RATE

        return income

# trainer = Trainer(Policy, TestGame)
# for _ in range(0, const.ITERATION):
#    trainer.train()

game = Game()
game.add(Bullet((0, 0), (20, 100)))

for _ in range(0, 1000):
    game.update()
