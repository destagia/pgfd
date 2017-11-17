"""
Canon game
"""

import sys
import pygame
import numpy as np
from pygame.locals import *

SCREEN_SIZE = (160, 90)

SAFE_AREA = ((30, 60), (130, 30))

def random_safe_area_position():
    left, right = SAFE_AREA
    x = np.random.uniform(left[0], right[0])
    y = np.random.uniform(left[1], right[1])
    return [x, y]

GRAVITY = 9.80665 * 10

class Bullet(object):
    def __init__(self, position, v0):
        self.__position = position
        self.__v0 = v0
        self.__v = (v0[0], v0[1])

    def update(self, dt):
        x, y = self.__position
        x += self.__v[0] * dt
        y += self.__v[1] * dt
        self.__position = (x, y)
        self.__v = (self.__v[0], self.__v[1] - GRAVITY * dt)

    def draw(self, display):
        xi = int(self.__position[0])
        yi = int(self.__position[1])
        display.circle((100, 20, 200), (xi, yi), 2)

class Enemy(object):
    def __init__(self, position):
        self.__position = position

    @property
    def position(self): return self.__position

    def update(self, dt):
        pass

    def draw(self, display):
        xi = int(self.__position[0])
        yi = int(self.__position[1])
        display.circle((100, 200, 0), (xi, yi), 2)

class Game(object):
    def __init__(self, display, enemy_position):
        self.__display = display
        self.__clock = pygame.time.Clock()
        self.__game_objects = []

        self.__enemy = Enemy(enemy_position)
        self.add(self.__enemy)

    def add(self, game_object):
        self.__game_objects.append(game_object)

    def shoot(self, action):
        self.__bullet = Bullet((0, 0), action)
        self.add(self.__bullet)

    def current_state(self):
        return self.__enemy.position

    def update(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()

        delta_ms = self.__clock.tick(30)
        delta_s = delta_ms / 1000

        self.__display.clear_buffer()

        for game_object in self.__game_objects:
            game_object.update(delta_s)
            game_object.draw(self.__display)

        self.__display.update()

        return 0.1

class Display(object):
    def __init__(self):
        self.__screen = pygame.display.set_mode(SCREEN_SIZE)
        self.__x = 0.0

    def clear_buffer(self):
        self.__screen.fill((0, 0, 0))

    def update(self):
        pygame.display.update()

    def circle(self, color, position, radius):
        x, y = position
        y = SCREEN_SIZE[1] - y
        pygame.draw.circle(self.__screen, color, (x, y), radius)
    

class GameManager(object):
    def __init__(self):
        self.__display = Display()

    def new_game(self):
        enemy_position = random_safe_area_position()
        game = Game(self.__display, enemy_position)
        return game
