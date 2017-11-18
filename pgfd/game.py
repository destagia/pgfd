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

    @property
    def position(self): return self.__position

    def update(self, dt):
        x, y = self.__position
        x += self.__v[0] * dt
        y += self.__v[1] * dt
        self.__position = (x, y)
        self.__v = (self.__v[0], self.__v[1] - GRAVITY * dt)

    def draw(self, display):
        xi = int(self.__position[0])
        yi = int(self.__position[1])
        display.circle((20, 100, 255), (xi, yi), 2)

class Enemy(object):
    def __init__(self, position):
        self.__position = position

        size = 10
        xi = int(self.__position[0]) - (size / 2)
        yi = int(self.__position[1]) - (size / 2)
        self.__rect = pygame.Rect(xi, yi, size, size)

    @property
    def position(self): return self.__position

    def collide(self, position):
        return self.__rect.collidepoint(position[0], position[1])

    def update(self, dt):
        pass

    def draw(self, display):
        display.rect((100, 200, 0), self.__rect)

class Game(object):
    def __init__(self, display, enemy_position):
        self.__display = display
        self.__game_objects = []

        self.__enemy = Enemy(enemy_position)
        self.add(self.__enemy)

        self.__is_terminal = False
        self.__time = 0

    def add(self, game_object):
        self.__game_objects.append(game_object)

    def shoot(self, action):
        self.__bullet = Bullet((0, 0), action)
        self.add(self.__bullet)

    def current_state(self):
        return self.__enemy.position

    @property
    def is_terminal(self): return self.__is_terminal

    def update(self):
        delta_s = self.__display.clear_buffer()

        for game_object in self.__game_objects:
            game_object.update(delta_s)
            game_object.draw(self.__display)

        self.__display.update()

        if self.__bullet is not None:
            if self.__enemy.collide(self.__bullet.position):
                self.__is_terminal = True
                return 1.0


        self.__is_terminal = self.__bullet.position[0] > (self.__enemy.position[0] + 20) or self.__time > 1000
        self.__time += 1

        return 0.0

class Display(object):
    def __init__(self):
        self.__clock = pygame.time.Clock()
        self.__screen = pygame.display.set_mode(SCREEN_SIZE)
        self.__x = 0.0

    def clear_buffer(self):
        for event in pygame.event.get():
            if event.type == QUIT:
                sys.exit()

        self.__screen.fill((0, 0, 0))

        delta_ms = self.__clock.tick(30)
        return delta_ms / 1000

    def update(self):
        pygame.display.update()

    def circle(self, color, position, radius):
        x, y = position
        y = SCREEN_SIZE[1] - y
        pygame.draw.circle(self.__screen, color, (x, y), radius)

    def rect(self, color, rect):
        y = SCREEN_SIZE[1] - rect.y - rect.height
        rect = pygame.Rect(rect.x, y, rect.width, rect.height)
        pygame.draw.rect(self.__screen, color, rect)

class DevNull(object):
    def __init__(self):
        self.__delta_time = 1.0 / 30.0
        
    def clear_buffer(self): return self.__delta_time
    def update(self): pass
    def circle(self, color, position, radius): pass
    def rect(self, color, rect): pass

class GameManager(object):
    def new_game(self, disp, enemy_position = None):
        if enemy_position is None:
            enemy_position = random_safe_area_position()

        if disp:
            display = Display()
        else:
            display = DevNull()

        game = Game(display, enemy_position)
        return game
