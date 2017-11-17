"""
Canon game
"""

import sys
import pygame
import numpy as np
from pygame.locals import *

SCREEN_SIZE = (160, 90)

class Game(object):
    def __init__(self, display = None):
        if display is None:
            display = Display()
        self.__display = display
        self.__clock = pygame.time.Clock()
        self.__game_objects = []


    def add(self, game_object):
        self.__game_objects.append(game_object)

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
    
