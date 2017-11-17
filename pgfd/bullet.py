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
