import gym
import numpy as np
from Box2D import Box2D
from gym.envs.classic_control import rendering
from gym.utils import seeding
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, distanceJointDef,
                      contactListener)


def rgb(r, g, b):
    return float(r) / 255, float(g) / 255, float(b) / 255


class Rocket:
    gimbal = 0.0
    throttle = 0.0
    power = 0.0
    force_direction = 0.0

    def __init__(self, observation_space, action_space, world, scale, initial_x, initial_y):
        self.observation_space = observation_space
        self.action_space = action_space
        self.world = world
        self.ROCKET_WIDTH = 3.66 * scale
        self.ROCKET_HEIGHT = self.ROCKET_WIDTH / 3.7 * 47.9
        self._init_rocket_body(initial_x, initial_y)

    def _init_rocket_body(self, initial_x, initial_y):
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=((-self.ROCKET_WIDTH / 2, 0),
                                             (+self.ROCKET_WIDTH / 2, 0),
                                             (self.ROCKET_WIDTH / 2, +self.ROCKET_HEIGHT),
                                             (-self.ROCKET_WIDTH / 2, +self.ROCKET_HEIGHT))),
                density=1.0,
                friction=0.5,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0)
        )

        self.lander.color1 = rgb(230, 230, 230)


class World:
    SKY_COLOR = rgb(126, 150, 233)

    def __init__(self, env, scale, width, height, fps):
        self.world = Box2D.b2World()
        self.env = env
        self.SCALE = scale
        self.WIDTH = width
        self.HEIGHT = height
        self.WATER_LEVEL = self.HEIGHT / 20
        self.FPS = fps
        initial_x = self.WIDTH / 2 + self.WIDTH * np.random.uniform(-0.3, 0.3)
        initial_y = self.HEIGHT * 0.95
        self.rocket = Rocket(env.observation_space, env.action_space, self.world, self.SCALE, initial_x, initial_y)
        self.SHIP_HEIGHT = self.rocket.ROCKET_WIDTH
        self.SHIP_WIDTH = self.SHIP_HEIGHT * 40
        self.drawlist = [self.rocket.lander]

    def render_frame(self, height, width):
        self._render_sky(height, width)
        self._render_drawlist()

    def step(self, action):
        self.world.Step(1.0 / self.FPS, 60, 60)

    def reset(self):
        self._add_water()
        self._add_ship()

    def state(self):
        return []

    def _render_sky(self, height, width):
        sky = rendering.FilledPolygon(((0, 0), (0, height), (width, height), (width, 0)))
        sky.set_color(*self.SKY_COLOR)
        self.env.viewer.add_geom(sky)

    def _render_drawlist(self):
        for obj in self.drawlist:
            for f in obj.fixtures:
                trans = f.body.transform
                path = [trans * v for v in f.shape.vertices]
                self.env.viewer.draw_polygon(path, color=obj.color1)

    def _add_water(self):
        self.water = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=((0, 0), (self.WIDTH, 0), (self.WIDTH, self.WATER_LEVEL), (0, self.WATER_LEVEL))),
                friction=0.1,
                restitution=0.0)
        )
        self.water.color1 = rgb(70, 96, 176)
        self.drawlist.append(self.water)

    def _add_ship(self):
        ship_pos = self.WIDTH / 2
        helipad_x1 = ship_pos - self.SHIP_WIDTH / 2
        helipad_x2 = helipad_x1 + self.SHIP_WIDTH
        helipad_y = self.WATER_LEVEL + self.SHIP_HEIGHT
        self.ship = self.world.CreateStaticBody(
            fixtures=fixtureDef(
                shape=polygonShape(
                    vertices=((helipad_x1, self.WATER_LEVEL),
                              (helipad_x2, self.WATER_LEVEL),
                              (helipad_x2, self.WATER_LEVEL + self.SHIP_HEIGHT),
                              (helipad_x1, self.WATER_LEVEL + self.SHIP_HEIGHT))),
                friction=0.5,
                restitution=0.0)
        )
        self.containers = []
        for side in [-1, 1]:
            self.containers.append(self.world.CreateStaticBody(
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=((ship_pos + side * 0.95 * self.SHIP_WIDTH / 2, helipad_y),
                                  (ship_pos + side * 0.95 * self.SHIP_WIDTH / 2, helipad_y + self.SHIP_HEIGHT),
                                  (ship_pos + side * 0.95 * self.SHIP_WIDTH / 2 - side * self.SHIP_HEIGHT,
                                   helipad_y + self.SHIP_HEIGHT),
                                  (ship_pos + side * 0.95 * self.SHIP_WIDTH / 2 - side * self.SHIP_HEIGHT, helipad_y)
                                  )),
                    friction=0.2,
                    restitution=0.0)
            ))
            self.containers[-1].color1 = rgb(206, 206, 2)

        self.ship.color1 = (0.2, 0.2, 0.2)
        self.drawlist.append(self.ship)
        self.drawlist.extend(self.containers)


class HybridLander(gym.Env):
    FPS = 60
    START_HEIGHT = 300.0
    SCALE = 0.5
    VIEWPORT_H = 720
    VIEWPORT_W = 500
    H = 1.1 * START_HEIGHT * SCALE
    W = float(VIEWPORT_W) / VIEWPORT_H * H

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': FPS
    }

    def __init__(self, a_space_type='continuous', add_vel_state=True):
        self.np_random = None
        self.seed()
        self.a_space_type = a_space_type
        self.vel_state = add_vel_state

        high = np.array([1, 1, 1, 1, 1, 1, 1, np.inf, np.inf, np.inf], dtype=np.float32)
        low = -high
        if not self.vel_state:
            high = high[0:7]
            low = low[0:7]
        self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)

        if self.a_space_type == 'continuous':
            self.action_space = gym.spaces.Box(-1.0, +1.0, (3,), dtype=np.float32)
        elif self.a_space_type == 'hybrid':
            # TODO: implement hybrid action space support
            raise NotImplementedError('Hybrid action spaces not supported yet.')
        else:
            self.action_space = gym.spaces.Discrete(7)

        self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
        self.viewer.set_bounds(0, self.W, 0, self.H)
        self.world = World(self, self.SCALE, self.W, self.H, self.FPS)
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.world.reset()

    def step(self, action):
        self.world.step(action)

        # should we normalize the state? not sure
        state = self.world.state()

        reward = self._compute_reward(state)
        reward = np.clip(reward, -1.0, 1.0)

        done = self._compute_done(state)

        return np.array(state), reward, done, {}

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(self.VIEWPORT_W, self.VIEWPORT_H)
            self.viewer.set_bounds(0, self.W, 0, self.H)
            self.world.viewer = self.viewer
        self.world.render_frame(self.H, self.W)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer is not None:
            self.viewer.close()
            self.viewer = None

    def _compute_reward(self, state):
        return 0.0

    def _compute_done(self, state):
        return False
