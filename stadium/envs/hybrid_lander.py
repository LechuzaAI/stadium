import gym
import numpy as np
from Box2D import Box2D
from gym.envs.classic_control import rendering
from gym.spaces import Box
from gym.utils import seeding
from Box2D.b2 import (edgeShape, circleShape, fixtureDef, polygonShape, revoluteJointDef, distanceJointDef,
                      contactListener)


def rgb(r, g, b):
    return float(r) / 255, float(g) / 255, float(b) / 255


def rgba(r, g, b, a):
    return float(r) / 255, float(g) / 255, float(b) / 255, float(a) / 255


class ContactDetector(contactListener):
    def __init__(self, world, rocket):
        contactListener.__init__(self)
        self.world = world
        self.rocket = rocket

    def BeginContact(self, contact):
        if self.world.water in [contact.fixtureA.body, contact.fixtureB.body] \
                or self.rocket.lander in [contact.fixtureA.body, contact.fixtureB.body] \
                or self.world.containers[0] in [contact.fixtureA.body, contact.fixtureB.body] \
                or self.world.containers[1] in [contact.fixtureA.body, contact.fixtureB.body]:
            self.world.game_over = True
        else:
            for i in range(2):
                if self.rocket.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                    self.rocket.legs[i].ground_contact = True

    def EndContact(self, contact):
        for i in range(2):
            if self.rocket.legs[i] in [contact.fixtureA.body, contact.fixtureB.body]:
                self.rocket.legs[i].ground_contact = False


class Rocket:
    # CONTROL
    gimbal = 0.0
    throttle = 0.0
    power = 0.0
    force_direction = 0.0

    # ROCKET
    MIN_THROTTLE = 0.4
    GIMBAL_THRESHOLD = 0.4

    # LEGS
    BASE_ANGLE = -0.27
    SPRING_ANGLE = 0.27
    MAX_SMOKE_LIFETIME = 120

    def __init__(self, observation_space, action_space, world, sky_color, fps, scale, height, initial_x, initial_y):
        self.observation_space = observation_space
        self.action_space = action_space
        self.world = world
        self.ROCKET_WIDTH = 3.66 * scale
        self.ROCKET_HEIGHT = self.ROCKET_WIDTH / 3.7 * 47.9
        self.ENGINE_HEIGHT = self.ROCKET_WIDTH * 0.5
        self.ENGINE_WIDTH = self.ENGINE_HEIGHT * 0.7
        self.THRUSTER_HEIGHT = self.ROCKET_HEIGHT * 0.86
        self.LEG_AWAY = self.ROCKET_WIDTH / 2
        self.LEG_LENGTH = self.ROCKET_WIDTH * 2.2
        self.MAIN_ENGINE_POWER = 1600 * scale
        self.SIDE_ENGINE_POWER = 100 / fps * scale
        self.FPS = fps
        self.HEIGHT = height
        self.legs = []
        self.smoke = []
        self.stepnumber = 0
        self.sky_color = sky_color
        self.sky_color_half_transparent = np.array((np.array(self.sky_color) + rgb(255, 255, 255))) / 2
        self._init_rocket_body(initial_x, initial_y)

    def _init_rocket_body(self, initial_x, initial_y):
        self.initial_x = initial_x
        self.initial_y = initial_y
        self.lander = self.world.CreateDynamicBody(
            position=(initial_x, initial_y),
            angle=0.0,
            fixtures=fixtureDef(
                shape=polygonShape(vertices=((-self.ROCKET_WIDTH / 2, 0),
                                             (+self.ROCKET_WIDTH / 2, 0),
                                             (+self.ROCKET_WIDTH / 2, +self.ROCKET_HEIGHT),
                                             (0, +self.ROCKET_HEIGHT * 1.1),
                                             (-self.ROCKET_WIDTH / 2, +self.ROCKET_HEIGHT))),
                density=1.0,
                friction=0.5,
                categoryBits=0x0010,
                maskBits=0x001,
                restitution=0.0)
        )

        self.lander.color1 = rgb(230, 230, 230)

        for i in [-1, +1]:
            leg = self.world.CreateDynamicBody(
                position=(initial_x - i * self.LEG_AWAY, initial_y + self.ROCKET_WIDTH * 0.2),
                angle=(i * self.BASE_ANGLE),
                fixtures=fixtureDef(
                    shape=polygonShape(
                        vertices=((0, 0), (0, self.LEG_LENGTH / 25), (i * self.LEG_LENGTH, 0),
                                  (i * self.LEG_LENGTH, -self.LEG_LENGTH / 20),
                                  (i * self.LEG_LENGTH / 3, -self.LEG_LENGTH / 7))),
                    density=1,
                    restitution=0.0,
                    friction=0.2,
                    categoryBits=0x0020,
                    maskBits=0x001)
            )
            leg.ground_contact = False
            leg.color1 = (0.25, 0.25, 0.25)
            rjd = revoluteJointDef(
                bodyA=self.lander,
                bodyB=leg,
                localAnchorA=(i * self.LEG_AWAY, self.ROCKET_WIDTH * 0.2),
                localAnchorB=(0, 0),
                enableLimit=True,
                maxMotorTorque=2500.0,
                motorSpeed=-0.05 * i,
                enableMotor=True
            )
            djd = distanceJointDef(bodyA=self.lander,
                                   bodyB=leg,
                                   anchorA=(i * self.LEG_AWAY, self.ROCKET_HEIGHT / 8),
                                   anchorB=leg.fixtures[0].body.transform * (i * self.LEG_LENGTH, 0),
                                   collideConnected=False,
                                   frequencyHz=0.01,
                                   dampingRatio=0.9
                                   )
            if i == 1:
                rjd.lowerAngle = -self.SPRING_ANGLE
                rjd.upperAngle = 0
            else:
                rjd.lowerAngle = 0
                rjd.upperAngle = + self.SPRING_ANGLE
            leg.joint = self.world.CreateJoint(rjd)
            leg.joint2 = self.world.CreateJoint(djd)

            self.legs.append(leg)

    def reset(self):
        self.stepnumber = 0

    def render(self, viewer):
        for l in zip(self.legs, [-1, 1]):
            path = [self.lander.fixtures[0].body.transform * (l[1] * self.ROCKET_WIDTH / 2, self.ROCKET_HEIGHT / 8),
                    l[0].fixtures[0].body.transform * (l[1] * self.LEG_LENGTH * 0.8, 0)]
            viewer.draw_polyline(path, color=self.legs[0].color1, linewidth=2)

        self.rockettrans = rendering.Transform()
        engine = rendering.FilledPolygon(((0, 0),
                                          (self.ENGINE_WIDTH / 2, -self.ENGINE_HEIGHT),
                                          (-self.ENGINE_WIDTH / 2, -self.ENGINE_HEIGHT)))
        self.enginetrans = rendering.Transform()
        engine.add_attr(self.enginetrans)
        engine.add_attr(self.rockettrans)
        engine.set_color(.4, .4, .4)
        viewer.add_onetime(engine)

        self.fire = rendering.FilledPolygon(((self.ENGINE_WIDTH * 0.4, 0), (-self.ENGINE_WIDTH * 0.4, 0),
                                             (-self.ENGINE_WIDTH * 1.2, -self.ENGINE_HEIGHT * 5),
                                             (0, -self.ENGINE_HEIGHT * 8),
                                             (self.ENGINE_WIDTH * 1.2, -self.ENGINE_HEIGHT * 5)))
        self.fire.set_color(*rgb(255, 230, 107))
        self.firescale = rendering.Transform(scale=(1, 1))
        self.firetrans = rendering.Transform(translation=(0, -self.ENGINE_HEIGHT))
        self.fire.add_attr(self.firescale)
        self.fire.add_attr(self.firetrans)
        self.fire.add_attr(self.enginetrans)
        self.fire.add_attr(self.rockettrans)
        viewer.add_onetime(self.fire)

        smoke = rendering.FilledPolygon(((self.ROCKET_WIDTH / 2, self.THRUSTER_HEIGHT * 1),
                                         (self.ROCKET_WIDTH * 3, self.THRUSTER_HEIGHT * 1.03),
                                         (self.ROCKET_WIDTH * 4, self.THRUSTER_HEIGHT * 1),
                                         (self.ROCKET_WIDTH * 3, self.THRUSTER_HEIGHT * 0.97)))
        smoke.set_color(*self.sky_color_half_transparent)
        self.smokescale = rendering.Transform(scale=(1, 1))
        smoke.add_attr(self.smokescale)
        smoke.add_attr(self.rockettrans)
        viewer.add_onetime(smoke)

        self.gridfins = []
        for i in (-1, 1):
            finpoly = (
                (i * self.ROCKET_WIDTH * 1.1, self.THRUSTER_HEIGHT * 1.01),
                (i * self.ROCKET_WIDTH * 0.4, self.THRUSTER_HEIGHT * 1.01),
                (i * self.ROCKET_WIDTH * 0.4, self.THRUSTER_HEIGHT * 0.99),
                (i * self.ROCKET_WIDTH * 1.1, self.THRUSTER_HEIGHT * 0.99)
            )
            gridfin = rendering.FilledPolygon(finpoly)
            gridfin.add_attr(self.rockettrans)
            gridfin.set_color(0.25, 0.25, 0.25)
            self.gridfins.append(gridfin)

        for g in self.gridfins:
            viewer.add_onetime(g)

        if self.stepnumber % round(self.FPS / 10) == 0 and self.power > 0:
            s = [self.MAX_SMOKE_LIFETIME * self.power,  # total lifetime
                 0,  # current lifetime
                 self.power * (1 + 0.2 * np.random.random()),  # size
                 np.array(self.lander.position)
                 + self.power * self.ROCKET_WIDTH * 10 * np.array((np.sin(self.lander.angle + self.gimbal),
                                                                   -np.cos(self.lander.angle + self.gimbal)))
                 + self.power * 5 * (np.random.random(2) - 0.5)]  # position
            self.smoke.append(s)

        for s in self.smoke:
            s[1] += 1
            if s[1] > s[0]:
                self.smoke.remove(s)
                continue
            t = rendering.Transform(translation=(s[3][0], s[3][1] + self.HEIGHT * s[1] / 2000))
            cir = viewer.draw_circle(radius=0.05 * s[1] + s[2])
            cir._color.vec4 = rgba(128, 128, 128, 64)
            cir.add_attr(t)

        self.rockettrans.set_translation(*self.lander.position)
        self.rockettrans.set_rotation(self.lander.angle)
        self.enginetrans.set_rotation(self.gimbal)
        self.firescale.set_scale(newx=1, newy=self.power * np.random.uniform(1, 1.3))
        self.smokescale.set_scale(newx=self.force_direction, newy=1)

        self.stepnumber += 1

    def set_v0(self, env, initial_random, start_speed, width):
        self.lander.linearVelocity = (
            -env.np_random.uniform(0, initial_random) * start_speed * (self.initial_x - width / 2) / width,
            -start_speed)

        self.lander.angularVelocity = (1 + initial_random) * np.random.uniform(-1, 1)

    def apply_action(self, action, fps):
        # CONTINUOUS
        if isinstance(self.action_space, Box):
            np.clip(action, -1, 1)
            self.gimbal += action[0] * 0.15 / fps
            self.throttle += action[1] * 0.5 / fps
            if action[2] > 0.5:
                self.force_direction = 1
            elif action[2] < -0.5:
                self.force_direction = -1
        # TODO add hybrid
        # elif hybrid
        # DISCRETE
        else:
            if action == 0:
                self.gimbal += 0.01
            elif action == 1:
                self.gimbal -= 0.01
            elif action == 2:
                self.throttle += 0.01
            elif action == 3:
                self.throttle -= 0.01
            elif action == 4:  # left
                self.force_direction = -1
            elif action == 5:  # right
                self.force_direction = 1

        self.gimbal = np.clip(self.gimbal, -self.GIMBAL_THRESHOLD, self.GIMBAL_THRESHOLD)
        self.throttle = np.clip(self.throttle, 0.0, 1.0)
        self.power = 0 if self.throttle == 0.0 else self.MIN_THROTTLE + self.throttle * (1 - self.MIN_THROTTLE)

        # main engine force
        force_pos = (self.lander.position[0], self.lander.position[1])
        force = (-np.sin(self.lander.angle + self.gimbal) * self.MAIN_ENGINE_POWER * self.power,
                 np.cos(self.lander.angle + self.gimbal) * self.MAIN_ENGINE_POWER * self.power)
        self.lander.ApplyForce(force=force, point=force_pos, wake=False)

        # control thruster force
        force_pos_c = self.lander.position + self.THRUSTER_HEIGHT * np.array(
            (np.sin(self.lander.angle), np.cos(self.lander.angle)))
        force_c = (-self.force_direction * np.cos(self.lander.angle) * self.SIDE_ENGINE_POWER,
                   self.force_direction * np.sin(self.lander.angle) * self.SIDE_ENGINE_POWER)
        self.lander.ApplyLinearImpulse(impulse=force_c, point=force_pos_c, wake=False)


class World:
    SKY_COLOR = rgb(126, 150, 233)

    def __init__(self, env, scale, width, height, start_speed, fps, initial_random):
        self.world = Box2D.b2World()
        self.env = env
        self.SCALE = scale
        self.WIDTH = width
        self.HEIGHT = height
        self.WATER_LEVEL = self.HEIGHT / 20
        self.FPS = fps
        self.START_SPEED = start_speed
        self.INITIAL_RANDOM = initial_random
        self.initial_x = self.WIDTH / 2 + self.WIDTH * np.random.uniform(-0.3, 0.3)
        self.initial_y = self.HEIGHT * 0.95
        self.SHIP_HEIGHT = 3.66 * scale
        self.SHIP_WIDTH = self.SHIP_HEIGHT * 40
        self.drawlist = []
        self.water = None

    def render_frame(self, height, width):
        self._render_sky(height, width)
        self._render_drawlist()
        self.rocket.render(self.env.viewer)

    def step(self, action):
        self.rocket.apply_action(action, self.FPS)
        self.world.Step(1.0 / self.FPS, 60, 60)

    def reset(self):
        self._destroy()
        self._add_water()
        self._add_ship()
        self._add_rocket()
        self.world.contactListener_keepref = ContactDetector(self, self.rocket)
        self.world.contactListener = self.world.contactListener_keepref

    def state(self):
        pos = self.rocket.lander.position
        vel_l = np.array(self.rocket.lander.linearVelocity) / self.START_SPEED
        vel_a = self.rocket.lander.angularVelocity
        x_distance = (pos.x - self.WIDTH / 2) / self.WIDTH
        y_distance = (pos.y - self.SHIP_HEIGHT) / (self.HEIGHT - self.SHIP_HEIGHT)
        distance = np.linalg.norm((3 * x_distance, y_distance))  # weight x position more
        angle = (self.rocket.lander.angle / np.pi) % 2
        speed = np.linalg.norm(vel_l)
        landed = self.rocket.legs[0].ground_contact and self.rocket.legs[
            1].ground_contact and speed < 0.1
        groundcontact = self.rocket.legs[0].ground_contact or self.rocket.legs[1].ground_contact
        # brokenleg = (self.rocket.legs[0].joint.angle < 0 or self.rocket.legs[1].joint.angle > -0) and groundcontact
        brokenleg = False
        outside = abs(
            self.rocket.lander.position.x - self.WIDTH / 2) > self.WIDTH / 2 or self.rocket.lander.position.y > self.HEIGHT
        if angle > 1:
            angle -= 2
        fuelcost = 0.1 * (0 * self.rocket.power + abs(self.rocket.force_direction)) / self.FPS
        crashed = vel_a < 0.1 and speed < 0.1 and abs(angle) > 0.5
        state = {'pos': pos,
                 'distance': distance,
                 'speed': speed,
                 'vel_l': vel_l,
                 'x': 2 * x_distance,
                 'y': 2 * (y_distance - 0.5),
                 'v_x': vel_l[0],
                 'v_y': vel_l[1],
                 'vel_a': vel_a,
                 'angle': angle,
                 'g_0': 1.0 if self.rocket.legs[0].ground_contact else 0.0,
                 'g_1': 1.0 if self.rocket.legs[1].ground_contact else 0.0,
                 'throttle': 2 * (self.rocket.throttle - 0.5),
                 'gimbal': self.rocket.gimbal / self.rocket.GIMBAL_THRESHOLD,
                 'landed': landed,
                 'crashed': crashed,
                 'brokenleg': brokenleg,
                 'outside': outside,
                 'fuelcost': fuelcost,
                 'groundcontact': groundcontact
                 }
        return state

    def _render_sky(self, height, width):
        sky = rendering.FilledPolygon(((0, 0), (0, height), (width, height), (width, 0)))
        sky.set_color(*self.SKY_COLOR)
        self.env.viewer.add_onetime(sky)

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

    def _add_rocket(self):
        self.rocket = Rocket(self.env.observation_space,
                             self.env.action_space,
                             self.world,
                             self.SKY_COLOR,
                             self.FPS,
                             self.SCALE,
                             self.HEIGHT,
                             self.initial_x,
                             self.initial_y)
        self.rocket.set_v0(self.env, self.INITIAL_RANDOM, self.START_SPEED, self.WIDTH)
        self.drawlist.append(self.rocket.lander)
        self.drawlist.extend(self.rocket.legs)

    def _destroy(self):
        if not self.water:
            return
        self.world.DestroyBody(self.water)
        self.water = None
        self.world.DestroyBody(self.rocket.lander)
        self.world.DestroyBody(self.rocket.legs[0])
        self.world.DestroyBody(self.rocket.legs[1])
        self.rocket = None
        self.world.DestroyBody(self.ship)
        self.ship = None
        self.world.DestroyBody(self.containers[0])
        self.world.DestroyBody(self.containers[1])
        self.containers = []
        self.drawlist = []


class HybridLander(gym.Env):
    FPS = 30
    START_HEIGHT = 500.0
    START_SPEED = 40
    INITIAL_RANDOM = 0.4
    SCALE = 0.35
    VIEWPORT_H = 800
    VIEWPORT_W = 450
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
        self.world = World(self, self.SCALE, self.W, self.H, self.START_SPEED, self.FPS, self.INITIAL_RANDOM)
        self.game_over = False
        self.prev_shaping = None
        self.landed_ticks = 0
        self.reset()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.world.reset()
        self.game_over = False
        self.prev_shaping = None

    def step(self, action):
        self.world.step(action)

        # should we normalize the state? not sure
        state_dict = self.world.state()

        state = [state_dict['x'],
                 state_dict['y'],
                 state_dict['angle'],
                 state_dict['vel_l'][0],
                 state_dict['vel_l'][1],
                 state_dict['vel_a'],
                 state_dict['g_0'],
                 state_dict['g_1'],
                 state_dict['throttle'],
                 state_dict['gimbal']]

        reward = self._compute_reward(state_dict)
        reward = np.clip(reward, -1.0, 1.0)

        done = self._compute_done(state_dict)

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
        reward = -state['fuelcost']
        shaping = -0.5 * (state['distance'] + state['speed'] + abs(state['angle']) ** 2)
        shaping += 0.1 * (self.world.rocket.legs[0].ground_contact + self.world.rocket.legs[1].ground_contact)
        if self.prev_shaping is not None:
            reward += shaping - self.prev_shaping
        self.prev_shaping = shaping

        if state['landed']:
            self.landed_ticks += 1
        else:
            self.landed_ticks = 0
        if self.landed_ticks == self.FPS:
            reward = 1.0
            self.game_over = True

        if self._compute_done(state):
            reward += max(-1, 0 - 2 * (state['speed'] + state['distance'] + abs(state['angle']) + abs(state['vel_a'])))
        else:
            reward -= 0.25 / self.FPS

        reward = np.clip(reward, -1, 1)

        return reward

    def _compute_done(self, state):
        outside = state['outside']
        brokenleg = state['brokenleg']
        crashed = state['crashed']
        if outside or brokenleg or crashed:
            self.game_over = True
        if self.game_over:
            print(outside, brokenleg, crashed)
        return self.game_over
