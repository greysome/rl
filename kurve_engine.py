import random
import contextlib
from math import cos, sin, radians, floor
from enum import Enum
import numpy as np

# suppress welcome message
with contextlib.redirect_stdout(None):
    import pygame

class CollisionBin(object):
    def __init__(self, w, h, size):
        self.w, self.h = w, h
        self.buffer = [[[] for i in range(self.w)] for j in
                       range(self.h)]

    def add(self, rect, obj):
        x = rect.x // self.w
        y = rect.y // self.h
        self.buffer[y][x].append((rect, obj))

    def clear(self):
        self.buffer = [[[] for i in range(self.w)] for j in
                       range(self.h)]

    def collide_any(self, rect):
        x = rect.x // self.w
        y = rect.y // self.h
        offsets = ((-1,-1),(-1,0),(-1,1),
                   (0,-1),(0,0),(0,1),
                   (1,-1),(1,0),(1,1))
        for dx, dy in offsets:
            for buffer_rect, buffer_obj in self.buffer[y+dy][x+dx]:
                if rect.colliderect(buffer_rect):
                    return True, buffer_obj
        return False, None

class KurveError(Exception):
    pass

class KurvePlayer(object):
    n_instances = 0 

    def __init__(self, x, y, theta):
        self.id_ = KurvePlayer.n_instances
        KurvePlayer.n_instances += 1
        self.is_alive = True
        self.x = x
        self.y = y
        # the net movement of the player (in pixels) per tick
        self.dr = 4
        # the thickness of the player
        self.r = 5
        # how much the player rotates
        self.rotation = 5
        # the clockwise angle between x-axis and player
        self.theta = theta
        # number of ticks until a gap is created
        self.gap_interval = random.randint(30, 150)
        # number of ticks left to draw the gap
        self.gap_countdown = 10

    def get_rect(self):
        return pygame.Rect(self.x+self.r, self.y+self.r, self.r*2, self.r*2)

    def is_drawing_gap(self):
        return self.gap_interval == 0

    def update(self):
        self.x += self.dr * cos(radians(self.theta))
        self.y += self.dr * sin(radians(self.theta))

        # player has a gap
        if self.gap_interval == 0:
            if self.gap_countdown == 0:
                self.gap_interval = random.randint(30, 150)
                self.gap_countdown = 10
            else:
                self.gap_countdown -= 1
        else:
            self.gap_interval -= 1

class KurveEngine(object):
    ACTION_LEFT = 0
    ACTION_RIGHT = 1

    def __init__(self, n_players, w, h, sector_theta, fov):
        KurvePlayer.n_instances = 0
        self.w, self.h = w, h
        self.n_players = n_players
        # list of (id, rect, ticks) tuples
        # used to check for collisions between players
        self.n_ticks = 0
        self.players = self._init_players()
        self.sector_theta = sector_theta
        self.fov = fov
        self.collision_bin = CollisionBin(self.w, self.h, self.players[0].r)

    def reset(self):
        KurvePlayer.n_instances = 0
        for p in self.players:
            p.is_alive = True
        self.n_ticks = 0
        self.players = self._init_players()
        self.collision_bin.clear()

    def step(self, updates):
        for id_, action in updates:
            p = self._find_player(id_)
            if not p.is_alive:
                continue
            if action == KurveEngine.ACTION_LEFT:
                p.theta -= p.rotation
            elif action == KurveEngine.ACTION_RIGHT:
                p.theta += p.rotation

        self._update_players()
        self._check_collision()
        self.n_ticks += 1

    def observe(self, id_):
        p = self._find_player(id_)
        return self._get_max_dists(p)

    def out_of_bounds(self, x, y):
        return x <= 0 or x >= self.w or y <= 0 or y >= self.h

    def game_ended(self):
        p_alive = [p for p in self.players if p.is_alive]
        if self.n_players == 1:
            return len(p_alive) == 0, None
        else:
            survivor = (p_alive[0] if len(p_alive) == 1 else None)
            return len(p_alive) <= 1, survivor

    def _init_players(self):
        # divide the width and height into a and b sections respectively
        a = b = 3

        # the playing area is divided into equally sized spawn regions
        # at most 1 player can spawn in each spawn region
        spawn_regions = [(i,j) for i in range(a) for j in range(b)]

        # players can only spawn within the square bounded by
        # (x_offset, y_offset), (self.w-x_offset, self.h-y_offset)
        spawn_frac = 0.8
        x_offset = (1-spawn_frac)/2*self.w
        y_offset = (1-spawn_frac)/2*self.h
        section_frac = 0.3

        # width and height of each section
        w = (self.w-x_offset*2)/a
        h = (self.h-y_offset*2)/b

        players = []

        for id_ in range(self.n_players):
            # the jth section in the ith row
            i, j = random.choice(spawn_regions)
            theta = random.randint(0, 360)

            # spawn somewhere in the region but not too close to the edge
            x = random.randint(round(x_offset + w*i + section_frac*w),
                               round(x_offset + w*i + (1-section_frac)*w))
            y = random.randint(round(y_offset + h*j + section_frac*h),
                               round(y_offset + h*j + (1-section_frac)*h))

            p = KurvePlayer(x, y, theta)
            players.append(p)

            # ensure only that player spawns in the section
            spawn_regions.remove((i,j))

        return players

    def _check_collision(self):
        for idx, p in enumerate(self.players):
            if not p.is_alive:
                continue

            if self.out_of_bounds(p.x, p.y):
                p.is_alive = False

            collision, obj = self.collision_bin.collide_any(p.get_rect())
            if collision:
                trail_id, ticks = obj
                if p.id_ == trail_id:
                    # don't count if the current head collides with
                    # the previous head
                    if self.n_ticks-ticks < 6:
                        continue
                    p.is_alive = False
                else:
                    p.is_alive = False

    def _update_players(self):
        for p in self.players:
            if not p.is_alive:
                continue

            # add current head to trail
            if self.n_ticks % 2 == 0 and not p.is_drawing_gap():
                r = pygame.Rect(p.x+p.r, p.y+p.r, p.r*2, p.r*2)
                self.collision_bin.add(r, (p.id_, self.n_ticks))

            p.update()

    def _find_player(self, id_):
        for p in self.players:
            if p.id_ == id_:
                return p
        raise KurveError(f'could not find player with id {id_}')

    def _get_max_dists(self, p):
        n_sectors = self.fov//self.sector_theta
        # rays `self.sector_theta` angles apart emanate from the player head;
        # the distance before it collides with a trail or the border is the
        # "maximum distance"
        max_dists = np.zeros(n_sectors)

        thetas = range(p.theta-self.fov//2, p.theta+self.fov//2, self.sector_theta)

        # change in coordinates
        dxs = np.array([cos(radians(theta))*p.r for theta in thetas])
        dys = np.array([sin(radians(theta))*p.r for theta in thetas])

        # coords of rays emanating from player head
        xs = np.array([float(p.x) for _ in thetas])
        ys = np.array([float(p.y) for _ in thetas])

        # rays that have not collided with a trail or the borders yet
        remaining = [i for i in range(n_sectors)]
        
        while True:
            xs += dxs
            ys += dys

            for i in range(n_sectors):
                if i not in remaining:
                    continue
                
                collided = False

                cur_rect = pygame.Rect(xs[i]+p.r, ys[i]+p.r, 1, 1)
                collision, obj = self.collision_bin.collide_any(cur_rect)
                if collision:
                    id_, ticks = obj
                    if not (id_ == p.id_ and self.n_ticks-ticks < 6):
                        collided = True

                if self.out_of_bounds(xs[i], ys[i]):
                    collided = True

                if collided:
                    # don't update rays anymore
                    dxs[i] = dys[i] = 0
                    remaining.remove(i)

            max_dists += (dxs**2 + dys**2)
            if len(remaining) == 0:
                break

        return max_dists / 3000
