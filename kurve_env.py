import contextlib
import gym
from gym import spaces
from kurve_engine import KurveEngine

# suppress welcome message
with contextlib.redirect_stdout(None):
    import pygame
    from pygame.locals import DOUBLEBUF

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

class KurveSinglePlayer(gym.core.Env):
    def __init__(self, render_screen=True):
        self.render_screen = render_screen
        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=708, shape=(30,))
        self.engine = KurveEngine(n_players=1, w=500, h=500,
                                  fov=150, sector_theta=5)

        if self.render_screen:
            self._pygame_init()

        self.update_circle = None
    
    def reset(self):
        self.engine.reset()
        if self.render_screen:
            self.body_sfc.fill(BLACK)
            self.screen.blit(self.body_sfc, (0, 0))
            pygame.display.flip()
        return self.engine.observe(0)

    def step(self, a):
        self.engine.step([(0, a)])
        done, _ = self.engine.game_ended()
        r = (0 if done else 1)
        if self.render_screen:
            self._render_player()
        return self.engine.observe(0), r, done, {}

    def close(self):
        super().close()

    def render(self):
        self.screen.blit(self.body_sfc, (0, 0))
        pygame.display.update(self.update_circle)

    def _pygame_init(self):
        pygame.init()
        pygame.event.set_allowed([])
        pygame.display.set_caption('kurve')
        self.screen = pygame.display.set_mode((self.engine.w, self.engine.h),
                                              DOUBLEBUF)
        self.screen.set_alpha(None)
        self.body_sfc = pygame.Surface((self.engine.w, self.engine.h))

    def _render_player(self):
        p = self.engine._find_player(0)
        if not p.is_drawing_gap():
            self.update_circle = pygame.draw.circle(self.body_sfc,
                                                    WHITE,
                                                    (round(p.x), round(p.y)),
                                                    p.r)
