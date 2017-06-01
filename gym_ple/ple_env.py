import gym
from gym import spaces
from ple import PLE
import numpy as np

from PIL import Image


def state_preprocessor(game_dict):
    _, values = zip(*sorted(list(game_dict.items())))
    state = np.array(values)
    return state


class PLEEnv(gym.Env):
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, game_name='FlappyBird', display_screen=True, observe_state=False):
        # open up a game state to communicate with emulator
        import importlib
        game_module_name = ('ple.games.%s' % game_name).lower()
        game_module = importlib.import_module(game_module_name)
        game = getattr(game_module, game_name)()
        self.game_state = PLE(game, fps=30, display_screen=display_screen, state_preprocessor=state_preprocessor)
        self.game_state.init()
        self._action_set = self.game_state.getActionSet()
        self.action_space = spaces.Discrete(len(self._action_set))
        self.screen_width, self.screen_height = self.game_state.getScreenDims()
        if self.screen_height+self.screen_width > 500:
            img_scale = 0.25
        else:
            img_scale = 1.0
        self.screen_width = int(self.screen_width*img_scale)
        self.screen_height = int(self.screen_height*img_scale)
        self.observe_state = observe_state
        if self.observe_state:
            # the bounds are typically not infinity
            self.observation_space = spaces.Box(low=-float('inf'), high=float('inf'), shape=self.game_state.state_dim)
        else:
            self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.viewer = None

    def _step(self, a):
        reward = self.game_state.act(self._action_set[a])
        if self.observe_state:
            state = self.game_state.getGameState()
        else:
            state = self._get_image()
        terminal = self.game_state.game_over()
        return state, reward, terminal, {}

    def _resize_frame(self, frame):
        pil_image = Image.fromarray(frame)
        pil_image = pil_image.resize((self.screen_width, self.screen_height), Image.ANTIALIAS)
        return  np.array(pil_image)

    def _get_image(self):
        image_rotated = np.fliplr(np.rot90(self.game_state.getScreenRGB(), 3)) # Hack to fix the rotated image returned by ple
        return self._resize_frame(image_rotated)

    @property
    def _n_actions(self):
        return len(self._action_set)

    # return: (states, observations)
    def _reset(self, **kwargs):
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.screen_height, self.screen_width, 3))
        self.game_state.reset_game(**kwargs)
        if self.observe_state:
            state = self.game_state.getGameState()
        else:
            state = self._get_image()
        return state

    def _render(self, mode='human', close=False):
        if close:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            return
        img = self._get_image()
        if mode == 'rgb_array':
            return img
        elif mode == 'human':
            from gym.envs.classic_control import rendering
            if self.viewer is None:
                self.viewer = rendering.SimpleImageViewer()
            self.viewer.imshow(img)

    def _seed(self, seed):
        rng = np.random.RandomState(seed)
        self.game_state.rng = rng
        self.game_state.game.rng = self.game_state.rng

        self.game_state.init()
