import gymnasium as gym
from game_logic import Game
import numpy as np

#wrap env with a Monitor
class ordering_cards_env(gym.Env):
    """Custom Environment that follows gym interface."""

    metadata = {"render_modes": ["human"]}

    def __init__(self, id=0, seed=None, cards_amount=10, ts="0000000"):
        self.game = Game(id, seed, cards_amount, ts)
        self.cards = np.array([])

        self.action_space = gym.spaces.Discrete(cards_amount)
        self.observation_space = gym.spaces.MultiDiscrete([41] * cards_amount * 2, dtype=np.int32)

        self.reset()

        self.actions_history = []
        self.regrets = []
    
    def reset(self, seed=None):
        self.actions_history = []
        self.cards = self.game.start_game()

        observation = np.zeros(self.cards.size*2, dtype=np.int32) + 40
        observation[:self.cards.size] = self.cards

        return observation, {}
        # return observation, info
        
    def step(self, action):
        # Execute one time step within the environment
        # observation, reward, done, info = self.game.throw_card(action)
        # return observation, reward, done, info

        if action in self.actions_history:
            board = np.array(self.game.board)
            observation = np.zeros(self.cards.size*2, dtype=np.int32) + 40
            observation[self.cards.size : self.cards.size+len(board)] = board
            observation[:self.cards.size] = self.cards
            return observation, -1.0, False, False, {}
        
        self.actions_history.append(action)

        reward, terminated = self.game.throw_card(action)

        if terminated:
            self.regrets.append(0.0 - reward)    #1.0 is the maximum possible reward

        board = np.array(self.game.board)
        observation = np.zeros(self.cards.size*2, dtype=np.int32) + 40
        observation[self.cards.size : self.cards.size+len(board)] = board
        observation[:self.cards.size] = self.cards
        
        return observation, 1.0, terminated, False, {}
        # return observation, reward, terminated, False, {}
        # return observation, reward, terminated, truncated, info