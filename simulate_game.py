import numpy as np
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from environment import ordering_cards_env
from matplotlib import pyplot as plt
from os import makedirs

class Simulation:
    def __init__(self, seed=None, cards_amount=10, games_to_play=100, parallel_runs=1, ts="0000000") -> None:
        self.seed = seed
        self.cards_amount = cards_amount
        self.games_to_play = games_to_play
        self.parallel_runs = parallel_runs
        self.ts = ts
    
    def run(self, progress_bar=False):


        # env = ordering_cards_env(self.seed, self.cards_amount, self.ts)
        # env = gym.vector.AsyncVectorEnv([
        #     lambda:  ordering_cards_env(self.seed, self.cards_amount, self.ts)
        #                 for _ in range(self.parallel_runs)
        # ])
        #wrap env in SubprocVecEnv
        # env = DummyVecEnv([ 
        #     lambda :  ordering_cards_env(env_id, self.seed, self.cards_amount, self.ts)
        #                 for env_id in range(self.parallel_runs)
        # ])


        # Here we use the "fork" method for launching the processes, more information is available in the doc
        # This is equivalent to make_vec_env(env_id, n_envs=n_procs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))
        # make_myenv = lambda : ordering_cards_env(0, self.seed, self.cards_amount, self.ts)
        env = make_vec_env(     lambda : ordering_cards_env(0, self.seed, self.cards_amount, self.ts), 
                                n_envs=self.parallel_runs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))


        # check_env(env)
        model = PPO("MlpPolicy", env, verbose=1)
        timesteps = int(self.games_to_play * self.cards_amount * 2 // self.parallel_runs)
        real_timesteps = max(  (timesteps//(2048*self.parallel_runs) + 1) * 2048*self.parallel_runs,   2048 * self.parallel_runs  )
        model.learn(total_timesteps=real_timesteps, progress_bar=progress_bar)
        # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        
        print("done training, now plotting")

        # save plot in results/ts/plot.png
        plt.ioff()

        # n_regrets =[env.unwrapped.envs[i].regrets for i in range(len(env.unwrapped.envs))]
        n_regrets = env.unwrapped.get_attr('regrets')
        min_len = min([len(x) for x in n_regrets])
        n_regrets = [x[:min_len] for x in n_regrets]

        y = np.array(n_regrets)
        print(y.shape, [len(x) for x in y])
        y_mean = np.mean(y, axis=0)
        y_err = np.std(y, axis=0)

        # y_mean_ma = np.convolve(y_mean,  np.ones((timesteps//100,)) / timesteps//100 , mode='valid')
        ma_amt = y_mean.size // 100
        y_mean_ma = np.convolve(y_mean,  np.ones((ma_amt,)) / ma_amt , mode='valid')
        y_err_ma = np.convolve(y_err,  np.ones((ma_amt,)) / ma_amt , mode='valid')

        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.axhline(0, color='black', lw=1, alpha=.7)
        plt.plot(y_mean_ma)
        plt.fill_between(range(len(y_mean_ma)), y_mean_ma-y_err_ma, y_mean_ma+y_err_ma, alpha=0.2)

        makedirs(f"results/{self.ts}", exist_ok=True)
        plt.savefig(f"results/{self.ts}/plot.png")
        plt.close()

        # return mean_reward