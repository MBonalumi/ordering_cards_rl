import numpy as np
import gymnasium as gym

from stable_baselines3 import PPO
from myPPO import MyPPO

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
    
    def plot_results(self, n_regrets):
        # save plot in results/ts/plot.png
        plt.ioff()

        # n_regrets =[env.unwrapped.envs[i].regrets for i in range(len(env.unwrapped.envs))]
        print(f"regret lengths: {[len(x) for x in n_regrets]}")
        min_len = min([len(x) for x in n_regrets])
        n_regrets = [x[:min_len] for x in n_regrets]

        y = np.array(n_regrets)
        print(y.shape, [len(x) for x in y])
        y_mean = np.mean(y, axis=0)
        y_err = np.std(y, axis=0)

        # y_mean_ma = np.convolve(y_mean,  np.ones((timesteps//100,)) / timesteps//100 , mode='valid')
        ma_amt = y_mean.size // 100
        # ma_amt = 1000
        y_mean_ma = np.convolve(y_mean,  np.ones((ma_amt,)) / ma_amt , mode='valid')
        y_err_ma = np.convolve(y_err,  np.ones((ma_amt,)) / ma_amt , mode='valid')

        figsize = (10,6)
        plt.figure(figsize=figsize)

        plt.title(f"Regret over time, {self.cards_amount} cards, {self.parallel_runs} parallel runs")

        plt.grid(True, 'major', 'y', ls='--', lw=.5, c='k', alpha=.3)
        plt.axhline(0, color='black', lw=1, alpha=.7)
        plt.plot(y_mean_ma)
        plt.fill_between(range(len(y_mean_ma)), y_mean_ma-y_err_ma, y_mean_ma+y_err_ma, alpha=0.2)

        makedirs(f"results/{self.ts}", exist_ok=True)
        plt.savefig(f"results/{self.ts}/plot.png")
        plt.close()

        # return mean_reward

    def make_env(self, env_id, rank):
        """
        Utility function for multiprocessed env.
    
        :param env_id: (str) the environment ID
        :param num_env: (int) the number of environment you wish to have in subprocesses
        :param seed: (int) the inital seed for RNG
        :param rank: (int) index of the subprocess
        """
        def _init():
            env = ordering_cards_env(env_id, self.seed + rank, self.cards_amount, self.ts)
            # env.seed(seed + rank)
            return env
        return _init

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

        # env = make_vec_env(     [lambda : ordering_cards_env(i, self.seed + i, self.cards_amount, self.ts) for i in range(self.parallel_runs)], 
        #                         n_envs=self.parallel_runs, vec_env_cls=SubprocVecEnv, vec_env_kwargs=dict(start_method='fork'))

        # train_env = SubprocVecEnv( [lambda ordering_cards_env(i, self.seed + i, self.cards_amount, self.ts) for i in range(self.parallel_runs)] , 
        #                             start_method='spawn')

        train_env = SubprocVecEnv( [self.make_env(i, i) for i in range(self.parallel_runs)] ,  # type: ignore
                                    start_method='fork')


        # check_env(env)
        
        # model = PPO("MlpPolicy", train_env, verbose=1)
        model = MyPPO("MlpPolicy", train_env, verbose=1)

        timesteps = int(self.games_to_play * self.cards_amount * 2 // self.parallel_runs)
        real_timesteps = max(  (timesteps//(2048*self.parallel_runs) + 1) * 2048*self.parallel_runs,   2048 * self.parallel_runs  )
        model.learn(total_timesteps=real_timesteps,
                    progress_bar=progress_bar)
        # mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
        # print(f"mean_reward:{mean_reward:.2f} +/- {std_reward:.2f}")
        
        print("done training, now plotting")

        #save model
        model.save(f"results/{self.ts}/model")

        #load model
        # model = PPO.load("results/20210920-1518/model")

        self.plot_results(train_env.unwrapped.get_attr('regrets'))