import argparse
from simulate_game import Simulation
import time
import os

if __name__ == '__main__':
    print("Hello there!")

    parser = argparse.ArgumentParser()
    # parser.add_argument('--sameitem', action='store_const', const=True, help='')
    parser.add_argument('--cards-amount', type=int, default=10, help='the length of the cards the agent will play with')
    parser.add_argument('--games-to-play', type=int, default=100, help='the number of games to play')
    parser.add_argument('--parallel-runs', type=int, default=1, help='the number of games to play in parallel during training')
    parser.add_argument('--seed', type=int, default=0, help='random initialization seed')
    # parser.add_argument('--setting', type=str, default="ciao", help='')

    args = parser.parse_args()

    args.cards_amount = int(args.cards_amount)
    args.seed = int(args.seed)
    args.games_to_play = int(args.games_to_play)
    args.parallel_runs = int(args.parallel_runs)

    print(f"cards_amount: {args.cards_amount}, seed: {args.seed}")
    command_string = f"python3 run.py --cards-amount {args.cards_amount} --games-to-play {args.games_to_play} --parallel-runs {args.parallel_runs} --seed {args.seed}"

    ts = time.strftime("%Y%m%d-%H%M", time.localtime())
    folder = f"results/{ts}"
    os.makedirs(folder, exist_ok=True)
    with open(f"{folder}/command.txt", "w") as f:
        f.write(command_string)

    simulation = Simulation(args.seed, args.cards_amount, args.games_to_play, args.parallel_runs, f"not_invalid_{ts}")
    print("begin sim")
    simulation.run( progress_bar=True )
    print("sim is done")
