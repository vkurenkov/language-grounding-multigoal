import multiprocessing
import time
import os
import json

from multiprocessing import Pool
from argparse        import ArgumentParser
from collections     import defaultdict

### Utilized models
def dueling_dqn_ga():
    from experiments.dueling_dqn_ga.parameters import train_parameters
    from experiments.dueling_dqn_ga.parameters import instructions_parameters
    from experiments.dueling_dqn_ga.train      import start_training
    from experiments.dueling_dqn_ga.evaluate   import benchmark_all
    from experiments.dueling_dqn_ga.parameters import get_experiment_folder
    from experiments.dueling_dqn_ga.parameters import layouts_parameters 

    return "(Dueling DQN + GA)", train_parameters, instructions_parameters, start_training, benchmark_all, get_experiment_folder, layouts_parameters
def dueling_dqn_cat():
    from experiments.dueling_dqn_cat.parameters import train_parameters
    from experiments.dueling_dqn_cat.parameters import instructions_parameters
    from experiments.dueling_dqn_cat.train      import start_training
    from experiments.dueling_dqn_cat.evaluate   import benchmark_all
    from experiments.dueling_dqn_cat.parameters import get_experiment_folder
    from experiments.dueling_dqn_cat.parameters import layouts_parameters 

    return "(Dueling DQN + Cat)", train_parameters, instructions_parameters, start_training, benchmark_all, get_experiment_folder, layouts_parameters
def dueling_dqn_ga_per():
    from experiments.dueling_dqn_ga_per.parameters import train_parameters
    from experiments.dueling_dqn_ga_per.parameters import instructions_parameters
    from experiments.dueling_dqn_ga_per.train      import start_training
    from experiments.dueling_dqn_ga_per.evaluate   import benchmark_all
    from experiments.dueling_dqn_ga_per.parameters import get_experiment_folder
    from experiments.dueling_dqn_ga_per.parameters import layouts_parameters 

    return "(Dueling DQN + GA + PER)", train_parameters, instructions_parameters, start_training, benchmark_all, get_experiment_folder, layouts_parameters
def dueling_dqn_cat_per():
    from experiments.dueling_dqn_cat_per.parameters import train_parameters
    from experiments.dueling_dqn_cat_per.parameters import instructions_parameters
    from experiments.dueling_dqn_cat_per.train      import start_training
    from experiments.dueling_dqn_cat_per.evaluate   import benchmark_all
    from experiments.dueling_dqn_cat_per.parameters import get_experiment_folder
    from experiments.dueling_dqn_cat_per.parameters import layouts_parameters 

    return "(Dueling DQN + Cat + PER)", train_parameters, instructions_parameters, start_training, benchmark_all, get_experiment_folder, layouts_parameters

def experiment_name(params):
    seed = params["seed"]
    conjunction = params["conjunction"]
    unseen_proportion = params["unseen_proportion"]
    agent = params["agent"]
    level = params["level"]

    name, _, _, _, _, _, _ = agent()
    name += "-{}-{}-{}-{}".format(seed, conjunction, level, unseen_proportion)

    return name

def experiment_run(params):
    seed = params["seed"]
    conjunction = params["conjunction"]
    unseen_proportion = params["unseen_proportion"]
    agent = params["agent"]
    level = params["level"]

    _, train_params, instruction_params, start_training, benchmark_all, _, layout_params = agent()
    train_params["seed"]       = seed
    instruction_params["seed"] = seed
    layout_params["seed"]      = seed

    instruction_params["conjunctions"] = conjunction
    instruction_params["unseen_proportion"] = unseen_proportion
    instruction_params["level"] = level

    name = experiment_name(params)
    print("{}: Training is started.".format(name))
    start_time = time.time()
    start_training(erase_folder=True) # dangerous af
    end_time   = time.time()
    print("{}: Training is done ({}).".format(name, str(end_time - start_time)))

    experiment_benchmark(params)

def experiment_benchmark(params):
    seed = params["seed"]
    conjunction = params["conjunction"]
    unseen_proportion = params["unseen_proportion"]
    agent = params["agent"]
    level = params["level"]

    _, train_params, instruction_params, _, benchmark_all, _, layout_params = agent()
    train_params["seed"]       = seed
    instruction_params["seed"] = seed
    layout_params["seed"]      = seed

    instruction_params["conjunctions"] = conjunction
    instruction_params["unseen_proportion"] = unseen_proportion
    instruction_params["level"] = level

    name = experiment_name(params)
    print("{}: Benchmarking is started.".format(name))
    start_time = time.time()
    benchmark_all()
    end_time   = time.time()
    print("{}: Benchmarking is done ({}).".format(name, str(end_time - start_time)))


def run_training(setups, num_processes, ignore_trained):
    results = []
    with Pool(processes=num_processes) as pool:
        for args in setups:
            if ignore_trained:
                agent      = args["agent"]()
                get_folder = agent[5]
                if not os.path.exists(get_folder()):
                    results.append(pool.apply_async(experiment_run, (args,)))
            else:
                results.append(pool.apply_async(experiment_run, (args,)))

        for result in results:
            result.get()

def run_benchmarking(setups, num_processes, ignore_benchmarked):
    results = []
    with Pool(processes=num_processes) as pool:
        for args in setups:
            if ignore_benchmarked:
                agent      = args["agent"]()
                get_folder = agent[5]
                if not os.path.exists(os.path.join(get_folder(), "benchmarks.json")):
                    results.append(pool.apply_async(experiment_benchmark, (args,)))
            else:
                results.append(pool.apply_async(experiment_benchmark, (args,)))
        for result in results:
            result.get()

# def show_benchmarking_stats(setups):
#     results = defaultdict(defaultdict(list))
#     for args in setups:
#         agent      = args["agent"]
#         get_folder = agent[5]
#         benchmarks_path = os.path.join(get_folder(), "benchmarks.json")
#         if not os.path.exists(benchmarks_path):
#             pass

#         identity = (args["level"], args["conjunction"], agent[0])
#         with open(benchmarks_path, mode="r") as f:
#             benchmarks = json.load(f)
#         results[identity][args["unseen_proportion"]].append((success_rate(benchmarks),
#                     mean_traj_len_succ(benchmarks), 
#                     optimal_mean_traj_len_succ(benchmarks),
#                     optimal_mean_traj_len_all(benchmarks)))

    # 1 - I want to know the algorithm with the best training score for all proportions (harmonic mean)
    # 2 - I want to know the algorithm with the best unseen combinations score for all proportions (harmonic mean)
    # 3 - I want to know the algorithm with the best hight sub-goals score for all proportions (harmonic mean)

    # 4 - I want to know how the unseen proportion of instructions affect training/unseen combinations/higher sub-goals (draw few graphs)
    # 5 - I want to know how performance drops with the increase of instructions level for training/unseen/higher
    # 6 - I want to know how performance drops with the increase of synonumous for training/unseen/higher

    # 7 - I want to know how successfully algorithms handle "but first" conjunction

    #for regime in results.
    # for identity in sorted(results.keys()):
    #     name = "Level: {}; Conjunctions: {}; Testing Regime: {}".format(identity[0], identity[1], identity[2])

        # We should aggregate results by
            # Level + Conjunctions + Algorithm
            # For every Unseen Proportion
            #   - Training:     Success Rate, Mean Trajectory Length (for successful instructions), Optimal Mean Trajectory Length (for successful instructions), Optimal Mean Trajectory Length
            #   - Unseen Combs: Success Rate, Mean Trajectory Length (for successful instructions), Optimal Mean Trajectory Length (for successful instructions), Optimal Mean Trajectory Length
            #   - Unseen High:  Success Rate, Mean Trajectory Length (for successful instructions), Optimal Mean Trajectory Length (for successful instructions), Optimal Mean Trajectory Length

    # name = "Level: {}; Conjunctions: {}; Algorithm: {}".format(args["level"], args["conjunction"], agent[0])

PARAMETERS       = {
    "unseen_proportions": [0.1, 0.3, 0.5, 0.7, 0.9],
    "conjunctions":       ["only_comma"],
    "seeds":              [0],
    "methods":            [dueling_dqn_ga, dueling_dqn_cat, dueling_dqn_ga_per, dueling_dqn_cat_per],
    "levels":             [3, 4]
}

# Generate all setups from given parameters
ALL_SETUPS = []
for seed in PARAMETERS["seeds"]:
    for conjunction in PARAMETERS["conjunctions"]:
        for level in PARAMETERS["levels"]:
            for method in PARAMETERS["methods"]:
                for unseen_proportion in PARAMETERS["unseen_proportions"]:
                    args = {"seed": seed, "conjunction": conjunction, "unseen_proportion": unseen_proportion, "agent": method, "level": level}
                    ALL_SETUPS.append(args)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--mode", choices=["train", "benchmark_only"], required=True)
    parser.add_argument("--num-processes", type=int, required=True)
    parser.add_argument("--ignore-trained", type=bool, default=True)
    parser.add_argument("--ignore-benchmarked", type=bool, default=True)
    args = parser.parse_args()

    if args.mode == "train":
        run_training(ALL_SETUPS, args.num_processes, False)#args.ignore_trained)
    elif args.mode.startswith("benchmark"):
        print(args.ignore_benchmarked)
        print(args.num_processes)
        run_benchmarking(ALL_SETUPS, args.num_processes, False)#args.ignore_benchmarked)