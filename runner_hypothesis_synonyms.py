import multiprocessing
import time
from multiprocessing import Pool
from argparse        import ArgumentParser

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
    from experiments.dueling_dqn_ga_per.parameters import create_experiment_folder
    from experiments.dueling_dqn_ga_per.parameters import layouts_parameters 

    return "(Dueling DQN + GA + PER)", train_parameters, instructions_parameters, start_training, benchmark_all, create_experiment_folder, layouts_parameters
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

    name = experiment_name(seed, conjunction, unseen_proportion, agent)
    print("{}: Training is started.".format(name))
    start_time = time.time()
    start_training(erase_folder=True) # dangerous af
    end_time   = time.time()
    print("{}: Training is done ({}).".format(name, str(end_time - start_time)))

    experiment_benchmark(seed, conjunction, unseen_proportion, agent)
def experiment_validate(params):
    seed = params["seed"]
    conjunction = params["conjunction"]
    unseen_proportion = params["unseen_proportion"]
    agent = params["agent"]
    level = params["level"]

    _, train_params, instruction_params, _, _, get_folder, layout_params = agent()
    train_params["seed"]       = seed
    instruction_params["seed"] = seed
    layout_params["seed"]      = seed

    instruction_params["conjunctions"] = conjunction
    instruction_params["unseen_proportion"] = unseen_proportion
    instruction_params["level"] = level
    get_folder()
def experiment_benchmark(params):
    seed = params["seed"]
    conjunction = params["conjunction"]
    unseen_proportion = params["unseen_proportion"]
    agent = params["agent"]

    _, train_params, instruction_params, _, benchmark_all, _, layout_params = agent()
    train_params["seed"]       = seed
    instruction_params["seed"] = seed
    layout_params["seed"]      = seed

    instruction_params["conjunctions"] = conjunction
    instruction_params["unseen_proportion"] = unseen_proportion

    name = experiment_name(seed, conjunction, unseen_proportion, agent)
    print("{}: Benchmarking is started.".format(name))
    start_time = time.time()
    benchmark_all()
    end_time   = time.time()
    print("{}: Benchmarking is done ({}).".format(name, str(end_time - start_time)))

PARALLEL_WORKERS = 5
PARAMETERS       = {
    "unseen_proportions": [0.1, 0.3, 0.5, 0.7, 0.9],
    "conjunctions":       ["only_comma"],
    "seeds":              [0],
    "methods":            [dueling_dqn_ga, dueling_dqn_cat, dueling_dqn_ga_per, dueling_dqn_cat_per],
    "levels":             2
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

# if __name__ == "__main__":
#     parser = ArgumentParser()
#     parser.add_argument("--")

# Confirm what setups should be run
valid_setups = []
setup_action = []
for setup in ALL_SETUPS:
    # print("Setup: {}".format(experiment_name(*setup)))
    # experiment_validate(*setup) # Will allow us manually setup whether we want to erase the folder and etc.
    # print("What to do? (t - train, b - benchmark, n - skip")
    # confirm = input()
    # if confirm == "t" or confirm  == "b":
    #     setup_action.append(confirm)
    #     valid_setups.append(setup)
    setup_action.append("t")
    valid_setups.append(setup)

print("Num validated setups: {}".format(len(valid_setups)))

print("Enter number of processes: ")
num_processes = int(input())

print("Starting eperiments...")
results = []
with Pool(processes=num_processes) as pool:
    for args, action in zip(valid_setups, setup_action):
        if action == "b":
            results.append(pool.apply_async(experiment_benchmark, args))
        elif action == "t":
            results.append(pool.apply_async(experiment_run, args))

    for result in results:
        result.get()