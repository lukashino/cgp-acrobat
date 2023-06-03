import cgp
import gymnasium as gym
import numpy as np
import sympy

seed=8483992

population_params = {
    "n_parents": 8,
    "seed": 8188211
}


class ActionConverter(cgp.OperatorNode):
    _arity = 1
    _def_output = "x_0 % 3"

# class Sine(cgp.OperatorNode):
#     import math
#     _arity = 1
#     _def_output = "math.sin(x_0)"
#     _def_numpy_output = "np.sin(x_0)"


# class Cosine(cgp.OperatorNode):
#     import math
#     _arity = 1
#     _def_output = "math.cos(x_0)"
#     _def_numpy_output = "np.cos(x_0)"


class ConstantFloatMinusTorque(cgp.ConstantFloat):
    _def_output = "0.0"

class ConstantFloatNeutralTorque(cgp.ConstantFloat):
    _def_output = "1.0"

class ConstantFloatPlusTorque(cgp.ConstantFloat):
    _def_output = "2.0"


genome_params = {
    "n_inputs": 6,
    "n_outputs": 1,
    "n_columns": 24,
    "n_rows": 1,
    "levels_back": None,
    "primitives": (
            cgp.Add,
            cgp.Sub,
            cgp.Mul,
            cgp.Div,
            cgp.ConstantFloat,
            # Sine,
            # Cosine,
            # ConstantFloatMinusTorque,
            # ConstantFloatNeutralTorque,
            # ConstantFloatPlusTorque,
        )
}


ea_params = {
    "n_offsprings": 8,
    "tournament_size": 1,
    "mutation_rate": 0.08,
    "n_processes": 8
}

evolve_params = {
        "max_generations": 1500,
        "termination_fitness": 0,
    }

INDIVIDUAL_RUNS_CNT = 5
INDIVIDUAL_RUN_STEPS_CNT = 499

history = {}
history["fitness_parents"] = []
history["expr_champion"] = []
def recording_callback(pop):
    history["expr_champion"].append(pop.champion.to_sympy())
    history["fitness_parents"].append(pop.fitness_parents())










def objective(individual):
    if not individual.fitness_is_None():
        return individual


    cum_reward_all_episodes = []
    f = individual.to_func()

    env = gym.make('Acrobot-v1') #, render_mode="human")
    observation, _ = env.reset(seed=seed)
    try:
        for _ in range(INDIVIDUAL_RUNS_CNT):
            observation, _ = env.reset(seed=seed)

            cum_reward_this_episode = 0
            for _ in range(INDIVIDUAL_RUN_STEPS_CNT):
                next_action_npfloat = f(*observation)
                # if next_action_npfloat is np.NaN or next_action_npfloat is np.Inf or next_action_npfloat is -np.Inf:
                #     next_action_npfloat = 1.0
                try:
                    next_action = int(np.round(next_action_npfloat) % 3)
                except ValueError:
                    next_action = 1
                    next_action_npfloat = 1.0

                if next_action < 0 or next_action > 2:
                    next_action = 1
                
                # next_action = int(next_action_npfloat)
                observation, reward, terminated, truncated, _ = env.step(next_action)
                # print(f"Next action float {next_action_npfloat}, nexta {next_action}, {observation}")
                cos1 = observation[0]
                syn1 = observation[1]
                cos2 = observation[2]
                syn2 = observation[3]
                vel1 = observation[4]
                vel2 = observation[5]

                angle_reward = (1 / syn1) % 12
                velocity_reward = (vel1)

                # cum_reward_this_episode += angle_reward + velocity_reward
                # if reward == 100:
                #     reward = 10000

                cum_reward_this_episode += reward

                if terminated or truncated:
                    print("end story")
                    cum_reward_this_episode += 100
                    cum_reward_all_episodes.append(cum_reward_this_episode)
                    cum_reward_this_episode = 0
                    observation, _ = env.reset(seed=seed)

        n_episodes = float(len(cum_reward_all_episodes))
        # mean_cum_reward = np.mean(cum_reward_all_episodes)
        if n_episodes > 0:
            mean_cum_reward = np.amax(cum_reward_all_episodes)
            individual.fitness = mean_cum_reward # n_episodes / INDIVIDUAL_RUNS_CNT + 
        else:
            individual.fitness = -np.inf
    
    except ZeroDivisionError:
        individual.fitness = -np.inf

    env.close()
    return individual

def visualize_behaviour_for_evolutionary_jumps(seed, history, only_final_solution=True):
    n_runs_per_individual = 1
    n_total_steps = 499

    expr = history["expr_champion"][-1]
    expr_str = str(expr).replace("x_0", "x").replace("x_1", "dx/dt")

    print(f'visualizing behaviour for expression "{expr_str}" ')

    x_0, x_1, x_2, x_3, x_4, x_5 = sympy.symbols("x_0, x_1, x_2, x_3, x_4, x_5")
    f_lambdify = sympy.lambdify([x_0, x_1, x_2, x_3, x_4, x_5], expr)

    def f(x,y,z,a,b,c):
        return f_lambdify(x,y,z,a,b,c)
    
    cum_reward_all_episodes = []
    env = gym.make('Acrobot-v1', render_mode="human")
    observation, _ = env.reset(seed=seed)
    for _ in range(INDIVIDUAL_RUNS_CNT):
        observation, _ = env.reset(seed=seed)

        cum_reward_this_episode = 0
        for _ in range(INDIVIDUAL_RUN_STEPS_CNT):
            next_action_npfloat = f(*observation)
            try:
                next_action = int(np.round(next_action_npfloat) % 3)
            except ValueError:
                next_action = 1
                next_action_npfloat = 1.0

            if next_action < 0 or next_action > 2:
                next_action = 1
            
            # next_action = int(next_action_npfloat)
            observation, reward, terminated, truncated, _ = env.step(next_action)

            cum_reward_this_episode += reward

            if terminated or truncated:
                print("end story")
                cum_reward_this_episode += 50
                cum_reward_all_episodes.append(cum_reward_this_episode)
                cum_reward_this_episode = 0
                observation, _ = env.reset(seed=seed)

    env.close()
    # def f(x,y):
    #     return f_lambdify(x, y)

    #         inner_objective(f, seed, n_runs_per_individual, n_total_steps, render=True)

    # max_fitness = -np.inf
    # for i, fitness in enumerate(history["fitness_champion"]):

    #     if only_final_solution and i != (len(history["fitness_champion"]) - 1):
    #         continue

    #     if fitness > max_fitness:
    #         expr = history["expr_champion"][i]
    #         expr_str = str(expr).replace("x_0", "x").replace("x_1", "dx/dt")

    #         print(f'visualizing behaviour for expression "{expr_str}" (fitness: {fitness:.05f})')

    #         x_0, x_1 = sympy.symbols("x_0, x_1")
    #         f_lambdify = sympy.lambdify([x_0, x_1], expr)

    #         def f(x,y):
    #             return f_lambdify(x, y)

    #         inner_objective(f, seed, n_runs_per_individual, n_total_steps, render=True)

    #         max_fitness = fitness



pop = cgp.Population(**population_params, genome_params=genome_params)
ea = cgp.ea.MuPlusLambda(**ea_params)
print("all good")
cgp.evolve(pop, objective, ea, **evolve_params, print_progress=True, callback=recording_callback)
print("more than good")

visualize_behaviour_for_evolutionary_jumps(1, history, only_final_solution=True)
