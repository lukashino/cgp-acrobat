import random
import cgp
import gymnasium as gym
import numpy as np
import sympy
from typing import Callable, List
import argparse
from time import perf_counter
import csv
from multiprocessing import Lock


class ILogCGPIndividualRun:
    def __init__():
        pass

    def log(self, ind_run_id: int, ind_id: int, exp_id: int, replay_id: int, ttc_sec: float, stc: float):
        pass

class ILogCGPIndividual:
    def __init__():
        pass

    def log(self, ind_id: int, exp_id: int, replay_id: int, ttc_sec: float, fitness: float):
        pass

class ILogCGPEvolution:
    def __init__():
        pass

    def log(self, gene_id: int, experiment_id: int, ttc_sec: float):
        pass

# class ILogCGPExperiment:
#     def __init__():
#         pass

#     def log(self, gene_id: int, experiment_id: int, ttc_sec: float):
#         pass


class CSVLoggerCGPIndividualRun(ILogCGPIndividualRun):
    def __init__(self, fname: str):
        self.fname = fname
        self.fieldnames = [
            'Individual Run ID', 
            'Individual ID', 
            'Experiment ID',
            'Replay ID',
            'Time To Completion [seconds]', 
            'Steps to completion'
        ]

        with open(self.fname, "w", encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()

    def log(self, ind_run_id: int, ind_id: int, exp_id: int, replay_id: int, ttc_sec: float, stc: float):
        row = {
            'Individual Run ID': ind_run_id, 
            'Individual ID': ind_id, 
            'Experiment ID': exp_id,
            'Replay ID': replay_id,
            'Time To Completion [seconds]': ttc_sec, 
            'Steps to completion': stc
        }
        with open(self.fname, "a", encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

class CSVLoggerCGPIndividual(ILogCGPIndividual):
    def __init__(self, fname: str):
        self.fname = fname
        self.fieldnames = [ 
            'Individual ID',
            'Experiment ID', 
            'Replay ID',
            'Time To Completion [seconds]', 
            'Fitness',
        ]
        with open(self.fname, "w") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writeheader()
    
    def log(self, ind_id: int, exp_id: int, replay_id: int, ttc_sec: float, fitness: float):
        row = {
            'Individual ID': ind_id, 
            'Experiment ID': exp_id, 
            'Replay ID': replay_id,
            'Time To Completion [seconds]': ttc_sec, 
            'Fitness': fitness,
        }
        with open(self.fname, "a", encoding='UTF8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            writer.writerow(row)

class CGPIndividualEpisode:
    def __init__(self, id: int, ind_id: int, exp_id: int, replay_id: int, 
                 max_episode_steps: int, f: Callable, env: gym.Env, 
                 ep_logger: ILogCGPIndividualRun = None) -> None:
        self.id = id
        self.ind_id = ind_id
        self.exp_id = exp_id
        self.replay_id = replay_id
        self.f = f
        self.env = env
        self.max_episode_steps = max_episode_steps
        self.ep_logger: ILogCGPIndividualRun = ep_logger

    # on weird result we return neutral action
    def _sanitize_cgp_action(self, next_action: float) -> int:
        if next_action < 0 or next_action > 2:
            return 1

        try:
            return int(np.round(next_action) % 3)
        except ValueError:
            return 1
    
    def simulate(self, seed: int = None) -> bool:
        if seed is not None:
            observation, _ = self.env.reset(seed=seed)
        else:
            observation, _ = self.env.reset()
        res = False
        steps = 0
        start_time = perf_counter()
        for i in range(self.max_episode_steps):
            next_action = self._sanitize_cgp_action(self.f(*observation))
            observation, _, terminated, truncated, _ = \
                self.env.step(next_action)
            steps += 1
            
            if terminated or truncated:
                res = True if terminated else False
                break

        end_time = perf_counter()
        elapsed_time = end_time - start_time
        if self.ep_logger is not None:
            self.ep_logger.log(
                self.id, self.ind_id, self.exp_id, self.replay_id, elapsed_time, steps)
            
        return res

class CGPIndividual:
    def __init__(
            self, id: int, exp_id: int, replay_id: int, f: Callable, 
            episodes_cnt: int, max_episodes_steps: int, render: bool = False, 
            env_seed: int = None, ir_logger: ILogCGPIndividualRun = None, 
            i_logger: ILogCGPIndividual = None) -> List[bool]:
        self.exp_id = exp_id
        self.replay_id = replay_id
        self.id = id
        self.f = f
        self.human_render = render
        self.episodes_cnt = episodes_cnt
        self.max_episodes_steps = max_episodes_steps
        self.env_seed = env_seed
        self.ir_logger = ir_logger
        self.i_logger = i_logger

    def _fitness_get(self, episodes_rslts: List[bool]) -> float:
        return float(len(episodes_rslts)) / self.episodes_cnt

    def simulate_episodes(self):
        episodes_success: List[bool] = []
        env = gym.make(
            'Acrobot-v1', render_mode="human" if self.human_render else None)

        for i in range(self.episodes_cnt):
            ep = CGPIndividualEpisode(i, self.id, self.exp_id, self.replay_id, self.max_episodes_steps, self.f, env, self.ir_logger)
            res = ep.simulate(self.env_seed)
            if res:
                episodes_success.append(res)
        
        env.close()
        return self._fitness_get(episodes_success)
    


class CGPAgent:
    def __init__(self, exp_id: int=None, replay_id: int=None, seed=None, 
                 individual_runs_cnt=30, individual_run_steps_cnt=200, 
                 render=False, population_params=None, genome_params=None, 
                 ea_params=None, evolve_params=None):
        self.exp_id = exp_id
        self.replay_id = replay_id
        self.inter_time: float = 0.0
        self.individ_id = 0
        self.seed = seed
        self.render = render
        self.individual_runs_cnt = individual_runs_cnt
        self.individual_run_steps_cnt = individual_run_steps_cnt
        self.population_params = population_params
        self.genome_params = genome_params
        self.ea_params = ea_params
        self.evolve_params = evolve_params

        self.logger_ir: ILogCGPIndividualRun = CSVLoggerCGPIndividualRun(
            f"indrun"
            f"_p{self.population_params['n_parents']}"
            f"_gen_cols{self.genome_params['n_columns']}"
            f"_gen_rows{self.genome_params['n_rows']}"
            f"_gen_lbacks{self.genome_params['levels_back']}"
            f"_gen_pmtvs{len(self.genome_params['primitives'])}"
            f"_ea_offspgs{self.ea_params['n_offsprings']}"
            f"_ea_trnmtsz{self.ea_params['tournament_size']}"
            f"_ea_mutrt{self.ea_params['mutation_rate']}"
            f"_ea_proc_cnt{self.ea_params['n_processes']}"
            f"_tfit{self.evolve_params['termination_fitness']}"
            f"_exp_id{self.exp_id}"
            f"_replay_id{self.replay_id}"
            ".csv"
        )
        self.logger_i: ILogCGPIndividual = CSVLoggerCGPIndividual(
            f"individuals"
            f"_p{self.population_params['n_parents']}"
            f"_gen_cols{self.genome_params['n_columns']}"
            f"_gen_rows{self.genome_params['n_rows']}"
            f"_gen_lbacks{self.genome_params['levels_back']}"
            f"_gen_pmtvs{len(self.genome_params['primitives'])}"
            f"_ea_offspgs{self.ea_params['n_offsprings']}"
            f"_ea_trnmtsz{self.ea_params['tournament_size']}"
            f"_ea_mutrt{self.ea_params['mutation_rate']}"
            f"_ea_proc_cnt{self.ea_params['n_processes']}"
            f"_tfit{self.evolve_params['termination_fitness']}"
            f"_exp_id{self.exp_id}"
            f"_replay_id{self.replay_id}"
            ".csv")

        self.history = {
            "expr_champion": [],
            "node_champion": [],
        }

    def _recording_callback(self, pop: cgp.Population) -> None:
        self.history["expr_champion"].append(pop.champion.to_sympy())
        self.history["node_champion"].append(pop.champion)
        if pop.generation > 0:
            inter_diff = perf_counter() - self.inter_time
            self.logger_i.log(
                pop.generation, self.exp_id, self.replay_id, inter_diff, pop.champion.fitness)

        self.inter_time = perf_counter()

    def _objective(self, individual: cgp.individual) -> cgp.individual:
        if not individual.fitness_is_None():
            return individual
        
        ind_id = random.randint(0,2**63-1)
        
        f = individual.to_func()
        ivdl = CGPIndividual(
            ind_id, self.exp_id, self.replay_id, f, self.individual_runs_cnt, 
            self.individual_run_steps_cnt, render=self.render, 
            env_seed=self.seed, ir_logger=self.logger_ir, 
            i_logger=self.logger_i)
        try:
            individual.fitness = ivdl.simulate_episodes()
        except ZeroDivisionError:
            individual.fitness = -np.inf

        return individual

    def visualize_final_solution(self) -> None:
        ind: cgp.individual = self.history["node_champion"][-1]
        cg = cgp.CartesianGraph(ind.genome)
        print(f"CG pretty print - {cg.pretty_str()}")

        expr = self.history["expr_champion"][-1]
        expr_str = str(expr)

        print(f'visualizing behaviour for expression "{expr_str}" ')

        x_0, x_1, x_2, x_3, x_4, x_5 = sympy.symbols("x_0, x_1, x_2, x_3, x_4, x_5")
        f_lambdify = sympy.lambdify([x_0, x_1, x_2, x_3, x_4, x_5], expr)

        def f(x,y,z,a,b,c):
            return f_lambdify(x,y,z,a,b,c)
        
        cgp_ind = CGPIndividual(0, 0, 0, f, self.individual_runs_cnt, self.individual_run_steps_cnt, render=True, env_seed=None)
        cgp_ind.simulate_episodes()

    def evolve(self):
        pop = cgp.Population(**self.population_params, genome_params=self.genome_params)
        ea = cgp.ea.MuPlusLambda(**self.ea_params)
        cgp.evolve(self._objective, pop, ea, **self.evolve_params, print_progress=True, callback=self._recording_callback)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CGP Agent parameters.')

    # CGPAgent arguments
    parser.add_argument('--experiment-id', type=int, default=0, help='Experiment ID for later evaluation purposes. Logs information to CSV files.')
    parser.add_argument('--replay-id', type=int, default=0, help='Replay ID for later evaluation purposes. Logs information to CSV files.')
    parser.add_argument('--seed', type=int, default=None, help='Seed value for the simulator agent.')
    parser.add_argument('--individual_runs_cnt', type=int, default=30, help='Individual runs count.')
    parser.add_argument('--individual_run_steps_cnt', type=int, default=200, help='Individual run steps count.')
    parser.add_argument('--render', type=bool, action=argparse.BooleanOptionalAction, default=False, help='Visualizing the whole learning process.')
    parser.add_argument('--n_parents', type=int, default=8, help='Number of parents.')
    parser.add_argument('--pop_seed', type=int, default=8188211, help='Seed value for the population.')
    parser.add_argument('--n_columns', type=int, default=16, help='Number of columns.')
    parser.add_argument('--n_rows', type=int, default=1, help='Number of rows.')
    parser.add_argument('--levels_back', type=int, default=None, help='Levels back parameter.')
    parser.add_argument('--n_offsprings', type=int, default=4, help='Number of offsprings.')
    parser.add_argument('--tournament_size', type=int, default=2, help='Tournament size.')
    parser.add_argument('--mutation_rate', type=float, default=0.08, help='Mutation rate.')
    parser.add_argument('--n_processes', type=int, default=8, help='Number of processes.')
    parser.add_argument('--max_generations', type=int, default=1500, help='Max generations.')
    parser.add_argument('--termination_fitness', type=float, default=0.95, help='Termination fitness from 0 to 1.')
    parser.add_argument('--visualize', type=bool, action=argparse.BooleanOptionalAction, default=True, help='Whether to visualize the final solution.')

    args = parser.parse_args()

    agent = CGPAgent(
        exp_id=args.experiment_id,
        replay_id=args.replay_id,
        seed=args.seed,
        individual_runs_cnt=args.individual_runs_cnt,
        individual_run_steps_cnt=args.individual_run_steps_cnt,
        render=args.render,
        population_params={
            "n_parents": args.n_parents,
            "seed": args.pop_seed
        },
        genome_params={
            "n_inputs": 6,
            "n_outputs": 1,
            "n_columns": args.n_columns,
            "n_rows": args.n_rows,
            "levels_back": args.levels_back,
            "primitives": (
                cgp.Add,
                cgp.Sub,
                cgp.Mul,
                cgp.Div,
                cgp.ConstantFloat,
            )
        },
        ea_params={
            "n_offsprings": args.n_offsprings,
            "tournament_size": args.tournament_size,
            "mutation_rate": args.mutation_rate,
            "n_processes": args.n_processes
        },
        evolve_params={
            "max_generations": args.max_generations,
            "termination_fitness": args.termination_fitness,
        }
    )


    agent.evolve()

    if args.visualize:
        agent.visualize_final_solution()
