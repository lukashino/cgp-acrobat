import cgp
import gymnasium as gym
import numpy as np
import sympy
from typing import Callable, List, Dict, Optional, Any
import argparse



class CGPAgent:
    def __init__(self, seed=8483992, individual_runs_cnt=30, individual_run_steps_cnt=200, render=False, 
                 population_params=None, genome_params=None, ea_params=None, evolve_params=None):
        self.seed = seed
        self.render = render
        self.individual_runs_cnt = individual_runs_cnt
        self.individual_run_steps_cnt = individual_run_steps_cnt
        self.population_params = population_params
        self.genome_params = genome_params
        self.ea_params = ea_params
        self.evolve_params = evolve_params

        self.history = {
            "expr_champion": [],
        }

    def _recording_callback(self, pop: Any) -> None:
        self.history["expr_champion"].append(pop.champion.to_sympy())

    # on weird result we return neutral action
    def _sanitize_cgp_action(self, next_action: float) -> int:
        if next_action < 0 or next_action > 2:
            return 1

        try:
            return int(np.round(next_action) % 3)
        except ValueError:
            return 1
        
    # return True on successful end-reaching state
    def _simulate_individual_episode(
            self, f: Callable, env: Any = None) -> bool:
        observation, _ = env.reset()
        for _ in range(self.individual_run_steps_cnt):
            next_action = self._sanitize_cgp_action(f(*observation))
            observation, _, terminated, truncated, _ = \
                env.step(next_action)

            if terminated:
                return True
            elif truncated:
                return False
            
        return False

    def _simulate_individual_runs(
            self, f: Callable, render: bool = False) -> List[bool]:
        env = gym.make('Acrobot-v1', render_mode="human" if render else None)

        episodes_success: List[bool] = []

        for _ in range(self.individual_runs_cnt):
            res = self._simulate_individual_episode(f, env)
            if res:
                episodes_success.append(
                    res
                )
        
        env.close()
        return episodes_success

    def _objective(self, individual: Any) -> Any:
        if not individual.fitness_is_None():
            return individual
        
        f = individual.to_func()
        try:
            res = self._simulate_individual_runs(f, render=self.render)
        except ZeroDivisionError:
            res = []

        if len(res) > 0:
            individual.fitness = float(len(res)) / self.individual_runs_cnt
        else:
            individual.fitness = -np.inf

        return individual


    def visualize_final_solution(self) -> None:
        expr = self.history["expr_champion"][-1]
        expr_str = str(expr)

        print(f'visualizing behaviour for expression "{expr_str}" ')

        x_0, x_1, x_2, x_3, x_4, x_5 = sympy.symbols("x_0, x_1, x_2, x_3, x_4, x_5")
        f_lambdify = sympy.lambdify([x_0, x_1, x_2, x_3, x_4, x_5], expr)

        def f(x,y,z,a,b,c):
            return f_lambdify(x,y,z,a,b,c)
        
        self._simulate_individual_runs(f, render=True)

    def evolve(self):
        pop = cgp.Population(**self.population_params, genome_params=self.genome_params)
        ea = cgp.ea.MuPlusLambda(**self.ea_params)
        cgp.evolve(pop, self._objective, ea, **self.evolve_params, print_progress=True, callback=self._recording_callback)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CGP Agent parameters.')

    # CGPAgent arguments
    parser.add_argument('--seed', type=int, default=8483992, help='Seed value for the simulator agent.')
    parser.add_argument('--individual_runs_cnt', type=int, default=30, help='Individual runs count.')
    parser.add_argument('--individual_run_steps_cnt', type=int, default=200, help='Individual run steps count.')
    parser.add_argument('--render', type=bool, action=argparse.BooleanOptionalAction, default=False, help='Visualizing the whole learning process.')
    parser.add_argument('--n_parents', type=int, default=8, help='Number of parents.')
    parser.add_argument('--pop_seed', type=int, default=8188211, help='Seed value for the population.')
    parser.add_argument('--n_columns', type=int, default=64, help='Number of columns.')
    parser.add_argument('--n_rows', type=int, default=1, help='Number of rows.')
    parser.add_argument('--levels_back', default=None, help='Levels back parameter.')
    parser.add_argument('--n_offsprings', type=int, default=4, help='Number of offsprings.')
    parser.add_argument('--tournament_size', type=int, default=2, help='Tournament size.')
    parser.add_argument('--mutation_rate', type=float, default=0.08, help='Mutation rate.')
    parser.add_argument('--n_processes', type=int, default=8, help='Number of processes.')
    parser.add_argument('--max_generations', type=int, default=1500, help='Max generations.')
    parser.add_argument('--termination_fitness', type=float, default=0.95, help='Termination fitness from 0 to 1.')
    parser.add_argument('--visualize', type=bool, action=argparse.BooleanOptionalAction, default=False, help='Whether to visualize the final solution.')

    args = parser.parse_args()

    agent = CGPAgent(
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