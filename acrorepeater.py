import argparse
import subprocess
from time import perf_counter

params = {
    '--experiment-id': 0,
    '--replay-id': 0,
    '--seed': None,
    '--individual_runs_cnt': 30,
    '--individual_run_steps_cnt': 200,
    '--render': False,
    '--n_parents': 8,
    '--pop_seed': 8188211,
    '--n_columns': 64,
    '--n_rows': 1,
    '--levels_back': None,
    '--n_offsprings': 4,
    '--tournament_size': 1,
    '--mutation_rate': 0.08,
    '--n_processes': 8,
    '--max_generations': 1500,
    '--termination_fitness': 0.95,
    '--visualize': False
}

def benchmark(command: str):
    total_time = 0
    start = perf_counter()
    process = subprocess.run(command, shell=True)
    end = perf_counter()
    total_time += end - start
    return total_time
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark an external program.')
    parser.add_argument('--test-param', type=str, required=True, help='Parameter to test.')
    parser.add_argument('--test-values', type=str, required=True, help='Comma-separated list of values to test.')
    parser.add_argument('--test-replays', type=int, required=True, help='Number of times the base program should be evaluated.')
    args = parser.parse_args()

    # Convert the argument to list of ints
    test_values = [int(x) if x.isdigit() else x for x in args.test_values.split(',')]
    # Test the parameter under test
    test_param = args.test_param if args.test_param in params.keys() else exit(-1)

    for id_val, test_val in enumerate(test_values):
        for i in range (0, args.test_replays):
            benched_params = params
            benched_params['--experiment-id'] = id_val
            benched_params['--replay-id'] = i
            benched_params[test_param] = test_val
            # Format parameters for bash command
            
            formatted_params = ' '.join(
                f'{k}={v}' 
                if not isinstance(v, bool) 
                else f'{k}' 
                if v 
                else f'--no-{k[2:]}' 
                for k, v in benched_params.items() 
                if v is not None)
            # Command to benchmark 
            command = f'python3 acrobot.py {formatted_params}'
            print(command)
            cmd_time = benchmark(f"{command}")

