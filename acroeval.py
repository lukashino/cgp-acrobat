import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from typing import List

import glob

# import matplotlib.pyplot as plt


# Get a list of all CSV files
# i want to use and combine test-param/test-values
# files = glob.glob('*replay_id*.csv')
# files = glob.glob('individuals_p8_gen_cols64_gen_rows1_gen_lbacksNone_gen_pmtvs5_ea_offspgs1_ea_trnmtsz1_ea_mutrt0.08_ea_proc_cnt8_tfit0.95_exp_id*_replay_id*')

props_human = {
    '_p': 'Parents',
    '_gen_cols': 'CGP columns',
    '_gen_rows': 'CGP rows',
    '_gen_lbacks': 'CGP L-Backs',
    '_gen_pmtvs': 'CGP Primitives (operations)',
    '_ea_offspgs': 'Offsprings',
    '_ea_trnmtsz': 'Tournament size',
    '_ea_mutrt': 'Mutation rate',
    '_ea_proc_cnt': 'Parallel processes',
    '_tfit': 'Termination fitness',
}

# PUT - property under test
def get_PUT_name(filenames: List[str]) -> str:
    if len(filenames) < 1:
        raise Exception("Filename list is empty")
    
    # Store previous property values
    prev_values = {}
    # The property that changed
    changed_prop = None

    # Iterate over file names
    for file_name in filenames:
        # Iterate over properties
        for prop, _ in props_human.items():
            # Find the property in the file name
            start_index = file_name.find(prop)
            if start_index != -1:
                end_index = file_name.find("_", start_index + len(prop) + 1)
                end_index = end_index if end_index != -1 else len(file_name)

                # Extract and parse the property value
                prop_value = file_name[start_index + len(prop):end_index]
                print(prop_value)

                # Compare with the previous value
                if prop in prev_values and prev_values[prop] != prop_value:
                    changed_prop = prop
                    break

                # Store the current value
                prev_values[prop] = prop_value

        # Stop if a change was detected
        if changed_prop:
            break

    if changed_prop:
        return changed_prop
    else:
        raise ValueError('No property under test detected from file names')

# PUT - property under values
def get_PUT_values(filenames: List[str], queried_property: str) -> List[str]:
    if len(filenames) < 1:
        raise Exception("Filename list is empty")

    unique_values = set()

    # Iterate over file names
    for file_name in filenames:
        # Find the property in the file name
        start_index = file_name.find(queried_property)
        if start_index != -1:
            end_index = file_name.find("_", start_index + len(queried_property) + 1)
            end_index = end_index if end_index != -1 else len(file_name)

            # Extract and parse the property value
            prop_value = file_name[start_index + len(queried_property):end_index]

            # Add the current value to the set
            unique_values.add(float(prop_value))
            
    if not unique_values:
        raise ValueError(f"No values found for the property '{queried_property}' in the provided filenames.")
    
    # Convert set to list and return
    return sorted(list(unique_values))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Benchmark an external program.')
    parser.add_argument('--result-path', type=str, required=True, help='Path to the results for evaluation e.g. "results/gen_cols/"')
    args = parser.parse_args()

    files = glob.glob(f'{args.result_path}/individuals_*')
    test_prop_name = get_PUT_name(files)
    test_prop_vals = get_PUT_values(files, test_prop_name)
    frames = []

    for file in files:
        print(file)
        # Read each CSV file
        df = pd.read_csv(file)
        # Calculate total time and max fitness for each experiment
        total_time = df['Time To Completion [seconds]'].sum()
        max_fitness = df['Fitness'].max()
        num_generations = df['Individual ID'].nunique()

        # Get the experiment ID from the data (assuming all rows have the same experiment ID)
        experiment_id = df['Experiment ID'].iloc[0]

        # Get the replay ID, same assumption
        replay_id = df['Replay ID'].iloc[0]

        # Create a new DataFrame with the summary information
        summary_df = pd.DataFrame({
            'Experiment ID': [experiment_id],
            'Replay ID': [replay_id],
            'Test Property Value': [test_prop_vals[experiment_id]],
            'Experiment\'s Time To Completion [seconds]': [total_time],
            'Experiment Max Fitness': [max_fitness],
            'Number of Generations': [num_generations]
        })

        frames.append(summary_df)

    result = pd.concat(frames, ignore_index=True)
    result.to_csv(f'{args.result_path}/merged{test_prop_name}.csv', index=False)

    # Create boxplot
    plt.figure(figsize=(10, 6))
    # sns.boxplot(x='Experiment ID', y='Experiment\'s Time To Completion [seconds]', data=result)
    sns.boxplot(x='Test Property Value', y='Experiment\'s Time To Completion [seconds]', data=result).set(xlabel=f'{props_human[test_prop_name]}')
    # sns.boxplot(x='Experiment ID', y='Experiment\'s Time To Completion [seconds]', data=result, showfliers=False)

    plt.title('Boxplot of Experiment\'s Time To Completion [seconds] for Each Experiment ID')
    plt.show()