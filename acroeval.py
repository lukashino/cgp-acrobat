import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

import glob

# import matplotlib.pyplot as plt


# Get a list of all CSV files
# i want to use and combine test-param/test-values
# files = glob.glob('*replay_id*.csv')
# files = glob.glob('individuals_p8_gen_cols64_gen_rows1_gen_lbacksNone_gen_pmtvs5_ea_offspgs1_ea_trnmtsz1_ea_mutrt0.08_ea_proc_cnt8_tfit0.95_exp_id*_replay_id*')
files = glob.glob('individuals_p{1,2,4,8}_gen_cols64_gen_rows1_gen_lbacksNone_gen_pmtvs5_ea_offspgs4_ea_trnmtsz1_ea_mutrt0.08_ea_proc_cnt8_tfit0.95_exp_id*_replay_id*.csv')
# individuals_p{1,2,4,8}_gen_cols64_gen_rows1_gen_lbacksNone_gen_pmtvs5_ea_offspgs4_ea_trnmtsz1_ea_mutrt0.08_ea_proc_cnt8_tfit0.95_exp_id*_replay_id*.csv

frames = []

# # Create a list to hold dataframes
# df_list = []

# for file in files:
#     # Read each CSV file
#     df = pd.read_csv(file)
#     # df_list.append(df)
# # Concatenate all dataframes in the list
# # df_all = pd.concat(df_list, ignore_index=True)
# # Create boxplot
# # plt.figure(figsize=(10, 6))
# # sns.boxplot(x='Replay ID', y='Time To Completion [seconds]', data=df_all)
# # plt.title('Boxplot of Time To Completion [seconds] for Each Replay ID')
# # plt.show()

#     # Calculate total time and max fitness for each experiment
#     total_time = df['Time To Completion [seconds]'].sum()
#     max_fitness = df['Fitness'].max()
#     num_generations = df['Individual ID'].nunique()

#     # Get the experiment ID from the data (assuming all rows have the same experiment ID)
#     experiment_id = df['Experiment ID'].iloc[0]

#     # Get the replay ID, same assumption
#     replay_id = df['Replay ID'].iloc[0]

#     # Create a new DataFrame with the summary information
#     summary_df = pd.DataFrame({
#         'Experiment ID': [experiment_id],
#         'Replay ID': [replay_id],
#         'Experiment\'s Time To Completion [seconds]': [total_time],
#         'Experiment Max Fitness': [max_fitness],
#         'Number of Generations': [num_generations]
#     })

#     frames.append(summary_df)
#     print("ok")

# # Concatenate all dataframes
# result = pd.concat(frames, ignore_index=True)



# # # plt.figure(figsize=(10, 6))
# # # sns.boxplot(x='Experiment ID', y='Time To Completion [seconds]', hue='Replay ID', data=result)
# # # plt.show()

# # # Write the result to a new CSV file
# result.to_csv('merged.csv', index=False)

for file in files:
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
        'Experiment\'s Time To Completion [seconds]': [total_time],
        'Experiment Max Fitness': [max_fitness],
        'Number of Generations': [num_generations]
    })

    frames.append(summary_df)
    print("ok")

result = pd.concat(frames, ignore_index=True)
result.to_csv('merged.csv', index=False)

# Create boxplot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Experiment ID', y='Experiment\'s Time To Completion [seconds]', data=result)
# sns.boxplot(x='Experiment ID', y='Experiment\'s Time To Completion [seconds]', data=result, showfliers=False)

plt.title('Boxplot of Experiment\'s Time To Completion [seconds] for Each Experiment ID')
plt.show()