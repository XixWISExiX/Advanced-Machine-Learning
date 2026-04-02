import matplotlib.pyplot as plt
import scipy.stats as stats
from analysis_support import *

def figure_1_generation():
    '''
    Generates Figure 1 which contains the mean test set FVAF from Part 1 and Part 2.
    (No Regularization, and No Regularization + Early Stopping)
    '''
    # Load Experiment 1 and 2
    exp_1_df = load_results_df('results/part_1_bmi_aggregate.pkl')
    exp_2_df = load_results_df('results/exp_raw_bmi_aggregate.pkl')

    # Compute Experiment 1 and 2 Metrics
    exp1 = compute_means_1D(exp_1_df, 'Ntraining', ['predict_testing_eval_loss', 'predict_testing_eval_fvaf', 'predict_testing_eval_rmse'])
    exp2 = compute_means_1D(exp_2_df, 'Ntraining', ['predict_testing_eval_loss', 'predict_testing_eval_fvaf', 'predict_testing_eval_rmse'])

    # Get the mean fvaf scores of Experiment 1 and 2
    exp_1_mean_fvaf_arr = []
    for row in exp1['predict_testing_eval_fvaf_individual']:
        exp_1_mean_fvaf_arr.append(np.mean(row))

    exp_2_mean_fvaf_arr = []
    for row in exp2['predict_testing_eval_fvaf_individual']:
        exp_2_mean_fvaf_arr.append(np.mean(row))

    fig, ax = plt.subplots() 

    # Plot metrics
    ax.plot(exp1['key0_values'], exp_1_mean_fvaf_arr, label='Mean Test fvaf No Regularization', color='black', marker='o')
    ax.plot(exp2['key0_values'], exp_2_mean_fvaf_arr, label='Mean Test fvaf Early Stopping', color='red', marker='x')
    
    # Add Labels
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('fvaf')
    ax.set_title('fvaf vs Training Set Size')
    ax.legend()

    save_location = './results/fig1.png'
    fig.savefig(save_location)



def figure_2_generation():
    '''
    Generates Figure 2 which contains the mean validation set FVAF from Part 3.
    (Dropout)
    '''
    # Load Experiment 3
    exp_3_df = load_results_df('Part3_4_results/part_3_bmi_aggregate.pkl')

    # Init charting variables
    fig, ax = plt.subplots() 
    colors = ['black', 'blue', 'brown', 'red', 'green', 'purple']
    idx = 0

    # Load the combinations of 'NTraining' and 'dropout' into the tuples arr
    for drop_prob in [0.015625, 0.03125, 0.0675, 0.125, 0.25, 0.5]:
        tuples = []
        for training_size in [1,2,3,4,6,8,11,14,18]:
            tuples.append((training_size, drop_prob))
        # Compute Experiment 3 Metrics
        exp3 = compute_means_1D(exp_3_df, 'Ntraining', ['predict_validation_eval_fvaf'], key1='dropout', filter_tuples=tuples)

        # Get the mean fvaf scores of Experiment 3
        exp_3_mean_fvaf_arr = []
        for row in exp3['predict_validation_eval_fvaf_individual']:
            exp_3_mean_fvaf_arr.append(np.mean(row))

        # Plot metrics
        label = f"Mean Validation fvaf Dropout = {drop_prob}"
        ax.plot([1,2,3,4,6,8,11,14,18], exp_3_mean_fvaf_arr, label=label, color=colors[idx], marker='o')
        idx+=1
         
    # Add Labels
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('fvaf')
    ax.set_title('fvaf vs Training Set Size')
    ax.legend()

    save_location = './results/fig2.png'
    fig.savefig(save_location)
     


def figure_3_generation():
    '''
    Generates Figure 3 which contains the mean validation set FVAF from Part 4.
    (Dropout + Early Stopping)
    '''
    # Load Experiment 4
    exp_4_df = load_results_df('Part3_4_results/part_4_bmi_aggregate.pkl')

    # Init charting variables
    fig, ax = plt.subplots() 
    colors = ['black', 'blue', 'brown', 'red', 'green', 'purple']
    idx = 0

    # Load the combinations of 'NTraining' and 'dropout' into the tuples arr
    for drop_prob in [0.015625, 0.03125, 0.0675, 0.125, 0.25, 0.5]:
        tuples = []
        for training_size in [1,2,3,4,6,8,11,14,18]:
            tuples.append((training_size, drop_prob))
        # Compute Experiment 4 Metrics
        exp4 = compute_means_1D(exp_4_df, 'Ntraining', ['predict_validation_eval_fvaf'], key1='dropout', filter_tuples=tuples)

        # Get the mean fvaf scores of Experiment 4
        exp_4_mean_fvaf_arr = []
        for row in exp4['predict_validation_eval_fvaf_individual']:
            exp_4_mean_fvaf_arr.append(np.mean(row))

        # Plot metrics
        label = f"Mean Validation fvaf Early Stopping + Dropout = {drop_prob}"
        ax.plot([1,2,3,4,6,8,11,14,18], exp_4_mean_fvaf_arr, label=label, color=colors[idx], marker='o')
        idx+=1
         
    # Add Labels
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('fvaf')
    ax.set_title('fvaf vs Training Set Size')
    ax.legend()

    save_location = './results/fig3.png'
    fig.savefig(save_location)
     


def figure_4_generation():
    '''
    Generates Figure 4 which contains the mean validation set FVAF from Part 5.
    (L2 Regularization)
    '''
    # Load Experiment 5
    exp_5_df = load_results_df('part5_results/part_5_bmi_aggregate.pkl')

    # Init charting variables
    fig, ax = plt.subplots() 
    colors = ['black', 'blue', 'brown', 'red', 'green', 'purple']
    idx = 0

    # Load the combinations of 'NTraining' and 'L2' into the tuples arr
    for l2 in [0.1, 0.01, 0.001, 0.0001, 0.00001]:
        tuples = []
        for training_size in [1,2,3,4,6,8,11,14,18]:
            tuples.append((training_size, l2))
        # Compute Experiment 5 Metrics
        exp5 = compute_means_1D(exp_5_df, 'Ntraining', ['predict_validation_eval_fvaf'], key1='L2_regularization', filter_tuples=tuples)

        # Get the mean fvaf scores of Experiment 5
        exp_5_mean_fvaf_arr = []
        for row in exp5['predict_validation_eval_fvaf_individual']:
            exp_5_mean_fvaf_arr.append(np.mean(row))

        # Plot metrics
        label = f"Mean Validation fvaf l2 = {l2}"
        ax.plot([1,2,3,4,6,8,11,14,18], exp_5_mean_fvaf_arr, label=label, color=colors[idx], marker='o')
        idx+=1
         
    # Add Labels
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('fvaf')
    ax.set_title('fvaf vs Training Set Size')
    ax.legend()

    save_location = './results/fig4.png'
    fig.savefig(save_location)

     

def find_best_hyper_parameters(key0:str, key1:str, key0_arr:list[float], key1_arr:list[float], exp_df:pd.DataFrame)->list[float, float]:
    '''
    Find and store the best hyper parameters.

    :param key0: Key for the first category. 
    :param key1: Key for the second category. 
    :param key0_arr: List for the first category.
    :param key1_arr: List for the second category.
    :param exp_df: Dataframe that holds all experiments.
    :return: Returns both the best key0 value and the best key1 value in terms of FVAF.
    '''
    # Init Variables
    idx = 0
    max_val = -10000
    best_key0 = 0
    best_key1 = 0

    # Load the combinations of key0 and key1 into the tuples arr
    for a in key1_arr:
        tuples = []
        for b in key0_arr:
            tuples.append((b, a))
        # Compute Metrics
        exp = compute_means_1D(exp_df, key0, ['predict_validation_eval_fvaf'], key1=key1, filter_tuples=tuples)

        # Get the mean fvaf scores
        for i, row in enumerate(exp['predict_validation_eval_fvaf_individual']):
            # Find the highest fvaf score and store the parameters
            if max_val < np.mean(row):
                max_val = np.mean(row)
                best_key0 = key0_arr[i]
                best_key1 = key1_arr[idx]
        idx+=1
         
    return best_key0, best_key1
     


def print_p_value_and_mean_diff(arr1:list[float], arr2:list[float]):
    '''
    Print the p_value and mean difference of arrays 1 and 2 using a paired t-test.
    :param arr1: First array. 
    :param arr2: Second array. 
    '''
    t_stat, p_value = stats.ttest_rel(arr1, arr2)
    mean_diff = abs(np.mean(arr1) - np.mean(arr2))
    print(p_value, mean_diff)



def figure_5_generation():
    # Load Parts 1-5 Experiment DFs
    exp_1_df = load_results_df('results/part_1_bmi_aggregate.pkl')
    exp_2_df = load_results_df('results/exp_raw_bmi_aggregate.pkl')
    exp_3_df = load_results_df('Part3_4_results/part_3_bmi_aggregate.pkl')
    exp_4_df = load_results_df('Part3_4_results/part_4_bmi_aggregate.pkl')
    exp_5_df = load_results_df('part5_results/part_5_bmi_aggregate.pkl')

    # Compute Part 1 and 2 Metrics
    exp1 = compute_means_1D(exp_1_df, 'Ntraining', ['predict_testing_eval_loss', 'predict_testing_eval_fvaf', 'predict_testing_eval_rmse'])
    exp2 = compute_means_1D(exp_2_df, 'Ntraining', ['predict_testing_eval_loss', 'predict_testing_eval_fvaf', 'predict_testing_eval_rmse'])

    # Get the mean fvaf scores of Part 1 and 2
    exp_1_mean_fvaf_arr = []
    for row in exp1['predict_testing_eval_fvaf_individual']:
        exp_1_mean_fvaf_arr.append(np.mean(row))

    exp_2_mean_fvaf_arr = []
    for row in exp2['predict_testing_eval_fvaf_individual']:
        exp_2_mean_fvaf_arr.append(np.mean(row))

    # Find the best hyper-parameters using the validation set for Parts 3-5
    best_part_3_training_size, best_part_3_drop_prob = find_best_hyper_parameters(key0='Ntraining', key1='dropout', key0_arr=[1,2,3,4,6,8,11,14,18], key1_arr=[0.015625, 0.03125, 0.0675, 0.125, 0.25, 0.5], exp_df=exp_3_df)
    best_part_4_training_size, best_part_4_drop_prob = find_best_hyper_parameters(key0='Ntraining', key1='dropout', key0_arr=[1,2,3,4,6,8,11,14,18], key1_arr=[0.015625, 0.03125, 0.0675, 0.125, 0.25, 0.5], exp_df=exp_4_df)
    best_part_5_training_size, best_part_5_l2 = find_best_hyper_parameters(key0='Ntraining', key1='L2_regularization', key0_arr=[1,2,3,4,6,8,11,14,18], key1_arr=[0.1, 0.01, 0.001, 0.0001, 0.00001], exp_df=exp_5_df)


    # Make the various tuple combinations with the best hyper-parameters
    training_size_arr = [1,2,3,4,6,8,11,14,18]
    part_3_tuples = []
    part_4_tuples = []
    part_5_tuples = []
    for training_size in training_size_arr:
        part_3_tuples.append((training_size, best_part_3_drop_prob))
        part_4_tuples.append((training_size, best_part_4_drop_prob))
        part_5_tuples.append((training_size, best_part_5_l2))

    # Compute Part 3-5 Test Metrics
    exp3 = compute_means_1D(exp_3_df, 'Ntraining', ['predict_testing_eval_fvaf'], key1='dropout', filter_tuples=part_3_tuples)
    exp4 = compute_means_1D(exp_4_df, 'Ntraining', ['predict_testing_eval_fvaf'], key1='dropout', filter_tuples=part_4_tuples)
    exp5 = compute_means_1D(exp_5_df, 'Ntraining', ['predict_testing_eval_fvaf'], key1='L2_regularization', filter_tuples=part_5_tuples)

    # Get the mean fvaf scores of Part 3-5
    exp_3_mean_fvaf_arr = []
    for row in exp3['predict_testing_eval_fvaf_individual']:
        exp_3_mean_fvaf_arr.append(np.mean(row))

    exp_4_mean_fvaf_arr = []
    for row in exp4['predict_testing_eval_fvaf_individual']:
        exp_4_mean_fvaf_arr.append(np.mean(row))

    exp_5_mean_fvaf_arr = []
    for row in exp5['predict_testing_eval_fvaf_individual']:
        exp_5_mean_fvaf_arr.append(np.mean(row))


    # Print Best Metrics for Parts 3-5
    print("Part 3: Training Size =", best_part_3_training_size, "Drop Probability =", best_part_3_drop_prob)
    print("Part 4: Training Size =", best_part_4_training_size, "Drop Probability =", best_part_4_drop_prob)
    print("Part 5: Training Size =", best_part_5_training_size, "L2 =", best_part_5_l2)

    
    # Print p_value + mean differences 
    print("\nExperiment 2 comparison:")
    print_p_value_and_mean_diff(exp_1_mean_fvaf_arr, exp_2_mean_fvaf_arr)
    print("\nExperiment 3 comparison:")
    print_p_value_and_mean_diff(exp_1_mean_fvaf_arr, exp_3_mean_fvaf_arr)
    print("\nExperiment 4 comparison:")
    print_p_value_and_mean_diff(exp_1_mean_fvaf_arr, exp_4_mean_fvaf_arr)
    print("\nExperiment 5 comparison:")
    print_p_value_and_mean_diff(exp_1_mean_fvaf_arr, exp_5_mean_fvaf_arr)

    fig, ax = plt.subplots() 

    # Plot metrics
    ax.plot(exp1['key0_values'], exp_1_mean_fvaf_arr, label='Part 1: Mean Test fvaf No Regularization', color='black', marker='o')
    ax.plot(exp2['key0_values'], exp_2_mean_fvaf_arr, label='Part 2: Mean Test fvaf Early Stopping', color='blue', marker='o')
    ax.plot(training_size_arr, exp_3_mean_fvaf_arr, label='Part 3: Mean Test fvaf Best Dropout', color='brown', marker='o')
    ax.plot(training_size_arr, exp_4_mean_fvaf_arr, label='Part 4: Mean Test fvaf Best Dropout w/t Early Stopping', color='red', marker='o')
    ax.plot(training_size_arr, exp_5_mean_fvaf_arr, label='Part 5: Mean Test fvaf Best L2 Regularization', color='green', marker='o')
    
    # Add Labels
    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('fvaf')
    ax.set_title('fvaf vs Training Set Size')
    ax.legend()

    save_location = './results/fig5.png'
    fig.savefig(save_location)



if __name__ == "__main__":
    # Generate all figures (1-5)
    figure_1_generation()
    figure_2_generation()
    figure_3_generation()
    figure_4_generation()
    figure_5_generation()
