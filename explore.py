import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
from math import sqrt

from acquire import get_telco_data

############################### Chi-squared test & Evaluation Funtions #######################################


def chi2_and_visualize(train, cat_var, target, a=0.05):
    observed = pd.crosstab(train[cat_var], train[target])
    chi2, p, degf, e = stats.chi2_contingency(observed)

    print('-------------------------')
    
    print(f'Chi2 Statistic: {chi2}\n')
    print(f'P-Value: {p}\n')
    # print(f'Degrees of Freedom: {degf}\n')
    # print(f'Expected: {e}\n')

    # Plotting the countplot
    title = f'{cat_var.capitalize()} Churned vs. Not Churned'
    plot_cp(train, target, cat_var, title)
    
    eval_p(p)

    print('-------------------------')

def plot_cp(data, x_col, hue_col, title):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=x_col, hue=hue_col)
    plt.title(title)
    plt.show()

def eval_p(p, a=0.05):
    if p < a:
        print(f'\nWe reject the null hypothesis with a p-value of {round(p, 2)}.')
    else:
        print(f'\nWe failed to reject the null hypothesis with a p-value of {round(p, 2)}.')

############################### Pearson's r correlation test & Evaluation Funtions #######################################
def pearson_r_test(data, x_col, y_col, a=0.05):
    # Calculate Pearson's correlation coefficient and p-value
    pearson_r, p_value = stats.pearsonr(data[x_col], data[y_col])

    print(f'Pearson\'s r: {pearson_r:.2f}\n')
    
    # Plot a scatter plot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=data, x=x_col, y=y_col, hue='churn')
    plt.title(f'Pearson\'s r Analysis: {x_col} vs. {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.show()
    
    if p_value < a:
        print(f'\nReject the null hypothesis. There is a linear correlation (p-value: {p_value:.2f})')
    else:
        print(f'\nWe fail to reject the null hypothesis that there is a linear correlation (p-value: {p_value:.2f})')
    

################### Spearman's rank correlation coefficient test and visualize ##########################################
def spearman_r_analysis(data, x_col, y_col, alpha=0.05):
    # Calculate Spearman's rank correlation coefficient and p-value
    spearman_r, p_value = stats.spearmanr(data[x_col], data[y_col])

    print(f'Spearman\'s r: {spearman_r:.2f}\n')
    print(f'This suggests a moderate positive correlation between variables. As monthly charges increase, tenure tends to increase.')
    print(f'This does not mean there is a strong relationship. They are moderately associated.\n')
    
    # Plot a scatter plot
    plt.figure(figsize=(10, 6))
    ax = sns.scatterplot(data=data, x=x_col, y=y_col, hue='churn')
    
    # Add mean lines
    ax.axhline(data[y_col].mean(), color='red', linestyle='dashed', label='Mean ' + y_col)
    ax.axvline(data[x_col].mean(), color='blue', linestyle='dashed', label='Mean ' + x_col)
    
    plt.title(f'Spearman\'s r Analysis: {x_col} vs. {y_col}')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.legend()
    plt.show()
    
    if p_value < alpha:
        print(f'\nWe Reject the null hypothesis. There is a linear correlation (p-value: {p_value:.2f})')
    else:
        print(f'\nWe fail to reject the null hypothesis that there is a linear correlation (p-value: {p_value:.2f})')



################### parametric tests and visualize ##########################################
def one_sample_t_test(data, pop_mean, alpha=0.05):
    t_stat, p_value = stats.ttest_1samp(data, pop_mean)
    result = eval_result(alpha, p_value)
    
    print(f'1-Sample t-test:')
    print(f'T-Statistic: {t_stat:.4f}\n')
    print(result)

    # Plot a histogram
    plt.figure(figsize=(8, 6))
    sns.histplot(data, kde=True)
    plt.title(f'1-Sample t-test Analysis\nData: {data.name}')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    plt.show()

def two_sample_t_test(data1, data2, alpha=0.05):
    t_stat, p_value = stats.ttest_ind(data1, data2)
    result = eval_result(alpha, p_value)
    
    print(f'Independent 2-Sample t-test:')
    print(f'T-Statistic: {t_stat:.4f}\n')
    print(result)

    # Plot histograms
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(data1, kde=True)
    plt.title(f'Data 1 Distribution\nData: {data1.name}')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    
    plt.subplot(1, 2, 2)
    sns.histplot(data2, kde=True)
    plt.title(f'Data 2 Distribution\nData: {data2.name}')
    plt.xlabel('Values')
    plt.ylabel('Frequency')
    
    plt.tight_layout()
    plt.show()

################### baseline acc ##########################################

def baseline(target):
    """
    The function calculates and prints the accuracy of a baseline model that always predicts the most
    frequent class in the target variable.
    
    :param target: The "target" parameter is likely a Pandas Series or DataFrame column that contains
    the true labels or values that we are trying to predict or classify. The "baseline" function appears
    to calculate the accuracy of a simple baseline model that always predicts the most common value in
    the "target" column
    """
    print(f'Baseline: {round(((target==target.value_counts().idxmax()).mean())*100,2)}% Accuracy')



    
############################ Target var Dist Function ################################
def plot_tvd(data):
    data = get_telco_data()
    sns.set(style="whitegrid")
    plt.figure(figsize=(8, 6))
    sns.countplot(data=data, x='churn')
    plt.title('Distribution of Churn in Telco Churn Dataset')
    plt.xlabel('Churn')
    plt.ylabel('Count')
    plt.show()
