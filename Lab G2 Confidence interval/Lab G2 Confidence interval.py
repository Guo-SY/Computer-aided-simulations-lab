############################
#  lab_2
###########################
'''
Let X be the output of stochastic process (.e.g, the estimated average in an experiment).

Let us assume that X is uniformly distributed between 0 and 10.

We wish to study the effect on the accuracy of the estimation in function of the number of experiments and in function of the confidence level.

1. Define properly all the input parameters
2. Write all the adopted formulas
3. Explain which python function you use to compute average, standard deviation and confidence interval
4. Plot the confidence interval and the accuracy 

5. Discuss the main conclusions drawn from the graphs.
'''


import math
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt


def confi_level(number):

    Confidence_lower_bound = 0.9
    
    Confidence_upper_bound = 1

    # Create a uniform distribution
    uniform_distribution = stats.uniform(loc=Confidence_lower_bound, scale=Confidence_upper_bound-Confidence_lower_bound)

    # Generate random variates
    levels = uniform_distribution.rvs(number)
    Confidence_levels = [round(i,5) for i in levels]
    Confidence_levels.sort()
    return Confidence_levels

def random_data(size):
    lower_bound = 0
    upper_bound = 100

    # Create a uniform distribution
    uniform_distribution = stats.uniform(loc=lower_bound, scale=upper_bound - lower_bound)
    data = [ int (x)  for  x in uniform_distribution.rvs(size)]

    return data

def accuracy(data,confidence_level):

    # data = [12.0, 11.0, 14.0, 15.0, 10.0, 16.0]
    sample_mean = np.mean(data)
    sample_size = len(data)
    # confidence_level = 0.9
    degrees_of_freedom = sample_size - 1

    if sample_size <= 30:
        squared_deviations = np.std(data, ddof=1) 
        confidence_interval = stats.t.interval(confidence_level, df=sample_size - 1, loc=sample_mean, scale=squared_deviations / np.sqrt(sample_size))

    else:
        squared_deviations = np.std(data)
        confidence_interval = stats.norm.interval(confidence_level, loc=sample_mean, scale=squared_deviations/np.sqrt(sample_size))
        
    lower_bound, upper_bound = confidence_interval

    relative_error = ((upper_bound - lower_bound )/2)/sample_mean
    accuracy       = (1-relative_error)

    # print("Confidence Interval: ({:.2f}, {:.2f})".format(lower_bound, upper_bound))
    # print('sample mean: ', sample_mean)
    # print('squared_deviations: ', squared_deviations)
    # print('sample relative_error: ', relative_error)
    # print('sample accuracy : ', accuracy)
    # print('\n ')

    # return accuracy
    return (accuracy,relative_error,lower_bound, upper_bound)




# (accuracy,relative_error,lower_bound, upper_bound)

accuracy_summary = []
Confidence_levels = confi_level(number = 5)

for confidence_level in Confidence_levels:  #Confidence_levels
    
    accuracy_list_diffNumber = []
    relative_error_list_diffNumber = []
    # confidence_interval_list_diffNumber = []
    lower_bound_list_diffNumber = []
    upper_bound_list_diffNumber = []


    for size in np.arange(10, 1000, 20):
            
        data_list  = random_data(size)   # the list of random number
        accuracy_value  = round(accuracy(data_list, confidence_level)[0],3)

        relative_error = (accuracy(data_list, confidence_level)[1])
        # confidence_interval = accuracy(data_list, confidence_level)[2]
        lower_bound = round(accuracy(data_list, confidence_level)[2],3)
        upper_bound = round(accuracy(data_list, confidence_level)[3],3)

        accuracy_list_diffNumber.append(accuracy_value)
        relative_error_list_diffNumber.append(relative_error)

        lower_bound_list_diffNumber.append(lower_bound)
        upper_bound_list_diffNumber.append(upper_bound)
        # confidence_interval_list_diffNumber.append(confidence_interval)

    
    accuracy_summary.append((accuracy_list_diffNumber,relative_error_list_diffNumber,lower_bound_list_diffNumber,upper_bound_list_diffNumber))





# Create some example data
for index,(level, result) in enumerate(zip(Confidence_levels,accuracy_summary)):
    # print(level)
    # print(len(result))

    plt.figure(figsize=(18, 12))
    x = [i for i in np.arange(10, 1000, 20)] 
    y1 = result[0]
    y2 = result[1]
    y3 = result[2]
    y4 = result[3]

    # Create a 2x2 grid of subplots and activate the first subplot
    plt.subplot(2, 2, 1)
    plt.plot(x, y1, linestyle='-', color='r')
    plt.xlabel('Experiment Number')
    plt.ylabel('Accuracy')
    plt.title(f'accuracy analysist(Confidence_level:{level})')
    plt.grid(True)
    plt.legend()

    # Activate the second subplot
    plt.subplot(2, 2, 4)
    plt.plot(y2, y3,label=f'lower_bound', linestyle='-', color='b')
    plt.plot(y2, y4, label=f'Upper_bound',linestyle='-', color='g')
    # plt.title(f'Confidence_level:{Confidence_levels[1]}')
    plt.xlabel('Relative Error')
    plt.ylabel('Confidence Interval')
    plt.title(f'accuracy analysist(Confidence_level:{level})')
    plt.grid(True)
    plt.legend()

    # Activate the third subplot
    plt.subplot(2, 2, 2)
    plt.plot(x, y3,label=f'lower_bound', linestyle='-', color='b')
    plt.plot(x, y4, label=f'Upper_bound',linestyle='-', color='g')

    plt.xlabel('Experiment Number')
    plt.ylabel('Confidence Interval')
    plt.title(f'accuracy analysist(Confidence_level:{level})')
    plt.grid(True)
    plt.legend()

    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    plt.grid(True)
    plt.legend()

    # Activate the third subplot
    plt.subplot(2, 2, 3)
    plt.plot(y2, y1,label=f'lower_bound', linestyle='-', color='b')


    plt.xlabel('Relative Error')
    plt.ylabel('Accuracy')
    plt.title(f'accuracy analysist(Confidence_level:{level})')
    plt.grid(True)
    plt.legend()

    # Adjust layout to prevent overlapping titles and labels
    plt.tight_layout()
    plt.grid(True)
    plt.legend()

    # Show the entire figure
    plt.show()


accuracy_diff_CL = []
data_list  = random_data(size = 1000)   # the list of random number
confidence_level_list = confi_level( number = 20)         # the list of confidence level


for level in confidence_level_list:
    a = accuracy(data_list, level)[0] 
    accuracy_diff_CL.append(a)
    # level_list.append(level)

# print(accuracy_diff_CL)

x = confidence_level_list
y = accuracy_diff_CL

plt.figure(figsize=(10, 6))
# plt.plot(accuracy_analysis) #（x和y 和线的label）
plt.plot(x, y, marker='o', linestyle='-', color='b')


plt.xlabel('confidence_level')
plt.ylabel('accuracy')
plt.title(f'accuracy analysist')

plt.legend()
plt.grid(True)
plt.show()
    