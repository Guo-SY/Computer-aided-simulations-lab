import heapq
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.stats import probplot, chisquare, binom
from scipy.stats import chi2_contingency




#---------------------------------------------Function------------------------------------------------#
def generate_poisson_edges(n, average_rate):

    gnp_graph = [[0] * n for _ in range(n)]   # Generate a empty list for nodes 

    for i in range(n):
        for j in range(i + 1, n):
            # Generate edge with Poisson-distributed probability
            poisson_probability = np.random.poisson(average_rate)
            edge_probability = np.random.uniform(0, 1)

            #When the edge_probability less than the poisson_probability, the value is converted to 1
            if edge_probability < poisson_probability:
                gnp_graph[i][j] = 1
                gnp_graph[j][i] = 1

    return gnp_graph

def draw_gnp_graph(adjacency_matrix):

    n = len(adjacency_matrix)

    #Generate a position dictionary for each node based on the number of nodes
    positions = {i: (np.random.random(), np.random.random()) for i in range(n)}
    

    #Plot the edges according to position dictionary
    fig, ax = plt.subplots(figsize=(20, 18))

    for i in range(n):
        for j in range(i + 1, n):
            if adjacency_matrix[i][j] == 1:
                ax.plot([positions[i][0], positions[j][0]], [positions[i][1], positions[j][1]], 'bo-')

            elif adjacency_matrix[i][j] == 0:
                ax.plot([positions[i][0], positions[j][0]], [positions[i][1], positions[j][1]], 'bo')

    for i, pos in positions.items():
        ax.text(pos[0], pos[1], str(i), fontsize=8, ha='center', va='center', color='w', fontweight='bold', bbox=dict(facecolor='blue', edgecolor='black', boxstyle='circle'))

    ax.axis('off')
    plt.show()


def compute_degree_distribution(adjacency_matrix):
    #For each row(node), obtain the degree of edges.
    n = len(adjacency_matrix)
    degrees = [sum(row) for row in adjacency_matrix]

    return degrees

def plot_degree_distribution(n_nodes, degrees, p):

    fig, ax = plt.subplots(figsize=(12, 12))
    
    plt.hist(degrees, bins=range(min(degrees), max(degrees) + 2), align='left', density=True, rwidth=0.8, alpha=0.7, label='Histogram')

    unique_degrees, counts = zip(*sorted((degree, degrees.count(degree) / len(degrees)) for degree in set(degrees)))
    plt.plot(unique_degrees, counts, marker='o', linestyle='-', color='b', label='Degree Distribution')

    x = np.arange(0, n_nodes + 1)
    pmf_values = stats.binom.pmf(x, n_nodes, p)

    plt.plot(x, pmf_values, alpha=0.7, label=f'Binomial Distribution (n={n_nodes}, p={p})')

    plt.xlabel('Degree')
    plt.ylabel('Probability')
    plt.title('Degree Distribution and Binomial Distribution')
    plt.legend()
    plt.show()


def plot_qq_plot(empirical_degrees, theoretical_distribution, distribution_params):

    quantiles, values = probplot(empirical_degrees, dist=theoretical_distribution, sparams=distribution_params, fit=False)

    plt.scatter(values, quantiles, color='b')
    plt.plot([np.min(values), np.max(values)], [np.min(values), np.max(values)], '--', color='r')
    plt.xlabel('Theoretical Quantiles')
    plt.ylabel('Sample Quantiles')
    plt.title('Q-Q Plot')
    plt.show()





#*****************************************************Event Loop***********************************************#
n_nodes = 200
average_rate = 0.1 
gnp_graph = generate_poisson_edges(n_nodes, average_rate)
draw_gnp_graph(gnp_graph)



#---------------------Chi-square Test----------------------------------#
degrees = compute_degree_distribution(gnp_graph)

probability_for_binomial = 0.1  # Replace this with your desired probability for the binomial distribution
plot_degree_distribution(n_nodes, degrees, probability_for_binomial)

data = degrees   

# Perform the chi-square test
chi2_stat, p_value, dof, expected = chi2_contingency(data)

# Print the results
print(f"Chi2 Stat: {chi2_stat}")
print(f"P-value: {p_value}")
print(f"Degrees of Freedom: {dof}")
print("Expected Frequencies:")
print(expected)
print("----------------------")
print("----------------------")
#---------------------Q-Q plot----------------------------------#
empirical_degrees = compute_degree_distribution(gnp_graph)

# Assuming a binomial distribution for comparison
# theoretical_distribution = stats.binom
theoretical_distribution = stats.norm 


distribution_params = (np.mean(empirical_degrees), np.std(empirical_degrees))
plot_qq_plot(empirical_degrees, theoretical_distribution, distribution_params)