                                              
                                              
                                              # The entire simulation time is 25 seconds #


import numpy as np
from matplotlib import pyplot as plt

from scipy.stats import chisquare, binom, poisson 
from scipy.sparse import dok_matrix, lil_matrix
from collections import Counter
import random


import numpy as np
from scipy.sparse import lil_matrix
from collections import Counter

# pip install tqdm
# import tqdm
# import time


#--------------------------------------------------- G(n,p) Model----------------------------------------------#
class Gnp_MatrixUpdater:
    def __init__(self, n, lambda_parameter=1, probability_threshold=0.01):
        """
        Initialize MatrixUpdater.

        Parameters:
        - n: Size of the matrix (n x n).
        - lambda_parameter: Lambda parameter for the Poisson distribution.
        - probability_threshold: Probability threshold for matrix element assignment.
        """
        self.n = n
        self.lambda_parameter = lambda_parameter
        self.probability_threshold = probability_threshold
        self.p_1 = p_1
        self.matrix_dict = {(i, j): np.random.choice([-1, 1], p=[1-p_1, p_1]) for i in range(n) for j in range(n)}
        self.P_one = 0
        self.P_neg_one = 0
        self.iteration = 0
        self.orig_one_count = Counter(self.matrix_dict.values())[1]
        self.orig_neg_one_count = Counter(self.matrix_dict.values())[-1]

    def update_neighbors(self):
        """
        Update the matrix based on the Poisson distribution and probability threshold.

        Returns:
        - G_matrix: Updated matrix.
        """
        G_matrix = lil_matrix((self.n, self.n), dtype=int)
        for i in range(self.n):
            for j in range(self.n):
                num_ones = np.random.poisson(self.lambda_parameter)
                if np.random.rand() < self.probability_threshold:
                    G_matrix[i, j] = 1
        return G_matrix

    def G_neighborhoods(self, G_matrix):
        """
        Get the neighborhood state from the matrix.

        Parameters:
        - G_matrix: Input matrix.

        Returns:
        - G_neighbors_state: Neighborhood state.
        """
        return list(zip(*G_matrix.nonzero()))

    def wake_up(self, G_neighbors_state):
        """
        Update the matrix state based on the neighborhood state.

        Parameters:
        - G_neighbors_state: Neighborhood state.
        """
        count = sum(self.matrix_dict[v] for v in G_neighbors_state)
        new_value = 1 if count > 0 else -1

        for v in G_neighbors_state:
            self.matrix_dict[v] = new_value

    def main(self):
        """
        Run the simulation until the specified conditions are met.

        Returns:
        - matrix_dict: Final matrix state.
        - P_neg_one: Final probability of -1.
        - P_one: Final probability of 1.
        - iteration: Number of iterations performed.
        """
        print("*****************The Result of G(n,p) Model*******************************")
        print("Original One Count In G(n,p) Vector Model:", self.orig_one_count)
        print("Original Negative In G(n,p) Vector Model:", self.orig_neg_one_count)
        T = 0
        t = t = np.random.poisson(1)
        while self.P_one < 1 and self.P_neg_one < 1: 
            T += t
            all_values_count = Counter(self.matrix_dict.values())
            one_count = all_values_count[1]
            neg_one_count = all_values_count[-1]

            self.P_one = one_count / (one_count + neg_one_count)
            self.P_neg_one = neg_one_count / (one_count + neg_one_count)

            self.iteration += 1
            G_matrix = self.update_neighbors()
            G_neighbors_state = self.G_neighborhoods(G_matrix)
            self.wake_up(G_neighbors_state)

        print("Final Numberof Ones In G(n,p) Vector Model:", one_count)
        print("Final Number of Minus Ones In G(n,p) Vector Model:", neg_one_count )
        return self.matrix_dict, self.P_neg_one, self.P_one, T


#--------------------------------------------------- 2D Gride Model----------------------------------------------#
class G2d_VertexUpdater:
    def __init__(self, num_rows, num_cols, p_1):
        self.num_rows = num_rows
        self.num_cols = num_cols
        self.p_1 = p_1
        self.vertex_matrix = self.initialize_matrix()
        self.vertex_events = self.initialize_vertex_events()
        self.T = 0
        self.orig_one_count = np.count_nonzero(self.vertex_matrix.A==1)
        self.orig_neg_one_count = np.count_nonzero(self.vertex_matrix.A==-1)

    def initialize_matrix(self):
        vertex_matrix = dok_matrix((self.num_rows, self.num_cols), dtype=int)

        for i in range(self.num_rows):
            for j in range(self.num_cols):
                if np.random.rand() < self.p_1:
                    vertex_matrix[i, j] = 1
                else:
                    vertex_matrix[i, j] = -1

        return vertex_matrix

    def initialize_vertex_events(self):
        rows, cols = self.vertex_matrix.nonzero()
        return list(zip(rows, cols))

    def check_neighbors(self, rv):
        state = []
        directions = [(0, 1), (0, -1), (-1, 0), (1, 0)]

        for direction in directions:
            neighbor = (rv[0] + direction[0], rv[1] + direction[1])

            if 0 <= neighbor[0] < self.num_rows and 0 <= neighbor[1] < self.num_cols:
                state.append(self.vertex_matrix[neighbor])

        return state

    def wake_up(self, rv):
        # for each vertex, check all neighbors around the vertex
        neighbor_list = self.check_neighbors(rv)
        count_1 = sum(1 for v in neighbor_list if v == 1)

        # if the count of +1 is more than 2, set the vertex to +1
        if count_1 > 2:
            self.vertex_matrix[rv] = 1

        elif count_1 < 2:
            self.vertex_matrix[rv] = -1

        elif count_1 ==2:
            self.vertex_matrix[rv] = np.random.choice([-1, 1], p=[1 - self.p_1, self.p_1])

    def main(self):
        print("*****************The Result of 2D Model*******************************")
        print("Original One Count In 2D Vector Model:", self.orig_one_count)
        print("Original Negative One Count In 2D Vector Model:", self.orig_neg_one_count)

        P_lambda = 1
        percentage_ones = 0
        percentage_minus_ones = 0

        count_of_all = self.vertex_matrix.count_nonzero()

        while percentage_ones < 1 and percentage_minus_ones < 1:

            vertex_events = list(self.vertex_matrix.nonzero())
            random.shuffle(vertex_events)

            for rv in vertex_events:
                t = np.random.poisson(P_lambda)
                self.T += t
                self.wake_up(rv)

                count_of_ones = np.count_nonzero(self.vertex_matrix.A == 1)
                percentage_ones = count_of_ones / count_of_all

                count_of_neg_ones = np.count_nonzero(self.vertex_matrix.A == -1)
                percentage_minus_ones = count_of_neg_ones / count_of_all


        print("Final Numberof Ones In 2D Vector Model:", count_of_ones)
        print("Final Number of Minus Ones In 2D Vector Model:", count_of_neg_ones)

        return percentage_ones, percentage_minus_ones, self.T


#--------------------------------------------------- 3D Gride Model----------------------------------------------#

class G3d_VertexUpdater:
    def __init__(self, n, p_1):
        self.n = n
        self.p_1 = p_1
        self.random_3d_matrix = np.random.choice([-1, 1], size=(n, n, n), p=[1 - p_1, p_1])
        self.vertex_3d_dict = {(i, j, k): self.random_3d_matrix[i, j, k] for i in range(n) for j in range(n) for k in range(n)}
        self.vertex_3d_events = list(self.vertex_3d_dict.keys())
        self.orig_one_count = Counter(self.vertex_3d_dict.values())[1]
        self.orig_neg_one_count = Counter(self.vertex_3d_dict.values())[-1]
        self.T = 0

    def check_3d_neighbors(self, rv):
        x, y, z = rv
        neighbors_state = []

        directions = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]

        for dx, dy, dz in directions:
            n_x, n_y, n_z = x + dx, y + dy, z + dz

            if 0 <= n_x < self.n and 0 <= n_y < self.n and 0 <= n_z < self.n:
                neighbors_state.append((n_x, n_y, n_z))

        return neighbors_state

    def wake_up_3d(self, rv):
        state_event = self.check_3d_neighbors(rv)

        count_ones = sum(1 for v in state_event if self.random_3d_matrix[v] == -1)

        if count_ones > 3:
            self.random_3d_matrix[rv] = 1
        elif count_ones < 3:
            self.random_3d_matrix[rv] = -1
        elif count_ones == 3:
            self.random_3d_matrix[rv] = np.random.choice([-1, 1], p=[1 - self.p_1, self.p_1])

    def main(self):
        print("*****************The Result of 3D Model*******************************")
        print("Final P_1 of  G(n,p) Vector Model :", self.p_1)
        print("Original One Count In 3D Vector Model:", self.orig_one_count)
        print("Original Negative One Count In 3D Vector Model:", self.orig_neg_one_count)
        T = 0
        P_lambda = 1
        t = np.random.poisson(P_lambda)

        percentage_ones = 0
        percentage_minus_ones = 0
        iteration_3d = 0

        count_of_all = np.count_nonzero(self.random_3d_matrix)

        while percentage_ones < 1 and percentage_minus_ones < 1 and iteration_3d <= 3000:
            for rv in self.vertex_3d_events:
                iteration_3d += 1
                T += t
                self.wake_up_3d(rv)

                count_of_ones = np.count_nonzero(self.random_3d_matrix == 1)
                percentage_ones = count_of_ones / count_of_all

                count_of_neg_ones = np.count_nonzero(self.random_3d_matrix == -1)
                percentage_minus_ones = count_of_neg_ones / count_of_all

        print("Final Numberof Ones In 3D Vector Model:", count_of_ones)
        print("Final Number of Minus Ones  In 3D Vector Model:", count_of_neg_ones)

        return percentage_ones, percentage_minus_ones, T





#_______________________________________________________Simulation_______________________________________________________#
#-------------------------------------------------------------#
#---------------------G(n,p) simulation-----------------------#
np.random.seed(42)
# Generate a random size from the given choices
size = np.random.choice([10**3, 10**4])
# Find factors of size
factors = [(i, size // i) for i in range(1, int(size**0.5) + 1) if size % i == 0]
# Choose factors that make the matrix as square as possible
n_rows, n_columns = factors[len(factors) // 2]
# Calculate the average of n_rows and n_columns
n = (n_rows + n_columns) // 2

print('The number of n_rows and n_columns is:', (n_rows, n_columns))
print('The average of n_rows and n_columns is:', n)

probability_threshold = 1/size 
for p_1  in (0.51, 0.55, 0.6, 0.7):
    matrix_updater = Gnp_MatrixUpdater(n,probability_threshold)
    final_matrix_dict, final_P_neg_one, final_P_one, final_iteration = matrix_updater.main()

    # print("Final Matrix Dict:", final_matrix_dict)
    
    print("Final P_neg_one of  G(n,p) Vector Model :", final_P_neg_one)
    print("Final P_one of  G(n,p) Vector Model:", final_P_one)
    print("Final Iteration Time  of G(n,p) Vector Model:", final_iteration)
    print("****************************************************************************")
    print("****************************************************************************")







#---------------------------------------------------------#
#---------------------2d simulation-----------------------#

num_rows = 100
num_cols = 100
p_1 = 0.1


vertex_updater = G2d_VertexUpdater(num_rows, num_cols, p_1)
result_ones, result_neg_ones, result_T = vertex_updater.main()

print("Final Percentage of Ones In 2D Vector Model:", result_ones)
print("Final Percentage of neg Ones In 2D Vector Model:", result_neg_ones)
print("Final T In 2D Vector Model:", result_T)
print("****************************************************************************")
print("****************************************************************************")





#---------------------------------------------------------#
#---------------------3d simulation-----------------------#
n_value = 10
p_1_value = 0.51
vertex_updater_3d = G3d_VertexUpdater(n_value, p_1_value)
result_ones, result_neg_ones, result_T = vertex_updater_3d.main()

print("Final Percentage of Ones In 3D Vector Model:", result_ones)
print("Final Percentage of Minus Ones In 3D Vector Model:", result_neg_ones)
print("Total Time (T) In 3D Vector Model:", result_T)
print("****************************************************************************")
print("****************************************************************************")
#---------------------------------------------------------#