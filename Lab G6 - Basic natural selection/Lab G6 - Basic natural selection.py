import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt

import numpy as np
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix
np.random.seed(42)

class Children:
    def __init__(self, ID, kid):
        self.parent_ID = ID
        self.kid_ID = ID * 10 + kid                 #it can represent the generation of a child
        self.bornTime = 0
        self.generation = 0
        self.lifeTime = np.random.randint(1, 100)




def Born_child(born_event, parent_children,r_lambda,r_alpha, prob_improve):
    
    mean = r_lambda                                 #the average reprodution rate 
    generation = parent_children.generation
    P_ID = parent_children.kid_ID                    # Use the kid_ID of the parent as the parent_ID for the new children
    last_ft = 0                                      #the life time of parent
    last_born = 0                                    #the birth time of parent
    
    n  = st.poisson.rvs(mu=mean, size=1)            # Each parent will have n childrens
    i = 0
    generation += 1
    alpha = 0.1                                      #alpha is the improvement factor

    while i < n:                                    # I have know how many kids a parent will have, then assign them to a specific KID
        ID = P_ID                                 # Use the parent's ID as the base for the child's ID
        kid = i + 1
        next_children = Children(ID, kid)
        last_ft = parent_children.lifeTime          #the life time of parent
        last_born  = parent_children.bornTime        #the birth time of parent
        
        if np.random.random() < prob_improve:
            ft = np.random.uniform(last_ft, int(last_ft*(1+alpha)))           
        else:
            ft = np.random.uniform(0,last_ft)           

        # Ensure that the interval between T and bt is at least 1. The birth interval must be greater than 1 (the time of pregnancy).
        bt = np.random.randint(last_born, last_born + max(1, ft - 1) + 1, 1)[0]

        
        next_children.bornTime = bt
        next_children.lifeTime = ft
        born_event.append(next_children)
        i += 1  # Increment i inside the loop

    born_event.sort(key=lambda x: x.bornTime)

    return born_event          #Gen_dic


def gen_child_atmostgeneration(Gen_dic, max_generation, r_lambda, r_alpha, prob_improve):

    g = 0
    
    while g < max_generation:

        # in here, there are lots of parents actually. So the born_event should put outside of Born_child function
        born_event = []
        for parent_children in Gen_dic[g]:
            Events = Born_child(born_event, parent_children, r_lambda, r_alpha,prob_improve)  # Update Gen_dic directly
        
        Gen_dic[g+1] = Events
        g += 1

    return Gen_dic

def all_generation_population(population,max_generation, r_lambda,r_alpha,prob_improve):
    all_generation_dict = {}
    first_generation = {}
    
    num_p = 0
    popu_1 = Children(0, 0)
    first_generation = {0: [popu_1]}

    for num_p in range(population):
        # print(num_p)
        
        maxg_Gen_dic = gen_child_atmostgeneration(first_generation,max_generation, r_lambda, r_alpha,prob_improve)

        all_generation_dict[num_p+1] = maxg_Gen_dic

    # Create an empty dok_matrix to record the mount of  each person each generation
    num_rows = population
    num_cols = max_generation
    allpopu_matrix = dok_matrix((num_rows, num_cols), dtype=float) 

    for i in range(num_rows):                 # i represent the i-th person
        for j in range(num_cols):             # j represent the j-th generation
            allpopu_matrix[i,j] = len(list(all_generation_dict.items())[i][1][j])

    return all_generation_dict, allpopu_matrix




#--------------------------------------------------------------------------#
#------------------------different r_alpha Vs final Popuation-----------------------------#
print('Differrent r_alpha Vs fianl Popuation')
population = 10
max_generation = 10
r_lambda = 1.7
r_alpha = [0.1, 0.2, 0.3, 0.4, 0.5]
prob_improve = 0.5
final_result = {}

generation_sums = np.zeros((len(r_alpha), max_generation), dtype=int)

for i, al in enumerate(r_alpha):
    _, all_generation_matrix = all_generation_population(population, max_generation, r_lambda, al, prob_improve)
    
    # Sum along the columns to get the population for each generation
    generation_sums[i, :] = np.sum(all_generation_matrix, axis=0, dtype=int)

# Extract x values form all_generation
x_values = np.arange(max_generation)

# Plotting all lines on the same panel
plt.figure(figsize=(12, 8))
for i, key in enumerate(r_alpha):
    plt.plot(x_values, generation_sums[i, :], marker='o', label=f'r_alpha = {key}')

plt.xlabel('Generations')
plt.ylabel('Final Population')
plt.title('Differrent r_alpha Vs fianl Popuation')
plt.grid()
plt.legend()
plt.show()

#------------------------Differrent r_lambda Vs fianl Popuation-----------------------------#
print('---Differrent r_lambda Vs fianl Popuation----')
population = 10 #[10,20,50,100,200,500]
max_generation = 10
r_lambda  = [0.7, 0.9, 1, 1.5, 1.7]
r_alpha = 0.1
prob_improve = 0.5


final_result = {}


generation_sums = np.zeros((len(r_lambda), max_generation), dtype=int)

for i, la in enumerate(r_lambda):
    _, all_generation_matrix = all_generation_population(population, max_generation, la, r_alpha, prob_improve)
    
    generation_sums[i, :] = np.sum(all_generation_matrix, axis=0, dtype=int)

x_values = np.arange(max_generation)

plt.figure(figsize=(12, 8))
for i, key in enumerate(r_lambda):
    plt.plot(x_values, generation_sums[i, :], marker='o', label=f'r_lambda = {key}')

plt.xlabel('Generations')
plt.ylabel('Final Population')
plt.title('Differrent r_lambda Vs fianl Popuation')
plt.grid()
plt.legend()
plt.show()

#------------------------Different prob_improve Vs Popuation size-----------------------------#
print('---Different prob_improve Vs Popuation size----')
population = 100 #[10,20,50,100,200,500]
max_generation = 10
r_lambda  = 1.7
r_alpha = 0.1
prob_improve = [0.1,0.2,0.3,0.4,0.5]


final_result = {}

generation_sums = np.zeros((len(prob_improve), max_generation), dtype=int)

for i, p in enumerate(prob_improve):
    _, all_generation_matrix = all_generation_population(population, max_generation, r_lambda, r_alpha, p)
    

    generation_sums[i, :] = np.sum(all_generation_matrix, axis=0, dtype=int)


x_values = np.arange(max_generation)


plt.figure(figsize=(12, 8))
for i, key in enumerate(prob_improve):
    plt.plot(x_values, generation_sums[i, :], marker='o', label=f'prob_improve = {key}')

plt.xlabel('Generations')
plt.ylabel('Final Population')
plt.title('Different prob_improve Vs Popuation size')
plt.grid()
plt.legend()
plt.show()

#------------------------Different population Vs final Popuation size-----------------------------#
print('---Different population Vs final Popuation size----')
population = [10,20,50,100,200,500]
max_generation = 10
r_lambda  = 1.7
r_alpha = 0.1
prob_improve = 0.1


final_result = {}


generation_sums = np.zeros((len(population), max_generation), dtype=int)

for i, size in enumerate(population):
    _, all_generation_matrix = all_generation_population(size, max_generation, r_lambda, r_alpha, prob_improve)
    

    generation_sums[i, :] = np.sum(all_generation_matrix, axis=0, dtype=int)


x_values = np.arange(max_generation)


plt.figure(figsize=(12, 8))
for i, key in enumerate(population):
    plt.plot(x_values, generation_sums[i, :], marker='o', label=f'population = {key}')

plt.xlabel('Generations')
plt.ylabel('Final Population')
plt.title('Different population size Vs Final Popuation size')
plt.grid()
plt.legend()
plt.show()