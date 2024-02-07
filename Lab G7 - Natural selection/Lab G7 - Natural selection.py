import numpy as np
import pandas as pd
import random
import scipy.stats as st
import matplotlib.pyplot as plt
from scipy.sparse import dok_matrix


from itertools import permutations
from collections import defaultdict

np.random.seed(42)


#The Children include all information of an  individual
class Children:
    def __init__(self, ID, kid, r_lambda, position):
        self.parent_ID = ID                            #The The initial parent ID
        self.indi_ID = ID * 10 + kid
        self.bornTime = 0                              #The initial birth time
        self.generation = 0                            #The initial generation
        self.lifeTime = st.poisson.rvs(84)             #The average lifespan of Italian
        self.Nkids = 0                                 #How many kids will the parent have
        self.position = position                       # Newborns are in the same position as their parents
        self.Maxkids = st.poisson.rvs(r_lambda)        # The worldwide reproduction rate, I set how many descenvants will a parent will have 


# in a row x column region, this function can random choose an initial position for any individuals  
def get_position(row, column):
    x = np.random.randint(0, row) 
    y = np.random.randint(0, column)
    return (x, y)


'''
A parent can have a child, and the child can inherit some of the information from his parent. The child will get the initial 
position from the parent. He will also get his ID, lifespan, and theoretically how many children he will have in the future.    
'''
def Born_child(t, parent_children, r_lambda, r_alpha, prob_improve):
                             
    generation = parent_children.generation                # The generation of his parent
    P_ID = parent_children.indi_ID                         # Use the individual_ID of the parent as the parent_ID for the new children
    n = parent_children.Nkids                              # How many childrens the parent have had already
    

    last_ft = parent_children.lifeTime                    # the life time of parent                                      
                                              
    born_time = t                                         # the birth time of the children
    
    
    alpha = 0.1                                           # alpha is the improvement factor
    max_kids =  parent_children.Maxkids                   # Theoritically, the maximum number of children that parents can have is five.
    

    # If the amount of descendants of a parent reaches max_kids, the parent must stop producing descendants
    if n <= max_kids:                                     
        parent_children.Nkids = n + 1                     # Number of children of parents to date
        generation += 1
                                           
        ID = P_ID                                         # Record the parent's ID in the child
        kid =  n + 1                                  
        position = parent_children.position               # Inherit parent's position 

        next_children = Children(ID, kid, r_lambda, position)        # generate a new individual
        next_children.generation = generation             #Update the generation of new individual
        next_children.position = parent_children.position #The new individual inherits the position of the parent

        
        if np.random.random() < prob_improve:             # According to the improve probabality, generate a new lifesapn for new child
            ft = np.random.uniform(last_ft, int(last_ft*(1+alpha)))           
        else:
            ft = np.random.uniform(0,last_ft)           

        next_children.bornTime = born_time
        next_children.lifeTime = ft
 
    else:     
                                                        # If the number of childrens of parent exceed the maximum limit
                                                        # the parent can not have a new child                           
        next_children = parent_children

    return next_children      
 

def individuals_times(t, individuals_dict,r_lambda, r_alpha, prob_improve):
    # At t == 0, all individuals are parents, it need to creat initial parents. The initial_population is a stochastic inputs for any species
    if t == 0:                                               # t = 0, which means that all species are first generation
        initial_population = 10                                   # I set the initial population size for each species at 10

        for p in range(1, initial_population+1):                  # A child class will be defined for each individual
                                                    
            position = get_position(boundary,boundary)
            ID = 0                                                # The ID is the Id for parents
            kid = p                                               # The kid is the aequence of  for every individual
            r_lambda = r_lambda
            k = Children(ID, kid, r_lambda, position)

            individuals_dict[k.indi_ID] = k                       # For a specific species, the individuals_dict can collect all individuals

    else:
        # t != 0, which means that all species are not first generation                                 
        reproduction_size = np.random.randint(1, len(final_species_dict.keys()))  # I set the ''reproduction size'' to determine how many individuals      
                        
        random_items = random.sample(list(individuals_dict.items()), reproduction_size)    # At the specific time, the number of reproduction individuals  
        
        # print(random_items)                            
        #For each species, I have a lael for them, which 'A','B','C','D','E'...
        
        for sid, parent_children in random_items:

            # Each newly born child will inherit his species identity, with a birth time of T
            next_children = Born_child(t, parent_children, r_lambda, r_alpha, prob_improve)
            individuals_dict[sid[0] + str(next_children.indi_ID)] = next_children


    return individuals_dict  # new_individual_dict will include all newborns


# I can get an individuals dictionary for any specific species by the function: individuals_times
# Now I need a species_times
def species_times(species, t, species_dict, final_species_dict,r_lambda, r_alpha, prob_improve):
    if t == 0:                     # t = 0, which means that all species are first generation
        for s in species:          # Each species produces new individuals _dic and then they combine to form a Species_dic
            individuals_dict = {}
            individuals_dict = individuals_times(t, individuals_dict,r_lambda, r_alpha, prob_improve)
            individuals_dict = {s + str(key): value for key, value in individuals_dict.items()}  #Integration of individual 'species' and IDs 
            species_dict[s] = individuals_dict     


        for species in  species_dict.items():
            # print(species[1].items())
            for key,children in species[1].items():
                final_species_dict[key] = children


                                    # If t > 0,  which means that all species are not first generation.
    else:
        #Generate a new  final_species_dict based on original final_species_dict
        final_species_dict = individuals_times(t, final_species_dict, r_lambda, r_alpha, prob_improve)
                     
        # It have to remove the items that essced their lifespan (deleted death)
        # Deleting the item with the specified value
        
        #This formula checks how the current time compares to an individual's lifespan. 
        #Extract the key of dead indicvidual then remove them from final_species_dict
        key_to_delete = next((sid for sid, individual in final_species_dict.items() if individual.lifeTime <= t), None)

        if key_to_delete is not None:
            del final_species_dict[key_to_delete]

        else:
            print(f"The individual '{key_to_delete}' was dead.")

    
    return final_species_dict


'''
Design mobility model
The goal is create a dictinory {(x,y):(a1,b1,a2,a3,c1),1}
In the mobility model, all individuals can move in 4 directions (up, down, left, right) or stay in their current position.

up: y+1
down: y-1
left: x-1
right: x+1

maxminimum x = boundary
maxminimum y = boundary
'''

# move_position cuold update all information of final_species_dict, position_dic

def move_position(final_species_dict, position_dic):

    move_individuals = list(final_species_dict.items()) #All personnel will need to be relocated or retained in their current positions

    for individual in move_individuals:
        movement = (random.choice([-1, 0, 1]), random.choice([-1, 0, 1]))
        individual[1].position = tuple(map(lambda x, y: max(0, min(x + y, boundary)), individual[1].position, movement))
        individual[1].pposition = tuple(map(lambda x: max(0, x), individual[1].position))                               

    #Create a new position dictionary to record current location information of all individuals
    #It is necessary to create a new dictionary at each iteration 
    for sid, individual in final_species_dict.items():

        if individual.position not in position_dic:
            position_dic[individual.position] = [sid]
        else:
            position_dic[individual.position].append(sid)

    return final_species_dict, position_dic


'''
Design Fight Model
------when the number of species exceed 1, they may fight and may not survive
'''
def fighting_model(final_position_dic,final_species_dict):
    loser_list = []              # I set up a loser_list to keep track of all the losers in each duel.

    ### the goal of fighting_model is to update the final_position_dic and final_species_dict

    ## update the final_position_dic
    for position, sid_list in final_position_dic.items():

        # Extract alphabetic characters
        exist_species = [item[0] for item in sid_list]

        # Get unique types of species
        type_species = set(exist_species)

        # Count the number of unique types of letters
        num_species = len(type_species)

        # when the number of species exceed 1, they may fight and may not survive.
        if num_species > 1:
            loser_num = np.random.randint(0, len(sid_list))
            losers = np.random.choice(sid_list, size=loser_num, replace=False).tolist()

            # Remove losers from the sid_list
            new_sid_list = [species for species in sid_list if species not in losers]
            loser_list.append(losers)  #record loser SID

        else:
            new_sid_list = sid_list.copy()

        #Update all final_position_dic
        final_position_dic[position] = new_sid_list
    
    
    ### update the final_species_dict
    # len(loser_list) > 0 can avoid the empty list
    if len(loser_list) > 0:
        loser_list= list(np.concatenate(loser_list))
        # print(loser_list)

    for loser in loser_list:
    # Deleting the item with the specified key
        if loser in final_species_dict:
            del final_species_dict[loser]
            # final_species_dict[loser].lifeTime = t-final_species_dict[loser].bornTime


    return final_position_dic, final_species_dict
    

# How to  plot the location of all individuals at time t
def location_plot(final_posiotion_dic):

    # Rearrange the dictionary
    rearranged_data = {'Position': list(final_posiotion_dic.keys()), 'Species_Individuals_ID': list(final_posiotion_dic.values())}

    # Convert the rearranged dictionary to a DataFrame
    df = pd.DataFrame(rearranged_data)
    
    # Using apply and explode
    new_df = df.apply(lambda x: pd.Series({'Tuple': x['Position'], 'Individual': x['Species_Individuals_ID']}), axis=1)
    new_df = new_df.explode('Individual').reset_index(drop=True)

    # Keep only the first letter of each individual using str.get
    new_df['Individual'] = new_df['Individual'].str.get(0)

    # Define a color map based on individuals
    individual_colors = {individual: f'C{i}' for i, individual in enumerate(new_df['Individual'].unique())}

    # Set Matplotlib style
    plt.style.use('seaborn-whitegrid')

    # Plot using Matplotlib scatter plot with jitter
    plt.figure(figsize=(10, 8))
    jitter = 0.1  # Adjust the jitter amount

    # Create a set to store unique labels
    unique_labels = set()

    for _, row in new_df.iterrows():
        individual = row['Individual']     
        color = individual_colors[individual]
        x_jitter = row['Tuple'][0] + np.random.uniform(-jitter, jitter)
        y_jitter = row['Tuple'][1] + np.random.uniform(-jitter, jitter)
        plt.scatter(x_jitter, y_jitter, label=None, color=color, s=100)

        # Annotate each point with corresponding species individuals
        # plt.annotate(individual, (x_jitter, y_jitter), fontsize=8, ha='right', va='bottom', color=color)

        # Add unique labels to the legend
        if individual not in unique_labels:
            plt.scatter([], [], label=individual, color=color, s=100)
            unique_labels.add(individual)

    # Add a unique label outside the grid
    plt.text(plt.xlim()[1] + 2, plt.ylim()[1] - 2, ' ', fontsize=12, fontweight='bold', color='black')  #Lable Title is '' empty 

    plt.title('The Position for Each Individual (Matplotlib with Jitter)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.legend(title='Individual', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.show()




#*****************************************************************************************************#
'''
                                                 Natural Selection
'''
#*****************************************************************************************************#
#Step_1  Setting the initial conditions
species = ['A','B','C','D','E']                    # All species 
species_dict = {}                                  # Dictionary of all individuals of a given species
final_species_dict = {}                            # Dictionaries record all individuals of all species
final_position_dic = {}

plot_before_position_dic = {}
plot_after_position_dic = {}

time_population_acount = {}                      # Initialize a dictionary to store populations for each key type
time_lifespan_counts= {}  


t = 0          
r_lambda = 0.1                                       # The average reproductin rate
r_alpha = 0.3                                     # Average lifespan improvement rate
prob_improve =0.5                                 # Average lifespan improvement probability

simulation_year = 200                             #THe siimulation year
boundary = 20                                     #The boundary of mobile regions


def measure(species, t, species_dict, final_species_dict, r_lambda, r_alpha, prob_improve, simulation_year):
    while t <= simulation_year:

        final_species_dict = species_times(species, t, species_dict, final_species_dict, r_lambda, r_alpha, prob_improve)

# '------------------------------------#Step_2 update the position of all individuals------------------------------------------'

        posiotion_dic = {}   # After making a new move, it is necessary to set an empty dictionary to record the current position

        final_species_dict,  final_posiotion_dic = move_position(final_species_dict, posiotion_dic)


#-----------------------------------------------------------------#
        '''
        Attention this please :

        The following two commands will assist me in obtaining the post-fight positions of 
        all individuals at various times. There are five pictures will be generated. You can remove the "#" if you wish to see them.
        
        '''
        if t % 50 == 0:
            plot_before_position_dic[t] = posiotion_dic
            # print(f'At the {t}_th year, before the fight')
            # location_plot(final_posiotion_dic)
#-----------------------------------------------------------------#


#'------------------------------------Step_3 update the sid of all individuals------------------------------------------'

        final_posiotion_dic, final_species_dict= fighting_model(final_posiotion_dic, final_species_dict)
        


#-------------------------------Measure the population size for each species -----------------------------------------#
    # Initialize a species_counts dictionary to store counts for each key species at each time
        species_counts = {}                          # Set a {} record the population at times


        count_species_type = 0
        count_species_lifespan = 0


                                                     # Count the occurrences of each species
        for species in final_species_dict.keys():
            species_type = species[0]
            if species_type in species_counts:
                species_counts[species_type] += 1
                # lifespan_counts[species_type] += final_species_dict[species].lifeTime


            else:
                species_counts[species_type] = 1
                # lifespan_counts[species_type] = final_species_dict[species].lifeTime
    
          #{species:amount_lifespan}


                                                      # Store populations in the dictionary for a whole simulation time
        for species, population in species_counts.items():
            species_type = species          
            if species_type in time_population_acount:
                time_population_acount[species_type].append(population) # If the species is present, add the population size to the list
            else:
                time_population_acount[species_type] = [population]   
        
        # print(time_population_acount)

            


#-------------------------------Measure the average lifespan for each species -----------------------------------------#
    # Initialize a species_counts dictionary to store counts for each key species at each time
        lifespan_counts = {}                          # Set a {} record the population at times
        ave_lifespan = {}

        #I want to get the lifespan_counts {species:amount_lifespan}
        for species in final_species_dict.keys():
            species_type = species[0]
            if species_type in lifespan_counts:
                lifespan_counts[species_type] += final_species_dict[species].lifeTime
            else:
                lifespan_counts[species_type] = final_species_dict[species].lifeTime
        
        # print(lifespan_counts)

        #we want to get the ave_lifespan {species:avg_lifespan}
        for species_type in lifespan_counts.keys():
            avg = lifespan_counts[species_type] / species_counts[species_type]

            if species_type in time_lifespan_counts:
                time_lifespan_counts[species_type].append(avg)
            else:
                time_lifespan_counts[species_type] = [avg]
            
        # print(time_lifespan_counts)


#-----------------------------------------------------------------#
        '''
        Attention this please :

        The following two commands will assist me in obtaining the post-fight positions of 
        all individuals at various times. There are five pictures will be generated. You can remove the "#" if you wish to see them.
        
        '''
        #After fightinig, plot the position
        if t % 50 == 0:
            plot_after_position_dic[t] = final_posiotion_dic
            # print(f'At the {t}_th year, after the fight')
            # location_plot(final_posiotion_dic)
#-----------------------------------------------------------------#


        # print("Initial values - t:", t)
        t += 10
    print("Initial values - t:", t, "simulation_year:", simulation_year)

    return final_species_dict,  final_posiotion_dic, time_population_acount

#***************************************************************************************************************************************************************************#
                                #The Final Results#
#***************************************************************************************************************************************************************************#

final_species_dict,  final_posiotion_dic, time_population_acount = measure(species, t, species_dict, final_species_dict, r_lambda, r_alpha, prob_improve, simulation_year)



#-------------------Time Vs Population Size of all Species--------------------------------#
#-----------------------------------------------------------------------------------------#

# Generate x values from the given range
plt.figure(figsize=(12, 10))

x = list(range(10, simulation_year+20, 10))

# Plot the data for each key
for species, population_acount in time_population_acount.items():
    plt.plot(x, population_acount , label=species)

# Set labels and title
plt.xlabel('The simulation time(years)')
plt.ylabel('Population_size')
plt.title('The trend of Population Size for all species at different year')
plt.legend()
plt.grid(True)
plt.show()

#-------------------Time Vs Average Lifespan of all Species--------------------------------#
#-----------------------------------------------------------------------------------------#
# Plot the data for each key
plt.figure(figsize=(12, 10))
for species, lifespan in time_lifespan_counts.items():
    plt.plot(x, lifespan, label=species)

# Set labels and title
plt.xlabel('The simulation time(years)')
plt.ylabel('Average Lifespan')
plt.title('The trend of Average Lifespan for all species at different year')
plt.legend()
plt.grid(True)
plt.show()