import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
np.random.seed(42)

#For the part, I design a function to help to compulate the ordinary differential equations (ODEs) over time 
def Epidemic_model(y, t, beta, gamma, delta, hospitalized_rate, it_rate, hospital_capacity, it_capacity, death_limit):
    S, I, R, H, D, IT = y

    leaving_H = (H / hospitalized_rate) * gamma   #The  individuals who leave hospital
    leaving_IT = (IT / it_rate) * gamma           #The  individuals who leave hospital Intensive Treatments
    
    dSdt = -beta * S * I / N                       #Susceptible Individuals
    dIdt = beta * S * I / N - (gamma + delta) * I  #Infected Individuals
    dRdt = gamma * I                               #Recovered/ Individuals

    dHdt = hospitalized_rate * I - leaving_H  #Individuals needs to be Hospitalized
    dDdt = delta * I                               #Deceased individuals
    dITdt = it_rate * I - leaving_IT               #Individuals needs to be Intensive Treatments 

    return [dSdt, dIdt, dRdt, dHdt, dDdt, dITdt]

def plot_Epidemic_results(t, S, I, R, H, D, IT):
    plt.figure(figsize=(12, 8))
    plt.plot(t, S, label='Susceptible (S)')
    plt.plot(t, I, label='Infectious (I)')
    plt.plot(t, R, label='Recovered (R)')
    # plt.plot(t, H, label='Hospitalized (H)')
    plt.plot(t, D, label='Deceased (D)')
    # plt.plot(t, IT, label='Intensive Treatments (IT)')
    plt.title('SIR Epidemic Trajectory')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.grid()
    plt.legend()
    plt.show()

def plot_h_it_results(t, H, D, IT):
    plt.figure(figsize=(12, 8))
    plt.plot(t, H, label='Hospitalized (H)')
    plt.plot(t, D, label='Deceased (D)')
    plt.plot(t, IT, label='Intensive Treatments (IT)')
    plt.axhline(y=100000, color='r', linestyle='--', label='Limit (y = 100000)')
    plt.title('The Individuals Under Non Pharmaceutical Intervention Strategy ')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.grid()
    plt.legend()
    plt.show()



#***********************************************************************************#

#----------------------------Initial Condition---------------------------------#
# Model parameters
N = 50000000   # Total population
I0 = 1         # Initial number of infected individuals
R0 = 0         # Initial number of recovered individuals
H0 = 0         # Initial number of hospitalized individuals
D0 = 0         # Initial number of deceased individuals
IT0 = 0        # Initial number of Intensive Treatments individuals

S0 = N - I0 - R0 - H0 - D0  # Initial number of susceptible individuals

hospitalized_rate = 0.1  # Fraction of infected individuals requiring hospitalization
it_rate = 0.06          # Fraction of infected individuals requiring Intensive Treatments
gamma = 1/14            # Recovery rate
delta = 0.03            # Fatality rate
lam = 0.1               # Effective contact rate
beta = 4/14             # Transmission rate

# Intervention capacity
hospital_capacity = 10000
it_capacity = 50000

# Death limit
death_limit = 1000000

# Time points
t = np.linspace(0, 365, 365)

# Solve the SIRH model equations
y = odeint(Epidemic_model, [S0, I0, R0, H0, D0, IT0], t, args=(beta, gamma, delta, hospitalized_rate, it_rate, hospital_capacity, it_capacity, death_limit))
S, I, R, H, D, IT = y.T

# Plot results
plot_Epidemic_results(t, S, I, R, H, D, IT)
plot_h_it_results(t, H, D, IT)


print('The maximum of Fatality individuals: ', max(D))
print('The cumulative Fatality individuals: ',D[-1])

#--------------------------Find out the suitable Beta-------------------------------------------------------#
#-----------------------------------------------------------------------------------------------------------#
max_iterations = 100  # make a simulation loop 
beta_list = []
beta_initial = 4/14

for iteration in range(max_iterations):
    # Simulate with the current beta
    
    y = odeint(Epidemic_model, [S0, I0, R0, H0, D0, IT0], t, args=(beta_initial, gamma, delta, hospitalized_rate, it_rate, hospital_capacity, it_capacity, death_limit))

    S, I, R, H, D, IT = y.T

    # Check if cumulative deaths meet the  fatality limitation
    death = D[-1]                 # D list record all fatality in a year

    if death <= death_limit:
        # print(f"Simulation successful with beta = {beta_initial:.4f}")
        beta_list.append(round(beta_initial,2))
        beta_list.sort()

    # Adjust beta based on the difference from the target
    beta_adjustment = death_limit / death
    beta_initial *= beta_adjustment

# print(beta_list)

#----------------------------- non pharmaceutical intervention strategy-------------------------------------------------#
#----------------------------------------------------------------------------------------------------------------------#
'''
In my non-pharmacological intervention strategy, recovery and mortality rates are fixed,
We can only change transmission rates and effective exposure rates. Transmission rates are related to
the number of reproductions. If I can limit people's mobility, I can quickly reduce the number of reproductions.
Eventually, I devised a function to assist me in updating the transmission rate. The physical meaning is
I can use some quarantine measures when infected people exceed the capacity of hospitals and intensive treatment.
'''

def update_beta(H, IT, D, gamma,hospital_capacity, IT_capacity,lam):
    limit_death = 10000
    beta = 4/14

    if (H <= hospital_capacity or IT <=  IT_capacity) and D <= limit_death:
        beta = beta

    else:
        beta = 0.081
    
    return beta


def Modify_Epidemic_model(y, t, gamma, delta, hospitalized_rate, it_rate, hospital_capacity, IT_capacity, death_limit):
    S, I, R, H, D, IT = y
    beta = update_beta(H, IT, D, gamma,hospital_capacity, IT_capacity,lam)

    rate_leaving_H = (H / hospitalized_rate) * gamma
    rate_leaving_IT = (IT / it_rate) * gamma
    
    dSdt = - beta * S * I / N
    dIdt = beta * S * I / N - (gamma + delta) * I
    dRdt = gamma * I 

    dHdt = hospitalized_rate * I - rate_leaving_H
    dDdt = delta * I
    dITdt = it_rate * I - rate_leaving_IT

    return [dSdt, dIdt, dRdt, dHdt, dDdt, dITdt]


# Solve the SIRH model equations
y = odeint(Modify_Epidemic_model, [S0, I0, R0, H0, D0, IT0], t, args=( gamma, delta, hospitalized_rate, it_rate, hospital_capacity, it_capacity, death_limit))
S, I, R, H, D, IT = y.T

plot_h_it_results(t, H, D, IT)

print('The maximun of hospitalized individuals: ', max([h for h in H]))
print('The account of hospitalized individuals who exceeding hospital capacity: ',len([h for h in H if h >10000]))
print('The maximun of Intensive Treatments  individuals: ', max(IT))
print('The account of hospitalized individuals who exceeding hospital capacity: ', len([it for it in IT if it >50000]))