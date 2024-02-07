import random
import numpy as np
import pandas as pd
from queue import Queue,PriorityQueue
import matplotlib.pyplot as plt


import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import expon
from scipy.stats import poisson
from scipy.optimize import fsolve

import scipy.stats as stats
from scipy.stats import poisson
from scipy.stats import expon

                                                  #----------------------Service Time----------------------#


# Hyper-exponential PDF
def hyper_exp_pdf(x,lambda1,lambda2):
    return p * lambda1 * np.exp(-lambda1 * x) + (1 - p) * lambda2 * np.exp(-lambda2 * x)

# Hyper-exponential CDF
def hyper_exp_cdf(x,lambda1,lambda2):
    return p * (1 - np.exp(-lambda1 * x)) + (1 - p) * (1 - np.exp(-lambda2 * x))

# Inverse CDF for hyper-exponential
def inverse_hyper_exp_cdf(q,p,lambda1,lambda2):
    if q < p:
        return -np.log(1 - q/p) / lambda1
    else:
        return -np.log(1 - (q-p)/(1-p)) / lambda2

def equations(params):
    lambda1, lambda2, p = params

    # Mean equation
    mean_eq = (1/lambda1) * p + (1/lambda2) * (1 - p) - average

    # Variance equation
    variance_eq = (1/lambda1**2) * p + (1/lambda2**2) * (1 - p) - std_deviation**2

    return [mean_eq, variance_eq, lambda1 + lambda2 + p - 1]  # Additional constraint


# Parameters
def hyper_exp_service_time(average, std_deviation, initial_guess, num_samples):

    lambda1, lambda2, p = fsolve(equations, initial_guess)

    # Generate samples
    uniform_samples = np.random.rand(num_samples)
    for u in uniform_samples:
        hyper_exp_samples = abs(inverse_hyper_exp_cdf(u,p,lambda1,lambda2)) 

    return hyper_exp_samples


def exponential_service_time(average, num_samples):
    # Set the average (mean) for the exponential distribution
    average = 1

    # Calculate the scale parameter from the average (mean)
    scale_parameter = 1 / average

    # Create an exponential distribution with the specified scale parameter
    exp_dist = expon(scale=scale_parameter)

    # Generate random samples from the exponential distribution
    exp_samples = exp_dist.rvs(size=num_samples)

    return float(exp_samples)


def deterministic_service_time(num_samples):
    return 1


#--------------------------------------------------
#                   ''Measure''
#--------------------------------------------------
class Measure:
    def __init__(self, Narr, Ndep, NAveraegClients, OldTimeEvent, AverageDelay):
        self.arr = Narr                 # number of arrival
        self.dep = Ndep                 # number of departures
        self.ut = NAveraegClients       # number of average customers
        self.oldT = OldTimeEvent
        self.delay = AverageDelay       #average delay
        self.utq = 0          # number of average customers in waiting line
        self.st = 0          # service time
        self.delay = 0
        self.dropouts = 0 
        self.loss = []          

        self.delays = []    #entities spend waiting in a queue before they are serviced 
        self.delayed = []   #the arrival_time of client 
        self.queue = []     #The waitling queue
        self.servers_queue = [] #the list record all server
        self.servered_queue = []
        self.record_list = []

#--------------------------------------------------
#                   ''# Client

# - type: for future use
# - time of arrival (for computing the delay, i.e., time in the queue)''
#--------------------------------------------------

class Client:
    def __init__(self, type, arrival_time,ID):
        self.arrival_ID = ID
        self.type = type         #arrival 
        self.arrival_time = arrival_time
        # self.departuretime = departure_time

#--------------------------------------------------
#                   ''Server''
# whether the server is idle or not
#--------------------------------------------------


class Server(object): 
    def __init__(self,n):
        self.server_ID = n
        self.endtime = 0
        self.service_time=0
        self.oldT = 0


#--------------------------------------------------
'''
# Event function: ARRIVAL 

# - the FES_FIFO, for possibly schedule new events
# - the queue of the clients
'''
#--------------------------------------------------

def arrival(time, FES_FIFO, ID, arrival_value):
    global clients, in_service, data
    global loss
    # global service_time_func

    

    # cumulate statistics
    data.arr += 1                               
    data.utq += (clients-in_service) * (time - data.oldT)   # number of average customers in waiting line
    data.ut += clients*(time-data.oldT)                     # number of average customers
    # data.oldT = time                 

    # define the arrivel time for customer
    inter_arrival =  random.expovariate(arrival_value)


    # schedule the next arrival
    next_arrival_time = time + inter_arrival
    new_ID = ID + 1
    FES_FIFO.append((next_arrival_time, "arrival",new_ID))

    clients += 1

    client = Client(TYPE1, time, ID)
    data.queue.append(client)
    sorted(data.queue, key=lambda x:x.arrival_time)



        # else:   #waiting line
    if len(data.queue)<waiting_line:
      
        client_servered = data.queue.pop(0)

                                                # I can get a new server object, server_ID always be 1   
        server= Server(ID)
        service_time = hyper_exp_service_time(1, 10, [0.5,0.5,0.5], 1)


        service_time = float(service_time)
        server.service_time = service_time

        # print(f'Serveice time {service_time}')
        data.st += service_time


        sorted(data.servers_queue, key=lambda x: x.endtime)
        last_server = data.servers_queue.pop(0)
        delay = 0
        
        if  time >= last_server.endtime:

            # data.oldT = client_servered.arrival_time
            departure_time = client_servered.arrival_time + service_time
            delay  = 0
            data.delays.append (delay)


        else:
            
            departure_time = last_server.endtime + service_time
            delay = last_server.endtime - client_servered.arrival_time
            data.delay += delay
            data.delays.append (delay)
        
        # print(delay)
            
        server.endtime = departure_time
        server.oldT = client_servered.arrival_time
        

        
        data.servers_queue.append(server)
        sorted(data.servers_queue, key=lambda x: x.endtime )
        
        
        data.queue.append(client_servered)
        

        FES_FIFO.append((departure_time, "departure", ID))
        # print(f'server {server.server_ID} endtime {server.endtime}')
        # break 

    else:
        client_left = data.queue.pop(0)
        data.dropouts +=1    

#--------------------------------------------------
'''
# Event function:  DEPARTURE

# - the FES_FIFO, for possibly schedule new events
# - the queue of the clients
'''
#--------------------------------------------------
def departure(time, FES_FIFO, ID):
    global clients, arrival_value
    global temp_st, in_service, data

    # global service_time_func

    # cumulate statistics
    data.dep += 1
    data.ut += clients*(time-data.oldT)
    data.utq += (clients - in_service) * (time - data.oldT)

    clients -= 1
    in_service -= 1
    data.oldT = time
    


    for server in data.servers_queue:
        if ID == server.server_ID:      #time  = client.departure_time
                                        # this mean one of sever is end up serving
            current_server = server
            break 
        else:
            current_server = Server(ID)



    # print(len(data.queue))
    if len(data.queue)>0:                         

        # we need pop one client who is waiting

        check_client = data.queue.pop(0)
    # if check_client[1]=='arrival':

        next_ID  = check_client.arrival_ID    #check_client[2]
        next_server = Server(next_ID)

        service_time = hyper_exp_service_time(1, 10, [0.5,0.5,0.5], 1)
        data.st += float(service_time)

        # data.oldT = client_servered.arrival_time
        next_departure_time = max(check_client.arrival_time,time) + service_time

        next_server.endtime = next_departure_time
        next_server.service_time = service_time
        next_server.oldT = check_client.arrival_time

        next_delay = max((next_server.endtime - check_client.arrival_time),0)

        data.delay += next_delay
        data.delays.append (next_delay)


        data.servers_queue.append(next_server)


#--------------------------------------------------
'''
                  # Event-loop 
'''
#--------------------------------------------------


#-------------------------------------------------------------------

combined_df = []
def event_loop(arrival_list):
    global data
    # global average, std_deviation, num_samples, initial_guess
    

    for index, arrival_value in enumerate(arrival_list):  
        check_point_1 = []
        check_point_2 = []
        check_point_3 = []
        check_point_4 = []
        check_point_5 = []


        # global arrival_value
        # arrival_value = value

        data = Measure(0, 0, 0, 0, 0)


        FES_FIFO = []
        FES_FIFO.append((0, "arrival",1))

        ## initialization
        arrivals = 0
        clients = 0
        dropouts = 0
        time = 0 
        ID  = 0

        
        original_server = Server(0)
        data.servers_queue.append(original_server)
        count = 0
        while time <=SIM_TIME:   #SIM_TIME
            count += 1
            
            
            FES_FIFO.sort()
            (time, event_type, ID) = FES_FIFO.pop(0)
            data.record_list.append((time, event_type, ID))
        
            
            # print(FES_FIFO)
            
            if event_type == "arrival":
                arrival(time, FES_FIFO, ID, arrival_value)


            elif event_type == "departure":
                departure(time, FES_FIFO, ID)

#--------------------------------do all summary----------------------------------#


            if count == 50:
                check_point_1.append((len(data.delays), len(data.delays)/data.dep, len(data.queue)))

            elif count == 100:
                check_point_2.append((len(data.delays),len(data.delays)/data.dep,len(data.queue)))

            elif count == 150:
                check_point_3.append((len(data.delays),len(data.delays)/data.dep,len(data.queue)))

            elif count == 200:
                check_point_4.append((len(data.delays),len(data.delays)/data.dep,len(data.queue)))

            elif count == 250:
                check_point_5.append((len(data.delays),len(data.delays)/data.dep,len(data.queue)))
        
        average_delay = data.delay/data.dep
        average_no_cust = data.ut/SIM_TIME
        
        arrival_value_list.append(arrival_value)
        arrivals_list.append(data.arr)
        departures_list.append(data.dep)
        average_delay_list.append(average_delay)
        average_no_cust_list.append(average_no_cust)
        
        proportation_of_delay = data.delay / (data.st + data.delay)


        delay_rate = sum(1 for i in data.delays if i > 0) / len(data.delays) if data.delays else 0 
        # print('the data delay rate is: ',delay_rate)

        delay_rate_list.append(round((  delay_rate ),4))

        dropouts_probability_list.append(round(data.dropouts/data.arr,4))
        # print(data.dropouts)


#---------------------------- do some short summary----------------------------------------#
        data = [check_point_1,check_point_2,check_point_3,check_point_4,check_point_5]

        # Flatten the nested lists
        flat_data = [item[0] if item else (None, None, None) for item in data]

        # Define column names
        columns = ['the number of delay', 'averger delay number', 'the length of queue']
        index_names = ['check_point_1', 'check_point_2', 'check_point_3', 'check_point_4', 'check_point_5']

        df = pd.DataFrame(flat_data, columns=columns, index=index_names)

        # Plotting the DataFrame
        df.plot(kind='line', y=['the number of delay', 'averger delay number', 'the length of queue'], marker='o')

        # Setting labels and title
        plt.xlabel('Check Points')
        plt.ylabel('Values')
        plt.title('Performance Metrics at Check Points')

        # Rotating x-axis labels
        plt.xticks(rotation=45, ha='right')

        # Adding grid and legend
        plt.grid()
        plt.legend(loc='upper left')

        # Display the plot
        plt.show()



                                #   Start Simulation   #
arrival_value_list = []
arrivals_list = []
departures_list = []
average_delay_list = []
average_no_cust_list = []
delay_rate_list = []
dropouts_probability_list = []

arrival_list = [0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99,0.998]

#define the initiation
SERVICE = 1  #np.random.uniform(service_time_max, service_time_min)    # average service time
clients = 0
in_service = 0
Total_servers = 1     #10
last_service_time=0
delay = 0

TYPE1 = 1 #
SIM_TIME = 1000     #simulation time
waiting_line = 5
  # the maximum of queue


average = 1
std_deviation = 10
num_samples = 1
initial_guess = [1, 1, 0.5]    # Initial guess for parameters


event_loop(arrival_list)


#--------------------------------------------------
'''
                  # conclusion
'''
#--------------------------------------------------

table = [arrival_value_list,
        arrivals_list,
        departures_list,
        average_delay_list,
        delay_rate_list,
        dropouts_probability_list]

columns =  ['arrival lambda','Number of arrivals', 'Number of departures', 'average delay', 'delay rate','dropping probability']

df_summary = pd.DataFrame(table).transpose()
df_summary.columns = columns
# df.index = row_names
df_summary.head(8)


fig, ax = plt.subplots(1, 2, figsize=(16, 7))

# First subplot
ax[0].scatter(df_summary['arrival lambda'], df_summary['average delay'])
ax[0].plot(df_summary['arrival lambda'], df_summary['average delay'])

# Adding labels and title
ax[0].set_xlabel('Arrival Lambda')
ax[0].set_ylabel('Average Delay')
ax[0].set_title('Arrival Lambda vs Average Delay')
ax[0].grid(True)
ax[0].legend()

# Second subplot
ax[1].scatter(df_summary['arrival lambda'], df_summary['dropping probability'])
ax[1].plot(df_summary['arrival lambda'], df_summary['dropping probability'])

# Adding labels and title
ax[1].set_xlabel('Arrival Lambda')
ax[1].set_ylabel('Dropping Probability')
ax[1].set_title('Arrival Lambda vs Dropping Probability')
ax[1].grid(True)
ax[1].legend()

# Display the plot

plt.show()

