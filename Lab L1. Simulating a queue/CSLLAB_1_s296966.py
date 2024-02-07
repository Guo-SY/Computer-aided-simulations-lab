import random
import numpy as np
import pandas as pd
from queue import Queue,PriorityQueue
import matplotlib.pyplot as plt


import scipy.stats as stats
from scipy.stats import poisson
from scipy.stats import expon

## define the inter_arrival time 
     
SERVICE = 1  #np.random.uniform(service_time_max, service_time_min)    # average service time

clients = 0
in_service = 0
Total_servers = 10
last_service_time=0
delay = 0

TYPE1 = 1 #
SIM_TIME = 20000     #simulation time
waiting_line = 1000  # the maximum of queue




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
        self.queue = []     #

#--------------------------------------------------
#                   ''# Client

# - type: for future use
# - time of arrival (for computing the delay, i.e., time in the queue)''
#--------------------------------------------------

class Client:
    def __init__(self, type, arrival_time):
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

def arrival(time, FES_FIFO, services):
    global clients, in_service, data
    global loss

    # cumulate statistics
    data.arr += 1                               
    data.utq += (clients-in_service) * (time - data.oldT)   # number of average customers in waiting line
    data.ut += clients*(time-data.oldT)                     # number of average customers
    # data.oldT = time                 

    # define the arrivel time for customer
    inter_arrival =  random.expovariate(arrival_value)


    # schedule the next arrival
    next_arrival_time = time + inter_arrival
    FES_FIFO.append((next_arrival_time, "arrival"))

    clients += 1

    client = Client(TYPE1, time)
    data.queue.append(client)


        # else:   #waiting line
    if len(data.queue)<waiting_line:
        


        for server in services:

            if  time >= server.endtime:
                client_servered = data.queue.pop(0)
                # data.oldT = client_servered.arrival_time

                service_time = random.expovariate(1.0)
                # print(f'Serveice time {service_time}')
                data.st += service_time

                departure_time = client.arrival_time + service_time
                
                server.endtime = departure_time
                server.service_time = service_time
                server.oldT = client.arrival_time

                FES_FIFO.append((departure_time, "departure"))
                # print(f'server {server.server_ID} endtime {server.endtime}')
                break 

    else:
        data.dropouts +=1    

#--------------------------------------------------
'''
# Event function:  DEPARTURE

# - the FES_FIFO, for possibly schedule new events
# - the queue of the clients
'''
#--------------------------------------------------
def departure(time, FES_FIFO, services):
    global clients
    global temp_st, in_service, data

    # cumulate statistics
    data.dep += 1
    data.ut += clients*(time-data.oldT)
    data.utq += (clients - in_service) * (time - data.oldT)

    clients -= 1
    in_service -= 1
    data.oldT = time
    


    for server in services:
        if time == server.endtime:      #time  = client.departure_time
                                        # this mean one of sever is end up serving

            if len(data.queue)>0:                         

                # we need pop one client who is waiting
                client_waited = data.queue.pop(0)
                delay = server.endtime - client_waited.arrival_time

                # last_service_time = server.service_time
                data.delay += delay
                data.delays.append (delay)
                # print(delay)
                # print(data.delay)

                service_time = random.expovariate(1.0)
                data.st += service_time
                next_departure_time = time + service_time

                server.endtime = next_departure_time
                server.service_time = service_time
                FES_FIFO.append((next_departure_time, "departure"))
                
                break 

#--------------------------------------------------
'''
                  # Event-loop 
'''
#--------------------------------------------------

arrival_value_list = []
arrivals_list = []
departures_list = []
average_delay_list = []
average_no_cust_list = []
delay_rate_list = []

for arrival_value in [5, 7, 9, 10, 12, 15]:   

        # schedule the first arrival at t=0
    data = Measure(0, 0, 0, 0, 0) 
    FES_FIFO = []
    FES_FIFO.append((0, "arrival"))


    # the simulation time
    services = [] 

    for i in range(Total_servers):
        services.append(Server(i+1))

    ## initialization
    arrivals = 0
    clients = 0
    dropouts = 0
    time = 0 

    while time <=SIM_TIME :   #SIM_TIME
        FES_FIFO.sort()
        (time, event_type) = FES_FIFO.pop(0)
        # print(FES_FIFO)
        
        if event_type == "arrival":
            arrival(time, FES_FIFO, services)

        elif event_type == "departure":
            departure(time, FES_FIFO, services)

    average_delay = data.delay/data.dep
    average_no_cust = data.ut/SIM_TIME
    
    arrival_value_list.append(arrival_value)
    arrivals_list.append(data.arr)
    departures_list.append(data.dep)
    average_delay_list.append(average_delay)
    average_no_cust_list.append(average_no_cust)
    delay_rate_list.append(round(data.delay / (data.st + data.delay) * 100, 2))


    print('\n ========= M/M/10 =========')
    print('\n Number of arrivals per second: ', arrival_value)
    print("\n Number of arrivals:",data.arr)
    print("\n Number of departures: ", data.dep)

    print("\n Number of average customers: ", round(data.ut/time, 2))  
    print("\n Average system delay:",round(data.delay/data.dep, -1) ,"ms") 
    print("\n time:",time)


    print("\n Average queue delay", round(((data.delay - data.st)/data.dep),3),"ms" )   
    print("\n Average number of customers in queue", round(data.utq / time,2))  
    print("\n busy time:", round(data.st / (data.st + data.delay) * 100, 2),"%")
    print("\n")
    print("\n all server time:",data.st)
    print("\n all delay time:",data.delay)



    # print("Missing service probability: ",data.loss/data.arr)
    print("\nArrival rate: ",data.arr/time)
    print("Average delay: ",data.delay/data.dep)


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

columns =  ['number of arrival per second','Number of arrivals', 'Number of departures', 'average delay', 'delay rate','dropping probability']

df = pd.DataFrame(table).transpose()
df.columns = columns
# df.index = row_names
df.head(10)