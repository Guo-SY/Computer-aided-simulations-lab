import random
import numpy as np
import pandas as pd
from queue import Queue,PriorityQueue
import matplotlib.pyplot as plt


import scipy.stats as stats
from scipy.stats import poisson
from scipy.stats import expon



                         ########### Function ###################
#                   ''# Client

# - type: for future use
# - time of arrival (for computing the delay, i.e., time in the queue)''
#--------------------------------------------------

class Client:
    def __init__(self, type, color, arrival_time):
        self.type = type         #arrival 
        self.color = color
        self.arrival_time = arrival_time
        # self.departure_time = 0
        # self.service_time = 0

        # self.departuretime = departure_time
        

#                   ''Server''
# whether the server is idle or not
#--------------------------------------------------

class Server(object): 
    def __init__(self):
        self.endtime = 0
        self.service_time=0
        # self.color = none
        self.oldT = 0



def Interrupt_service(client, FES):
    global data 

    ## it always is Y_client or G_client, but sometime is red
    curr_event_arrival = client.arrival_time
    curr_event_color   = client.color
    curr_event_deaprture_time = 0

    ############## initialization#######
    
    Inter_client  = client                         
    
    found_red_event = False                   #we need to find the  Red_event_arrival_time 
    Red_event_arrival_time = 0                # avoid local variable 'Red_event_arrival_time' referenced before assignment


    for event_1 in FES:   
        
        event_1_type = event_1[2]      
        event_1_color = event_1[1]
        event_1_arrival_time = event_1[0]

        if event_1_color == 'red' and event_1_type == 'arrival':
            Red_event_arrival_time = event_1_arrival_time
            found_red_event = True
            break

        else:
            found_red_event = False

                            ######## we need find the curr_event_deaprture_time
    departure_FES =[]
    departure_FES = [event for event in FES if event[2]== 'departure']
    for event_2 in departure_FES: 

        event_2_type = event_2[2]      
        event_2_color = event_2[1]
        event_2_arrival_time = event_2[3]

        if event_2_color == curr_event_color and event_2_arrival_time == curr_event_arrival:
            curr_event_deaprture_time = event_2[0]
            break

        else:
            # service_time = 3
            curr_event_deaprture_time = curr_event_arrival + hospital_server.service_time
            
                                      
        # if red event arrival when other color clients are servered, 
        # it is necessary to determine if the red event arrives within curr_event_arrival and curr_event_deaprture_time

    if found_red_event == True:

        if curr_event_color == 'yellow':
            if curr_event_arrival <= Red_event_arrival_time < curr_event_deaprture_time:
            
                # the yellow event have been interrupted,so we need to put it back to Yellow_Q
                data.Yellow_Q.append(client)
                # data.Yellow_Q.sort()

                FES.pop(FES.index((Red_event_arrival_time, 'red', 'arrival')))
                Inter_red_client = Client(type = 'arrival', color = 'red', arrival_time=Red_event_arrival_time)

            else:
                # the yellow event have not been interrupted,so we will keep it as current client
                Inter_red_client = client
        
        elif curr_event_color == 'green':

            if curr_event_arrival <= Red_event_arrival_time <= curr_event_deaprture_time:
                # the yellow event have been interrupted,so we need to put it back to Yellow_Q
                data.Green_Q.append(client)
                # data.Green_Q.sort()
                
                
                FES.pop(FES.index((Red_event_arrival_time, 'red', 'arrival')))
                Inter_red_client = Client(type = 'arrival', color = 'red', arrival_time=Red_event_arrival_time)

            else:
                # the yellow event have not been interrupted,so we will keep it as current client
                Inter_red_client = client

        elif curr_event_color == 'red':
            if curr_event_arrival <= Red_event_arrival_time:
                Inter_red_client = client
                wait_red_client = Client(type = 'arrival', color = 'red', arrival_time=Red_event_arrival_time)
                data.Red_Q.append(wait_red_client)
                # data.Red_Q.sort()


            else:
                Inter_red_client = Client(type = 'arrival', color = 'red', arrival_time=Red_event_arrival_time)
                wait_red_client = client

                data.Red_Q.append(wait_red_client)
                Inter_red_client = Client(type = 'arrival', color = 'red', arrival_time=Red_event_arrival_time)
                # data.Red_Q.sort()

    else:
        Inter_client  = client     
    
    return data.Red_Q, data.Yellow_Q, data.Green_Q, Inter_client 

def Next_arrival_client(curr_arrival_time, arrival_value):
    

                            # Use random.choices with weights to select a color 
    inter_arrival =  random.expovariate(arrival_value)
    
    color_list = ['red', 'yellow', 'green']
    weights = [1/6, 1/3, 1/3]
    color = random.choices(color_list, weights, k=1)[0]

    next_arrival_time = round((curr_arrival_time + inter_arrival),4)
    
    return (next_arrival_time, color, "arrival")



def Strict_priority_service(client, FES):
    global data

    urge_client = client
    
    if len(data.Red_Q) > 0:
        urge_client = data.Red_Q.pop(0)
        
    else: 
        if len(data.Yellow_Q) > 0:
            Y_client = data.Yellow_Q.pop(0)
            
            data.Red_Q, data.Yellow_Q, data.Green_Q, Inter_client  = Interrupt_service(Y_client, FES) # whether we need input a uragent red_event
            urge_client = Inter_client 
            
            # return curr_client 

        else: 
            if len(data.Green_Q) > 0:
                G_client = data.Green_Q.pop(0)
                data.Red_Q, data.Yellow_Q, data.Green_Q, Inter_client = Interrupt_service(G_client, FES)
                urge_client = Inter_client 



    return data.Red_Q, data.Yellow_Q, data.Green_Q, urge_client


class Measure:
    def __init__(self, Narr, Ndep, NAveraegClients, OldTimeEvent, AverageDelay):
        self.arr = Narr                 # number of arrival
        self.dep = Ndep                 # number of departures
        self.ut = NAveraegClients       # number of average customers
        self.oldT = OldTimeEvent
        self.delay = AverageDelay       #average delay
        self.utq = 0          # number of average customers in waiting line
        self.st = 0          # service time
        self.R_delay = 0
        self.Y_delay = 0
        self.G_delay = 0
        self.dropouts = 0 
        self.loss = []
        self.departure_event  = []  
        
        self.Red_Q = [] 
        self.Yellow_Q = []  
        self.Green_Q = []      

        self.delays = []    #entities spend waiting in a queue before they are serviced 
        self.R_delayed = []   #the arrival_time of client 
        self.Y_delayed = []   #the arrival_time of client 
        self.G_delayed = []   #the arrival_time of client 
        self.queue = [] 

def arrival(client, FES, a):
    global clients, in_service, data
    data.arr += 1                               
    data.utq += (clients-in_service) * (client.arrival_time - data.oldT)   # number of average customers in waiting line
    data.ut += clients*(client.arrival_time-data.oldT)                     # number of average customers
    data.oldT = client.arrival_time  
    
    clients += 1
    in_service +=1

    service_time = random.expovariate(1.0) 

    

                    ###### schedule the next arrival  ###############
                    
    arrival_value = a
    curr_arrival_time = client.arrival_time

    next_arrival_client = Next_arrival_client(curr_arrival_time, arrival_value)
    FES.append(next_arrival_client)

    curr_client = client



    
                   ##### we put a new arrival client into different queue############
    if client.arrival_time <= hospital_server.endtime:
        if client.color == 'red':
            data.Red_Q.append(client)
            # data.Red_Q.sort()

        
        elif client.color == 'yellow':
            data.Yellow_Q.append(client)
            # data.Yellow_Q.sort()

        elif client.color == 'green':
            data.Green_Q.append(client)
            # data.Green_Q.sort()


                 ###### we choose a patient who have a priority############

  
        data.Red_Q, data.Yellow_Q, data.Green_Q, curr_client = Strict_priority_service(client, FES)

    else:
        data.Red_Q, data.Yellow_Q, data.Green_Q, curr_client = Strict_priority_service(client, FES)

    

  

                           ######After service, the system create a new departure############


    if curr_client.arrival_time <= hospital_server.endtime:
        curr_departure_time = round((hospital_server.endtime + service_time), 5)

    else:
        curr_departure_time = round((curr_client.arrival_time + service_time), 7)
    
    hospital_server.endtime = curr_departure_time
    hospital_server.service_time = service_time
    hospital_server.oldT = curr_client.arrival_time

    FES.append((curr_departure_time, curr_client.color, "departure", curr_client.arrival_time, service_time))
    FES.sort()



                   ###### arrival ans deaprture  #######

def departure(de_client, FES, a):   # here we need a departure time 
    global clients
    global temp_st, in_service, data

        # cumulate statistics
    data.dep += 1
    data.ut += clients*(de_client[0]-data.oldT)
    data.utq += (clients - in_service) * (de_client[0] - data.oldT)

    clients -= 1
    in_service -= 1
    
    de_patient = de_client
    de_patient_color = de_patient[1]
    de_patient_type = de_patient[2]

    de_patient_arrival_time = de_patient[3]
    de_patient_departure_time = de_patient[0]
    
    de_patient_service_time = de_patient[4]

    data.oldT = de_patient[3]

    service_time = round(random.expovariate(1.0))                   #create a random service time
                                                                 # get the current client are serrvered
    
    hospital_server.endtime = de_patient_departure_time
    hospital_server.service_time = de_patient_service_time
    hospital_server.oldT = de_patient_arrival_time




    # if hospital_server.endtime == time:               #time == server.endtime means that we could get service_time from server
                                                    # this mean one of sever is end up serving
    if de_patient_color == 'red':
        data.R_delay = round((de_patient_departure_time  -  de_patient_arrival_time  -  de_patient_service_time),6)
        # data.R_delayed.append((de_patient_departure_time, de_patient_arrival_time, de_patient_service_time, data.R_delay))
        data.R_delayed.append(data.R_delay)
        data.R_delay += data.R_delay 
        data.st += de_patient_service_time


    elif de_patient_color == 'yellow':
        data.Y_delay = round((de_patient_departure_time  -  de_patient_arrival_time  -  de_patient_service_time),6)
        data.Y_delayed.append(data.Y_delay)
        data.Y_delay += data.Y_delay 
        # data.delay += data.Y_delay 
        data.st += de_patient_service_time


    else:
        data.G_delay = round((de_patient_departure_time  -  de_patient_arrival_time  -  de_patient_service_time),6)
        data.G_delayed.append(data.G_delay)
        data.G_delay += data.G_delay 
        data.st += de_patient_service_time


                               # now the hospital_server is free, we could service next client

    if (len(data.Red_Q)+len(data.Yellow_Q)+len(data.Green_Q)) > 0:
            # cumulate statistics
 
        service_time = round(random.expovariate(1.0))                   #create a random service time
                                                                 # get the current client are serrvered
        

        if len(data.Red_Q)>0:
            new_client = data.Red_Q.pop(0)

        else:
            if len(data.Yellow_Q)>0:
                new_client = data.Yellow_Q.pop(0)

            else:
                if len(data.Green_Q)>0:
                    new_client = data.Green_Q.pop(0)


       
        data.Red_Q, data.Yellow_Q, data.Green_Q, next_departure_client = Interrupt_service(new_client, FES)

        if next_departure_client.arrival_time <= hospital_server.endtime:
            next_departure_time = round((hospital_server.endtime + service_time), 5)

        else:
            next_departure_time = round((next_departure_client.arrival_time + service_time), 7)
        
        # hospital_server.endtime = next_departure_time
        # hospital_server.service_time = service_time
        # hospital_server.oldT = curr_client.arrival_time

        FES.append((next_departure_time, next_departure_client.color, "departure", next_departure_client.arrival_time, service_time))
        FES.sort()



    
        arrival_value = a
        curr_arrival_time = de_patient_arrival_time

        next_arrival_client = Next_arrival_client(curr_arrival_time, arrival_value)
        FES.append(next_arrival_client)

    data.departure_event.append((de_patient_departure_time, de_patient_color, de_patient_type, de_patient_arrival_time, de_patient_service_time))




                                 #######  Event Loop ######

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


for arrival_value in [5, 7, 9, 10, 12, 15]:   #, 7, 9, 10, 12, 15
    a = arrival_value
    ## initialization
    arrivals = 0
    clients = 0
    dropouts = 0
    in_service = 0

    hospital_server = Server()
    data = Measure(0,0,0,0,0)

    SERVICE = 1   #np.random.uniform(service_time_max, service_time_min)    # average service time
    clients = 0
    in_service = 0

    SIM_TIME = 200                #simulation time
    # waiting_line = 10         # 00  # the maximum of queue


    FES = []
    color_list = ['red', 'yellow', 'green']
    weights = [1/6, 1/3, 1/2]
    color = random.choices(color_list, weights, k=1)[0]


    FES.append((0, color, "arrival"))
    # print(FES)
    patient_time = 0


    #SIM_TIME
    while patient_time <= SIM_TIME :

        patient = FES.pop(0)

        # print(patient)
        patient_color = patient[1]
        patient_type = patient[2]
        patient_time = patient[0]

        # print(patient)

        if patient_type == "arrival":
            client = Client(type = patient_type, color= patient_color, arrival_time = patient_time)      #type, color, arrival_time
            arrival(client, FES, a)

        elif patient_type == "departure":
            departure(patient, FES, a)

#######  Event Loop ######

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
Red_Q_delay_list = []
Yellow_Q_delay_list = []
Green_Q_delay_list = []


for arrival_value in [5, 7, 9, 10, 12, 15, 20]:   #, 7, 9, 10, 12, 15
    a = arrival_value
    ## initialization
    arrivals = 0
    clients = 0
    dropouts = 0
    in_service = 0

    hospital_server = Server()
    data = Measure(0,0,0,0,0)

    SERVICE = 1   #np.random.uniform(service_time_max, service_time_min)    # average service time
    clients = 0
    in_service = 0

    SIM_TIME = 100                #simulation time
    # waiting_line = 10         # 00  # the maximum of queue


    FES = []
    color_list = ['red', 'yellow', 'green']
    weights = [1/6, 1/3, 1/2]
    color = random.choices(color_list, weights, k=1)[0]


    FES.append((0, color, "arrival"))
    # print(FES)
    patient_time = 0


    #SIM_TIME
    while patient_time <= SIM_TIME :

        patient = FES.pop(0)

        # print(patient)
        patient_color = patient[1]
        patient_type = patient[2]
        patient_time = patient[0]

        # print(patient)

        if patient_type == "arrival":
            client = Client(type = patient_type, color= patient_color, arrival_time = patient_time)      #type, color, arrival_time
            arrival(client, FES, a)

        elif patient_type == "departure":
            departure(patient, FES, a)


                      ###conclusion

    average_delay = (data.R_delay + data.Y_delay + data.G_delay)/data.dep
    average_no_cust = data.ut/SIM_TIME

    arrival_value_list.append(arrival_value)
    arrivals_list.append(data.arr)
    departures_list.append(data.dep)
    average_delay_list.append(average_delay)

    Red_Q_delay_list.append(data.R_delay)
    Yellow_Q_delay_list.append(data.Y_delay)
    Green_Q_delay_list.append(data.G_delay)

    average_no_cust_list.append(average_no_cust)
    delay_rate_list.append(round((data.R_delay + data.Y_delay + data.G_delay) / (data.st + data.R_delay + data.Y_delay + data.G_delay) * 100, 2))
     

    print('\n ========= M/3/1 =========')
    print('\n Number of arrivals per second: ', arrival_value)
    print("\n Number of arrivals:",data.arr)
    print("\n Number of departures: ", data.dep)

    print("\n Number of average customers: ", round(data.ut/patient_time, 2))  
    print("\n R_queue delay", round(data.R_delay,3),"ms" )
    print("\n Y_queue delay", round(data.Y_delay,3),"ms" )
    print("\n G_queue delay", round(data.G_delay,3),"ms" )
    print("\n Average system delay:",round((data.R_delay + data.Y_delay + data.G_delay)/data.dep, 3) ,"ms") 

    print("\n time:",patient_time)
    print("\n Average queue delay", round(((data.st -  - data.Y_delay - data.G_delay)/data.dep),3),"ms" )



    print("\n Average number of customers in queue", round(data.utq / patient_time,2))  
    print("\n busy time:", round(data.st / (data.st + data.delay) * 100, 2),"%")
    print("\n")
    print("\n all server time:",data.st)
    print("\n all delay time:",data.R_delay + data.Y_delay + data.G_delay)



    # print("Missing service probability: ",data.loss/data.arr)
    print("\n Arrival rate: ",data.arr/patient_time)

table = [arrival_value_list,
        arrivals_list,
        departures_list,
        average_delay_list,
        delay_rate_list]    #        Red_Q_delay,Yellow_Q_delay,Green_Q_delay,

columns =  ['number of arrival per second','Number of arrivals','Number of departures', 'average delay', 'delay rate'] #, 'The delay of Red_Q_delay','The delay of Yellow_Q_delay','The delay of Green_Q_delay',

df = pd.DataFrame(table).transpose()
df.columns = columns
# df.index = row_names
df.head(10)

import matplotlib.pyplot as plt

x = arrival_value_list
y1 = Red_Q_delay_list
y2 = Yellow_Q_delay_list
y3 = Green_Q_delay_list

fig, ax = plt.subplots()

ax.plot(x, y1, label='Line 1', linestyle='-', marker='o', color='red')
ax.plot(x, y2, label='Line 2', linestyle='--', marker='s', color='yellow')
ax.plot(x, y3, label='Line 3', linestyle='-.', marker='^', color='green')

ax.legend()
ax.set_xlabel('different Inter_arrival lambda')
ax.set_ylabel('The delay time')
ax.set_title('The delay time of patients')

plt.show()
