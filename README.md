# Computer-aided-simulations-lab

                      ######## Lab L1. Simulating a queue  ########
Write a piece of code that represents a multi-server FIFO queue  (K servers) and finite/infinite waiting line.

Please, use  list structures to represent both the queue and the FES.

Test your simulator in the following setting:

i)  K=10;

ii) waiting line size N= 1000  customers;

ii) arrival process of customers  is a Poisson process  at rate lambda  in { 5, 7, 9, 10, 12, 15 }, (i.e. interarrivals are exponentially distributed with average 1/lambda);

 iii) service times are expontentially distributed with average 1.

Evaluate the average delay and the dropping probability.



                      ######## Lab L2. Hospital Emergency Room  ########
Write a  simple simulator of of the queueing process within a Hospital Emergency Room.

Arriving customers can be classified into tree categories:

red code  (very urgent)

yellow  code  (moderately urgent)

green code   (not urgent)

Customers arrive upon time according to some  unknown stationary arrival process. 

1/6 of arriving customers are red code;

1/3 are yellow code, the remaining fraction (1/2) are green code.  

K different medical teams operate in parallel.

Customers are served according to a strict-priority service discipline (a yellow-code customer enters service only when there are no red-code customers waiting, etc.)

Upon the arrival of a red-code customer, if a yellow or green code customer is being served; the service of the latter customer is interrupted and the service of the just arrived  red-code customers is immediately started.  The interrupted service will be resumed later on.

Make  reasonable assumptions (and justify them) on the arrival process(es) of customers and on the service times of customers (choose suitable distribution(s)).

For $K=1$, simulate a few scenarios, by varying the rate of arrival of customers and/or service times distributions.  Write a brief  comment on the assumptions you did and results you got.



                      ######## Lab L3 Generation of Random Variables  ########
Write functions that generate  R.V.s distributed as follows: 
-  Rayleigh(sigma)  
-  Lognormal(mu, sigma^2)
-  Beta(alpha, beta) distributed random variable, with alpha>1 and beta>1,
-  Chi square (n)   n>=1
-  Rice distribution (nu,sigma)  for nu>=0 sigma>=0

For one of the previously listed R.V.s, test  your generator by 
evaluating  the empirical first two moments you obtain after n\in {1000,10000,100000} random extractions, and comparing them with analytical predictions.  In addition compare also the empirical Cdf/pdf to the analytical one.

Write a brief report in which you describe and justify  the method/algorithm you have used. In addition report the outcomes of your test.



                      ######## Lab L4: Transient elimination and batch means  ########

Re-adapt the code you have developed within lab L1 to simulate a M/G/1 queue.
Add  procedures to eliminate the transient in run time (by inventing a proper algorithm) and to evaluate confidence intervals by adopting a batch means technique.

Consider the following  distributions for service time:
Deterministic equal to 1.
Exponential  with average 1.
Hyper-exponential with two branches; average=1  and standard deviation=10.
 
For each of  three considered systems produce a plot in which you report the average queueing delay  vs lambda. 

Choose the following vales  for  lambda:  {0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 0.99, 0,999} 

Compare the results you have got with theoretical predictions given in:

https://en.wikipedia.org/wiki/Pollaczek%E2%80%93Khinchine_formula

Verify experimentally the Little formula:

https://en.wikipedia.org/wiki/Little%27s_law ;
 
At last, consider a M/M/1  with finite waiting line N=1000,  and produce two  plots: 
average delay vs lambda;
dropping probability vs lambda. 
Use the following values for  lambda: {0.2, 0.4, 0.6, 0.8, 0.9, 0.95, 1, 1.1, 1.2}. 



Deliver the code and a brief report in which
you report  the results you got (with a brief comment) and you describe:
i)  the algorithm you have implemented for the transient elimination;
ii) how you compute confidence intervals;
iii) the parameters  of the hyper exponential and how you obtained them;




                      ######## Lab L5: Dynamic processes on graphs  ########
                      
Write a piece of code to simulate dynamical processes on graphs.
The simulator should be able to:
i)  generate in an efficient way a either G(n,p) graphs or regular  grids  (with order of 100k nodes);
ii)  handle in  an efficient way the FES (resort on properties of Poisson processes). 

Deliver the code along with a brief report, in which you clearly describe:
i)   the data structure you use;
ii)  which are the events and how the FES is handled.
ii)  the algorithm according to which you generate samples of  G(n,p) graphs.

Furthermore for n=100k, p= 10^{-4}  compare the empirical distribution of the degree with analytical predictions.  Build a q-q plot and execute a \chi^2 test.


                      ######## Lab L6: Dynamic processes on graphs   ########
The goal of Lab L6 is to study properties of  the dynamic processes 
 over graphs.

Consider a voter model over a G(n,p) with  n chosen in the range [10^3, 10^4]  ( in such a way that simulation last a reasonable time i.e. 5/10 min at most) and p_g= 10/n.
According to the initial condition,   each node  has a probability p_1  of being in state +1 with p_1\in {0.51, 0.55, 0.6, 0.7}.
Evaluate the probability of reaching  a +1-consensus  (if the graph is not connected consider only the giant component). Evaluate, as well, the time needed to reach consensus.
 
Then consider a voter model over finite portion of   Z^2 and Z^3.
Fix p_1=0.51 and, for 2/3 values of n \in[10^2, 10^4], estimate, as before,
  the probability of reaching  a +1-consensus  and the time needed to reach consensus.

Deliver the code along with a brief report, in which you present and comment your results. Are the obtained results in line with  theoretical predictions?



                      ######## Lab L7:Epidemic processes  ########
The goal of Lab L7 is to define and simulate simple strategies 
to  control an  epidemic (SIR) process through non pharmaceutical interventions
(I.e. by introducing mobility restrictions).

Consider a homogeneous population of 50M individuals.
Fix R(0)=4 and \gamma= 1/14 days (recovering rate).    
Assume that  10% (6%) of the infected individuals  needs to be Hospitalized (H)  (undergo Intensive Treatments (IT).)
  
Fix the fatality rate of the epidemic to 3%.
H/IT places are limited (10k/50 k). Design some  non pharmaceutical intervention strategy that avoids H/IT overloads, and limits the number of death in 1 year to 100K.
To design your strategy you can use a mean-field SIR model.

Then, once you have defined your strategy simulate both the stochastic SIR and its mean field.  Are there significant differences, why? 
What happens if you scale down your population N to 10K (of course you have to scale down also other parameters, such as H and IT places)?



                      ######## Lab G2: Confidence interval  ########
Let X be the output of a stochastic process (.e.g, the estimated average in an experiment), being a real number.

Let us assume that X is uniformly distributed between 0 and 10.

We wish to study the effect on the accuracy of the estimation of the average in function of the number of experiments and in function of the confidence level.

1. Define properly all the input parameters
2. Write all the adopted formulas
3. Explain which python function you use to compute average, standard deviation and confidence intervals
4. Plot the confidence interval and the accuracy

5. Discuss the main conclusions drawn from the graphs.





                      ######## Lab G4. Student career  ########
Develop a simulator to evaluate the graduation time and the final grade at a MSc course.

Explain in details the random elements in the simulated system
Explain all the input parameters
Explain the main assumptions of the adopted simulation model
Explain all all the output metrics
Explain all the main data structures
Explain some interesting correlations between output metrics and input parameters
For the simulation, use realistic values for all the parameters.

You must produce a document with all the answers, with no more than 2 pages.




                      ######## Lab G5. Student career  ########
Develop a simulator of the student career in the university, that should be an enhanced version of LABG4.

Q1. Explain the purpose of the simulator, i.e.. the main questions to address
Q2. Explain in details the random elements in the simulated system
Q3. Explain the main assumptions of the adopted simulation model
Q4. Explain all the input parameters
Q5. Describe the (eventual) open data sources adopted in the simulation
Q6. Explain all the output metrics
Q7. Describe the main adopted data structures

Consider a realistic/interesting simulation scenario:
R1. Motivate the choice of all the input parameters
R2. Comment in details the  numerical results, focusing on the main questions described above.



                      ######## Lab G6 - Basic natural selection  ########
  Consider a simulator for natural selection with the following simplified simulation model:

All the individuals belong to the same species
The initial population is equal to P
The reproduction rate for each individual is lambda
The lifetime LF(k) of individual k whose parent is d(k) is distributed according to the following distribution:
LF(k)=
uniform(LF(d(k),LF(d(k)*(1+alpha)) with probability prob_improve   
uniform(0,LF(d(k)) with probability 1-prob_improve

where prob_improve  is the probability of improvement for a generation and alpha is the improvement factor (>=0)

Answer to the following questions:

Describe some interesting questions to address based on the above simulator.
List the corresponding output metrics.
Develop the simulator
Define some interesting scenario in terms of input parameters.
Show and comment some numerical results addressing the above questions.
Upload only the py code and the report (max 2 pages).


                      ######## Lab G7 - Basic natural selection  ########
Consider a simulator for natural selection with the following simplified simulation model:

All the individuals belong to S different species
Let s be the index of each species, s=1...S
The initial population of s is equal to P(s)
The reproduction rate for each individual is lambda
The theoretical lifetime LF(k) of individual k whose parent is d(k) is distributed according to the following distribution:
LF(k)=
uniform(LF(d(k)),LF(d(k))*(1+alpha)) with probability prob_improve   
uniform(0,LF(d(k)) with probability 1-prob_improve

where prob_improve  is the probability of improvement for a generation and alpha is the improvement factor (>=0)

The individuals move randomly in a given region and when individuals of different species meet, they may fight and may not survive. In such a case, the actual lifetime of a individual may be lower than its theoretical lifetime. A died individual cannot reproduce.
Answer to the following questions:

Describe some interesting questions to address based on the above simulator.
List the corresponding input metrics.
List the corresponding output metrics.
Describe in details the mobility model with finite speed
Describe in details the fight model and the survivabilty model
Develop the simulator
Define all the events you used and their attribute
Define some interesting scenario in terms of input parameters.
Show and comment some numerical results addressing the above questions.
