# Q-learning Algorithm for Detection and Defense Tasks in a Smart Power System
A Model-Free Reinforcement Learning algorithm for detection and defense tasks in a smart power system.
## Description
For each possible observation-action pair (o; a), we propose to learn a Q(o; a) value, i.e., the expected future cost, using an RL algorithm where all Q(o; a) values are stored in a Q-table. After learning the Q-table, the policy of the defender will be choosing the action a with the minimum Q(o; a) for each observation o.
## Challenges
Generally, Reinforcement Learning uses for sequential, and It has some complexity to map our application to it. Since we need to discrete the space of values. 
Another challenge is that since high voltage and power factor change can affect the power system in a dangerous mode, our actions are low and fixed changing the parameters.

## Details of the Model
In this step of this project, we focus on two parameters as our input: voltage and power factor. We consider 4 states and 3 actions for each of these two parameters.\
The 4 states are 1-Healthy 2-Acceptable 3-Critical 4-Compromised. To discern the space of values for voltage and make our states we split the overall range of voltage into 4 ranges as we shoe in the following:\
![image](https://user-images.githubusercontent.com/20415408/43529592-7f97f8ce-9560-11e8-9700-302cd75c7eda.png)\
Also, for the voltage parameter, we have 3 action: 1-No defense 2-Increase by fixed step 3-Decrease by a fixed parameter.
Therefore, if we consider each parameter we have a Q-table with 4 states and 3 actions:
![image](https://user-images.githubusercontent.com/20415408/43530158-c0357626-9561-11e8-9f65-85b1189bb89b.png)\
We do the same process for the power factor parameter. We can consider the ranges (0.93, 0.94), (0.95, 0.96), (0.97,0.98), (0.99, 100) as the Healthy,  Acceptable, Critical,and Compromised states.\
However, we are going to control these two parameters simultaneously, because these two parameters at the same time specify the status of our system. As we explained before, for each of these two parameters we have 4 states and 3 actions. If we consider them as some pairs of states and action, we have 16 states and 9 actions totally. 
Then we can define our state machine and determine that from one state to another state we can go by which action.\
The reward for each action for every parameter equals to -1. In this way, if we go from one state to another one by an action which needs two changes (one for each parameter), therefore; the reward for this action equals to -2. 
For training, we assume a determined number of trials in which we have some steps to go through a specific state. During these trials, we update the values of the Q-table and train the system.\
In the online detection phase: based on the observations, the action with the lowest expected future cost (Q value) is chosen at each time using the previously learned Q-table


## Results
We need to measure 3 metrics:\
1- Data Accuracy\
2- Data Efficiency: the amount of data used for the actual controlled system during learning (to get a specific accuracy)\
3- Learning Cost: how long the algorithm needs to train (depends on the amount of computation- computation efficiency)\
For the Data Accuracy, we train the system by 40 trials and during these training process, we measure the accuracy (since we know the ground truth states). The following curve is the result for these 40 trails:\
![image](https://user-images.githubusercontent.com/20415408/43531451-a6e27324-9564-11e8-9739-1357def50366.png)\
In order to measure the data efficiency of our system, we define a different number of steps for each trial. The number of steps starts from 10 to 250 steps. In each experiment, we count the number of trails to reach a specific accuracy. Then we plot these numbers of trials (amount of data to train the system) based on the number of steps.\
![image](https://user-images.githubusercontent.com/20415408/43533158-01129eb0-9569-11e8-8f4f-e3b69265b461.png)
In the end, for measuring the learning cost, we run each trial until it goes to state 0 (safe state) and measures the total reward. We observe that after around 40 trials, our system converges.\
![image](https://user-images.githubusercontent.com/20415408/43532802-33780dd2-9568-11e8-8202-664aa096149f.png)


