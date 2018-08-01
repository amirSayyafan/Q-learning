# Q-learning
A Model-Free Reinforcement Learning algorithm for detect and defense task in a smart power system
## Description
For each possible observation-action pair (o; a), we propose to learn a Q(o; a) value, i.e., the expected future cost, using an RL algorithm where all Q(o; a) values are stored in a Q-table. After learning the Q-table, the policy of the defender will be choosing the action a with the minimum Q(o; a) for each observation o.
## Challenges
Generally, Reinforcement Learning uses for sequential and It has some kind of complexity to map our application to it. Since we need to discrete the space of values. 
Another challenges is that since high voltage and power factor change can affect the power system in a dangerous mode, our actions are low and fixed changing the parameters.

## Details
In this step of project, we just focuse on two parameters as our input: voltage and power factor. We consider 4 states and 3 actions for each of these two parameters.\
The 4 states are: 1- Healthy 2- Acceptable 3- Critical 4- Compromised. In order to disceret the space of values for voltage and make our states we slpit the overall range of voltage into 4 ranges as we shoe in the following:\
![image](https://user-images.githubusercontent.com/20415408/43529592-7f97f8ce-9560-11e8-9700-302cd75c7eda.png)
Also, for the voltage parameter we have 3 action: 1- no defense 2- increse by fixed step 3- decrese by fixed parameter
Therefore, if we consider each paratmer we have a Q-table with 4 states and 3 actions:


