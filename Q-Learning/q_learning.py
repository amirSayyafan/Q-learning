import numpy as np
import random
import matplotlib.pyplot as plt
import time


def generate_random_state():
    '''
    generate a random value for initial state
    '''
    state = random.randint(0, 15)
    return state


def select_action(current_state, N):
    '''
    select the action randomly
    '''
    flag_accept = 0
    while (flag_accept == 0):
        action = random.randint(0, 8)
        next_state = N[current_state][action]
        if next_state  != -1:
            flag_accept = 1
    return next_state, action


def find_max(next_state, Q):
    '''
    find the max reward for the next action in order to update Q table
    '''
    max_value = Q[next_state][1]
    for i in range (1, 9):
        if max_value < Q[next_state][i]:
            max_value = Q[next_state][i]
    return max_value


def update_q_table(current_state, next_state, action, R, Q, gamma):
    '''
    update Q table based on the current state and action:
        Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
    '''
    max_value = find_max(next_state, Q)
    cost = R[current_state][action] + max_value - Q[current_state][action]
    rew = R[current_state][action] + max_value - Q[current_state][action]
    Q[current_state][action] = Q[current_state][action] + gamma * (R[current_state][action] + max_value - Q[current_state][action])
    return Q, cost, rew


def training(N, R, Q, A, num_steps, num_trials, gamma, flag_num_steps):
    updated_Q = Q
    result, index, time_itr, err = [], [], [], []
    if (flag_num_steps == 0):
        for i in range(num_trials):
            start = time.time()
            initial_state = generate_random_state()
            current_state = initial_state
            cost , rew = 0, 0
            while (current_state != 0):
                next_state, action = select_action(current_state, N)
                updated_Q, cost_per_itr, rew_per_itr = update_q_table(current_state, next_state, action, R, updated_Q, gamma)
                cost = cost + cost_per_itr
                rew = rew + rew_per_itr
                current_state = next_state
            end = time.time()
            time_per_itr = end - start
            time_itr.append(time_per_itr)
            result.append(cost)
            index.append(i)
            err.append(rew - A[initial_state])
    else:
        i , cost = 0, -10
        while (cost < -5):
            start = time.time()
            initial_state = generate_random_state()
            current_state = initial_state
            cost, rew = 0, 0
            for j in range(num_steps):
                next_state, action = select_action(current_state, N)
                updated_Q, cost_per_itr, rew_per_itr = update_q_table(current_state, next_state, action, R, updated_Q, gamma)
                cost = cost + cost_per_itr
                rew = rew + rew_per_itr
                current_state = next_state
            end = time.time()
            time_per_itr = end - start
            time_itr.append(time_per_itr)
            result.append(cost)
            i = i + 1
            index.append(i)
            err.append(rew - A[initial_state])
    return updated_Q, result, index, time_itr, err


def getQ(state, action):
        return Q.get((state, action), 0.0)


def learnQ(state, action, reward, value, alpha):
    '''
    Q-learning:
        Q(s, a) += alpha * (reward(s,a) + max(Q(s') - Q(s,a))
    '''
    oldv = Q.get((state, action), None)
    if oldv is None:
        Q[(state, action)] = reward
    else:
        Q[(state, action)] = oldv + alpha * (value - oldv)


def learn(self, state1, action1, reward, state2):
    maxqnew = max([self.getQ(state2, a) for a in self.actions])
    self.learnQ(state1, action1, reward, reward + self.gamma*maxqnew)


def chooseAction(state, epsilon, return_q=False):
    q = [getQ(state, a) for a in Q.actions]
    maxQ = max(q)

    if random.random() < epsilon:
        minQ = min(q); mag = max(abs(minQ), abs(maxQ))
        # add random values to all the actions, recalculate maxQ
        q = [q[i] + random.random() * mag - .5 * mag for i in range(len(Q.actions))]
        maxQ = max(q)

    count = q.count(maxQ)
    # In case there're several state-action max values
    # we select a random one among them
    if count > 1:
        best = [i for i in range(len(Q.actions)) if q[i] == maxQ]
        i = random.choice(best)
    else:
        i = q.index(maxQ)

    action = Q.actions[i]
    if return_q: # if they want it, give it!
        return action, q
    return action





if __name__ == "__main__":
    # N : matrix contains next states
    # R: matrix contains rewards
    # A: matrix contains the final rewards for each state
    # num_steps: number of steps for each trail
    N = np.array([[0, 1, -1, 4, 5, -1, -1, -1, -1],
         [1, 2, 0, 5, 6, 4, -1, -1, -1],
         [2, 3, 1, 6, 7, 5, -1, -1, -1],
         [3, 3, 2, 7, 7, 6, -1, -1, -1],
         [4, 5, -1, 8, 9, -1, 0, 1, -1],
         [5, 6, 4, 9, 10, 8, 1, 2, 0],
         [6, 7, 5, 10, 11, 9, 2, 3, 1],
         [7, 7, 6, 11, 11, 10, 3, 3, 2],
         [8, 9, -1, 12, 13, -1, 4, 5, -1],
         [9, 10, 8, 13, 14, 12, 5, 6, 4],
         [10, 11, 9, 14, 15, 13, 6, 7, 5],
         [11, 11, 10, 15, 15, 14, 7, 7, 6],
         [12, 13, -1, 12, 13, -1, 8, 9, -1],
         [13, 14, 12, 13, 14, 12, 9, 10, 8],
         [14, 15, 13, 14, 15, 13, 10, 11, 9],
         [15, 15, 14, 15, 15, 14, 11, 11, 10]])


    R = np.array([[0, -1, -3, -1, -2, -3, -3, -3, -3],
         [0, -1, -1, -1, -2, -2, -3, -3, -3],
         [0, -1, -1, -1, -2, -2, -3, -3, -3],
         [0, -1, -1, -1, -2, -2, -3, -3, -3],
         [0, -1, -3, -1, -2, -3, -1, -2, -3],
         [0, -1, -1, -1, -2, -2, -1, -2, -2],
         [0, -1, -1, -1, -2, -2, -1, -2, -2],
         [0, -1, -1, -1, -2, -2, -1, -2, -2],
         [0, -1, -3, -1, -2, -3, -1, -2, -3],
         [0, -1, -1, -1, -2, -2, -1, -2, -2],
         [0, -1, -1, -1, -2, -2, -1, -2, -2],
         [0, -1, -1, -1, -2, -2, -1, -2, -2],
         [0, -1, -3, -1, -2, -3, -1, -2, -3],
         [0, -1, -1, -1, -2, -2, -1, -2, -2],
         [0, -1, -1, -1, -2, -2, -1, -2, -2],
         [0, -1, -1, -1, -2, -2, -1, -2, -2]])

    A = [0 , -1, -2, -3,
         -1, -2, -3, -4,
         -2, -3, -4, -5,
         -3, -4, -5, -6]


    ######################################################### data accuracy and learning cost
    flag_num_steps = 0
    Q = np.zeros((16, 9))
    num_steps = 24
    num_trials = 40
    gamma = 0.8
    Q_table, result, index, time_itr, err = training (N, R, Q, A, num_steps, num_trials, gamma, flag_num_steps)
    plt.plot(np.array(index), np.array(result))
    # naming the x axis
    plt.xlabel('Trials')
    # naming the y axis
    plt.ylabel('Trial Reward')

    # giving a title to my graph
    plt.title('Learning Cost')

    # function to show the plot
    plt.show()

    error = [abs(number) for number in err]
    plt.plot(np.array(index), np.ones(len(error)) - (np.array(error)/max(error)))
    # naming the x axis
    plt.xlabel('Trials')
    # naming the y axis
    plt.ylabel('Accuracy')

    # giving a title to my graph
    plt.title('Data Accuracy')

    # function to show the plot
    plt.show()

    ######################################################### data efficiency
    data_efficiency, k_index = [], []
    flag_num_steps = 1
    final_num_steps = 250
    for k in range(10, final_num_steps):
        Q = np.zeros((16, 9))
        num_steps = k
        num_trials = 0
        gamma = 0.8
        Q_table, result, index, time_itr, acc = training (N, R, Q, A, num_steps, num_trials, gamma, flag_num_steps)
        data_efficiency.append(index[-1])
        k_index.append(k)
    plt.plot(np.array(k_index), np.array(data_efficiency))
    # naming the x axis
    plt.xlabel('Number of Steps to Train')
    # naming the y axis
    plt.ylabel('Amount of Data(Number of Trials)')

    # giving a title to my graph
    plt.title('Data Efficiency')

    # function to show the plot
    plt.show()


