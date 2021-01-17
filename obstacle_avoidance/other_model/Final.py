from __future__ import print_function
import math
import random
from numpy import inf
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import time
import numpy as np
import robobo
import cv2
import sys
import signal
# import prey
import csv
import sys
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


if not sys.warnoptions:
    import warnings
    warnings.simplefilter("ignore")

def terminate_program(signal_number, frame):
    print("Ctrl-C received, terminating program")
    sys.exit(1)

signal.signal(signal.SIGINT, terminate_program)
# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display


# if gpu is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def save_memory(self, path):


        with open(path, 'w', newline='') as myfile:
            writer = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            for val in self.memory:
                writer.writerow([val])


    def __len__(self):
        return len(self.memory)

class DQN(torch.nn.Module):
    def __init__(self, input_size, hidden_size_1, hidden_size_2, n_actions):
        super(DQN, self).__init__()
        self.input_size = input_size
        self.hidden_size_1 = hidden_size_1
        self.hidden_size_2 = hidden_size_2
        self.n_actions = n_actions
        self.fc1 = torch.nn.Linear(self.input_size, self.hidden_size_1)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(self.hidden_size_1, self.hidden_size_2)
        # self.softmax = torch.nn.Softmax(dim=1)
        self.fc3 = torch.nn.Linear(self.hidden_size_2, self.n_actions)

    def forward(self, x):
        hidden_1 = self.fc1(x)
        relu_1 = self.relu(hidden_1)
        hidden_2 = self.fc2(relu_1)
        relu_2 = self.relu(hidden_2)
        output = self.fc3(relu_2)
        # output = self.softmax(output)

        return output.view(output.size(0), -1)



def optimize_model(BATCH_SIZE, memory, policy_net, target_net, optimizer, GAMMA=0.999):

    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # print(f'state batch: {state_batch}')
    # print(f'action batch: {action_batch}')
    # print(f'NN output: {policy_net(state_batch)}')
    state_action_values = policy_net(state_batch).gather(1, action_batch)
    # print(f'State action values: {state_action_values}')
    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1)[0].
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch
    # print(f'State  expected_state_action_values: {expected_state_action_values}')
    # Compute Huber loss
    loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()

    for param in policy_net.parameters():
        param.grad.data.clamp_(-1, 1)

    optimizer.step()

    return loss.item()


def select_action(state, EPS_END, EPS_START, EPS_DECAY, n_actions, policy_net, steps=None, validation=False):

    steps_done = steps
    if validation:
        with torch.no_grad():

            return torch.tensor([[policy_net(state).argmax()]], device=device)
    else:
        # global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                print('FROM NN')
                return torch.tensor([[policy_net(state).argmax()]], device=device), steps_done
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long), steps_done


def plot_rewards(robot, rewards_all, average_per = 100):

    if robot == "":
        scene = 'Training. Scene: Only walls'
    elif robot == "#0":
        scene = 'Training. Scene: Walls and obstacles'
    elif robot == '#2':
        scene = 'Training. Scene: Maze'
    else:
        scene = 'Validation'
    plt.figure(5)
    plt.clf()
    rewards = torch.tensor(rewards_all, dtype=torch.float, device=device)
    plt.title(scene)
    plt.xlabel('Steps')
    plt.ylabel('Reward')
    plt.plot(rewards.cpu().numpy())

    # Take 100 episode averages and plot them too
    if len(rewards) >= average_per:
        means = rewards.unfold(0, average_per, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(average_per-1, device=device), means))
        plt.plot(means.cpu().numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())


def plot_loss(robot, loss_all, average_per = 100):
    if robot == "":
        scene = 'Training. Scene: Only walls'
    elif robot == "#0":
        scene = 'Training. Scene: Walls and obstacles'
    elif robot == '#2':
        scene = 'Training. Scene: Maze'
    else:
        scene = 'Validation'
    plt.figure(10)
    plt.clf()
    loss = torch.tensor(loss_all, dtype=torch.float, device=device)
    plt.title(scene)
    plt.xlabel('Steps')
    plt.ylabel('Loss')

    plt.plot(loss.cpu().numpy())
    # Take 100 episode averages and plot them too
    if len(loss) >= average_per:
        means = loss.unfold(0, average_per, 1).mean(1).view(-1)
        means = torch.cat((torch.zeros(average_per-1, device=device), means))
        plt.plot(means.cpu().numpy())

    plt.pause(0.001)  # pause a bit so that plots are updated
    if is_ipython:
        display.clear_output(wait=True)
        display.display(plt.gcf())

def get_reward(action, Dictionary, rob):

    # Get speeds for each wheel
    action_left = Dictionary[action][0]
    action_right = Dictionary[action][1]


    # Make a move with those speeds
    rob.play_simulation()
    rob.move(action_left, action_right, 1500)
    # next_state = torch.tensor([rob.position()])

    # Read sensors
    v_sens = np.log(rob.read_irs()) / 10
    v_sens[v_sens == -inf] = 0.5
    v_sens[v_sens == inf] = 0.5
    next_state = torch.tensor([list(v_sens)], dtype=torch.float, device=device)

    # Calculate s_rot to punish for too much rotation

    s_rot = abs(abs(action_left)-abs(action_right))/100

    # Calculate the reward

    ## Calculate if the robot is next to the obstacle
    if sum(v_sens) == 0.5*8:
        V = 0
    elif sum(i < 0 for i in v_sens) == 3:
        print('yes')
        V = 0.5
    else:
        V = 1

    ## Punishment for moving backwards
    if action_left == -25 and action_right == -25:
        s_back = 0.1
    else:
        s_back = 0

    ## In case if the robot is close to the obstacle reward is zero, in the other case formula

    if V == 0 or V == 0.5:
        reward = 1*(1-s_rot/2)*(1-V) - s_back
    else:
        reward = 0

    return reward, next_state, action_left, action_right, v_sens


def training(Dictionary, connection_address, num_runs_per_episode=5, num_episodes=2,
             eps_end=0.05, eps_start=0.9, eps_decay=2000,
             batch_size=64, memory_capacity=30000, size_hidden_layer_1=100, size_hidden_layer_2=100, target_update=20,
             average_for_graphs=100, gamma=0.999
             ):

    # Initialize NN and Memory
    policy_net = DQN(8, size_hidden_layer_1, size_hidden_layer_2, len(Dictionary)).to(device)
    target_net = DQN(8, size_hidden_layer_1, size_hidden_layer_2, len(Dictionary)).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.RMSprop(policy_net.parameters())
    memory = ReplayMemory(memory_capacity)

    rewards_all, loss_all = [], []
    steps_done = 0

    # Training

    for robot in ["", '#0', '#2']:

        if robot == "":
            print('SCENE 1: ONLY WALLS')
        elif robot == '#0':
            print('SCENE 2: WALLS AND OBSTACLES')
        elif robot == '#2':
            print('SCENE 3: MAZE')

        rob = robobo.SimulationRobobo(robot)
        rob.connect(address=connection_address, port=19997)
        # plt.ion()

        for i_episode in range(num_episodes):
            # print("using device " + str(device))
            rob.play_simulation()
            #Get the state
            # state = torch.tensor([rob.position()])
            rob.play_simulation()
            v_sens = np.log(rob.read_irs()) / 10
            v_sens[v_sens == -inf] = 0.5
            v_sens[v_sens == inf] = 0.5
            state = torch.tensor([list(v_sens)], dtype=torch.float, device=device)

            for run in range(num_runs_per_episode):

                # Select and perform an action, get the reward and the next state
                action, steps_done = select_action(state, eps_end, eps_start, eps_decay, len(Dictionary),steps=steps_done, policy_net=policy_net, validation=False)

                reward, next_state, action_left, action_right, v_sens = get_reward(action.item(), Dictionary, rob)
                reward = torch.tensor([reward], device=device, dtype=torch.float)

                # Plot reward
                rewards_all.append(reward.item())
                plot_rewards(robot, rewards_all, average_per=average_for_graphs)

                if run % num_runs_per_episode == num_runs_per_episode - 1:
                    next_state = None
                else:
                    next_state = next_state

                # Store the transition in memory and write to csv file
                memory.push(state, action, next_state, reward)
                memory.save_memory(r'./data_saved_during_training_validation/Training_Experience_Memory.csv')

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the target network)
                loss = optimize_model(batch_size, memory, policy_net, target_net, optimizer, GAMMA=gamma)


                # Plot training loss
                if len(rewards_all) >= BATCH_SIZE:
                    loss_all.append(loss)
                    plot_loss(robot, loss_all, average_per=average_for_graphs)
                if run % num_runs_per_episode == num_runs_per_episode - 1:
                    print('STOP')
                    rob.stop_world()
                    break

                print(f'EPISODE: {i_episode} \t RUN: {run} \t Speed - left wheel: {action_left} \t right wheel: {action_right} \t Reward: {reward.item()} \t Loss: {loss} '
                      f'\t Sensors: {v_sens}'
                      )
                # Update the target network, copying all weights and biases in DQN
            if i_episode % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        plt.figure(5).savefig(f'./images/Training_Reward_Robot{robot}')
        plt.figure(10).savefig(f'./images/Training_Loss_Robot{robot}')
        plt.ioff()
        rewards_all, loss_all = [], []
       # steps_done = 0

        rob.disconnect()

    # Save the weights
    PATH = './data_saved_during_training_validation/policy_net.pth'
    torch.save(policy_net.state_dict(), PATH)
    PATH = './data_saved_during_training_validation/target_net.pth'
    torch.save(target_net.state_dict(), PATH)

    print('FINISHED TRAINING')


def validation(Dictionary, connection_address, repeats = 3, runs_per_repeat = 5, eps_end=0.05,
               eps_start=0.9, eps_decay=2000, size_hidden_layer_1=100, size_hidden_layer_2=100, average_for_graphs_validation=100):

    # Initialize NN and load the state
    policy_net = DQN(8, size_hidden_layer_1, size_hidden_layer_2, len(Dictionary)).to(device)
    policy_net.load_state_dict(torch.load('./data_saved_during_training_validation/policy_net.pth'))

    # Initialize robot
    rob = robobo.SimulationRobobo('#1')
    rob.connect(address=connection_address, port=19997)
    plt.ion()
    rewards_all = []

    print('STARTED VALIDATION')

    for repeat in range(repeats):

        rob.play_simulation()
        #Get the state

        # state = torch.tensor([rob.position()])
        rob.play_simulation()
        v_sens = np.log(rob.read_irs()) / 10
        v_sens[v_sens == -inf] = 0.5
        v_sens[v_sens == inf] = 0.5
        state = torch.tensor([list(v_sens)], dtype=torch.float, device=device)

        for run in range(runs_per_repeat):

            # Select and perform an action
            action = select_action(state, eps_end, eps_start, eps_decay, n_actions=len(Dictionary),
                                   policy_net=policy_net, validation=True)
            reward, next_state, action_left, action_right, v_sens = get_reward(action.item(), Dictionary, rob)
            reward = torch.tensor([reward], device=device)

            # Plot reward
            rewards_all.append(reward.item())
            plot_rewards('#1', rewards_all, average_per=average_for_graphs_validation)

            with open(f'./data_saved_during_training_validation/Validation_rewards_repeat_{repeat}.csv', 'w', newline='') as myfile:
                writer = csv.writer(myfile, quoting=csv.QUOTE_ALL)
                for val in rewards_all:
                    writer.writerow([val])

            if run % runs_per_repeat == runs_per_repeat - 1:
                next_state = None
            else:
                next_state = next_state

            # Move to the next state
            state = next_state

            if run % runs_per_repeat == runs_per_repeat - 1:

                print('STOP')
                rob.stop_world()
                break

            print(
                f'REPEAT: {repeat} \t RUN: {run} \t Speed - left wheel: {action_left} \t right wheel: {action_right} \t Reward: {reward.item()} '
                f'\t Sensors: {v_sens}'
                )

        plt.savefig(f'./images/Validation_reward_repeat_{repeat}')
        plt.ioff()


# CONSTANTS
DICTIONARY = {0:[25, 25],
              1:[0, 40],
              2:[40, 0],
              3:[-25, -25]
              }

CONNECTION_ADDRESS = '127.0.0.1'
NUMBER_OF_RUNS_PER_EPISODE = 30
NUMBER_OF_EPISODES = 200
EPS_END = 0.05
EPS_START = 0.9
EPS_DECAY = 3000
GAMMA = 0.999

MEMORY_CAPACITY = 30000
SIZE_OF_HIDDEN_LAYER_1 = 6 + 30
SIZE_OF_HIDDEN_LAYER_2 = 20 + len(DICTIONARY)
TARGET_UPDATE = 10
AVERAGE_VALUES_FOR_GRAPHS_LOSS_REWARDS = 150
AVERAGE_VALUES_FOR_GRAPH_REWARDS_VALIDATION = 150
BATCH_SIZE = 32

REPEATS_FOR_VALIDATION = 3
RUNS_PER_REPEAT_IN_VALIDATION = 2000


#TRAINING
training(
    Dictionary=DICTIONARY, connection_address=CONNECTION_ADDRESS,
    num_runs_per_episode=NUMBER_OF_RUNS_PER_EPISODE,
    num_episodes=NUMBER_OF_EPISODES,
    eps_end=EPS_END, eps_start=EPS_START, eps_decay=EPS_DECAY, gamma=GAMMA,
    batch_size=BATCH_SIZE, memory_capacity=MEMORY_CAPACITY, size_hidden_layer_1=SIZE_OF_HIDDEN_LAYER_1,
    size_hidden_layer_2=SIZE_OF_HIDDEN_LAYER_2, target_update=TARGET_UPDATE,
    average_for_graphs=AVERAGE_VALUES_FOR_GRAPHS_LOSS_REWARDS
    )

#VALIDATION
validation(
    Dictionary=DICTIONARY,
    connection_address=CONNECTION_ADDRESS,
    repeats=REPEATS_FOR_VALIDATION,
    runs_per_repeat=RUNS_PER_REPEAT_IN_VALIDATION,
    size_hidden_layer_1=SIZE_OF_HIDDEN_LAYER_1,
    size_hidden_layer_2=SIZE_OF_HIDDEN_LAYER_2,
    average_for_graphs_validation=AVERAGE_VALUES_FOR_GRAPH_REWARDS_VALIDATION
    )

