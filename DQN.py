import os
import random
import numpy as np
import pygame
from tqdm import tqdm
from helpers.snakeGame import SnakeGame
from helpers.plot import plot

import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

pygame.font.init()

class replay_buffer():
    '''
    A deque storing trajectories
    '''

    def __init__(self, capacity):
        self.capacity = capacity  # the size of the replay buffer
        self.memory = deque(maxlen=capacity)  # replay buffer itself

    def insert(self, state, action, reward, next_state, done):
        '''
        Insert a sequence of data gotten by the agent into the replay buffer.

        Parameter:
            state: the current state
            action: the action done by the agent
            reward: the reward agent got
            next_state: the next state
            done: the status showing whether the episode finish
        
        Return:
            None
        '''
        self.memory.append([state, action, reward, next_state, done])

    def sample(self, batch_size):
        '''
        Sample a batch size of data from the replay buffer.

        Parameter:
            batch_size: the number of samples which will be propagated through the neural network
        
        Returns:
            observations: a batch size of states stored in the replay buffer
            actions: a batch size of actions stored in the replay buffer
            rewards: a batch size of rewards stored in the replay buffer
            next_observations: a batch size of "next_state"s stored in the replay buffer
            done: a batch size of done stored in the replay buffer
        '''
        batch = random.sample(self.memory, batch_size)
        observations, actions, rewards, next_observations, done = zip(*batch)
        return observations, actions, rewards, next_observations, done


class Net(nn.Module):
    '''
    The structure of the Neural Network calculating Q values of each state.
    '''

    def __init__(self,  num_actions, hidden_layer_size=50):
        super(Net, self).__init__()
        self.input_state = 3  # the dimension of state space
        self.num_actions = num_actions  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, 32)  # input layer
        self.fc2 = nn.Linear(32, hidden_layer_size)  # hidden layer
        self.fc3 = nn.Linear(hidden_layer_size, num_actions)  # output layer

    def forward(self, states):
        '''
        Forward the state to the neural network.
        
        Parameter:
            states: a batch size of states
        
        Return:
            q_values: a batch size of q_values
        '''
        x = F.relu(self.fc1(states))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)
        return q_values

class Agent():
    def __init__(self, game, epsilon=0.1, learning_rate=0.002, GAMMA=0.8, batch_size=32, capacity=10000):
        self.game = game
        self.n_game = 0
        self.block_size = 1
        self.n_action = 4
        self.count = 0
        
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.n_action)
        self.target_net = Net(self.n_action)

        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)
        self.lossse = []

        self.action = {
            0: "left", 
            1: "right",
            2: "up",
            3: "down"
        }


    def get_state(self,game):
        head = game.snake.body[0]
        dist_x = game.fruit_pos[1] - head[1]
        dist_y = game.fruit_pos[0] - head[0]
		
        if dist_x < 0:
            pos_x = 0
        elif dist_x > 0:
            pos_x = 1
        else:
            pos_x = 2
        
        if dist_y < 0:
            pos_y = 0
        elif dist_y > 0:
            pos_y = 1
        else:
            pos_y = 2
        
        sqs = [
            (head[0],                 head[1]-self.block_size),
            (head[0],                 head[1]+self.block_size),
            (head[0]-self.block_size, head[1]),
            (head[0]+self.block_size, head[1]),
        ]

        surrounding_list = []
        for sq in sqs:
            if sq[0] < 0 or sq[1] < 0:
                surrounding_list.append('1')
            elif sq[0] >= self.game.cols or sq[1] >= self.game.rows:
                surrounding_list.append('1')
            elif sq in self.game.snake.body[1:]:
                surrounding_list.append('1')
            else:
                surrounding_list.append('0')
        surroundings = ''.join(surrounding_list)
        surroundings = int(surroundings, 2)
        
        return (pos_x, pos_y, surroundings)

    def choose_action(self, state):
        if len(self.game.snake.directions) == 0:
            action = "right"
        else:
            with torch.no_grad():
                # Begin your code
                if random.random() < self.epsilon:
                    action = self.action[random.randint(0,3)]
                else:
                    x = torch.tensor(state).to(torch.float32)
                    print(torch.argmax(self.evaluate_net(x)))
                    action = self.action[int(torch.argmax(self.evaluate_net(x)))]
                # End your code
        return action

    def learn(self):
        if self.n_game % 100 == 0:
            self.target_net.load_state_dict(self.evaluate_net.state_dict())

        batch = self.buffer.sample(self.batch_size)
        state_batch = torch.tensor(np.array(batch[0])).to(torch.float32)
        action_batch = torch.tensor(np.array(batch[1])).to(torch.int64)
        reward_batch = torch.tensor(np.array(batch[2])).to(torch.float32)
        next_state_batch = torch.tensor(np.array(batch[3])).to(torch.float32)
        done_mask = [bool(x) for x in batch[4]]

        action_index = torch.stack((action_batch, action_batch),1)
        qtable = self.evaluate_net(state_batch)
        state_action_value = qtable.gather(1, action_index)[:, 0]

        max_q_values = self.target_net(next_state_batch).max(1)[0].detach()
        max_q_values[done_mask] = 0
        expected_state_action = reward_batch + (max_q_values * self.gamma)
        criterion = nn.MSELoss()
        loss = criterion(state_action_value, expected_state_action)
        self.losses.append(loss.detach().numpy())

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # End your code

def train():
    fps = 3000
    game = SnakeGame(fps)
    agent = Agent(game)
    episode = 1000

    state = agent.get_state(agent.game)
    while agent.game.play:
        if agent.n_game > 100:
            agent.epsilon = 0
        else:
            agent.epsilon = 0.1
        agent.game.clock.tick(fps)

        # get current state
        state = agent.get_state(game)

        # get move
        action = agent.choose_action(state)
        action_idx = list(agent.action.values()).index(action)

        # perform move and get new state
        reward, done, reason = game.move_snake(action)

        # get next state
        next_state = agent.get_state(game)
        agent.buffer.insert(state, action_idx, reward, next_state, int(done))

        if agent.count >= 100:
            agent.learn()

        agent.game.update_frames_since_last_fruit(agent.n_game)

        if done:
            agent.n_game += 1
            agent.game.game_over(agent.n_game, reason)
        
        if agent.game.restart == True:
            agent.game.restart = False
            continue
            
        agent.game.redraw_window()
        agent.game.event_handler()

        if agent.n_game == 1000:
            torch.save(agent.target_net.state_dict(), "./Tables/DQN.pt")
            break


def main():
    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")
    train()
        
if __name__ == "__main__":	
    main()
        