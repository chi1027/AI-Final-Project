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
        self.input_state = 12  # the dimension of state space
        self.num_actions = num_actions  # the dimension of action space
        self.fc1 = nn.Linear(self.input_state, 64)  # input layer
        self.fc2 = nn.Linear(64, hidden_layer_size)  # hidden layer
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
    def __init__(self, game, learning_rate=0.001, GAMMA=0.95, batch_size=32, capacity=10000):
        self.game = game
        self.epsilon = 1.0
        self.eps_discount = 0.98
        self.min_eps = 0.001     
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.batch_size = batch_size
        self.capacity = capacity
        self.n_action = 4
        self.count = 0

        self.buffer = replay_buffer(self.capacity)
        self.evaluate_net = Net(self.n_action)
        self.target_net = Net(self.n_action)
        self.optimizer = torch.optim.Adam(
            self.evaluate_net.parameters(), lr=self.learning_rate)
        self.n_game = 0
        self.action = {
            0: "left", 
            1: "right",
            2: "up",
            3: "down"
        }
        self.losses = []

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
                    action = self.action[int(torch.argmax(self.evaluate_net(x)))]
                # End your code
        return action

    def learn(self):
        if self.count % 100 == 0:
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

    plot_scores = []
    plot_mean_scores = []
    total_score = 0

    state = game.get_state()
    for i in tqdm(range(1000)):
        agent.epsilon = max(agent.epsilon * agent.eps_discount, agent.min_eps)
        game.play = True
        while game.play:
            game.clock.tick(fps)
            agent.count += 1

            # get current state
            state = game.get_state()

            # get move
            action = agent.choose_action(state)
            action_idx = list(agent.action.values()).index(action)

            # perform move and get new state
            reward, done, reason = game.move_snake(action)

            # get next state
            next_state = game.get_state()
            agent.buffer.insert(state, action_idx, reward, next_state, int(done))

            if agent.count >= 1000:
                agent.learn()

            game.update_frames_since_last_fruit()

            if done:
                agent.n_game += 1
                total_score += game.score
                mean_score = total_score / agent.n_game
                plot_scores.append(game.score)
                plot_mean_scores.append(mean_score)
                # print(f"Games: {agent.n_game}; Score: {game.score}; Reason: {reason}")
                game.game_over()

            if game.restart == True:
                game.restart = False
                continue
                
            # game.redraw_window()
            game.event_handler()
    torch.save(agent.target_net.state_dict(), f"./Tables/DQN.pt")
    print("Mean Score: {}, Highest Score: {}".format(
        total_score/agent.n_game, game.high_score))
    print("-" * 20)
    plot(plot_scores, plot_mean_scores, "DQN_train")

def test():
    fps = 3000
    game = SnakeGame(fps)
    testing_agent = Agent(game)
    testing_agent.target_net.load_state_dict(torch.load(f"./Tables/DQN.pt"))
    
    total_score = 0
    plot_scores = []
    plot_mean_score = []
    num = 100
    for _ in tqdm(range(num)):
        game.play = True
        while game.play:
            game.clock.tick(fps)
            # get current state
            state = game.get_state()
            # get move
            x = torch.tensor(state).to(torch.float32)
            action = testing_agent.action[
                int(torch.argmax(testing_agent.target_net(x)))
            ]
            # perform move and get new state
            reward, done, reason = game.move_snake(action)

            # get next state
            next_state = game.get_state()

            game.update_frames_since_last_fruit()

            if done:
                testing_agent.n_game += 1
                total_score += game.score
                mean_score = total_score / testing_agent.n_game
                plot_scores.append(game.score)
                plot_mean_score.append(mean_score)
                # print(f"Games: {i + 1}; Score: {game.score}; Reason: {reason}")
                game.game_over()

            if game.restart == True:
                game.restart = False
                continue
                
            game.redraw_window()
            game.event_handler()

            state = next_state
    print("-" * 20)
    print("Mean Score: {:.1f}, Highest Score: {}".format(
        total_score/(num), game.high_score))

def display():
    fps = 30
    game = SnakeGame(fps)
    testing_agent = Agent(game)
    testing_agent.target_net.load_state_dict(torch.load("./Tables/DQN.pt"))
    
    total_score = 0
    plot_scores = []
    plot_mean_score = []
    num = 10

    for i in range(num):
        game.play = True
        while game.play:
            game.clock.tick(fps)
            # get current state
            state = game.get_state()
            # get move
            x = torch.tensor(state).to(torch.float32)
            action = testing_agent.action[
                int(torch.argmax(testing_agent.target_net(x)))
            ]
            # perform move and get new state
            reward, done, reason = game.move_snake(action)

            # get next state
            next_state = game.get_state()

            game.update_frames_since_last_fruit()

            if done:
                testing_agent.n_game += 1
                total_score += game.score
                mean_score = total_score / testing_agent.n_game
                plot_scores.append(game.score)
                plot_mean_score.append(mean_score)
                print(f"Games: {i + 1}; Score: {game.score}; Reason: {reason}")
                game.game_over()

            if game.restart == True:
                game.restart = False
                continue
                
            game.redraw_window()
            game.event_handler()

            state = next_state
    print("-" * 20)
    print("Mean Score: {:.1f}, Highest Score: {}".format(
        total_score/(num), game.high_score))

def seed(seed=20):
    '''
    It is very IMPORTENT to set random seed for reproducibility of your result!
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
if __name__ == "__main__":
    seed(100)
    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")
    # train()
    # test()
    display()
        