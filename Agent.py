#*********************************************************************************
#trainGeneticAlgorithm.py
#Author: Craig Haber
#5/9/2020
#This program was used to train a genetic algorithm in order to create 
#the intelligent Snake Game agents that can be observed in test_trained_agents.py
#For more detailed information, check out:
#https://craighaber.github.io/AI-for-Snake-Game/
#*********************************************************************************
#Instructions: 
#Simply run the module to observe how the genetic algorithm was trained in action,
#starting from randomized chromosomes. 
#Specific information about each population is saved in Gadata.txt in the same 
#folder as this program, and the file will be created if it does not already exist
#after the first generation.
#Also, for every 10 populations, the population is saved in a file 
#in the populations directory.
#*********************************************************************************
#Dependecies: 
#
#To run this module, you must have the module pygame installed.
#Type pip install pygame in the command prompt or terminal to install it.
#If necessary, more specific instructions for installing pygame are here:
#https://www.pygame.org/wiki/GettingStarted 
#
#Also, a Python version of 3.7 or higher is required.
#*********************************************************************************
import os
import random
import numpy as np
import pygame
from tqdm import tqdm
from helpers.snakeGame import SnakeGame
from helpers.plot import plot

pygame.font.init()

"""
self.epsilon = 0.1
self.lr = 0.7
"""

class Agent():
    def __init__(self, game, learning_rate=0.01, GAMMA=0.95):
        self.game = game
        self.epsilon = 1.0
        self.eps_discount = 0.98
        self.min_eps = 0.001
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.n_game = 0
        self.action = {
            0: "left", 
            1: "right",
            2: "up",
            3: "down"
        }

        self.qtable = np.zeros((2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 4))

    def choose_action(self, state):
        if len(self.game.snake.directions) == 0:
            action = "right"
        else:
            if random.random() < self.epsilon:
                action = self.action[random.randint(0,3)]
            else:
                action = self.action[np.argmax(self.qtable[tuple(state)])]
        return action

    def learn(self, state, action, reward, next_state, done):
        action_idx = list(self.action.values()).index(action)
        cur_q = self.qtable[tuple(state) + (action_idx,)]
        max_q = 0 if done else np.max(self.qtable[tuple(next_state)])
        new_q = (1 - self.learning_rate) * cur_q + self.learning_rate * (reward + self.gamma * max_q)
        self.qtable[tuple(state) + (action_idx, )] = new_q


def train():
    fps = 3000
    game = SnakeGame(fps)
    agent = Agent(game)

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0

    for _ in tqdm(range(5000)):
        agent.epsilon = max(agent.epsilon * agent.eps_discount, agent.min_eps)
        game.play = True
        while game.play:
            agent.game.clock.tick(game.fps)

            if agent.n_game > 100:
                agent.epsilon = 0
            else:
                agent.epsilon = 0.1

            # get current state
            state = game.get_state()

            # get move
            action = agent.choose_action(state)

            # perform move and get new state
            reward, done, reason = game.move_snake(action)

            next_state = game.get_state()

            agent.learn(state, action, reward, next_state, done)

            #check if snake is killed for not eating a fruit in a while
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
        
    np.save(f"./Tables/snake_table.npy", agent.qtable)
    print("-" * 20)
    print(f"Average Score: {mean_score}, Highest Score: {agent.game.high_score}")
    plot(plot_scores, plot_mean_scores)

def test():
    fps = 3000
    game = SnakeGame(fps)
    game.high_score = 0
    testing_agent = Agent(game)
    testing_agent.qtable = np.load(f"./Tables/snake_table.npy")

    total_score = 0

    for i in range(10):
        state = game.get_state()
        testing_agent.epsilon = 0
        while game.play:
            testing_agent.game.clock.tick(game.fps)

            # choose action
            action = testing_agent.choose_action(state)

            # move snake
            reward, done, reason = testing_agent.game.move_snake(action)

            # get next state
            next_state = game.get_state()

            #check if snake is killed for not eating a fruit in a while
            testing_agent.game.update_frames_since_last_fruit()

            if done:
                testing_agent.n_game += 1
                total_score += game.score
                print(f"Games: {testing_agent.n_game}; Score: {game.score}; Reason: {reason}")
                game.game_over()
                break

            if game.restart == True:
                game.restart = False
                continue

            game.redraw_window()
            game.event_handler()

            state = next_state
    print("-" * 20)
    print("Mean Score: {}, Highest Score: {}".format(
        total_score/testing_agent.n_game, game.high_score))

def seed(seed=40):
    '''
    It is very IMPORTENT to set random seed for reproducibility of your result!
    '''
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)

if __name__ == "__main__":
    seed()
    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")
    train()
    test()