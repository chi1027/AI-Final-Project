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
from helpers import snake
import pygame
from tqdm import tqdm
from helpers.snakeGameQL import SnakeGameQL

directs = ["left", "right", "up", "down"]

class Agent():
    def __init__(self, game, epsilon=0.05, learning_rate=0.5, GAMMA=0.97):
        self.game = game
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA

        self.qtable = np.zeros((10, 10, 9, 4))

    def get_state(self):
        fruit = self.game.fruit_pos
        head = self.game.snake.body[0]

        if fruit == head:
            fruit_direct = 0
        elif fruit[0] > head[0] and fruit[1] > head[1]:
            fruit_direct = 1
        elif fruit[0] > head[0] and fruit[1] < head[1]:
            fruit_direct = 2
        elif fruit[0] < head[0] and fruit[1] < head[1]:
            fruit_direct = 3
        elif fruit[0] < head[0] and fruit[1] > head[1]:
            fruit_direct = 4
        elif fruit[0] == head[0] and fruit[1] < head[1]:
            fruit_direct = 5  # x axis left
        elif fruit[0] == head[0] and fruit[1] > head[1]:
            fruit_direct = 6  # x axis right
        elif fruit[1] == head[1] and fruit[0] < head[0]:
            fruit_direct = 7  # y axis left
        elif fruit[1] == head[1] and fruit[0] > head[0]:
            fruit_direct = 8  # y axis right
        return (head[0], head[1], fruit_direct)

    def choose_action(self, state):
        if len(self.game.snake.directions) == 0:
            action = "right"
        else:
            if random.random() < self.epsilon:
                action = directs[random.randint(0,3)]
            else:
                action = directs[np.argmax(self.qtable[tuple(state)])]
        return action

    def learn(self, state, action, reward, next_state, done):
        # print(tuple(state) + (action,))
        cur_q = self.qtable[tuple(state) + (action,)]
        max_q = 0 if done else np.max(self.qtable[tuple(next_state)])
        new_q = (1 - self.learning_rate) * cur_q + self.learning_rate * (reward + self.gamma * max_q)
        self.qtable[tuple(state) + (action, )] = new_q


def train():
    fps = 1
    game = SnakeGameQL(fps)
    training_agent = Agent(game)
    pygame.font.init()
    

    while game.play:
        # get current state
        state = training_agent.get_state()

        # get move
        action = training_agent.choose_action(state)

        # perform move and get new state
        reward, done = training_agent.game.move_snake(action)

        next_state = training_agent.get_state()

        training_agent.learn(
            state, directs.index(action), reward, next_state, done)

        #check if snake is killed for not eating a fruit in a while
        training_agent.game.update_frames_since_last_fruit()

        if training_agent.game.restart == True:
            training_agent.game.restart = False
            continue
    
        training_agent.game.redraw_window()
        training_agent.game.event_handler()

def test():
        pass

def main():
    train()
        
if __name__ == "__main__":	
    main()