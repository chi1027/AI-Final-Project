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
    def __init__(self, game, epsilon=0.1, learning_rate=0.75, GAMMA=0.8):
        self.game = game
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.n_game = 0
        self.block_size = 1
        self.action = {
            0: "left", 
            1: "right",
            2: "up",
            3: "down"
        }

        self.qtable = np.zeros((3, 3, 16, 4))

    def get_state(self, game):
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
    
    def is_collision(self, pt, game):
        if pt[0] < 0 or pt[0] >= game.cols or pt[1] < 0 or pt[1] >= game.cols or pt in game.snake.body[1:]:
            return True
        return False

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
        # print(self.qtable[tuple(next_state)])
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
    
    while agent.game.play:
        agent.game.clock.tick(game.fps)

        if agent.n_game > 100:
            agent.epsilon = 0
        else:
            agent.epsilon = 0.1

        # get current state
        state = agent.get_state(agent.game)

        # get move
        action = agent.choose_action(state)

        # perform move and get new state
        reward, done, reason = game.move_snake(action)

        next_state = agent.get_state(game)

        agent.learn(state, action, reward, next_state, done)

        #check if snake is killed for not eating a fruit in a while
        game.update_frames_since_last_fruit(agent.n_game)

        if done:
            agent.n_game += 1
            total_score += game.score
            mean_score = total_score / agent.n_game
            plot_scores.append(game.score)
            plot_mean_scores.append(mean_score)
            game.game_over(agent.n_game, reason)    

        if game.restart == True:
            game.restart = False
            continue
    
        game.redraw_window()
        game.event_handler()

        if agent.n_game == 500:
            np.save("./Tables/cartpole_table.npy", agent.qtable)
            print("-" * 20)
            print(f"Average Score: {mean_score}, Highest Score: {agent.game.high_score}")
            break
    plot(plot_scores, plot_mean_scores)

def test():
    fps = 15
    game = SnakeGame(fps)
    testing_agent = Agent(game)
    testing_agent.qtable = np.load("./Tables/cartpole_table.npy")

    for i in range(30):
        state = testing_agent.get_state(testing_agent.game)
        testing_agent.epsilon = 0
        scores = []
        while testing_agent.game.play:
            testing_agent.game.clock.tick(game.fps)

            # choose action
            action = testing_agent.choose_action(state)

            # move snake
            reward, done, reason = testing_agent.game.move_snake(action)

            # get next state
            next_state = testing_agent.get_state(testing_agent.game)

            #check if snake is killed for not eating a fruit in a while
            testing_agent.game.update_frames_since_last_fruit(testing_agent.n_game)

            if done:
                testing_agent.n_game += 1
                scores.append(testing_agent.game.score)
                testing_agent.game.game_over(testing_agent.n_game, reason)
                break

            if testing_agent.game.restart == True:
                testing_agent.game.restart = False
                continue

            testing_agent.game.redraw_window()
            testing_agent.game.event_handler()

            state = next_state
    print("-" * 20)
    print("Mean Score: {}, Highest Score: {}".format(sum(scores)/len(scores), testing_agent.game.high_score))

    

def main():
    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")
    # for i in range(1):
    # train()
    test()
        
if __name__ == "__main__":	
    main()