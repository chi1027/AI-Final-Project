import os
import random
import numpy as np
from helpers import snake
import pygame
from tqdm import tqdm
from helpers.snakeGame2 import SnakeGame
from helpers.plot import plot

"""
self.epsilon = 0.1
self.lr = 0.7
"""


class Agent():
    def __init__(self, game, epsilon=0.1, learning_rate=0.5, GAMMA=0.7):
        self.game = game
        self.epsilon = epsilon
        self.learning_rate = learning_rate
        self.gamma = GAMMA
        self.n_game = 0
        self.block_size = 1
        self.action = {0:0,1:1,2:2}

        self.qtable = np.zeros((3, 3, 8, 4, 3))

    def get_state(self,game):
        head = game.snake.body[0]
        dist_x = game.fruit_pos[1] - head[1]
        dist_y = game.fruit_pos[0] - head[0]

        sur = ""

        if len(self.game.snake.directions) == 0:
            dir = 1
            sur += "000" #直左右
        else:
            d = game.snake.directions[0]
            if d == "right":
                dir = 1
                if self.is_collision((head[0], head[1]+1), game):
                    sur += "1"
                else:
                    sur += "0"
                if self.is_collision((head[0]-1, head[1]), game):
                    sur += "1"
                else:
                    sur += "0"
                if self.is_collision((head[0]+1, head[1]), game):
                    sur += "1"
                else:
                    sur += "0"
                
            elif d == "left":
                dir = 0
                if self.is_collision((head[0], head[1]-1), game):
                    sur += "1"
                else:
                    sur += "0"
                if self.is_collision((head[0]+1, head[1]), game):
                    sur += "1"
                else:
                    sur += "0"
                if self.is_collision((head[0]-1, head[1]), game):
                    sur += "1"
                else:
                    sur += "0"
            elif d == "up":
                dir = 2
                if self.is_collision((head[0]-1, head[1]), game):
                    sur += "1"
                else:
                    sur += "0"
                if self.is_collision((head[0], head[1]-1), game):
                    sur += "1"
                else:
                    sur += "0"
                if self.is_collision((head[0], head[1]+1), game):
                    sur += "1"
                else:
                    sur += "0"
                
            elif d == "down":
                dir = 3
                if self.is_collision((head[0]+1, head[1]), game):
                    sur += "1"
                else:
                    sur += "0"
                if self.is_collision((head[0], head[1]+1), game):
                    sur += "1"
                else:
                    sur += "0"
                if self.is_collision((head[0], head[1]-1), game):
                    sur += "1"
                else:
                    sur += "0"

        surroundings = int(sur, 2)
                
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
        
        '''sqs = [
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
        surroundings = int(surroundings, 2)'''
        
        return (pos_x, pos_y, surroundings, dir)
    
    def is_collision(self, pt, game):
        if pt[0] < 0 or pt[0] >= game.cols or pt[1] < 0 or pt[1] >= game.cols or pt in game.snake.body[1:]:
            return True
        return False

    def choose_action(self, state):
        if len(self.game.snake.directions) == 0:
            action = 1
        else:
            if random.random() < self.epsilon:
                action = self.action[np.random.randint(0,3)]
            else:
                action = self.action[np.argmax(self.qtable[tuple(state)])]
        return action

    def learn(self, state, action, reward, next_state, done):
        action_idx = list(self.action.values()).index(action)
        if done:
            self.qtable[tuple(state) + (action_idx,)] -= 10
        else:
            max_q = np.max(self.qtable[tuple(next_state)])
            self.qtable[tuple(state) + (action_idx, )] = (1 - self.learning_rate) * self.qtable[tuple(state) + (action_idx,)] + self.learning_rate * (reward + self.gamma * max_q)


def train():
    env = SnakeGame(100)
    agent = Agent(env)
    pygame.font.init()

    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    
    while agent.game.play:
        agent.game.clock.tick(agent.game.fps)
        # get current state
        state = agent.get_state(agent.game)

        # get move
        action = agent.choose_action(state)

        # perform move and get new state
        reward, done = agent.game.move_snake(action)

        next_state = agent.get_state(agent.game)

        agent.learn(state, action, reward, next_state, done)

        #check if snake is killed for not eating a fruit in a while
        agent.game.update_frames_since_last_fruit()

        if done:
            agent.n_game += 1
            if agent.epsilon > 0.02:
                agent.epsilon -= 0.0008
            total_score += agent.game.score
            mean_score = total_score / agent.n_game
            plot_scores.append(agent.game.score)
            plot_mean_scores.append(mean_score)
            plot(plot_scores, plot_mean_scores)
            agent.game.game_over()      

        if agent.game.restart == True:
            agent.game.restart = False
            continue
    
        agent.game.redraw_window()
        agent.game.event_handler()

        if agent.n_game == 3000:
            np.save("./Tables/cartpole_table.npy", agent.qtable)
            break

def test():
    env = SnakeGame(500)
    testing_agent = Agent(env)
    
    pygame.font.init()

    testing_agent.qtable = np.load("./Tables/cartpole_table.npy")

    for i in range(100):
        state = testing_agent.get_state(testing_agent.game)
        scores = []
        while testing_agent.game.play:
            # choose action
            action = np.argmax(testing_agent.qtable[tuple(state)])

            # move snake
            reward, done = testing_agent.game.move_snake(action)

            # get next state
            next_state = testing_agent.get_state(testing_agent.game)

            #check if snake is killed for not eating a fruit in a while
            testing_agent.game.update_frames_since_last_fruit()

            if done:
                scores.append(testing_agent.game.score)
                testing_agent.game.game_over() 

            if testing_agent.game.restart == True:
                testing_agent.game.restart = False
                continue

            testing_agent.game.redraw_window()
            testing_agent.game.event_handler()

            state = next_state
    print("-" * 20)
    print("Mean Score: {}".format(sum(scores)/100))

    

def main():
    if not os.path.exists("./Tables"):
        os.mkdir("./Tables")

    for i in range(1):
        train()
    test()
        
if __name__ == "__main__":	
    main()
