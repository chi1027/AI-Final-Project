
import torch 
import random 
import numpy as np
from collections import deque
#from snake_gameai import SnakeGameAI,Direction,Point,BLOCK_SIZE
from helpers.snake import Snake
from helpers.snakeGame import SnakeGame
from model import Linear_QNet, QTrainer
#from Helper import plot
MAX_MEMORY = 100_000
BATCH_SIZE = 1000
LR = 0.001

class Agent:
    def __init__(self):
        self.n_game = 0
        self.epsilon = 0 # Randomness
        self.gamma = 0.9 # discount rate
        self.memory = deque(maxlen=MAX_MEMORY) # popleft()
        self.model = Linear_QNet(11,256,3) 
        self.trainer = QTrainer(self.model,lr=LR,gamma=self.gamma)
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n) 
        # self.model.to('cuda')   
        # for n,p in self.model.named_parameters():
        #     print(p.device,'',n)         
        # TODO: model,trainer

    # state (11 Values)
    #[ danger straight, danger right, danger left,
    #   
    # direction left, direction right,
    # direction up, direction down
    # 
    # food left,food right,
    # food up, food down]
    def get_state(self,game):
        head = game.snake.body[0]
        point_l = (head[0],head[1]-1)
        point_r = (head[0], head[1]+1)
        point_u = (head[0]-1, head[1])
        point_d = (head[0]+1, head[1])

        dir_l = 0
        dir_r = 2
        dir_u = 1
        dir_d = 3

        state = [
            # Danger Straight
            (dir_u and self.is_collision(point_u, game))or
            (dir_d and self.is_collision(point_d, game))or
            (dir_l and self.is_collision(point_l, game))or
            (dir_r and self.is_collision(point_r, game)),

            # Danger right
            (dir_u and self.is_collision(point_r, game))or
            (dir_d and self.is_collision(point_l, game))or
            (dir_u and self.is_collision(point_u, game))or #weird?
            (dir_d and self.is_collision(point_d, game)),

            #Danger Left
            (dir_u and self.is_collision(point_r, game))or
            (dir_d and self.is_collision(point_l, game))or
            (dir_r and self.is_collision(point_u, game))or
            (dir_l and self.is_collision(point_d, game)),

            # Move Direction
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            #Food Location
            game.fruit_pos[0] < head[0], # food is in left
            game.fruit_pos[0] > head[0], # food is in right
            game.fruit_pos[1] < head[1], # food is up
            game.fruit_pos[1] > head[1]  # food is down
        ]
        return np.array(state,dtype=int)

    def is_collision(self, pt, game):
        if pt[0] < 10 and pt[0] >= 0 and pt[1] < 10 and pt[1] >= 0 and pt not in game.snake.body:
            return False
        else:
            return True

    def remember(self,state,action,reward,next_state,done):
        self.memory.append((state,action,reward,next_state,done)) # popleft if memory exceed

    def train_long_memory(self):
        if len(self.memory) > BATCH_SIZE:
            mini_sample = random.sample(self.memory,BATCH_SIZE)
        else:
            mini_sample = self.memory
        states,actions,rewards,next_states,dones = zip(*mini_sample)
        self.trainer.train_step(states,actions,rewards,next_states,dones)

    def train_short_memory(self,state,action,reward,next_state,done):
        self.trainer.train_step(state,action,reward,next_state,done)

    def get_action(self,state):
        # random moves: tradeoff explotation / exploitation
        self.epsilon = 80 - self.n_game
        final_move = [0,0,0]
        if random.randint(0,200)<self.epsilon:
            move = random.randint(0,2)
            final_move[move]=1
        else:
            state0 = torch.tensor(state,dtype=torch.float)
            prediction = self.model(state0) # prediction by model 
            move = torch.argmax(prediction).item()
            final_move[move]=1 
        return final_move

def train():
    plot_scores = []
    plot_mean_scores = []
    total_score = 0
    record = 0
    agent = Agent()
    game = SnakeGame(60)
    while game.play:
        # Get Old state
        state_old = agent.get_state(game)

        # get move
        final_move = agent.get_action(state_old)

        # perform move and get new state
        game.move_snake(final_move)
        reward, score, done = game.check_collisions()
        state_new = agent.get_state(game)

        # train short memory
        agent.train_short_memory(state_old,final_move,reward,state_new,done)

        #remember
        agent.remember(state_old,final_move,reward,state_new,done)

        if done:
            # Train long memory,plot result
            game.game_over()
            agent.n_game += 1
            agent.train_long_memory()
            if(score > record): # new High score 
                record = score
                agent.model.save()
            print('Game:',agent.n_game,'Score:',score,'Record:',record)
            
            #plot_scores.append(score)
            total_score+=score
            mean_score = total_score / agent.n_game
            #plot_mean_scores.append(mean_score)
            #plot(plot_scores,plot_mean_scores)


if __name__=="__main__":
    train()