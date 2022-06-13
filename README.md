# AI-Final-Project
## Snake Game AI

Snake is a classical arcade game that originated in 1976.  
In this project, we want to implement and compared three methods, BFS, Q Learning, and DQN for this classical game.

### Reference:
Envionment: https://github.com/craighaber/AI-for-Snake-Game

Q Learning state: https://github.com/techtribeyt/snake-q-learning

### Requirements

pip install -r requirements.txt

### Usage

#### BFS:

python3 bfs.py

#### Q-Learning:

python3 Agent.py

#### DQN:

python3 DQN.py

#### If you want to play the game:

python3 playSnakeGame.py

### Hyperparameters

#### Q-Learning:

learning＿rate = 0.01  
GAMMA = 0.95  
epsilon = 1.0  
eps_decay = 0.98  
epsilon_min = 0.001  

#### DQN:

epsilon=0.1  
learning_rate = 0.002  
GAMMA = 0.8  
batch_size = 32  
capacity(memory) = 10000

### Result:

#### BFS:

average score: 71.44

![alt text](https://github.com/chi1027/AI-Final-Project/blob/main/image/BFS.png)



