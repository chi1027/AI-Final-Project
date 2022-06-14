# AI-Final-Project
## Snake Game AI

Snake is a classical arcade game that originated in 1976.  
In this project, we want to implement and compared three methods, BFS, Q Learning, and DQN for this classical game.

### Reference:
Envionment: https://github.com/craighaber/AI-for-Snake-Game

Q Learning state: https://github.com/techtribeyt/snake-q-learning

### Requirements

```
pip install -r requirements.txt
```

### Usage

#### BFS:
```
python3 bfs.py
```
#### Q-Learning:
```
python3 Agent.py
```
#### DQN:
```
python3 DQN.py
```
#### If you want to play the game:
```
python3 playSnakeGame.py
```
### Hyperparameters

#### Q-Learning:
```
learning＿rate = 0.01  
GAMMA = 0.95  
epsilon = 1.0  
eps_discount = 0.98  
epsilon_min = 0.001  
```
#### DQN:
```
learning_rate = 0.001
GAMMA = 0.95
epsilon = 1.0
eps_discount = 0.98
epsilon_min = 0.001
batch_size = 32  
capacity(memory) = 10000
```
### Result:

| | BFS | Q-Learning | DQN  |
| --- | --- | --- | --- |
| Average Score | 71.4 | 34.4 | 34.5 |
| Highest Score | 107  | 67   | 66   |

#### BFS:
![alt text](https://github.com/chi1027/AI-Final-Project/blob/main/image/BFS.png)

#### Q-Learning

**Training process figure**

![alt text](https://github.com/chi1027/AI-Final-Project/blob/main/image/qlearning_train.png)


#### DQN
**Training process figure**

![alt text](https://github.com/chi1027/AI-Final-Project/blob/main/image/DQN_train.png)




