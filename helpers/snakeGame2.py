import pygame
import random
import collections
from helpers import neuralNetwork  as nn
from helpers import geneticAlgorithm as ga 
from helpers.snake import Snake


class SnakeGame():
    """Class framework for creating the Snake Game.
    Attributes:
        self.width: The width of the pygame screen.
        self.height: The height of the pygame screen.
        self.grid_start_y: The y position that indicicates where the game grid begins (since the score info is above the grid).
        self.play = An attribute to determine if the game is being played.
        self.restart = An attribute to determine if the snake got a game over and needs to start over.
        self.clock = A pygame Clock object.
        self.fps = The frames per second of the game.
        self.rows = The number of rows in the game grid.
        self.cols = The number of columns in the game grid.
        self.snake = A Snake object representing the snake in the game.
        self.fruit_pos = The position of the frit in the grid, in (row,column) format.
        self.score = The current score in the game based on how much fruit has been eaten.
        self.high_score = The highest score achieved since the module was opened.
    """

    def __init__(self, fps):
        """Initializes the SnakeGame class."""

        self.width = 500
        self.height = 600
        self.grid_start_y = 100
        self.win = pygame.display.set_mode((self.width, self.height))
        self.play = True
        self.restart = False
        self.clock = pygame.time.Clock()
        self.fps = fps
        self.rows = 10
        self.cols = self.rows
        # self.rows = 40
        # self.cols = 60
        self.snake = Snake(self.rows,self.cols)
        self.previous_head = None
        self.fruit_pos = (0,0)
        self.generate_fruit()
        self.score = 0
        self.high_score = 0
        self.frames_since_last_fruit = 0
        
    def redraw_window(self):
        """Function to update the pygame window every frame, called from playSnakeGame.py."""

        self.win.fill(pygame.Color(10, 49, 245))
        self.draw_data_window()
        self.draw_grid()
        self.draw_grid_updates()
        pygame.display.update()

    def draw_data_window(self):
        """Function to draw the segment of the pygame window with the score and high score."""

        pygame.draw.rect(self.win, pygame.Color(20, 20, 20), (0,0,self.width, self.grid_start_y))

        #Add the score and high score
        font = pygame.font.SysFont('calibri', 20)
        score_text = font.render('Score: ' + str(self.score),1, (255,255,255))
        high_score_text = font.render('High Score: ' + str(self.high_score), 1, (255,255,255))
        self.win.blit(score_text, (30, 50))
        self.win.blit(high_score_text, (self.width - 140, 50))

    def draw_grid(self):
        """Function to draw the grid in the pygame window where the game is played."""

        space_col = self.width//self.cols
        space_row = (self.height - self.grid_start_y)//self.rows

        for i in range(self.rows):
            #draw horizontal line
            pygame.draw.line(self.win, pygame.Color(100,100,100), (0, space_row*i + self.grid_start_y),  (self.width, space_row*i + self.grid_start_y))

        for i in range(self.cols):
            #draw vertical line
            pygame.draw.line(self.win, pygame.Color(100,100,100), (space_col*i, self.grid_start_y), (space_col*i, self.height))

        #draw last lines so they are not cut off
        pygame.draw.line(self.win, pygame.Color(100,100,100), (space_col*self.rows-2, self.grid_start_y), (space_col*self.rows-2, self.height))
        pygame.draw.line(self.win, pygame.Color(100,100,100), (0, self.height -2),  (self.width, self.height -2))

    def generate_fruit(self):
        """Function to generate a new random position for the fruit."""

        fruit_row = random.randrange(0,self.rows)
        fruit_col = random.randrange(0,self.cols)

        #Continually generate a location for the fruit until it is not in the snake's body
        while (fruit_row, fruit_col) in self.snake.body:

            fruit_row = random.randrange(0,self.rows)
            fruit_col = random.randrange(0,self.cols)


        self.fruit_pos = (fruit_row,fruit_col)

    def move_snake(self, action):
        """Function to allow the user to move the snake with the arrow keys."""

        self.previous_head = self.snake.body[0]
        if len(self.snake.directions) == 0:
            direct = "right"
        else:
            dire = self.snake.directions[0]
            if action == 2:
                direct = dire
            elif action == 1:
                if dire == "up":
                    direct = "right"
                if dire == "left":
                    direct = "up"
                if dire == "right":
                    direct = "down"
                if dire == "down":
                    direct = "left"
            elif action == 0:
                if dire == "up":
                    direct = "left"
                if dire == "left":
                    direct = "down"
                if dire == "right":
                    direct = "up"
                if dire == "down":
                    direct = "right"

        self.snake.directions.appendleft(direct)
        if len(self.snake.directions) > len(self.snake.body):
            self.snake.directions.pop()

        self.snake.update_body_positions()
        done = False
        reward = 0
        
        if self.check_wall_collision():
            reward = -100
            done = True
            #print("wall")

        elif self.check_body_collision():
            reward = -100
            done = True
            #print("body")

        elif self.check_fruit_collision():
            reward = 20

        else:
            prev_head = self.previous_head  # previous state
            curr_head = self.snake.body[0]  # current state
            d0 = abs(self.fruit_pos[0] - prev_head[0]) + abs(self.fruit_pos[1] - prev_head[1])
            d1 = abs(self.fruit_pos[0] - curr_head[0]) + abs(self.fruit_pos[1] - curr_head[1])
            reward = 1 if d1 < d0 else -1
        
        return reward, done




    def draw_grid_updates(self):
        """Function called from redraw_window() to update the grid area of the window."""

        space_col = self.width//self.cols
        space_row = (self.height - self.grid_start_y)//self.rows

        #Draw the fruit
        fruit_y = self.fruit_pos[0]
        fruit_x = self.fruit_pos[1]
        pygame.draw.rect(self.win, pygame.Color(250,30,30), (space_col*fruit_x+1, self.grid_start_y + space_row*fruit_y+1, space_col-1, space_row-1))

        #Draw the updated snake since last movement
        for pos in self.snake.body:
            pos_y = pos[0]
            pos_x = pos[1]
            
            pygame.draw.rect(self.win, pygame.Color(31,240,12), (space_col*pos_x+1, self.grid_start_y + space_row*pos_y+1, space_col-1, space_row-1))

        
        head = self.snake.body[0]
        head_y = head[0]
        head_x = head[1]
        head_dir = self.snake.directions[0]

        #Draw eyes on the head of the snake, determining which direction they should face

        #if head facing left
        if head_dir == "left":
            #draw left eye
            pygame.draw.circle(self.win, pygame.Color(100,100,100), (space_col*head_x+space_col//10, self.grid_start_y + space_row*head_y + (space_row*4)//5), 2)
            #draw right eye
            pygame.draw.circle(self.win, pygame.Color(100,100,100), (space_col*head_x+space_col//10, self.grid_start_y + space_row*head_y + space_row//5), 2)
        #if head facing up
        elif head_dir == "up":
            #draw left eye
            pygame.draw.circle(self.win, pygame.Color(100,100,100), (space_col*head_x+space_col//5, self.grid_start_y + space_row*head_y + space_row//10), 2)
            #draw right eye
            pygame.draw.circle(self.win, pygame.Color(100,100,100), (space_col*head_x+(space_col*4)//5, self.grid_start_y + space_row*head_y + space_row//10), 2)
        #if head facing right
        elif head_dir == "right":
            #draw left eye
            pygame.draw.circle(self.win, pygame.Color(100,100,100), (space_col*head_x+(space_col*9)//10, self.grid_start_y + space_row*head_y + space_row//5), 2)
            #draw right eye
            pygame.draw.circle(self.win, pygame.Color(100,100,100), (space_col*head_x+(space_col*9)//10, self.grid_start_y + space_row*head_y + (space_row*4)//5), 2)
        #if head is facing down
        else:
            #draw left eye
            pygame.draw.circle(self.win, pygame.Color(100,100,100), (space_col*head_x+space_col//5, self.grid_start_y + space_row*head_y + (space_row*9)//10), 2)
            #draw right eye
            pygame.draw.circle(self.win, pygame.Color(100,100,100), (space_col*head_x+(space_col*4)//5, self.grid_start_y + space_row*head_y + (space_row*9)//10), 2)


    def check_collisions(self):
        """Function that consecutively calls all the functions that detect collisions."""

        self.check_fruit_collision()
        self.check_wall_collision()
        self.check_body_collision()

    def check_fruit_collision(self):
        """Function that detects and handles if the snake has collided with a fruit."""

        #If we found a fruit
        if self.snake.body[0] == self.fruit_pos:
            #Add the new body square to the tail of the snake
            self.snake.extend_snake()
            #Generate a new fruit in a random position
            self.generate_fruit()
            self.score += 1

            return True
        return False

    def check_wall_collision(self):
        """Function that checks and handles if the snake has collided with a wall."""

        #Only need to check the colisions of the head of the snake
        head = self.snake.body[0]
        head_y = head[0]
        head_x = head[1]

        #If there is a wall collision, game over
        if head_x == self.cols or head_y == self.rows or head_x < 0 or head_y < 0:
            # self.game_over()
            return True
        return False

    def check_body_collision(self):
        """Function that checks and handles if the snake has collided with its own body."""

        if len(self.snake.body) > 1:
            #Only need to check the colisions of the head of the snake
            head = self.snake.body[0]
            body_without_head = self.snake.body[1:]

            if head in body_without_head:
                # self.game_over()
                return True
        return False

    def event_handler(self):
        """Function for cleanly handling the event of the user quitting."""

        for event in pygame.event.get():
            #Check if user has quit the game
            if event.type == pygame.QUIT:
                self.run = False
                pygame.quit()
                quit()

    def update_frames_since_last_fruit(self):
        """Function to check if the snake needs to be killed for not eating a fruit in a while."""
        
        self.frames_since_last_fruit += 1
        if (self.frames_since_last_fruit == 50 and self.score < 6) or self.frames_since_last_fruit == 250:
            self.game_over()

    def game_over(self):
        """Function that restarts the game upon game over."""

        self.snake = Snake(self.rows,self.cols)
        self.generate_fruit()
        self.restart = True
        if self.score > self.high_score:
            self.high_score = self.score
        self.score = 0