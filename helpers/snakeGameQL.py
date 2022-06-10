#**************************************************************************************
#snakeGame.py
#Author: Craig Haber
#5/9/2020
#Module with the SnakeGame class that is instantiated in playSnakeGame.py to play
#the game, and is a super class for the SnakeGameGATest and SnakeGameGATrain classes.
#*************************************************************************************

import pygame
import random
import collections
from helpers import neuralNetwork  as nn
from helpers.snakeGame import SnakeGame
from helpers.snake import Snake


class SnakeGameQL(SnakeGame):
	"""Class framework for creating the Snake Game.

	Attributes:
		self.width: The width of the pygame screen.
		self.height: The height of the pygame screen.
		self.grid_start_y: The y position that indicicates where the game grid begins (since the score info is above the grid).
		self.An attribute to determine if the game is being played.
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
		super().__init__(fps)
		self.frames_since_last_fruit = 0
		
	def move_snake(self, direct):
		self.snake.directions.appendleft(direct)
		if len(self.snake.directions) > len(self.snake.body):
			self.snake.directions.pop()
		self.snake.update_body_positions()
		done = False
		if self.check_fruit_collision():
			reward = 20
		elif self.is_collision():
			reward = -10
			done = True
		else:
			reward = -1
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
		eat = self.check_fruit_collision()
		die = self.is_collision()
		return eat, die

	def check_fruit_collision(self, head=None):
		"""Function that detects and handles if the snake has collided with a fruit."""
		if head is None:
			head = self.snake.body[0]
		#If we found a fruit
		if head == self.fruit_pos:
			#Add the new body square to the tail of the snake
			self.snake.extend_snake()
			#Generate a new fruit in a random position
			self.generate_fruit()

			self.score += 1

		return head == self.fruit_pos

	def is_collision(self, head=None):
		if head is None:
			head = self.snake.body[0]
		#Only need to check the colisions of the head of the snake
		head_y = head[0]
		head_x = head[1]

		#If there is a wall collision, game over
		if head_x == self.cols or head_y == self.rows or head_x < 0 or head_y < 0:
			self.game_over()
			return True

		if len(self.snake.body) > 1:
			#Only need to check the colisions of the head of the snake
			if head in self.snake.body[1:]:
				self.game_over()
			return head in self.snake.body[1:]
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