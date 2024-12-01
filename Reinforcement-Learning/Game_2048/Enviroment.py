import gym
from gym import spaces
import numpy as np
import random
import pygame

class Board:
    def __init__(self):
        self.n = 4
        self.gridCell = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.compress = False
        self.merge = False
        self.moved = False

    def reverse(self):
        for ind in range(4):
            i = 0
            j = 3
            while i < j:
                self.gridCell[ind][i], self.gridCell[ind][j] = self.gridCell[ind][j], self.gridCell[ind][i]
                i += 1
                j -= 1

    def transpose(self):
        self.gridCell = [list(t) for t in zip(*self.gridCell)]

    def compressGrid(self):
        self.compress = False
        temp = [[0] * 4 for _ in range(4)]
        for i in range(4):
            cnt = 0
            for j in range(4):
                if self.gridCell[i][j] != 0:
                    temp[i][cnt] = self.gridCell[i][j]
                    if cnt != j:
                        self.compress = True
                    cnt += 1
        self.gridCell = temp

    def mergeGrid(self):
        self.merge = False
        for i in range(4):
            for j in range(3):
                if self.gridCell[i][j] == self.gridCell[i][j + 1] and self.gridCell[i][j] != 0:
                    self.gridCell[i][j] *= 2
                    self.gridCell[i][j + 1] = 0
                    self.score += self.gridCell[i][j]
                    self.merge = True

    def random_cell(self):
        cells = [(i, j) for i in range(4) for j in range(4) if self.gridCell[i][j] == 0]
        if cells:
            i, j = random.choice(cells)
            self.gridCell[i][j] = 2

    def can_merge(self):
        for i in range(4):
            for j in range(3):
                if self.gridCell[i][j] == self.gridCell[i][j + 1]:
                    return True
        for i in range(3):
            for j in range(4):
                if self.gridCell[i + 1][j] == self.gridCell[i][j]:
                    return True
        return False

    def reset(self):
        self.gridCell = [[0] * 4 for _ in range(4)]
        self.score = 0
        self.random_cell()
        self.random_cell()

    def step(self, action):
        # Translate the action to the corresponding key press
        if action == 0:  # Up
            self.transpose()
            self.compressGrid()
            self.mergeGrid()
            self.compressGrid()
            self.transpose()
        elif action == 1:  # Down
            self.transpose()
            self.reverse()
            self.compressGrid()
            self.mergeGrid()
            self.compressGrid()
            self.reverse()
            self.transpose()
        elif action == 2:  # Left
            self.compressGrid()
            self.mergeGrid()
            self.compressGrid()
        elif action == 3:  # Right
            self.reverse()
            self.compressGrid()
            self.mergeGrid()
            self.compressGrid()
            self.reverse()

        self.random_cell()

        done = self.is_done()
        return self.gridCell, self.score, done, {}

    def is_done(self):
        # Check if the game is over
        for i in range(4):
            for j in range(4):
                if self.gridCell[i][j] == 0:
                    return False
        if not self.can_merge():
            return True
        return False


class Game2048Env(gym.Env):
    def __init__(self):
        super(Game2048Env, self).__init__()
        self.board = Board()

        # Action space: 4 possible moves (up, down, left, right)
        self.action_space = spaces.Discrete(4)

        # Observation space: 4x4 grid with integer values from 0 to 2048 (the game grid)
        self.observation_space = spaces.Box(low=0, high=2048, shape=(4, 4), dtype=np.int32)

        # Initialize pygame
        pygame.init()
        self.font = pygame.font.SysFont("arial", 40)
        self.screen = pygame.display.set_mode((400, 400))

    def reset(self):
        self.board.reset()
        return np.array(self.board.gridCell)

    def step(self, action):
        next_state, score, done, info = self.board.step(action)
        return np.array(next_state), score, done, info

    def render(self, mode='human'):
        # Initialize the display
        self.screen.fill((187, 173, 160))  # Background color (light gray)

        # Draw the grid
        for i in range(4):
            for j in range(4):
                value = self.board.gridCell[i][j]
                x = j * 100
                y = i * 100
                pygame.draw.rect(self.screen, (204, 192, 179), pygame.Rect(x, y, 100, 100), 0)
                if value != 0:
                    label = self.font.render(str(value), True, (255, 255, 255))
                    self.screen.blit(label, (x + 35, y + 35))

        # Draw the score
        score_label = self.font.render(f"Score: {self.board.score}", True, (255, 255, 255))
        self.screen.blit(score_label, (10, 10))

        pygame.display.update()

    def close(self):
        pygame.quit()


# Testing the environment
if __name__ == "__main__":
    env = Game2048Env()
    state = env.reset()
    done = False
    while not done:
        action = env.action_space.sample()  # Random action
        state, reward, done, info = env.step(action)
        env.render()
        pygame.time.delay(100)  # Delay to make rendering visible
