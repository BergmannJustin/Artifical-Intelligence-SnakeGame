import pygame
import random
import numpy as np
from collections import namedtuple
from enum import Enum

pygame.init()
font = pygame.font.SysFont('Arial', 30)

# Farben
WHITE = (255, 255, 255)
BEIGE = (245, 245, 220)  # Hintergrund
LIGHT_GREEN = (144, 238, 144)  # für den Körper
DARK_GREEN = (0, 100, 0)  # für den Kopf
RED = (200, 0, 0)  # für das Futter
BLACK = (0, 0, 0)  # für den Score
GOLD = (255, 215, 0)  # Bonus
DARK_GRAY = (169, 169, 169)  # für den Rahmen und den Hintergrund

BLOCK_SIZE = 20
SPEED = 15
MAX_FRAME_ITERATION = 100 * BLOCK_SIZE  # Maximale Schritte pro Schlange


class Direction(Enum):
    RIGHT = 1
    LEFT = 2
    UP = 3
    DOWN = 4

Point = namedtuple('Point', 'x, y')

class SnakeGameAI:
    def __init__(self, w=640, h=480):
        self.w = w  # Setzt die Breite des Fensters
        self.h = h  # Setzt die Höhe des Fensters
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Snake Game AI')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        """Setzt den Zustand des Spiels zurück."""
        self.direction = Direction.RIGHT
        self.head = Point(self.w / 2, self.h / 2)  # Setzt die Position des Schlangenkopfes
        self.snake = [self.head,
                      Point(self.head.x - BLOCK_SIZE, self.head.y),
                      Point(self.head.x - (2 * BLOCK_SIZE), self.head.y)]
        self.score = 0
        self.food = None
        self._place_food()
        self.frame_iteration = 0
        self.level = 1
        self.speed_increase = 2  # Geschwindigkeitserhöhung pro Level
        self.speed_by_level = {
            1: 10,
            2: 20,
            3: 30,
            4: 40,
            5: 50,
            6: 70,
            7: 100
        }
        self.reset_speed()  # Setzt die Geschwindigkeit entsprechend dem Level

    def reset_speed(self):
        """Setzt die Geschwindigkeit des Spiels auf den Wert des aktuellen Levels."""
        global SPEED
        SPEED = self.speed_by_level.get(self.level, 15)  # Setzt die Geschwindigkeit basierend auf dem Level

    def _place_food(self):
        """Platziert das Futter an einer zufälligen Position."""
        x = random.randint(0, (self.w - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE  # Berechnet die x-Position des Futters.
        y = random.randint(0, (self.h - BLOCK_SIZE) // BLOCK_SIZE) * BLOCK_SIZE
        self.food = Point(x, y)
        if self.food in self.snake:  # Überprüft, ob das Futter zufällig auf einem Schlangensegment liegt und platziert es dann neu.
            self._place_food()

    def play_step(self, action):
        """Ein Schritt im Spiel."""
        self.frame_iteration += 1
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        reward = 0
        game_over = False

        self._move(action)
        self.snake.insert(0, self.head)

        if self.is_collision() or self.frame_iteration > MAX_FRAME_ITERATION:
            game_over = True
            reward = -10
            return reward, game_over, self.score

        if self.head == self.food:
            self.score += 1
            reward = 10
            self._place_food()

            # Steigerung der Geschwindigkeit jedes Mal, wenn die Schlange wächst
            if self.score % 10 == 0:
                self.level += 1
                self.reset_speed()  # Geschwindigkeit an das neue Level anpassen
                print(f"Level up! Level: {self.level}, Speed: {SPEED}")

        else:
            self.snake.pop()

        self._update_ui()
        self.clock.tick(SPEED)
        return reward, game_over, self.score

    def is_collision(self, pt=None):
        """Überprüft Kollisionen mit den Wänden oder der Schlange selbst."""
        if pt is None:
            pt = self.head
        if pt.x > self.w - BLOCK_SIZE or pt.x < 0 or pt.y > self.h - BLOCK_SIZE or pt.y < 0:
            return True
        if pt in self.snake[1:]:
            return True
        return False

    def _update_ui(self):
        """Aktualisiert das UI, einschließlich Hintergrund, Schlange, Futter und Punktzahl."""
        self.display.fill(BEIGE)  # Beige Hintergrund
        self._draw_border()
        self._draw_snake()
        self._draw_food()
        self._draw_score()

        pygame.display.flip()

    def _draw_border(self):
        """Zeichnet einen Rahmen um das Spielfeld."""
        pygame.draw.rect(self.display, DARK_GRAY, pygame.Rect(0, 0, self.w, self.h), 5)

    def _draw_snake(self):
        """Zeichnet die Schlange auf dem Bildschirm."""
        for i, pt in enumerate(self.snake):
            if i == 0:  # Kopf der Schlange
                pygame.draw.circle(self.display, DARK_GREEN, (pt.x + BLOCK_SIZE // 2, pt.y + BLOCK_SIZE // 2), BLOCK_SIZE // 2)
            else:  # Körper der Schlange
                pygame.draw.circle(self.display, LIGHT_GREEN, (pt.x + BLOCK_SIZE // 2, pt.y + BLOCK_SIZE // 2), BLOCK_SIZE // 2)

    def _draw_food(self):
        """Zeichnet das Futter."""
        pygame.draw.circle(self.display, RED, (self.food.x + BLOCK_SIZE // 2, self.food.y + BLOCK_SIZE // 2), BLOCK_SIZE // 2)

    def _draw_score(self):
        """Zeigt den aktuellen Punktestand an."""
        score_text = font.render(f"Score: {self.score}  Level: {self.level}", True, BLACK)
        self.display.blit(score_text, [10, 10])

    def _move(self, action):
        """Bewegt die Schlange basierend auf der Eingabeaktion."""
        clock_wise = [Direction.RIGHT, Direction.DOWN, Direction.LEFT, Direction.UP]
        idx = clock_wise.index(self.direction)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # Keine Änderung
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            new_dir = clock_wise[next_idx]  # Rechtsdrehung
        else:
            next_idx = (idx - 1) % 4
            new_dir = clock_wise[next_idx]  # Linksdrehung

        self.direction = new_dir

        x = self.head.x
        y = self.head.y
        if self.direction == Direction.RIGHT:
            x += BLOCK_SIZE
        elif self.direction == Direction.LEFT:
            x -= BLOCK_SIZE
        elif self.direction == Direction.DOWN:
            y += BLOCK_SIZE
        elif self.direction == Direction.UP:
            y -= BLOCK_SIZE

        self.head = Point(x, y)
