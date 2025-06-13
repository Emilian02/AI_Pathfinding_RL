import pygame
import sys
import numpy as np
import gym
from gym import spaces
import random

# Initialize Pygame
pygame.init()

# Screen dimensions
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 700
WINDOW = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Isometric Ai Pathfinding")

# Colors
WHITE = (255, 255, 255)
GRAY = (169, 169, 169)
LIGHT_BLUE = (173, 216, 230)
RED = (255, 0, 0)

# Tile sizes (smaller tiles)
TILE_WIDTH = 64
TILE_HEIGHT = 64
TILE_WIDTH_HALF = TILE_WIDTH // 2
TILE_HEIGHT_HALF = TILE_HEIGHT // 4

# Increase grid sizes to add more tiles
GRID_WIDTH = 15
GRID_HEIGHT = 15

# Conversion of grid to isometric screen coordinates
def grid_to_iso(x: int, y: int) -> tuple[int, int]:
    screen_x = (x - y) * TILE_WIDTH_HALF + WINDOW_WIDTH // 2
    screen_y = (x + y) * TILE_HEIGHT_HALF + WINDOW_HEIGHT // 2 - (GRID_HEIGHT * TILE_HEIGHT_HALF) // 2
    return screen_x, screen_y

class Renderer:
    def __init__(self, window: pygame.Surface):
        self.window = window
        self.load_tiles()
        self.reset_scene()

    def load_tiles(self):
        try:
            # Load tile images
            self.floor_tile = pygame.image.load("Isometric_Tiles_Pixel_Art\\Blocks\\blocks_1.png")
            self.objective_tile = pygame.image.load("Isometric_Tiles_Pixel_Art\\Blocks\\blocks_28.png")
        except pygame.error as e:
            print(f"Error loading tiles: {e}")
            sys.exit()

    def reset_scene(self):
        self.holes = set()
        self.objective_position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
        for _ in range(random.randint(5, 15)):
            hole_position = (random.randint(0, GRID_WIDTH - 1), random.randint(0, GRID_HEIGHT - 1))
            if hole_position != self.objective_position:
                self.holes.add(hole_position)

    def draw_tiles(self, player):
        tiles = []  # Clear the tiles list at the start

        # Iterate over the grid to add floor tiles
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                if (x, y) in self.holes:
                    continue  # Skip drawing holes
                screen_x, screen_y = grid_to_iso(x, y)
                tile_x = screen_x - TILE_WIDTH_HALF
                tile_y = screen_y

                tiles.append((tile_y, self.floor_tile, (tile_x, tile_y)))

        # Add objective tile
        obj_x, obj_y = self.objective_position
        obj_screen_x, obj_screen_y = grid_to_iso(obj_x, obj_y)
        tiles.append((obj_screen_y, self.objective_tile, (obj_screen_x - TILE_WIDTH_HALF, obj_screen_y)))

        # Add player to the list of tiles
        player_screen_x, player_screen_y = grid_to_iso(player.x, player.y)
        player_frame = player.frames[player.direction]
        player_tile_y = player_screen_y - player_frame.get_height() // 2

        tiles.append((player_tile_y + TILE_HEIGHT_HALF, player_frame, (player_screen_x - player_frame.get_width() // 2, player_screen_y - player_frame.get_height() // 2)))

        # Sort tiles by the y-coordinate
        tiles.sort(key=lambda tile: tile[0])

        # Draw tiles in sorted order
        for _, tile, pos in tiles:
            self.window.blit(tile, pos)

    def draw_grid_lines(self):
        # Draw grid lines for visual reference
        for x in range(GRID_WIDTH):
            for y in range(GRID_HEIGHT):
                # Get the 4 corners of the tile
                top = grid_to_iso(x, y)
                right = grid_to_iso(x + 1, y)
                bottom = grid_to_iso(x + 1, y + 1)
                left = grid_to_iso(x, y + 1)

                # Draw the lines connecting the 4 corners
                pygame.draw.line(self.window, WHITE, top, right, 1)
                pygame.draw.line(self.window, WHITE, right, bottom, 1)
                pygame.draw.line(self.window, WHITE, bottom, left, 1)
                pygame.draw.line(self.window, WHITE, left, top, 1)

class Player:
    FRAME_HEIGHT = 22  # Define constant for frame height

    def __init__(self, holes: set, objective_position: tuple[int, int]):
        self.x, self.y = self.find_valid_spawn(holes, objective_position)
        self.direction = 'down'
        self.frame_index = 0
        self.move_speed = 1
        self.last_update = pygame.time.get_ticks()
        self.load_sprites()

    def find_valid_spawn(self, holes: set, objective_position: tuple[int, int]) -> tuple[int, int]:
        while True:
            x = random.randint(0, GRID_WIDTH - 1)
            y = random.randint(0, GRID_HEIGHT - 1)
            distance = abs(x - objective_position[0]) + abs(y - objective_position[1])
            if distance >= min(GRID_WIDTH, GRID_HEIGHT) // 2:
                if (x, y) not in holes:
                    return x, y

    def load_sprites(self):
        try:
            # Load player sprite sheets for different directions
            self.sprite_sheets = {
                'up': pygame.image.load("PlayerAnim\\UpAnim\\Up_Anim.png").convert_alpha(),
                'down': pygame.image.load("PlayerAnim\\DownAnim\\Down_Anim.png").convert_alpha(),
                'left': pygame.image.load("PlayerAnim\\LeftAnim\\Left_Anim.png").convert_alpha(),
                'right': pygame.image.load("PlayerAnim\\RightAnim\\Right_Anim.png").convert_alpha()
            }
        except pygame.error as e:
            print(f"Error loading player sprites: {e}")
            sys.exit()

        # Extract the first frame from each sprite sheet
        self.frames = {direction: self.extract_first_frame(sheet) for direction, sheet in self.sprite_sheets.items()}

    def extract_first_frame(self, sheet: pygame.Surface) -> pygame.Surface:
        # Extract the first frame from the sprite sheet
        frame_width = sheet.get_width()
        frame = sheet.subsurface(pygame.Rect(0, 0, frame_width, self.FRAME_HEIGHT))
        return frame

    def update(self, action: int):
        if action == 0:  # Left
            self.x = max(0, self.x - self.move_speed)
            self.direction = 'left'
        elif action == 1:  # Right
            self.x = min(GRID_WIDTH - 1, self.x + self.move_speed)
            self.direction = 'right'
        elif action == 2:  # Up
            self.y = max(0, self.y - self.move_speed)
            self.direction = 'up'
        elif action == 3:  # Down
            self.y = min(GRID_HEIGHT - 1, self.y + self.move_speed)
            self.direction = 'down'

    def draw(self, window: pygame.Surface):
        # Draw the player on the screen
        screen_x, screen_y = grid_to_iso(self.x, self.y)
        frame = self.frames[self.direction]
        window.blit(frame, (screen_x - frame.get_width() // 2, screen_y - frame.get_height() // 2))

class PathfindingEnv(gym.Env):
    def __init__(self):
        super(PathfindingEnv, self).__init__()
        self.renderer = Renderer(WINDOW)
        self.player = Player(self.renderer.holes, self.renderer.objective_position)
        self.action_space = spaces.Discrete(4)  # up, down, left, right
        self.observation_space = spaces.Box(low=0, high=max(GRID_WIDTH, GRID_HEIGHT), shape=(2,), dtype=np.int32)
        self.step_limit = 50
        self.previous_state = np.array([self.player.x, self.player.y])
        self.visitted = set()

    def reset_scene(self):
        self.renderer.reset_scene()

    def reset(self) -> np.ndarray:
        self.player = Player(self.renderer.holes, self.renderer.objective_position)
        self.step_count = 0  # Reset step counter
        self.previous_state = np.array([self.player.x, self.player.y])
        self.visited = set()
        self.visited.clear()
        self.visited.add((self.player.x, self.player.y))
        return np.array([self.player.x, self.player.y])
    
    

    def step(self, action: int) -> tuple[np.ndarray, int, bool, dict,]:
        self.player.update(action)
        state = np.array([self.player.x, self.player.y])
        reward = -1  # Default step penalty
        done = False

        if (self.player.x, self.player.y) in self.renderer.holes:
            reward -= 10  # Penalty for falling into a hole
            done = True
        elif (self.player.x, self.player.y) == self.renderer.objective_position:
            reward += 35  # Reward for reaching the objective
            done = True
        elif self.player.x < 0 or self.player.x >= GRID_WIDTH or self.player.y < 0 or self.player.y >= GRID_HEIGHT:
            reward -= 15  # Penalty for leaving the grid
            done = True
        else:
            # Dynamic reward based on distance to the objective
            current_distance = abs(self.player.x - self.renderer.objective_position[0]) + abs(self.player.y - self.renderer.objective_position[1])
            previous_distance = abs(self.previous_state[0] - self.renderer.objective_position[0]) + abs(self.previous_state[1] - self.renderer.objective_position[1])
            
            if current_distance < previous_distance:
                reward += 5  # Reward for moving closer to the objective
            else:
                reward -= 2  # Penalty for moving away from the objective

        if (self.player.x, self.player.y) in self.visited:
            reward -= 2  # Penalty for revisiting a tile

        if self.step_count > self.step_limit:
            reward -= 3  # Penalty for exceeding step limit
            done = True

        self.step_count += 1  # step counter

        self.visited.add((self.player.x, self.player.y)) 
        self.previous_state = state  # Update previous state

        return state, reward, done, {}

    def render(self, mode='human'):
        WINDOW.fill(LIGHT_BLUE)
        self.renderer.draw_tiles(self.player)
        self.renderer.draw_grid_lines()
        pygame.display.flip()

def train_q_learning(env: PathfindingEnv, episodes: int = 1000, alpha: float = 0.1, gamma: float = 0.9, epsilon: float = 1.0, epsilon_min: float = 0.05, epsilon_decay: float = 0.9) -> np.ndarray:
    q_table = np.zeros((GRID_WIDTH, GRID_HEIGHT, env.action_space.n))

    for episode in range(episodes):
        if episode % 75 == 0:
            env.reset_scene()  # Reset the scene every 100 episodes so that the agent adapts to new challenges
            epsilon = 1.0 # Reset epsilon to encourage exploration at the start of each new scene
            print(f"Resetting scene for episode {episode + 1}")
        state = env.reset()
        done = False
        step_count = 0
        total_reward = 0

        while not done:
            if random.uniform(0, 1) < epsilon:
                action = env.action_space.sample()  # Explore action space
            else:
                action = np.argmax(q_table[state[0], state[1]])  # Exploit learned values

            next_state, reward, done, _ = env.step(action)
            old_value = q_table[state[0], state[1], action]
            next_max = np.max(q_table[next_state[0], next_state[1]])

            # Update Q-Value
            new_value = (1 - alpha) * old_value + alpha * (reward + gamma * next_max)
            q_table[state[0], state[1], action] = new_value

            state = next_state
            step_count += 1
            total_reward += reward

            # Logging for debugging
            if done:
                print(f"Episode {episode + 1}/{episodes} finished after {step_count} steps with total reward {total_reward}")
                
        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    return q_table

def main():
    env = PathfindingEnv()

    q_table = train_q_learning(env)

    # Main game loop
    clock = pygame.time.Clock()
    running = True
    done = False

    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    pygame.quit()
                    sys.exit()
                if event.key == pygame.K_r:
                    env.reset()
                    env.reset_scene()
                    q_table = train_q_learning(env, 75)  # Retrain Q-table for new scene
                    done = False

        if not done:
            # Get the best action from the Q-table
            state = np.array([env.player.x, env.player.y])
            action = np.argmax(q_table[state[0], state[1]])
            # Update player position and draw tiles and grid lines
            _, _, done, _ = env.step(action)
            env.render()

        # Draw reset button
        pygame.draw.rect(WINDOW, RED, (WINDOW_WIDTH - 100, 10, 90, 30))
        font = pygame.font.Font(None, 24)
        text = font.render('Reset (R)', True, WHITE)
        WINDOW.blit(text, (WINDOW_WIDTH - 95, 15))

        # Update the display
        pygame.display.flip()
        clock.tick(2)

if __name__ == "__main__":
    main()