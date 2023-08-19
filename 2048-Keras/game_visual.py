from game_env import env_2048
from a2c import a2c
import numpy as np
import pygame
import cv2
import sys

import tensorflow as tf
import keras
from keras.models  import Model
from keras.layers import Input, Dense, Conv2D, Add, Flatten

# A2C implementation of 2048
game = env_2048()

observation_space = game.observation_space  # int
action_space = len(game.action_space)  # int

class ValueNetwork():
    def __init__(self):
        input_layer = Input(shape=(4,4,11))
        conv_layer = Conv2D(filters = 128, kernel_size = 2, activation = "relu", data_format = "channels_last")(input_layer)
        conv_layer = Conv2D(filters = 64, kernel_size = 2, activation = "relu", data_format = "channels_last")(conv_layer)
        flattened_layer = Flatten()(conv_layer)
        dense_layer = Dense(256, activation = "relu")(flattened_layer)
        output_layer = Dense(1, activation = "relu")(dense_layer)

        self.model = Model(inputs = input_layer, outputs = output_layer)

        self.optimizer = keras.optimizers.Adam(learning_rate = .0001)

        # Compile so the model can be saved. I do not intend to use the defined loss function.
        self.model.compile(optimizer=self.optimizer, loss = "mse")

    @tf.function
    def forward(self, one_input):
        return self.model(one_input)

    @tf.function
    def train(self, one_input, bootstrap):
        # Pass bootstrapped return to exclude it from gradient calculations.
        with tf.GradientTape() as tape:
            prediction = self.model(one_input)
            advantage = bootstrap - prediction
            loss = tf.square(advantage)

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))

        # Return advantage to pass directly to policy
        return advantage

class PolicyNetwork():
    def __init__(self):
        input_layer = Input(shape = (4,4,11))
        input_mask = Input(shape = (4,))
        conv_layer = Conv2D(filters = 128, kernel_size = 2, activation = "relu", data_format = "channels_last")(input_layer)
        conv_layer = Conv2D(filters = 64, kernel_size = 2, activation = "relu", data_format = "channels_last")(conv_layer)
        flattened_layer = Flatten()(conv_layer)
        dense_layer = Dense(256, activation = "relu")(flattened_layer)
        logits_layer = Dense(4, activation = "linear")(dense_layer)
        masked_logits_layer = Add()([logits_layer, input_mask])
        output_layer = keras.layers.Activation("softmax")(masked_logits_layer)

        self.model = Model(inputs = [input_layer, input_mask], outputs = output_layer)

        self.optimizer = keras.optimizers.Adam(learning_rate = .00001)

        # Compile model so it can be saved. A custom loss function will be used below.
        self.model.compile(optimizer=self.optimizer, loss = "categorical_crossentropy")

    @tf.function
    def forward(self, one_input, mask):
        return self.model([one_input, mask])

    @tf.function
    def train(self,one_input, advantage, action_index, mask = np.ones((4,))):
        # Pass bootstrapped return to exclude it from gradient calculations.
        with tf.GradientTape() as tape:
            prediction = self.model([one_input, mask])
            chosen_action = prediction[0][action_index]
            loss = -advantage * tf.math.log(chosen_action) 

        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
    
value_network = ValueNetwork()
policy_network = PolicyNetwork()

reward_step = 0.01
gamma = .9

_a2c_ = a2c(value_network, policy_network, reward_step, gamma, False, False)

# Pygame board:

# Initialize pygame
pygame.init()

# Dimensions
TILE_SIZE = 100
TILES_DIM = 4
SCREEN_DIM = TILE_SIZE * TILES_DIM
FONT_SIZE = 32

# Colors for 2048 tiles (You can add more colors or change these based on your requirements)
COLORS = {
    0: (204, 192, 179),
    2: (238, 228, 218),
    4: (237, 224, 200),
    8: (242, 177, 121),
    16: (245, 149, 99),
    32: (246, 124, 95),
    64: (246, 94, 59),
    128: (237, 207, 114),
    256: (237, 204, 97),
    512: (237, 200, 80),
    1024: (237, 197, 63),
    2048: (237, 194, 46)
}

# Colors
WHITE = (255,255,255)
BLACK = (0, 0, 0)

# Font
font = pygame.font.SysFont(None, FONT_SIZE)


# Create the game screen and clock
screen = pygame.display.set_mode((SCREEN_DIM, SCREEN_DIM))

pygame.display.set_caption('2048 Game Display')
clock = pygame.time.Clock()

def draw_board(grid):
    for row in range(TILES_DIM):
        for col in range(TILES_DIM):
            value = grid[row][col]
            color = COLORS.get(value, (0, 0, 0))  # Default to black if value not in COLORS dict
            pygame.draw.rect(screen, color, (col*TILE_SIZE, row*TILE_SIZE, TILE_SIZE, TILE_SIZE))
            if value:
                text_surface = font.render(str(value), True, BLACK if value in [2, 4] else WHITE)
                screen.blit(text_surface, (col*TILE_SIZE + TILE_SIZE//2 - text_surface.get_width()//2,
                                           row*TILE_SIZE + TILE_SIZE//2 - text_surface.get_height()//2))

    pygame.display.flip()

# ----- # 



if True:
    # Import model
    saved_policy_net = keras.models.load_model("saved_models/model 1/policy_network/")

    # Set model
    policy_network.model = saved_policy_net

    
performance = []
blocks = {2:0, 4:0, 8:0, 16:0, 32:0, 64:0, 128:0, 256:0, 512:0, 1024:0, 2048:0}

video_number = 0
while True:

    # Initialize with start state
    game.reset_game()
    state = game.get_3d_feature()

    # Expand dims for batch size dimension
    state = np.expand_dims(state, axis =0)

    _a2c_.agent_init(state)
    invalid_actions = game.get_invalid_actions()

    # Expand dims for batch size dimension
    invalid_actions = np.expand_dims(invalid_actions, axis =0)

    # Play round
    total_reward = 0
    steps = 0

    while True:
        
        
        # Display the state using pygame
        draw_board(game.grid)
        clock.tick(5)  # This will make the game run at 60 FPS

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                exit(0)


        action = _a2c_.policy(state, invalid_actions)
        
        # Observe next state and reward
        terminal, reward = game.update_state(action)
        next_state = game.get_3d_feature()

        # Expand dims for batch size dimension
        next_state = np.expand_dims(next_state, axis =0)


        total_reward += reward

        if terminal:

            largest_block = max(game.grid.flatten())
            out.release()
            
            
            if largest_block >= 2048:
                sys.exit()

            break

        state = next_state
        invalid_actions = game.get_invalid_actions()

        # Expand dims for batch size dimension
        invalid_actions = np.expand_dims(invalid_actions, axis =0)

        steps += 1


