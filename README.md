# lunarlander-v2-reinforcement-learning
Involves the development of a reinforcement learning algorithm to play the LunarLander-V2 game within the OpenAI Gym simulation environment. The goal is to train a three-layer neural network using "Policy Gradient" to solve this game.

**Introduction:** The LunarLander-V2 game is a simulation environment provided by OpenAI Gym, where the agent controls a lunar lander module with the objective of landing it safely on a landing pad. The environment provides various parameters representing the position, speed, angle, and leg states of the lander.

For more details about the game, refer to the LunarLander-V2 documentation.

**Project Objective:**
Train a three-layer neural network using "Policy Gradient" to solve the LunarLander-V2 game. The neural network model consists of:
   
            model = keras.models.Sequential([
                keras.layers.Dense(64, activation="relu", input_shape=[n_inputs]),
                keras.layers.Dense(32, activation="relu"),
                keras.layers.Dense(n_outputs, activation="softmax"),
            ])
            
  Design Parameters:
  
  - Discount rate: 0.95
  - Model training: 200 iterations
  - Maximum steps per episode: 1000
  - Weight updates every 15 episodes
  - Random weight initialization

The neural network is trained to control the lunar lander module to successfully land on the landing pad while maximizing rewards.

![Picture1](https://github.com/divya-kapur/lunarlander-v2-reinforcement-learning/assets/47482776/27d9b5c5-ce5a-4c87-b5f1-8e361abc53f7)
      
