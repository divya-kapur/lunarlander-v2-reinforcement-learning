#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec  5 17:56:02 2021

@author: divyakapur
"""

# Pip installing all the necessary libraries
#!pip3 install Box2D

#!pip3 install box2d-py

#!pip3 install 'gym[all]'

#!pip3 install 'gym[Box_2D]'

# Importing numpy
import numpy as np

# Importing tensorflow
import tensorflow as tf

# Importing keras
from tensorflow import keras

# These two import statements are used to set up our simulation environment
import gym
import Box2D

# To get smooth animations
import matplotlib as mpl
import matplotlib.animation as animation
mpl.rc('animation', html='jshtml')

# To plot pretty figures
#%matplotlib inline
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.rc('axes', labelsize=14)
mpl.rc('xtick', labelsize=12)
mpl.rc('ytick', labelsize=12)

# Creates lunar lander simulation environment
myEnv = gym.make("LunarLander-v2")

# Initializes environment
obs = myEnv.reset()
print(obs)

# 8 parameters: x,y=coordinates of spaceship (randomly initialized)
              # h,v=horizontal & vertical speed of spaceship (randomly initialized)
              # a,w=spaceship's angle & angular velocity 
              # l,r=whether left or right leg is touches ground (0 or 1)
              
# To visualize the lunar lander
try:
    # Importing library to display simulation environment
    import pyvirtualdisplay
    # Initializing simulation display
    display = pyvirtualdisplay.Display(visible=0, size=(1400, 900)).start()
except ImportError:
    pass

# This command displays a pop-up window
myEnv.render()

# Defining a function to plot the simulation environment with matplotlib
def plot_environment(myEnv, figsize=(5,4)):
    # Sets up plot of appropriate size
    plt.figure(figsize=figsize)
    # Assigns RGB pixel representation of current state of environment to a variable
    img = myEnv.render(mode="rgb_array")
    # Displays RGB pixel representation with matplotib
    plt.imshow(img)
    # Removes x and y axes
    plt.axis("off")
    # Returns RBG pixel representation object
    return img

# Plotting the simulation environment with the custom function defined above
plot_environment(myEnv)
plt.show()

# To get the number of actions possible
myEnv.action_space

# 4 actions available: 0: Do nothing
                     # 1: Fire left orientation engine
                     # 2: Fire right orientation engine
                     # 3: Fire main engine

# Basic policy to solve the problem
def basic_policy(obs):
    x_coord = obs[0]
    # If x-coordinate is to the left of the landing pad, fire right orientation engine
    if x_coord < 0:
        action = 2
    # If x-coordinate is to the right of the landing pad, fire left orientation engine
    if x_coord > 0:
        action = 1
    # Else (if x-coordinate == 0), fire main engine
    else:
        action = 3
    return action

# Seeing how well our basic policy solves the lunar lander problem

# Initializing a list of all rewards for each episode run
totals = []
# Running our simulation for 500 episodes
for episode in range(500):
    # Initializing rewards from current episode
    episode_rewards = 0
    # Initializes simulation environment
    obs = myEnv.reset()
    # Each episode runs for 200 steps 
    for step in range(200):
        # An action is chosen for this step based on our basic policy
        action = basic_policy(obs)
        # Updating our environment parameters after the action has been taken
        obs, reward, done, info = myEnv.step(action)
        # Adding the reward after the action has been taken to the variable for total rewards from the episode
        episode_rewards += reward
        # If the agent landed on the landing pad or if it crashed, exit out of the loop
        if done:
            break
    # Adding our total reward for the episode to our list of rewards for all the episodes
    totals.append(episode_rewards)
totals

# Defining a function to play one step of the lunar lander simulation 
def play_one_step(myEnv, obs, model, loss_fn):
    # This function uses Tensorflow's automatic differentiation to compute gradients of each node of model
    with tf.GradientTape() as tape:
        # Calls the model as a single observation and assigns it to a variable
        left_proba = model(obs[np.newaxis])
        # Calculate the action based on the predicted probabiliy and a random number
        action = (tf.random.uniform([1, 1]) > left_proba)
        # Casting the action to a number to set the target such that the y_target and left_proba is very low
        y_target = tf.constant([[1.]]) - tf.cast(action, tf.float32)
        # Calculating the loss by computing the mean of the elements passed to the loss function
        loss = tf.reduce_mean(loss_fn(y_target, left_proba))
    # Computing the gradients after the model has taken the action
    grads = tape.gradient(loss, model.trainable_variables)
    # Agent takes the stepp and our environment parameters are updates
    obs, reward, done, info = myEnv.step(int(action[0, 0].numpy()))
    # Function returns updated enviornment parameters
    return obs, reward, done, grads

# Defining a function to play multiple episodes using the play_one_step function above
def play_multiple_episodes(myEnv, n_episodes, n_max_steps, model, loss_fn):
    # Initializing list for all rewards for each episode run
    all_rewards = []
    # Initializing list for all gradients for each episode run
    all_grads = []
    # This function will run for as many episodes you input
    for episode in range(n_episodes):
        # Initializing list for rewards for current episode
        current_rewards = []
        # Initializing list for gradients for current episode
        current_grads = []
        # Initializing simulation environment
        obs = myEnv.reset()
        # The number of maximum steps for each episode in also user-defined
        for step in range(n_max_steps):
            # Updates environment parameters after one step is played using play_one_step function defined above
            obs, reward, done, grads = play_one_step(myEnv, obs, model, loss_fn)
            # Appends reward after one step to list of rewards for current episode
            current_rewards.append(reward)
            # Appends gradient after one step to list of graidents for current episode
            current_grads.append(grads)
            # If the agent landed on the landing pad or if it crashed, exit out of the loop
            if done:
                break
        # Appending total reward from episode to list of all rewards from each episode
        all_rewards.append(current_rewards)
        # Appending total gradient from episode to list of all gradients from each episode
        all_grads.append(current_grads)
    # Returns list of all rewards from each episode and list of all gradients from each episode
    return all_rewards, all_grads

# Define function to return discounted rewards given a list of rewards and a discount rate 
def discount_rewards(rewards, discount_rate):
    # Converting list of rewards to numpy array
    discounted = np.array(rewards)
    for step in range(len(rewards) - 2, -1, -1):
        # Applying our discount rate to our rewards
        discounted[step] += discounted[step + 1] * discount_rate
    return discounted

# Define function to return discounted and normalized rewards using discount_rewards function defined above
def discount_and_normalize_rewards(all_rewards, discount_rate):
    # Applying the discount rate to our list of rewards using the discount_rewards function above
    all_discounted_rewards = [discount_rewards(rewards, discount_rate)
                              for rewards in all_rewards]
    # Concatenating our values of discounted rewards into one single numpy array
    flat_rewards = np.concatenate(all_discounted_rewards)
    # Computing mean of our discounted rewards
    reward_mean = flat_rewards.mean()
    # Computing standard deviation of discounted rewards
    reward_std = flat_rewards.std()
    # Returns normalized value for discounted rewards
    return [(discounted_rewards - reward_mean) / reward_std
            for discounted_rewards in all_discounted_rewards]

# Set the initial values of the model

# Model will be trained for 200 iterations
n_iterations = 200

# The weights will be updated after every 15 epidsodes
n_episodes_per_update = 15

# Each landing must be within a maximum of 1000 steps 
n_max_steps = 1000

# The discount rate of 0.95
discount_rate = 0.95

# Our model uses Adam optimization with a learning rate of 0.01
optimizer = keras.optimizers.Adam(learning_rate=0.01)

# Our model uses binary cross entropy as the loss function
loss_fn = keras.losses.binary_crossentropy

# Defining our neural network
model = keras.models.Sequential([
    # The first layer has 64 neurons, uses relu activation and has an input shape equal to the number of parameters our observation space contains
    keras.layers.Dense(64, activation="relu", input_shape=[myEnv.observation_space.shape[0]]),
    # The second layer has 32 neurons and also uses relu activation
    keras.layers.Dense(32, activation="relu"),
    # Our output layer has one output neuron and uses softmax activation
    keras.layers.Dense(1, activation="softmax"), 
])

# Our model can be trained for as many iterations as the user inputs
for iteration in range(n_iterations):
    # Playing multiple episodes of the lunar lander simulation using the function defined above
    all_rewards, all_grads = play_multiple_episodes(
        myEnv, n_episodes_per_update, n_max_steps, model, loss_fn)
    # Mapping the sum of the rewards for each episode to the list of total rewards
    total_rewards = sum(map(sum, all_rewards))
    # Printing the iteration number and the mean rewards for current iteration                
    print("\rIteration: {}, mean rewards: {:.1f}".format(          
        iteration, total_rewards / n_episodes_per_update), end="") 
    # Discounted and normalized rewards are computed and assigned to a variable
    all_final_rewards = discount_and_normalize_rewards(all_rewards,
                                                       discount_rate)
    # Initializing list of mean gradients for all gradients in current iteration
    all_mean_grads = []
    for var_index in range(len(model.trainable_variables)):
        # Calculating the mean gradient by computing the mean of all gradients from each episode and the final reward for each episode
        mean_grads = tf.reduce_mean(
            [final_reward * all_grads[episode_index][step][var_index]
             for episode_index, final_rewards in enumerate(all_final_rewards)
                 for step, final_reward in enumerate(final_rewards)], axis=0)
        all_mean_grads.append(mean_grads)
    optimizer.apply_gradients(zip(all_mean_grads, model.trainable_variables))

# Displays simulation environment after our model has been trained and run
plot_environment(myEnv)
plt.show()

