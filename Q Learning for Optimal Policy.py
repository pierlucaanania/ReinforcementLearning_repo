import numpy as np
import random

# Q-learning parameters
alpha = 0.1  # Learning rate
gamma = 0.95  # Discount factor
epsilon = 0.1  # Exploration rate (for epsilon-greedy)
num_episodes = 1000  # Number of episodes

# Example environment
# Assume we have an environment with 5 states and 2 possible actions
num_states = 5
num_actions = 2

# Initialize the Q-table with zeros
Q = np.zeros((num_states, num_actions))


# Simulate the environment: for this example, we assume a deterministic reward function
def take_action(state, action):
    """A mock environment's step function."""
    # Define rewards based on state-action pair (for example purposes)
    reward_table = np.array([
        [0, 1],  # State 0
        [1, -1],  # State 1
        [-1, 2],  # State 2
        [2, 0],  # State 3
        [0, 1],  # State 4
    ])
    # Simulate next state
    next_state = (state + action) % num_states
    # Get reward
    reward = reward_table[state, action]
    return next_state, reward


# Q-learning algorithm
for episode in range(num_episodes):
    # Start with a random initial state
    state = np.random.randint(0, num_states)

    done = False
    while not done:
        # Select an action using epsilon-greedy policy
        if random.uniform(0, 1) < epsilon:
            action = np.random.randint(0, num_actions)  # Random action
        else:
            action = np.argmax(Q[state, :])  # Action with the highest Q-value

        # Take the action and observe the next state and reward
        next_state, reward = take_action(state, action)

        # Q-learning update rule
        best_next_action = np.argmax(Q[next_state, :])
        Q[state, action] = Q[state, action] + alpha * (
                    reward + gamma * Q[next_state, best_next_action] - Q[state, action])

        # Move to the next state
        state = next_state

        # End condition (example): if the episode reaches a certain state
        if state == 4:  # You can define your own terminal state
            done = True

# After training, the Q-table should contain the learned Q-values
print("Learned Q-Table:")
print(Q)

# The optimal policy can be extracted by choosing actions with the highest Q-value for each state
optimal_policy = np.argmax(Q, axis=1)
print(f"Optimal Policy (best action for each state): {optimal_policy}")

