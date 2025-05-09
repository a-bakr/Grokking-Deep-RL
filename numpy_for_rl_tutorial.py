"""
NumPy for Reinforcement Learning: A Comprehensive Tutorial
===========================================================

This tutorial covers how to effectively use NumPy for implementing reinforcement learning algorithms,
from basic concepts to practical implementations.
"""

import numpy as np


# ========================= PART 1: NUMPY BASICS FOR RL =========================

"""
1. NumPy Fundamentals for RL
----------------------------
NumPy provides efficient data structures and operations that are essential for RL:
- Arrays for representing states, actions, rewards
- Vectorized operations for faster computations
- Random number generation for exploration strategies
"""

# Creating state representations
def numpy_basics():
    # 1D arrays (for simple state/action spaces)
    simple_state = np.array([0, 0, 1, 0])  # One-hot encoding of a state
    print("Simple state representation:", simple_state)

    # 2D arrays (for grid worlds or image-based environments)
    grid_world = np.zeros((4, 4))  # 4x4 grid world
    grid_world[0, 0] = 1  # Agent position
    grid_world[3, 3] = 10  # Goal position
    print("Grid world representation:\n", grid_world)

    # 3D arrays (for frame stacking in visual RL)
    frame_stack = np.zeros((4, 84, 84))  # 4 frames of 84x84 pixels
    print("Frame stack shape:", frame_stack.shape)
    
    # Handling probability distributions (for policies)
    action_probs = np.array([0.1, 0.2, 0.5, 0.2])
    print("Action probabilities:", action_probs)
    print("Sum of probabilities:", np.sum(action_probs))
    
    # Random sampling (for exploration)
    random_action = np.random.choice(4, p=action_probs)  # Sample from discrete action space
    print("Randomly sampled action based on probabilities:", random_action)


# ========================= PART 2: IMPLEMENTING Q-TABLES =========================

"""
2. Q-Tables with NumPy
----------------------
Q-tables are fundamental for tabular reinforcement learning methods.
NumPy enables efficient creation and manipulation of these tables.
"""

def q_table_operations():
    # Creating a Q-table for a simple environment
    # Rows = states, Columns = actions
    n_states, n_actions = 10, 4
    q_table = np.zeros((n_states, n_actions))
    print("Initial Q-table shape:", q_table.shape)
    
    # Initializing with optimistic initial values to encourage exploration
    optimistic_q = np.ones((n_states, n_actions)) * 10.0
    print("Optimistic Q-table example:\n", optimistic_q[:3, :])
    
    # Accessing and updating Q-values
    state, action = 0, 1
    q_table[state, action] = 1.5
    print(f"Q-value for state {state}, action {action}:", q_table[state, action])
    
    # Finding the best action for a state (greedy policy)
    best_action = np.argmax(q_table[state])
    print(f"Best action for state {state}:", best_action)
    
    # Implementing epsilon-greedy action selection
    def epsilon_greedy(state, q_table, epsilon=0.1):
        if np.random.random() < epsilon:
            # Explore: select random action
            return np.random.randint(q_table.shape[1])
        else:
            # Exploit: select best action
            return np.argmax(q_table[state])
    
    # Example of multiple selections with epsilon-greedy
    actions = [epsilon_greedy(0, q_table) for _ in range(10)]
    print("10 actions selected with epsilon-greedy:", actions)


# ========================= PART 3: HANDLING TRANSITIONS =========================

"""
3. State Transitions and Rewards
-------------------------------
Efficiently managing and processing state transitions, rewards, and experiences
is crucial for RL algorithms.
"""

def manage_transitions():
    # Simple experience replay buffer using NumPy
    buffer_size = 1000
    state_dim = 4
    
    # Initialize buffer components
    states = np.zeros((buffer_size, state_dim))
    actions = np.zeros(buffer_size, dtype=np.int32)
    rewards = np.zeros(buffer_size)
    next_states = np.zeros((buffer_size, state_dim))
    dones = np.zeros(buffer_size, dtype=np.bool_)
    
    # Function to store a transition
    def store_transition(buffer_idx, state, action, reward, next_state, done):
        states[buffer_idx] = state
        actions[buffer_idx] = action
        rewards[buffer_idx] = reward
        next_states[buffer_idx] = next_state
        dones[buffer_idx] = done
    
    # Example of storing a transition
    store_transition(
        0, 
        np.array([0.1, 0.2, 0.3, 0.4]),  # state
        1,                               # action
        0.5,                             # reward
        np.array([0.2, 0.3, 0.4, 0.5]),  # next_state
        False                            # done
    )
    
    print("Stored state:", states[0])
    print("Stored action:", actions[0])
    
    # Sampling batches for learning
    def sample_batch(batch_size=32):
        indices = np.random.randint(0, buffer_size, size=batch_size)
        return (
            states[indices],
            actions[indices],
            rewards[indices],
            next_states[indices],
            dones[indices]
        )
    
    # Example batch
    batch = sample_batch(4)
    print("Sampled batch shapes:")
    print("- States:", batch[0].shape)
    print("- Actions:", batch[1].shape)
    print("- Rewards:", batch[2].shape)


# ========================= PART 4: VALUE FUNCTION APPROXIMATION =========================

"""
4. Value Function Approximation
------------------------------
Using NumPy for implementing linear value function approximation 
for states when the state space is too large for tabular methods.
"""

def value_approximation():
    # Linear function approximation for state values
    def linear_value_function(state, weights):
        return np.dot(state, weights)
    
    # Example state and weights
    state = np.array([0.1, 0.5, 0.2, 0.8])
    weights = np.array([1.0, 2.0, -0.5, 1.5])
    
    # Approximate value
    value = linear_value_function(state, weights)
    print(f"Approximate value for state: {value:.3f}")
    
    # Updating weights with gradient descent
    def update_weights(state, target, prediction, weights, alpha=0.01):
        # TD error
        error = target - prediction
        # Gradient is just the feature vector (state) for linear approximation
        gradient = state
        # Update rule: w += α * δ * ∇_w V(s)
        return weights + alpha * error * gradient
    
    # Example update
    target_value = 1.2
    prediction = linear_value_function(state, weights)
    new_weights = update_weights(state, target_value, prediction, weights)
    print("Original weights:", weights)
    print("Updated weights:", new_weights)
    print("New prediction:", linear_value_function(state, new_weights))


# ========================= PART 5: PRACTICAL RL EXAMPLE =========================

"""
5. Implementing Q-Learning with NumPy
-----------------------------------
A complete example of tabular Q-learning for a simple environment.
"""

def q_learning_example():
    # Environment parameters (e.g., for a simple grid world)
    n_states = 16  # 4x4 grid
    n_actions = 4  # up, right, down, left
    
    # Learning parameters
    alpha = 0.1    # Learning rate
    gamma = 0.99   # Discount factor
    epsilon = 0.1  # Exploration rate
    
    # Initialize Q-table
    q_table = np.zeros((n_states, n_actions))
    
    # Epsilon-greedy policy
    def select_action(state, q_table, epsilon):
        if np.random.random() < epsilon:
            return np.random.randint(n_actions)
        else:
            return np.argmax(q_table[state])
    
    # Q-learning update rule
    def update_q_value(state, action, reward, next_state, q_table, alpha, gamma):
        # Q-learning formula: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        best_next_action = np.argmax(q_table[next_state])
        td_target = reward + gamma * q_table[next_state, best_next_action]
        td_error = td_target - q_table[state, action]
        q_table[state, action] += alpha * td_error
        return q_table
    
    # Simulating one step of Q-learning (in a real implementation, you'd get these from the environment)
    state = 0
    action = select_action(state, q_table, epsilon)
    next_state = 1  # Assume this is the result of taking action from state
    reward = 0.5    # Assume this is the reward received
    
    # Update Q-table
    q_table = update_q_value(state, action, reward, next_state, q_table, alpha, gamma)
    
    print("Updated Q-table for state 0:")
    print(q_table[0])
    
    # In a complete implementation, you would repeat this process for many episodes
    # to converge to an optimal policy.


# ========================= PART 6: ADVANCED TECHNIQUES =========================

"""
6. Advanced NumPy Techniques for RL
---------------------------------
More sophisticated operations useful for implementing RL algorithms.
"""

def advanced_techniques():
    # Vectorized operations for batch processing
    states = np.random.rand(100, 4)  # 100 states of dimension 4
    weights = np.random.rand(4)
    
    # Vectorized value computation for all states at once
    values = np.dot(states, weights)
    print("Computed values for 100 states at once, first 5:", values[:5])
    
    # Computing advantages (A = Q - V) for advantage actor-critic
    q_values = np.random.rand(100, 3)  # Q-values for 100 states, 3 actions
    v_values = np.random.rand(100)     # V-values for 100 states
    
    # Broadcasting to compute advantages
    advantages = q_values - v_values[:, np.newaxis]
    print("Advantages shape:", advantages.shape)
    print("Example advantages for one state:", advantages[0])
    
    # Softmax for policy output
    def softmax(x):
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
    
    logits = np.random.randn(5, 3)  # Logits for 5 states, 3 actions
    action_probs = softmax(logits)
    print("Softmax output (action probabilities):")
    print(action_probs)
    print("Sum of probabilities for each state:", np.sum(action_probs, axis=1))
    
    # One-hot encoding for discrete actions
    actions = np.array([1, 0, 2, 1, 0])
    one_hot_actions = np.eye(3)[actions]
    print("One-hot encoded actions:\n", one_hot_actions)


# ========================= MAIN EXECUTION =========================

if __name__ == "__main__":
    print("\n===== NUMPY BASICS FOR RL =====")
    numpy_basics()
    
    print("\n===== Q-TABLE OPERATIONS =====")
    q_table_operations()
    
    print("\n===== MANAGING TRANSITIONS =====")
    manage_transitions()
    
    print("\n===== VALUE FUNCTION APPROXIMATION =====")
    value_approximation()
    
    print("\n===== Q-LEARNING EXAMPLE =====")
    q_learning_example()
    
    print("\n===== ADVANCED TECHNIQUES =====")
    advanced_techniques()
    
    print("\nTutorial complete! You now have a solid foundation for using NumPy in RL implementations.") 