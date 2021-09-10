ENV_TYPE = 'unity'      # enum ('unity', 'gym') = choose which environment to run
CLOUD = False           # True if running in Udacity venv
BUFFER_SIZE = int(1e6)  # Replay buffer size
BATCH_SIZE = 128        # minibatch size
N_EPISODES = 3000       # max number of episodes to run
MAX_T = 1000            # Max time steps within an episode
N_WORKERS = 20          # number of workers to run in environment
LEARN_F = 1             # Learning Frequency within epiodes
GAMMA = 0.99            # discount factor
TAU = 1e-3              # soft update target parameter
LR_ACTOR = 1e-4         # learning rate for the actor
LR_CRITIC = 1e-3        # learning rate for the critic
WEIGHT_DECAY = 0.0      # L2 weight decay parameter
WINDOW_LEN = 100        # window length for averaging