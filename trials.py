import easydict
import gym
import numpy as np
from itertools import count
from collections import namedtuple
import logging
import logging.handlers

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Normal
import matplotlib.pyplot as plt

from quad_env import QuadEnv

# Cart Pole
# based on:
# https://github.com/pytorch/examples/blob/master/reinforcement_learning/actor_critic.py

# args = parser.parse_args()

args = easydict.EasyDict({
    "gamma": 0.99,
    "seed": 203,
    "render": False,
    "log_interval": 10,
    "write_logger":True
})

# env = gym.make('LunarLanderContinuous-v2')
env = QuadEnv()

# env.seed(args.seed)
torch.manual_seed(args.seed)

SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])
# state_dim = env.observation_space.shape[0]
state_dim = 21
# action_dim = env.action_space.shape[0]
action_dim = 4

class Policy(nn.Module):
    """
    implements both actor and critic in one model
    """

    def __init__(self, hidden_dim1=64, hidden_dim2=32, output_dim=128):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(state_dim, hidden_dim1)
        self.affine2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.affine3 = nn.Linear(hidden_dim2, output_dim)
        self.act1 = nn.ReLU()
        # actor's layer
        self.action_mean = nn.Linear(output_dim, action_dim)
        self.action_var = nn.Linear(output_dim, action_dim)
        # critic's layer
        self.value_head = nn.Linear(output_dim, 1)
        # action & reward buffer
        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        """
        forward of both actor and critic
        """
        # TODO map input to:
        # mean of action distribution,
        # variance of action distribution (pass this through a non-negative function),
        # state value

        input_x = x
        x = self.act1(self.affine1(x))
        x = self.act1(self.affine2(x))
        x = self.act1(self.affine3(x))
        action_mean = self.action_mean(x)
        action_var = F.softplus(self.action_var(x))
        state_values = self.value_head(x)  # <= Value Function not value of state
        if any(torch.isnan(x)) or any(torch.isnan(action_mean)) or any(torch.isnan(action_var)):
            print('NaN in forward pass')

        return 100.0 * action_mean, 100.0 * action_var, state_values


model = Policy().float()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    mu, sigma, state_value = model(state)

    # create a normal distribution over the continuous action space
    m = Normal(loc=mu, scale=sigma)

    # and sample an action using the distribution
    action = m.sample()

    # save to action buffer
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))

    # the action to take (left or right)
    return action.data.numpy()


def finish_episode():
    """
    Training code. Calculates actor and critic loss and performs backprop.
    """
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []  # list to save actor (policy) loss
    value_losses = []  # list to save critic (value) loss
    returns = []  # list to save the true values

    # calculate the true value using rewards returned from the environment
    for r in model.rewards[::-1]:
        # TODO compute the value at state x
        # via the reward and the discounted tail reward
        R = args.gamma * R + r

        returns.insert(0, R)

    # whiten the returns
    returns = torch.tensor(returns).float()
    returns = (returns - returns.mean()) / (returns.std() + eps)

    for (log_prob, value), R in zip(saved_actions, returns):
        # TODO compute the advantage via subtracting off value
        advantage = R - value.item()

        # TODO calculate actor (policy) loss, from log_prob (saved in select action)
        # and from advantage
        policy_loss = -log_prob * advantage
        # append this to policy_losses
        policy_losses.append(policy_loss)
        # TODO calculate critic (value) loss
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([R])))
    # reset gradients

    optimizer.zero_grad()

    # sum up all the values of policy_losses and value_losses
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()

    # perform backprop
    loss.backward()
    # gradient clipping to solve exploding gradient problem
    torch.nn.utils.clip_grad_norm_(model.parameters(), 0.25)
    optimizer.step()

    # reset rewards and action buffer
    del model.rewards[:]
    del model.saved_actions[:]


def main():
    if args.write_logger:
        log_filename = 'training_log.out'
        train_logger = logging.getLogger('TrainLogger')
        train_logger.setLevel(logging.DEBUG)
        handler = logging.handlers.RotatingFileHandler(log_filename, maxBytes=10*1024*1024, backupCount=5)
        train_logger.addHandler(handler)

    running_reward = -8000

    # run infinitely many episodes, until performance criteria met
    episodic_rewards = []
    episodes = []

    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0

        for t in range(1, 300):
            # select action from policy
            action = select_action(state)
            if any(np.isnan(action)):
                print('action is NaN')

            # take the action
            state, reward, done, _ = env.step(action)

            if args.render and i_episode % 100 == 0:
                env.render()

            if args.write_logger:
                train_logger.debug('episode {0}, step {1}, state {2}, action {3}, reward {4}'.format(i_episode, t, state, action, reward))

            model.rewards.append(reward)
            ep_reward += reward
            if done:
                episodes.append(i_episode)  # added
                episodic_rewards.append(ep_reward)
                break

        # update cumulative reward
        running_reward = 0.05 * ep_reward + (1 - 0.05) * running_reward

        # perform backprop
        finish_episode()

        # log results
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast reward: {:.2f}\tAverage reward: {:.2f}'.format(
                i_episode, ep_reward, running_reward))

        # check if we have "solved" the problem
        # if running_reward > 200:
        if i_episode > 6000:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))

            # TODO plot episodic_rewards --- submit this plot with your code
            plt.figure()
            plt.plot(episodes, episodic_rewards)
            break


if __name__ == '__main__':
    main()
