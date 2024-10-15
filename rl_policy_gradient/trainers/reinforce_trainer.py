from typing import Union
import numpy as np
import torch
from torch import nn
from rl_policy_gradient.models.stochastic_policy_network import StochasticPolicyNetwork
from rl_policy_gradient.trainers.rltrainer import RLTrainer
from rl_policy_gradient.util.rollout_buffer import RolloutBuffer

""""
IMPORTANT: in pseudocode gradient ascent (on the actor objective function) is performed. But PyTorch automatic differentiation
facilities perform gradient descent by default. Therefore, you should reverse the signs to turn gradient ascent
in the pseudocode to gradient descent.
"""


class ReinforceTrainer(RLTrainer):
    """
    REINFORCE with Baseline Trainer based on Sutton and Barto's algorithm.
    """

    def __init__(self,
                 policy: StochasticPolicyNetwork,
                 value_fun: nn.Module,
                 learning_rate_actor: float,
                 learning_rate_critic: float,
                 discount_factor: float):
        """
        Initialize the REINFORCE with Baseline Trainer.

        :param policy: The policy approximation.
        :param value_fun: The value function approximation for the baseline.
        :param learning_rate_actor: Learning rate for the actor.
        :param learning_rate_critic: Learning rate for the critic.
        :param discount_factor: Discount factor for future rewards.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy = policy.to(self.device)
        self.value_fun = value_fun.to(self.device)
        self.discount_factor = discount_factor

        self.buf = RolloutBuffer()

        # Optimizers for policy and value function (actor and critic)
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate_actor)
        self.critic_optimizer = torch.optim.Adam(self.value_fun.parameters(), lr=learning_rate_critic)

    def add_transition(self,
                       state: Union[torch.Tensor, np.ndarray],
                       action: Union[torch.Tensor, np.ndarray],
                       reward: Union[torch.Tensor, np.ndarray],
                       next_state: Union[torch.Tensor, np.ndarray],
                       done: bool) -> None:
        """
        Add a transition to the buffer for storing.

        :param state: The current state.
        :param action: The action taken.
        :param reward: The reward received.
        :param next_state: The next state after taking the action.
        :param done: Whether the episode has ended (boolean).
        """
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.double)
        action_t = torch.as_tensor(action, device=self.device, dtype=torch.double)
        reward_t = torch.as_tensor(reward, device=self.device, dtype=torch.double)
        next_state_t = torch.as_tensor(next_state, device=self.device, dtype=torch.float32)
        done_t = torch.as_tensor(done, device=self.device, dtype=torch.bool)
        self.buf.push(state_t, action_t, reward_t, next_state_t, done_t)

    def _compute_returns(self, rewards: torch.Tensor) -> torch.Tensor:
        """
        Compute the cumulative discounted returns (G_t).

        :param rewards: The rewards for each step in the trajectory.
        :return: Discounted returns (G_t) for each step.
        """
        G = torch.zeros_like(rewards)
        cumulative_return = 0

        for t in reversed(range(len(rewards))):
            cumulative_return = rewards[t] + self.discount_factor * cumulative_return
            G[t] = cumulative_return

        return G

    def _compute_loss(self) -> torch.Tensor:
        """
        Compute the actor and critic losses and return the sum of both losses.

        :return: Combined loss for actor and critic.
        """
        # Get the trajectory from the buffer
        states, actions, rewards, _, _ = self.buf.get()

        # Add code here.

        pass

    def _optimize(self, loss: torch.Tensor) -> None:
        """
        Backpropagate and optimize the actor and critic losses.

        :param loss: Combined actor and critic loss.
        """
        # Add code here.

        pass

    def train(self) -> bool:
        """
        Perform one training step after a full episode is finished.

        :return: True if optimized occurred, False otherwise.
        """
        _, _, _, _, done = self.buf.latest()
        
        if done:        # End of episode
            # Compute the combined loss
            loss = self._compute_loss()
            # Optimize
            self._optimize(loss)
            return True

        return False
