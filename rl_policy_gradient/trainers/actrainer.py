from typing import Union
import numpy as np
import torch
from torch import nn
from rl_policy_gradient.models.stochastic_policy_network import StochasticPolicyNetwork
from rl_policy_gradient.trainers.rltrainer import RLTrainer
from rl_policy_gradient.util.rollout_buffer import RolloutBuffer


class ACTrainer(RLTrainer):
    """
    One-step Actor-Critic Trainer based on Sutton and Barto's algorithm.
    """

    def __init__(self,
                 policy: StochasticPolicyNetwork,
                 value_fun: nn.Module,
                 learning_rate_actor: float,
                 learning_rate_critic: float,
                 discount_factor: float,
                 batch_size: int = 1):
        """
        Initialize the Actor-Critic Trainer.

        :param policy: The actor model (policy).
        :param value_fun: The critic model (value function).
        :param learning_rate_actor: Learning rate for the actor.
        :param learning_rate_critic: Learning rate for the critic.
        :param discount_factor: Discount factor for future rewards.
        """
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.policy = policy
        self.value_fun = value_fun
        self.discount_factor = discount_factor

        self._batch_size = batch_size
        self.buf = RolloutBuffer(batch_size=self._batch_size)

        # Optimizes policy parameters
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=learning_rate_actor)
        # Optimizes critic parameters
        self.value_fun_optimizer = torch.optim.Adam(self.value_fun.parameters(), lr=learning_rate_critic)

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
        state_t = torch.as_tensor(state, device=self.device, dtype=torch.float64)
        action_t = torch.as_tensor(action, device=self.device, dtype=torch.float64)
        reward_t = torch.as_tensor(reward, device=self.device, dtype=torch.float64)
        next_state_t = torch.as_tensor(next_state, device=self.device, dtype=torch.float64)
        done_t = torch.as_tensor(done, device=self.device, dtype=torch.bool)
        self.buf.push(
            state_t, action_t, reward_t, next_state_t, done_t
        )

    def _compute_loss(self) -> torch.Tensor:
        """
        Compute losses for actor and critic, and write to metrics tracker.
        You can return the sum of the two losses.
        In terms of gradient flow, as long as the losses are independent (i.e., calculated correctly),
        summing them will properly propagate gradients to their respective parameters.

        :return: a tuple of the actor loss and the critic loss.
        """
        state, action, reward, next_state, _ = self.buf.get()

        # Add code here.

        pass

    def _optimize(self, loss: torch.Tensor) -> None:
        """
        Backpropagate the critic and actor loss.
        :param loss: actor and critic loss.
        """
        # Add code here.

        pass

    def train(self) -> bool:
        """
        Perform a training step.

        :return: True if optimized occurred, False otherwise.
        """
        loss = self._compute_loss()
        self._optimize(loss)
        return True

