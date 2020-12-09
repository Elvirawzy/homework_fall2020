import abc
import itertools
from torch import nn
from torch.nn import functional as F
from torch import optim

import numpy as np
import torch
from torch import distributions

from cs285.infrastructure import pytorch_util as ptu
from cs285.policies.base_policy import BasePolicy
torch.manual_seed(1)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
torch.cuda.manual_seed(1)

class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):

    def __init__(self,
                 ac_dim,
                 ob_dim,
                 n_layers,
                 size,
                 discrete=False,
                 learning_rate=1e-4,
                 training=True,
                 nn_baseline=False,
                 **kwargs
                 ):
        super().__init__(**kwargs)

        # init vars
        self.ac_dim = ac_dim
        self.ob_dim = ob_dim
        self.n_layers = n_layers
        self.discrete = discrete
        self.size = size
        self.learning_rate = learning_rate
        self.training = training
        self.nn_baseline = nn_baseline

        if self.discrete:
            self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
                                           output_size=self.ac_dim,
                                           n_layers=self.n_layers,
                                           size=self.size)
            self.logits_na.to(ptu.device)
            self.mean_net = None
            self.logstd = None
            self.optimizer = optim.Adam(self.logits_na.parameters(),
                                        self.learning_rate)
        else:
            self.logits_na = None
            self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
                                      output_size=self.ac_dim,
                                      n_layers=self.n_layers, size=self.size)
            self.logstd = nn.Parameter(
                torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
            )
            self.mean_net.to(ptu.device)
            self.logstd.to(ptu.device)
            self.optimizer = optim.Adam(
                itertools.chain([self.logstd], self.mean_net.parameters()),
                self.learning_rate
            )
        if nn_baseline:
            self.baseline = ptu.build_mlp(
                input_size=self.ob_dim,
                output_size=1,
                n_layers=self.n_layers,
                size=self.size,
            )
            self.baseline.to(ptu.device)
            self.baseline_optimizer = optim.Adam(
                self.baseline.parameters(),
                self.learning_rate,
            )
        else:
            self.baseline = None

    ##################################

    def save(self, filepath):
        torch.save(self.state_dict(), filepath)

    ##################################

    # query the policy with observation(s) to get selected action(s)
    def get_action(self, obs: np.ndarray) -> np.ndarray:
        # TODO: get this from Piazza
        obs = np.array(obs)
        if len(obs.shape) > 1:
            observation = obs
        else:
            observation = obs[None]

        # TODO return the action that the policy prescribes
        action, distribution = self(observation)

        return action

    # update/train this policy
    def update(self, observations, actions, **kwargs):
        raise NotImplementedError

    # This function defines the forward pass of the network.
    # You can return anything you want, but you should be able to differentiate
    # through it. For example, you can return a torch.FloatTensor. You can also
    # return more flexible objects, such as a
    # `torch.distributions.Distribution` object. It's up to you!
    def forward(self, observation: torch.FloatTensor):
        # TODO: get this from Piazza
        obs = ptu.from_numpy(observation)
        if self.discrete:
            output = self.logits_na(obs)
            actions = F.softmax(output, 1)
            distribution = distributions.Categorical(actions)
            action = distribution.sample()
        else:
            output = self.mean_net(obs)
            distribution = distributions.Normal(output, torch.exp(self.logstd))
            action = distribution.rsample()

        action = ptu.to_numpy(action)

        return action, distribution


#####################################################
#####################################################


class MLPPolicyPG(MLPPolicy):
        def update(self, observations, actions, advantages, q_values=None):
            actions = ptu.from_numpy(actions)
            advantages = ptu.from_numpy(advantages)

            _, distribution = self(observations)
            log_pi = -distribution.log_prob(actions) * advantages
            loss = log_pi.mean()

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            train_log = {
                'Training Loss': ptu.to_numpy(loss),
            }
            return loss.item()

        def run_baseline_prediction(self, obs):
            obs = ptu.from_numpy(obs)
            predictions = self.baseline(obs)
            return ptu.to_numpy(predictions)[:, 0]

# import abc
# import itertools
# from torch import nn
# from torch.nn import functional as F
# from torch import optim
#
# import numpy as np
# import torch
# from torch import distributions
#
# from cs285.infrastructure import pytorch_util as ptu
# from cs285.policies.base_policy import BasePolicy
# torch.manual_seed(1)  # cpu 为CPU设置种子用于生成随机数，以使得结果是确定的
# torch.cuda.manual_seed(1)
#
# class MLPPolicy(BasePolicy, nn.Module, metaclass=abc.ABCMeta):
#
#     def __init__(self,
#                  ac_dim,
#                  ob_dim,
#                  n_layers,
#                  size,
#                  discrete=False,
#                  learning_rate=1e-4,
#                  training=True,
#                  **kwargs
#                  ):
#         super().__init__(**kwargs)
#
#         # init vars
#         self.ac_dim = ac_dim
#         self.ob_dim = ob_dim
#         self.n_layers = n_layers
#         self.discrete = discrete
#         self.size = size
#         self.learning_rate = learning_rate
#         self.training = training
#
#         if self.discrete:
#             self.logits_na = ptu.build_mlp(input_size=self.ob_dim,
#                                            output_size=self.ac_dim,
#                                            n_layers=self.n_layers,
#                                            size=self.size)
#             self.logits_na.to(ptu.device)
#             self.mean_net = None
#             self.logstd = None
#             self.optimizer = optim.Adam(self.logits_na.parameters(),
#                                         self.learning_rate)
#         else:
#             self.logits_na = None
#             self.mean_net = ptu.build_mlp(input_size=self.ob_dim,
#                                       output_size=self.ac_dim,
#                                       n_layers=self.n_layers, size=self.size)
#             self.logstd = nn.Parameter(
#                 torch.zeros(self.ac_dim, dtype=torch.float32, device=ptu.device)
#             )
#             self.mean_net.to(ptu.device)
#             self.logstd.to(ptu.device)
#             self.optimizer = optim.Adam(
#                 itertools.chain([self.logstd], self.mean_net.parameters()),
#                 self.learning_rate
#             )
#
#     ##################################
#
#     def save(self, filepath):
#         torch.save(self.state_dict(), filepath)
#
#     ##################################
#
#     # query the policy with observation(s) to get selected action(s)
#     def get_action(self, obs: np.ndarray) -> np.ndarray:
#         # TODO: get this from Piazza
#         obs = np.array(obs)
#         if len(obs.shape) > 1:
#             observation = obs
#         else:
#             observation = obs[None]
#
#         # TODO return the action that the policy prescribes
#         action, distribution = self(observation)
#
#         return action
#
#     # update/train this policy
#     def update(self, observations, actions, **kwargs):
#         raise NotImplementedError
#
#     # This function defines the forward pass of the network.
#     # You can return anything you want, but you should be able to differentiate
#     # through it. For example, you can return a torch.FloatTensor. You can also
#     # return more flexible objects, such as a
#     # `torch.distributions.Distribution` object. It's up to you!
#     def forward(self, observation: torch.FloatTensor):
#         # TODO: get this from Piazza
#         obs = ptu.from_numpy(observation)
#         if self.discrete:
#             output = self.logits_na(obs)
#             actions = F.softmax(output, 1)
#             distribution = distributions.Categorical(actions)
#             action = distribution.sample()
#         else:
#             output = self.mean_net(obs)
#             distribution = distributions.Normal(output, torch.exp(self.logstd))
#             action = distribution.rsample()
#
#         action = ptu.to_numpy(action)
#
#         return action, distribution
#
#
# #####################################################
# #####################################################
#
#
# class MLPPolicyAC(MLPPolicy):
#     def update(self, observations, actions, adv_n=None):
#         # TODO: update the policy and return the loss
#         actions = ptu.from_numpy(actions)
#         adv_n = ptu.from_numpy(adv_n)
#
#         _, distribution = self(observations)
#         log_pi = -distribution.log_prob(actions) * adv_n
#         loss = log_pi.mean()
#
#         self.optimizer.zero_grad()  # loss默认requires_grad是false
#         loss.backward()
#         self.optimizer.step()
#
#         return loss.item()