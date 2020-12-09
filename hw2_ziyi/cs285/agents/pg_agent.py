import numpy as np

from .base_agent import BaseAgent
from .MLP_policy import MLPPolicyPG
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import normalize


class PGAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(PGAgent, self).__init__()

        # init vars
        self.env = env
        self.agent_params = agent_params
        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']
        self.nn_baseline = self.agent_params['nn_baseline']
        self.reward_to_go = self.agent_params['reward_to_go']

        # actor/policy
        self.actor = MLPPolicyPG(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            discrete=self.agent_params['discrete'],
            learning_rate=self.agent_params['learning_rate'],
            nn_baseline=self.agent_params['nn_baseline']
        )

        # replay buffer
        self.replay_buffer = ReplayBuffer(1000000)

    def train(self, observations, actions, rewards_list, next_observations, terminals):

        """
            Training a PG agent refers to updating its actor using the given observations/actions
            and the calculated qvals/advantages that come from the seen rewards.
        """

        # step 1: calculate q values of each (s_t, a_t) point, using rewards (r_0, ..., r_t, ..., r_T)
        q_values = self.calculate_q_vals(rewards_list)

        # step 2: calculate advantages that correspond to each (s_t, a_t) point
        advantages = self.estimate_advantage(observations, q_values)

        # TODO: step 3: use all datapoints (s_t, a_t, q_t, adv_t) to update the PG actor/policy
        ## HINT: `train_log` should be returned by your actor update method

        train_log = self.actor.update(observations, actions, advantages, q_values)

        return train_log

    def calculate_q_vals(self, rewards_list):

        """
            Monte Carlo estimation of the Q function.
        """
        if not self.reward_to_go:
            q_values = np.concatenate([self._discounted_return(r) for r in rewards_list])
        else:
            q_values = np.concatenate([self._discounted_cumsum(r) for r in rewards_list])
            rewards_list_before = rewards_list[:]

        return q_values

    def estimate_advantage(self, obs, q_values):

        """
            Computes advantages by (possibly) subtracting a baseline from the estimated Q values
        """

        # Estimate the advantage when nn_baseline is True,
        # by querying the neural network that you're using to learn the baseline
        if self.nn_baseline:
            baselines_unnormalized = self.actor.run_baseline_prediction(obs)
            ## ensure that the baseline and q_values have the same dimensionality
            ## to prevent silent broadcasting errors
            assert baselines_unnormalized.ndim == q_values.ndim
            ## baseline was trained with standardized q_values, so ensure that the predictions
            ## have the same mean and standard deviation as the current batch of q_values
            baselines = baselines_unnormalized * np.std(q_values) + np.mean(q_values)
            ## TODO: compute advantage estimates using q_values and baselines
            advantages = q_values - baselines

        # Else, just set the advantage to [Q]
        else:
            advantages = q_values.copy()

        # Normalize the resulting advantages
        if self.standardize_advantages:
            ## TODO: standardize the advantages to have a mean of zero
            ## and a standard deviation of one
            ## HINT: there is a `normalize` function in `infrastructure.utils`
            mean = np.mean(advantages)
            std = np.std(advantages)
            advantages = normalize(advantages, mean, std)

        return advantages

    #####################################################
    #####################################################

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    #####################################################
    ################## HELPER FUNCTIONS #################
    #####################################################

    def _discounted_return(self, rewards):
        """
            Helper function

            Input: list of rewards {r_0, r_1, ..., r_t', ... r_T} from a single rollout of length T

            Output: list where each index t contains sum_{t'=0}^T gamma^t' r_{t'}
        """

        # TODO: create list_of_discounted_returns
        # Hint: note that all entries of this output are equivalent
        # because each sum is from 0 to T (and doesnt involve t)
        list_of_discounted_returns = np.zeros(len(rewards))
        R = 0
        gm = 1
        for i in range(len(rewards)):
            R += gm * rewards[i]
            gm *= self.gamma
        for i in range(len(rewards)):
            list_of_discounted_returns[i] = R

        return list_of_discounted_returns

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        # TODO: create `list_of_discounted_returns`
        # HINT1: note that each entry of the output should now be unique,
        # because the summation happens over [t, T] instead of [0, T]
        # HINT2: it is possible to write a vectorized solution, but a solution
        # using a for loop is also fine
        T = len(rewards)
        list_of_discounted_cumsums = np.zeros(T)
        list_of_discounted_cumsums[T - 1] = rewards[T - 1]
        for i in range(T):
            if i > 0:
                list_of_discounted_cumsums[T - i - 1] = rewards[T - i - 1] + self.gamma * list_of_discounted_cumsums[
                    T - i]

        return list_of_discounted_cumsums

# from collections import OrderedDict
#
# from cs285.infrastructure.replay_buffer import ReplayBuffer
# from cs285.infrastructure.utils import *
# from cs285.policies.MLP_policy import MLPPolicyAC
# from .base_agent import BaseAgent
#
#
# class ACAgent(BaseAgent):
#     def __init__(self, env, agent_params):
#         super(ACAgent, self).__init__()
#
#         self.env = env
#         self.agent_params = agent_params
#
#         self.gamma = self.agent_params['gamma']
#         self.standardize_advantages = self.agent_params['standardize_advantages']
#
#         self.actor = MLPPolicyAC(
#             self.agent_params['ac_dim'],
#             self.agent_params['ob_dim'],
#             self.agent_params['n_layers'],
#             self.agent_params['size'],
#             self.agent_params['discrete'],
#             self.agent_params['learning_rate'],
#         )
#
#         self.replay_buffer = ReplayBuffer()
#
#     def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
#         # TODO Implement the following pseudocode:
#         # for agent_params['num_critic_updates_per_agent_update'] steps,
#         #     update the critic
#         # for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
#             # critic_log = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)
#
#         # advantage = estimate_advantage(...)
#         advantage = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)
#
#         # for agent_params['num_actor_updates_per_agent_update'] steps,
#         #     update the actor
#         # for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
#         #     actor_log = self.actor.update(ob_no, ac_na, advantage)
#         actor_log = self.actor.update(ob_no, ac_na, advantage)
#
#         loss = OrderedDict()
#         # loss['Critic_Loss'] = critic_log
#         loss['Actor_Loss'] = actor_log
#
#         return loss
#
#     def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
#         # TODO Implement the following pseudocode:
#         # 1) query the critic with ob_no, to get V(s)
#         # 2) query the critic with next_ob_no, to get V(s')
#         # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
#         # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
#         # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
#         # v = self.critic.forward_np
#         # adv_n = re_n + self.gamma * v(next_ob_no) * (1-terminal_n) - v(ob_no)
#         advantages = np.concatenate([self._discounted_cumsum(r) for r in re_n])
#
#         if self.standardize_advantages:
#             # adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
#             mean = np.mean(advantages)
#             std = np.std(advantages)
#             advantages = normalize(advantages, mean, std)
#         return advantages
#
#     def add_to_replay_buffer(self, paths):
#         self.replay_buffer.add_rollouts(paths)
#
#     def sample(self, batch_size):
#         return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)
#
#     def _discounted_cumsum(self, rewards):
#         """
#             Helper function which
#             -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
#             -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
#         """
#
#         # TODO: create `list_of_discounted_returns`
#         T = len(rewards)
#         list_of_discounted_cumsums = np.zeros(T)
#         list_of_discounted_cumsums[T - 1] = rewards[T - 1]
#         for i in range(T):
#             if i > 0:
#                 list_of_discounted_cumsums[T - i - 1] = rewards[T - i - 1] + self.gamma * list_of_discounted_cumsums[
#                     T - i]
#
#         return list_of_discounted_cumsums
#
