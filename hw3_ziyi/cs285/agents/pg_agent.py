from collections import OrderedDict

from cs285.critics.bootstrapped_continuous_critic import \
    BootstrappedContinuousCritic
from cs285.infrastructure.replay_buffer import ReplayBuffer
from cs285.infrastructure.utils import *
from cs285.policies.MLP_policy import MLPPolicyAC
from .base_agent import BaseAgent


class ACAgent(BaseAgent):
    def __init__(self, env, agent_params):
        super(ACAgent, self).__init__()

        self.env = env
        self.agent_params = agent_params

        self.gamma = self.agent_params['gamma']
        self.standardize_advantages = self.agent_params['standardize_advantages']

        self.actor = MLPPolicyAC(
            self.agent_params['ac_dim'],
            self.agent_params['ob_dim'],
            self.agent_params['n_layers'],
            self.agent_params['size'],
            self.agent_params['discrete'],
            self.agent_params['learning_rate'],
        )
        self.critic = BootstrappedContinuousCritic(self.agent_params)

        self.replay_buffer = ReplayBuffer()

    def train(self, ob_no, ac_na, re_n, next_ob_no, terminal_n):
        # TODO Implement the following pseudocode:
        # for agent_params['num_critic_updates_per_agent_update'] steps,
        #     update the critic
        # for _ in range(self.agent_params['num_critic_updates_per_agent_update']):
            # critic_log = self.critic.update(ob_no, ac_na, next_ob_no, re_n, terminal_n)

        # advantage = estimate_advantage(...)
        advantage = self.estimate_advantage(ob_no, next_ob_no, re_n, terminal_n)

        # for agent_params['num_actor_updates_per_agent_update'] steps,
        #     update the actor
        # for _ in range(self.agent_params['num_actor_updates_per_agent_update']):
        #     actor_log = self.actor.update(ob_no, ac_na, advantage)
        actor_log = self.actor.update(ob_no, ac_na, advantage)

        loss = OrderedDict()
        # loss['Critic_Loss'] = critic_log
        loss['Actor_Loss'] = actor_log

        return loss

    def estimate_advantage(self, ob_no, next_ob_no, re_n, terminal_n):
        # TODO Implement the following pseudocode:
        # 1) query the critic with ob_no, to get V(s)
        # 2) query the critic with next_ob_no, to get V(s')
        # 3) estimate the Q value as Q(s, a) = r(s, a) + gamma*V(s')
        # HINT: Remember to cut off the V(s') term (ie set it to 0) at terminal states (ie terminal_n=1)
        # 4) calculate advantage (adv_n) as A(s, a) = Q(s, a) - V(s)
        # v = self.critic.forward_np
        # adv_n = re_n + self.gamma * v(next_ob_no) * (1-terminal_n) - v(ob_no)
        advantages = np.concatenate([self._discounted_cumsum(r) for r in re_n])

        if self.standardize_advantages:
            # adv_n = (adv_n - np.mean(adv_n)) / (np.std(adv_n) + 1e-8)
            mean = np.mean(advantages)
            std = np.std(advantages)
            advantages = normalize(advantages, mean, std)
        return advantages

    def add_to_replay_buffer(self, paths):
        self.replay_buffer.add_rollouts(paths)

    def sample(self, batch_size):
        return self.replay_buffer.sample_recent_data(batch_size, concat_rew=False)

    def _discounted_cumsum(self, rewards):
        """
            Helper function which
            -takes a list of rewards {r_0, r_1, ..., r_t', ... r_T},
            -and returns a list where the entry in each index t' is sum_{t'=t}^T gamma^(t'-t) * r_{t'}
        """

        # TODO: create `list_of_discounted_returns`
        T = len(rewards)
        list_of_discounted_cumsums = np.zeros(T)
        list_of_discounted_cumsums[T - 1] = rewards[T - 1]
        for i in range(T):
            if i > 0:
                list_of_discounted_cumsums[T - i - 1] = rewards[T - i - 1] + self.gamma * list_of_discounted_cumsums[
                    T - i]

        return list_of_discounted_cumsums
