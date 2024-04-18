import torch
import time
import logging
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np
import math
import csv
from scipy.signal import savgol_filter

f = open('mppi.csv', 'a', encoding='utf-8', newline='')
wr = csv.writer(f)

logger = logging.getLogger(__name__)

def is_tensor_like(x):
    return torch.is_tensor(x) or type(x) is np.ndarray


class MPPI():
    """
    Model Predictive Path Integral control
    This implementation batch samples the trajectories and so scales well with the number of samples K.

    Implemented according to algorithm 2 in Williams et al., 2017
    'Information Theoretic MPC for Model-Based Reinforcement Learning',
    based off of https://github.com/ferreirafabio/mppi_pendulum
    """

    def __init__(self, dynamics, running_cost, nx, nu, noise_sigma, num_samples=100, horizon=15, device="cpu",
                 terminal_state_cost=None,
                 lambda_=1.,
                 gamma_=1.,
                 noise_mu=None,
                 u_min=None,
                 u_max=None,
                 u_init=None,
                 U_init=None,
                 u_scale=1,
                 u_per_command=1,
                 step_dependent_dynamics=False,
                 rollout_var_cost=0,
                 rollout_var_discount=0.95,
                 sample_null_action=False,
                 smooth="no filter"):
        """
        :param dynamics: function(state, action) -> next_state (K x nx) taking in batch state (K x nx) and action (K x nu)
        :param running_cost: function(state, action) -> cost (K x 1) taking in batch state and action (same as dynamics)
        :param nx: state dimension
        :param noise_sigma: (nu x nu) control noise covariance (assume v_t ~ N(u_t, noise_sigma))
        :param num_samples: K, number of trajectories to sample
        :param horizon: T, length of each trajectory
        :param device: pytorch device
        :param terminal_state_cost: function(state) -> cost (K x 1) taking in batch state
        :param lambda_: temperature, positive scalar where larger values will allow more exploration
        :param noise_mu: (nu) control noise mean (used to bias control samples); defaults to zero mean
        :param u_min: (nu) minimum values for each dimension of control to pass into dynamics
        :param u_max: (nu) maximum values for each dimension of control to pass into dynamics
        :param u_init: (nu) what to initialize new end of trajectory control to be; defeaults to zero
        :param U_init: (T x nu) initial control sequence; defaults to noise
        :param step_dependent_dynamics: whether the passed in dynamics needs horizon step passed in (as 3rd arg)
        :param rollout_var_cost: Cost attached to the variance of costs across trajectory rollouts
        :param rollout_var_discount: Discount of variance cost over control horizon
        :param sample_null_action: Whether to explicitly sample a null action (bad for starting in a local minima)
        :param smooth: Whether to explicitly sample a null action (bad for starting in a local minima)

        * option
            :param rollout_samples: M, number of state trajectories to rollout for each control trajectory
            (should be 1 for deterministic dynamics and more for models that output a distribution)
        """

        self.smooth_method = smooth
        self.device = device
        self.dtype = noise_sigma.dtype
        self.K = num_samples  # N_SAMPLES
        self.T = horizon  # TIMESTEPS

        # dimensions of state and control
        self.nx = nx
        self.nu = 1 if len(noise_sigma.shape) == 0 else noise_sigma.shape[0]
        self.lambda_ = lambda_
        self.gamma_ = gamma_

        if noise_mu is None:
            noise_mu = torch.zeros(self.nu, dtype=self.dtype)

        if u_init is None:
            u_init = torch.zeros_like(noise_mu)

        if U_init is None:                                           
            U_init = torch.zeros(self.T, self.nu).to(device)

        # handle 1D edge case
        if self.nu == 1:                                             
            noise_mu = noise_mu.view(-1)
            noise_sigma = noise_sigma.view(-1, 1)

        # bounds
        self.u_min = u_min
        self.u_max = u_max
        self.u_scale = u_scale
        self.u_per_command = u_per_command
        # make sure if any of them is specified, both are specified
        if self.u_max is not None and self.u_min is None:
            if not torch.is_tensor(self.u_max):
                self.u_max = torch.tensor(self.u_max)
            self.u_min = -self.u_max
        if self.u_min is not None and self.u_max is None:
            if not torch.is_tensor(self.u_min):
                self.u_min = torch.tensor(self.u_min)
            self.u_max = -self.u_min
        if self.u_min is not None:
            self.u_min = self.u_min.to(device=self.device)
            self.u_max = self.u_max.to(device=self.device)

        self.noise_mu = noise_mu.to(self.device)
        self.noise_sigma = noise_sigma.to(self.device)
        self.noise_sigma_inv = torch.inverse(self.noise_sigma)
        self.noise_dist = MultivariateNormal(
            self.noise_mu, covariance_matrix=self.noise_sigma)
        # T x nu control sequence
        self.U = U_init
        self.u_init = u_init.to(self.device)

        if self.U is None:
            self.U = self.noise_dist.sample((self.T,))
            self.U = torch.zeros_like(self.U)

        self.U_history = torch.zeros_like(self.U)[:5]

        self.step_dependency = step_dependent_dynamics
        self.F = dynamics
        self.running_cost = running_cost
        self.terminal_state_cost = terminal_state_cost
        self.sample_null_action = sample_null_action
        self.state = None

        # handling dynamics models that output a distribution (take multiple trajectory samples)
        self.rollout_var_cost = rollout_var_cost
        self.rollout_var_discount = rollout_var_discount

        # sampled results from last command
        self.cost_total = None
        self.cost_total_non_zero = None
        self.omega = None
        self.states = None
        self.actions = None

    def _dynamics(self, state, u, t):
        return self.F(state, u, t) if self.step_dependency else self.F(state, u)

    def _running_cost(self, state):
        return self.running_cost(state)

    def command(self, state):
        """
        :param state: (nx) or (K x nx) current state, or samples of states (for propagating a distribution of states)
        :returns action: (nu) best action
        """
        # shift command 1 time step
        self.U = torch.roll(self.U, -1, dims=0)
        self.U[-1] = self.u_init

        perturbed_action = self.action_sampling()

        cost_total, states = self._compute_batch_rollout_costs(
            perturbed_action, state)
        self.omega = self._compute_weighting(cost_total)

        weighted_noise = torch.sum(
            self.omega.view(-1, 1, 1) * self.noise, dim=0)

        if self.smooth_method == "no filter":
            self.U += weighted_noise
        elif self.smooth_method == "smooth u":
            self.U += weighted_noise
            U_filtered = savgol_filter(
                            torch.cat([self.U_history, self.U]).to('cpu'), 5, 3, axis=0)
            self.U = torch.tensor(U_filtered[-self.T:]).to(self.device)

            self.U_history = torch.roll(self.U_history, -1, dims=0)
            self.U_history[-1] = self.U[0]
        elif self.smooth_method == "smooth noise":
            self.U += torch.tensor(savgol_filter(weighted_noise.to('cpu'),
                                                5, 3, axis=0)).to(self.device)
        else:
            raise Exception("Wrong smooth option !!!")


        action = self.U[0]

        return action

    def reset(self):
        """
        Clear controller state after finishing a trial
        """
        self.U = torch.zeros_like(self.U)

    def _compute_weighting(self, cost_total):
        beta = torch.min(cost_total)
        cost_total_non_zero = torch.exp(-1/self.lambda_ * (cost_total - beta))
        eta = torch.sum(cost_total_non_zero)
        omega = (1. / eta) * cost_total_non_zero
        return omega

    def _compute_batch_rollout_costs(self, perturbed_actions, state):
        K, T, nu = perturbed_actions.shape
        assert nu == self.nu

        cost_total = torch.zeros(K, device=self.device, dtype=self.dtype)
        cost_samples = torch.zeros(K, device=self.device, dtype=self.dtype)

        # allow propagation of a sample of states (ex. to carry a distribution), or to start with a single state
        # state -> nx
        if state.shape == (K, self.nx):
            state = state
        else:
            state = state.view(1, -1).repeat(K, 1)
        # state -> K*nu

        states = []
        actions = []

        prev_map_mask = torch.zeros((K), device=self.device)
        for t in range(T):
            # perturbed_actions -> K*T*nu
            # perturbed_actions[:, t] -> K*nu
            u = self.u_scale * perturbed_actions[:, t]  # v -> K*nu

            state = self._dynamics(state, u, t)
            c = self._running_cost(state)  # c -> K

            cost_samples += c  # cost_samples -> K

            # Save total states/actions
            states.append(state)
            actions.append(u)

        # actions -> [K*nu, K*nu ...] with size T
        # torch.stack(actions, dim=-2) -> K*T*nu
        actions = torch.stack(actions, dim=-2)
        states = torch.stack(states, dim=-2)

        # terminal state cost
        if self.terminal_state_cost:
            phi = self.terminal_state_cost(states, actions)
            cost_samples += phi

        action_cost = self.gamma_ * self.noise @ self.noise_sigma_inv

        # action_cost -> K*T*nu
        # U -> T*nu
        perturbation_cost = torch.sum(self.U * action_cost, dim=(1, 2))

        cost_total += cost_samples + perturbation_cost  # K dim

        return cost_total, states

    def _bound_action(self, action):

        return torch.max(torch.min(action, self.u_max), self.u_min)  # action

    def action_sampling(self):
        # parallelize sampling across trajectories
        # resample noise each time we take an action

        # Small portion are just guanssian perturbation aroung zero
        self.noise = self.noise_dist.sample(
            (round(self.K*0.99), self.T))  # K*T*nu (noise_dist has nu-dim)
        # broadcast own control to noise over samples; now it's K x T x nu
        perturbed_action = self.U + self.noise
        perturbed_action = torch.cat(
            [perturbed_action, self.noise_dist.sample((round(self.K*0.01), self.T))])

        if self.sample_null_action:
            perturbed_action[self.K - 1] = 0

        perturbed_action = self._bound_action(perturbed_action)

        # remove U to earn bounded noise
        self.noise = perturbed_action - self.U

        return perturbed_action


def angle_normalize(x):
    return (((x + math.pi) % (2 * math.pi)) - math.pi)

def run_mppi_episode(mppi, env, dataset_append, retrain_dynamics, cost, model_save, cost_tolerance, SUCCESS_CRITERION, retrain_after_iter=50, num_episode=30, render=True):
    dataset_count = 0
    cost_history = []
    cost_ = 0.
    for ep in range(num_episode):
        env.reset()
        success_count = 0
        cost_episode = []

        while True:
            if render:
                env.render()           
            state = env.state
            state = torch.tensor(state, dtype=mppi.noise_sigma.dtype).to(device=mppi.device)
            command_start = time.perf_counter()
            action = mppi.command(state)
            elapsed = time.perf_counter() - command_start
            s, _, done, _ = env.step(action.cpu().numpy())
            next_state = env.state
            next_state = torch.tensor(next_state, dtype=mppi.noise_sigma.dtype).to(device=mppi.device)

            # Collect Training datas
            dataset_append(state, action, next_state)

            logger.debug(
                "action taken: %.4f cost received: %.4f time taken: %.5fs", action, cost_, elapsed)

            dataset_count += 1
            di = dataset_count % retrain_after_iter
            if di == 0 and dataset_count > 0:
                retrain_dynamics()

            cost_ = cost(next_state.view(1, -1))
            cost_episode.append(cost_.item())

            if cost_ < cost_tolerance:
                success_count += 1
                if success_count >= SUCCESS_CRITERION:
                    print("Task completed")
                    cost_history.append(cost_episode)
                    model_save()
                    return cost_history
            else:
                success_count = 0

            if done:
                print("Episode {} terminated".format(ep + 1))
                break
        wr.writerow(cost_episode)
        cost_history.append(cost_episode)
    return cost_history