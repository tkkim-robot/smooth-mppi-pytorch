# SMPPI Implementation in Pytorch

This repository implements the idea of Smooth Model Predictive Path Integral control (SMPPI), using neural network dynamics model in pytorch. SMPPI is a general framework that is able to obtain smooth actions using sampling-based MPC without any extra smoothing algorithms (e.g. Savitzky-Golay Filter). The related paper will be relased soon.

# Installation

Clone repository, then 'pip install -e .' or 'pip3 install -e .' based on your environment.

Or you can manually install dependencies:

    - pytorch
    - numpy
    - gym
    - scipy

# How to Run Example

You can run our test example by:

For pendulum,
```bash
python gym_pendulum.py
```
For cartpole,
```bash
python gym_cartpole.py
```


It's an inverted pendulum in gym environment. The sample results of the four different controllers are shown below:

|                                                     MPPI w/o Smoothing                                                     |                                            MPPI (apply smoothing on noise sequence)                                            |
| :------------------------------------------------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------: |
|  <img src="https://user-images.githubusercontent.com/40379815/132696053-a7966a8d-28d9-43cd-a174-d8d6b16d1087.gif" width="200px" height="200px"> | <img src="https://user-images.githubusercontent.com/40379815/132696313-cc1c9876-c3e7-432b-bf66-c41443409d12.gif" width="200px" height="200px"> |
|                                        __MPPI (apply smoothing on control sequence)__                                         |                                                             __SMPPI__                                                              |
| <img src="https://user-images.githubusercontent.com/40379815/132696344-00f2631f-90dd-4950-ae86-5cda5450244f.gif" width="200px" height="200px"> |  <img src="https://user-images.githubusercontent.com/40379815/132696368-87c7840c-6706-4c96-8e1f-668c6820d03f.gif" width="200px" height="200px">     |

It's a cartpole (continuous action) environment. Since MPPI requires random noise sampling of actions, cartpole environment in openAI gym(which has only two discrete actions, Left or Right) is not suitable for MPPI test. So we made custom environment which provides continuous action. In this environment, action can vary continuously from -10.0 to 10.0. (For more detail, see envs/cartpole_cont.py)

The sample result of SMPPI controller is shown below:

<img src="https://user-images.githubusercontent.com/95032544/144233059-996d762b-11ce-4916-b11c-12dadde06757.gif" width="300px" height="200px">    


They are collecting the state-action pairs dataset with exploration. The dynamics models are retrained every 50 iterations. SMPPI can accurately find the optimal action sequence, right after re-training the neural network dynamics. 
# How to Use

Simply import SMPPI from 'pytorch_mppi', you can obtain sequence of smooth optimal actions from sampling-based MPC.

```python
from pytorch_mppi import smooth_mppi as smppi
# define your dynamics model (both work for nominal dynamics or neural network approximation)
# create controller with chosen parameters
mppi_env = smppi.MPPI(dynamics, running_cost, nx, nu, noise_sigma, 
                            num_samples=N_SAMPLES,
                            horizon=TIMESTEPS, lambda_=lambda_, gamma_=gamma_, device=device,
                            w_action_seq_cost=Omega,
                            u_min=torch.tensor(D_ACTION_LOW, dtype=dtype, device=device),
                            u_max=torch.tensor(D_ACTION_HIGH, dtype=dtype, device=device),
                            action_min=torch.tensor(ACTION_LOW, dtype=dtype, device=device),
                            action_max=torch.tensor(ACTION_HIGH, dtype=dtype, device=device))

# assuming you have a gym-like env
obs = env.reset()
for i in range(100):
    action = mppi_env.command(obs)
    obs, reward, done, _ = env.step(action.cpu().numpy())
```

Alternatively, you can test the original MPPI with different smoothing methods.

```python
from pytorch_mppi import mppi
# define your dynamics model (both work for nominal dyanmics or neural network approximation)
# create controller with chosen parameters
mppi_env = mppi.MPPI(dynamics, running_cost, nx, nu, noise_sigma, 
                            num_samples=N_SAMPLES,
                            horizon=TIMESTEPS, lambda_=lambda_, gamma_=gamma_, device=device,
                            u_min=torch.tensor(ACTION_LOW, dtype=dtype, device=device),
                            u_max=torch.tensor(ACTION_HIGH, dtype=dtype, device=device),
                            smooth=SMOOTHING_METHOD)
```

You have three options for the 'SMOOTHING_METHOD':

1.  __"no filter"__ : no smoothing
2.  __"smooth u"__ : smooth control sequence after adding noise
3.  __"smooth noise"__ : smooth noise sequence before adding noise

For the smoothing algorithm, we use convolutional Savitzky-Golay Filter (in scipy).

# Parameters Description

### lambda\_

- temperature, positive scalar where larger values will allow more exploration
- we recommend 10.0 ~ 20.0 when you have more than 1,000 samples

### gamma\_

- running action cost parameter
- see [MPPI paper](https://ieeexplore.ieee.org/abstract/document/8558663?casa_token=RTtdCK4jrykAAAAA:YgIhGuAKv_dPA_JjvaxHT2npZuaFVI0utE4JSnDkALwqbUvh676UydsOUg44ka5rawG7edPo) for more detail

### w_action_seq_cost

- (nu x nu) weight parameter for smoothing action sequence

### num_samples

- number of trajectories to sample; generally the more the better. (determine this parameter based on the size of your neural network model.)
- try to have it between 1K ~ 10K, if your GPU allows it to.

### noise_sigma

- (nu x nu) control noise covariance; larger covariance yeilds more exploration

| See our paper for further information (will be released soon).

# Requirements

- `next state <- dynamics(state, action)` function (doesn't have to be true dynamics)
  - `state` is `K x nx`, `action` is `K x nu`
- `cost <- running_cost(state, action)` function
  - `cost` is `K x 1`, state is `K x nx`, `action` is `K x nu`

| __The shapes of the important tensors (such as 'states', 'noise', 'actions') are all commented on the scripts.__

# Related Works

This repository was built based on the [project of pytorch implementation of MPPI](https://github.com/UM-ARM-Lab/pytorch_mppi), that I had contributed before. Thanks for the great work of [LemonPi](https://github.com/LemonPi).
