import gym
import numpy as np
import torch
import logging
import math
from gym import wrappers, logger as gym_log
from torch.utils.data import DataLoader
from torch.utils.data import Dataset

gym_log.set_level(gym_log.INFO)
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO,
                    format='[%(levelname)s %(asctime)s %(pathname)s:%(lineno)d] %(message)s',
                    datefmt='%m-%d %H:%M:%S')

SMPPI = True

downward_start = True
INIT_VEL = 0

# three options for control smoothing
# 1: "no filter"    : no smoothing
# 2: "smooth u"     : smooth control sequence after adding noise
# 3: "smooth noise" : smooth noise sequence before adding noise
# for more detail, please wait for our paper now under review.
if not SMPPI:
    SMOOTH = "no filter"


if SMPPI:
    from pytorch_mppi import smooth_mppi as mppi
else:
    from pytorch_mppi import mppi

if __name__ == "__main__":
    ENV_NAME = "Pendulum-v1"
    TIMESTEPS = 15  # T
    N_SAMPLES = 1000  # K
    ACTION_LOW = -2.0
    ACTION_HIGH = 2.0
    D_ACTION_LOW = -8.0
    D_ACTION_HIGH = 8.0

    device = torch.device("cuda") if torch.cuda.is_available(
    ) else torch.device("cpu")
    dtype = torch.double

    noise_sigma = torch.tensor([1.], device=device, dtype=dtype)
    # if size of action space is larger than 1:
    # noise_sigma = torch.tensor([[1, 0], [0, 2]], device=d, dtype=dtype)
    lambda_ = 10.
    gamma_ = 0.1

    import random

    randseed = 42
    if randseed is None:
        randseed = random.randint(0, 1000000)
    random.seed(randseed)
    np.random.seed(randseed)
    torch.manual_seed(randseed)
    logger.info("random seed %d", randseed)

    # new hyperparmaeters for approximate dynamics
    H_UNITS = 32
    TRAIN_EPOCH = 100  # 150
    BOOT_STRAP_ITER = 0
    EPISODE_CUT = 1000
    BATCH_SIZE = 50

    cost_tolerance = 0.1
    SUCCESS_CRITERION = 300

    nx = 2
    nu = 1
    # network output is state residual
    network = torch.nn.Sequential(
        torch.nn.Linear(nx + nu + 1, H_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(H_UNITS, H_UNITS),
        torch.nn.Tanh(),
        torch.nn.Linear(H_UNITS, nx)
    ).double().to(device=device)

    def dynamics(state, perturbed_action):
        u = torch.clamp(perturbed_action, ACTION_LOW, ACTION_HIGH)
        if state.dim() == 1 or u.dim() == 1:
            state = state.view(1, -1)
            u = u.view(1, -1)
        xu = torch.cat((state, u), dim=1)
        # feed in cosine and sine of angle instead of theta
        xu = torch.cat((torch.sin(
            xu[:, 0]).view(-1, 1), torch.cos(xu[:, 0]).view(-1, 1), xu[:, 1:]), dim=1)

        network.eval()
        with torch.no_grad():
            state_residual = network(xu)
        # output dtheta directly so can just add
        next_state = state.clone().detach() + state_residual
        next_state[:, 0] = angle_normalize(next_state[:, 0])
        return next_state

    def true_dynamics(state, perturbed_action):
        # true dynamics from gym
        th = state[:, 0].view(-1, 1)
        thdot = state[:, 1].view(-1, 1)

        g = 10
        m = 1
        l = 1
        dt = 0.05

        u = perturbed_action
        u = torch.clamp(u, -2, 2)

        newthdot = thdot + (-3 * g / (2 * l) *
                            torch.sin(th + np.pi) + 3. / (m * l ** 2) * u) * dt
        newth = th + newthdot * dt
        newthdot = torch.clamp(newthdot, -8, 8)

        next_state = torch.cat((newth, newthdot), dim=1)
        return next_state

    def angular_diff_batch(a, b):
        """Angle difference from b to a (a - b)"""
        d = a - b
        d[d > math.pi] -= 2 * math.pi
        d[d < -math.pi] += 2 * math.pi
        return d

    def angle_normalize(x):
        return (((x + math.pi) % (2 * math.pi)) - math.pi)

    def running_cost(state):
        theta = state[:, 0]
        theta_dt = state[:, 1]
        cost = angle_normalize(theta) ** 2 + 0.1 * theta_dt ** 2
        return cost

    dataset_xu = None
    dataset_Y = None
    # create some true dynamics validation set to compare model against
    Nv = 1000
    statev = torch.cat(((torch.rand(Nv, 1, dtype=dtype, device=device) - 0.5) * 2 * math.pi,
                        (torch.rand(Nv, 1, dtype=dtype, device=device) - 0.5) * 16), dim=1)
    actionv = (torch.rand(Nv, 1, dtype=dtype, device=device) -
               0.5) * (ACTION_HIGH - ACTION_LOW)

    class CustomDataset(Dataset):
        def __init__(self, x, y):
            self.x_data = x
            self.y_data = y

        def __len__(self):
            return len(self.x_data)

        def __getitem__(self, item):
            x_ = self.x_data[item]
            y_ = self.y_data[item]
            return x_, y_

    def dataset_append(state, action, next_state):
        global dataset_xu, dataset_Y
        state[0] = angle_normalize(state[0])
        next_state[0] = angle_normalize(next_state[0])
        action = torch.clamp(action.clone().detach(), ACTION_LOW, ACTION_HIGH)

        xu = torch.cat((state, action), dim=0)

        xu = torch.tensor(
            (torch.sin(xu[0]),
             torch.cos(xu[0]),
             xu[1],
             xu[2])).view(1, -1)
        dtheta = angular_diff_batch(next_state[0], state[0])
        dtheta_dot = next_state[1] - state[1]
        Y = torch.tensor((dtheta, dtheta_dot)).view(1, -1).clone().detach()

        if dataset_xu is None and dataset_Y is None:
            dataset_xu = xu
            dataset_Y = Y

        else:
            dataset_xu = torch.cat((dataset_xu, xu), dim=0)
            dataset_Y = torch.cat((dataset_Y, Y), dim=0)

    def train(epoch=TRAIN_EPOCH):
        global dataset_xu, dataset_Y, network
        # thaw network
        for param in network.parameters():
            param.requires_grad = True

        optimizer = torch.optim.Adam(network.parameters())
        train_dataset = CustomDataset(dataset_xu, dataset_Y)
        train_loader = DataLoader(
            train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

        network.train()
        for i in range(epoch):
            # MSE loss
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                yhat = network(x)
                loss = (y - yhat).norm(2, dim=1) ** 2
                optimizer.zero_grad()
                loss.mean().backward()
                optimizer.step()
                logger.debug("ds %d epoch %d loss %f",
                             dataset_xu.shape[0], i, loss.mean().item())

        # freeze network
        for param in network.parameters():
            param.requires_grad = False

        # evaluate network against true dynamics
        yt = true_dynamics(statev, actionv)
        yp = dynamics(statev, actionv)
        dtheta = angular_diff_batch(yp[:, 0], yt[:, 0])
        dtheta_dt = yp[:, 1] - yt[:, 1]
        E = torch.cat((dtheta.view(-1, 1), dtheta_dt.view(-1, 1)),
                      dim=1).norm(dim=1)
        logger.info("Error with true dynamics theta %f theta_dt %f norm %f", dtheta.abs().mean(),
                    dtheta_dt.abs().mean(), E.mean())
        logger.debug("Start next collection sequence")

    def model_save():
        global network
        torch.save(network.state_dict(), 'model_weights_pendulum.pth')

    env = gym.make(ENV_NAME).env  # bypass the default TimeLimit wrapper
    # bootstrap network with random actions
    if BOOT_STRAP_ITER:
        logger.info(
            "bootstrapping with random action for %d actions", BOOT_STRAP_ITER)
        data_count = 0
        while True:
            env.reset()
            for i in range(EPISODE_CUT):
                state = env.state
                state = torch.tensor(
                    state, dtype=torch.float64).to(device=device)
                action = np.random.uniform(low=ACTION_LOW, high=ACTION_HIGH)
                action = torch.tensor(
                    [action], dtype=torch.float64).to(device=device)
                s, _, done, _ = env.step(action.cpu().numpy())
                next_state = env.state
                next_state = torch.tensor(
                    next_state, dtype=torch.float64).to(device=device)
                dataset_append(state, action, next_state)
                data_count += 1
                if data_count == BOOT_STRAP_ITER:
                    break
                if done:
                    break
            if data_count == BOOT_STRAP_ITER:
                break
        train(epoch=500)
        logger.info("bootstrapping finished")

    env = wrappers.Monitor(env, '/tmp/mppi/', force=True)
    env.reset()
    if downward_start:
        env.env.state = [np.pi, INIT_VEL]

    if SMPPI:
        mppi_gym = mppi.MPPI(dynamics, running_cost, nx, nu, noise_sigma,
                             num_samples=N_SAMPLES,
                             horizon=TIMESTEPS,
                             lambda_=lambda_,
                             gamma_=gamma_,
                             device=device,
                             u_min=torch.tensor(
                                 D_ACTION_LOW, dtype=dtype, device=device),
                             u_max=torch.tensor(
                                 D_ACTION_HIGH, dtype=dtype, device=device),
                             action_min=torch.tensor(
                                 ACTION_LOW, dtype=dtype, device=device),
                             action_max=torch.tensor(ACTION_HIGH, dtype=dtype, device=device))
    else:
        mppi_gym = mppi.MPPI(dynamics, running_cost, nx, nu, noise_sigma,
                             num_samples=N_SAMPLES,
                             horizon=TIMESTEPS,
                             lambda_=lambda_,
                             gamma_=gamma_,
                             device=device,
                             u_min=torch.tensor(
                                 ACTION_LOW, dtype=dtype, device=device),
                             u_max=torch.tensor(
                                 ACTION_HIGH, dtype=dtype, device=device),
                             smooth=SMOOTH)

    cost_history = mppi.run_mppi_episode(
        mppi_gym, env, dataset_append, train, running_cost, model_save, cost_tolerance, SUCCESS_CRITERION)
