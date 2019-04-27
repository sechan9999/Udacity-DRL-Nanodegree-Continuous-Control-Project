# Report of the Implementation & Performance

This report is made as a part of this project.

## Agent Algorithm

The algorithms for the agent implemented in `ddqn_agent.py`, `model.py`, and `replay_buffer.py` are [Deep Deterministic Policy Gradient](https://arxiv.org/abs/1509.02971) and [Distributional Deep Deterministic Policy Gradient](https://arxiv.org/abs/1804.08617).

Both implementation incorporate [Prioritized Experienced Replay](https://arxiv.org/abs/1511.05952), [Multi-step Bootstrap Targets](https://arxiv.org/abs/1602.01783), and [Noisy Networks](https://arxiv.org/abs/1706.10295). Each component comes with its own set of hyperparameters and setting certain parameters to certain values can disable the effects of the components.

Most part of the implementation is from [my Rainbow implementation used in the previous project](https://github.com/wytyang00/Udacity-DRL-Nanodegree-Navigation-Project). The overall structures for the `ReplayBuffer` and `Agent` are very similar, and the `NoisyLinear` is used as it is.

The critic model contains both scalar and distributional network architectures, and you can choose which architecture to use via one of the hyperparameters, `distributional`.

There are two types of noise available: parameter space noise through noisy linear layers, and gaussian noise on the action space. The standard deviation of the gaussian noise is controlled by `eps` training parameters. Ornstein-Uhlenbeck noise is a popular and effective noise for continuous control tasks like this, but I have only implemented gaussian noise as I was following the methods used in [Distributional Deep Deterministic Policy Gradient](https://arxiv.org/abs/1804.08617).

For [my previous project](https://github.com/wytyang00/Udacity-DRL-Nanodegree-Navigation-Project), I wasn't especially interested in implementing the "Sum Tree" algorithm for efficient sampling as described in [Prioritized Experienced Replay](https://arxiv.org/abs/1511.05952) since I was too caught up in making the whole algorithm working. However, this time, the time cost for training was a huge issue, taking several days for training an agent, which made it hard for me to fiddle with the hyperparameters. So, I have implemented the "Sum Tree" algorithm using NumPy arrays and vectorized operations, and incorporated it in the `ReplayBuffer`. This kept the sampling and updating costs low and constant, significantly boosting the training speed.

## Hyperparameters

Most of the avalilable hyperparameters are similar to the ones I had in the [previous project](https://github.com/wytyang00/Udacity-DRL-Nanodegree-Navigation-Project), but a few hyperparameters, such as `eps`, have slightly different effects and there are also some additional hyperparameters added. I ran several training sessions with different hyperparameter settings and settled with these values using **20-Agents version**:

```python
hyperparams = {
    # Reproducibility
    'seed'                : 4,        # random seed for reproducible results

    # Agent basic parameters
    'batch_size'          : 256,      # batch size for each learning step
    'buffer_size'         : int(1e6), # up to how many recent experiences to keep
    'start_since'         : 256,      # how many experiences to collect before starting learning
    'gamma'               : 0.99,     # discount factor
    'update_every'        : 1,        # update step frequency
    'n_updates'           : 10,       # number of updates per update step
    'tau'                 : 1e-3,     # soft-update parameter [0, 1]

    'actor_lr'            : 5e-4,     # learning rate for the actor network
    'critic_lr'           : 5e-4,     # learning rate for the critic network
    'clip'                : 1,        # gradient clipping to prevent gradient spikes
    'weight_decay'        : 0,        # weight decay for the *critic* network

    'distributional'      : True,     # whether to use distributional learning

    # Prioritized Experience Replay Parameters
    'priority_eps'        : 1e-3,     # base priority in order to ensure nonzero priorities
    'a'                   : 1.,       # priority exponent parameter [0, 1]

    # n-step Bootstrap Target parameter
    'n_multisteps'        : 4,        # number of steps to bootstrap
    'separate_experiences': False,    # whether to store experiences with no overlap

    # Distributional Learning parameters
    'v_min'               : 0,        # minimum value for support
    'v_max'               : 10,       # maximum value for support
    'n_atoms'             : 51,       # number of atoms for distribution

    # Noisy Layer parameters
    'initial_sigma'       : 0.050,    # initial noisy parameter value
    'linear_type'         : 'noisy',  # either 'linear' or 'noisy'
    'factorized'          : False     # whether to use factorized gaussian noise or not(independent gaussian noise)
}

train_params = {
    'n_episodes'           : 30,    # number of total episodes to train
    'continue_after_solved': False, # whether to keep training after the environment is solved

    # Exploration using gaussian noise
    'eps_start'            : 0.0,   # initial epsilon value
    'eps_min'              : 0.0,   # minimum value for epsilon
    'eps_decay'            : 0.0,   # epsilon decay rate

    # Importance-Sampling Weight parameter for Prioritized Experience Replay
    'beta_start'           : 1.,   # starting value
    'beta_end'             : 1.    # end value
}
```

<br/>
The architecture of the Actor Network is as follows:
<table class="unchanged rich-diff-level-one">
  <tr>
    <td align="center">Input (, [<code>batch_size</code>, <code>state_size</code>])</td>
  </tr>
  <tr>
    <td align="center">NoisyLinear ([<code>batch_size</code>, <code>state_size</code>], [<code>batch_size</code>, 256])</td>
  </tr>
  <tr>
    <td align="center">LeakyReLU(neg_slope=0.01)</td>
  </tr>
  <tr>
    <td align="center">NoisyLinear ([<code>batch_size</code>, 256], [<code>batch_size</code>, 128])</td>
  </tr>
  <tr>
    <td align="center">LeakyReLU(neg_slope=0.01)</td>
  </tr>
  <tr>
    <td align="center">NoisyLinear ([<code>batch_size</code>, 128], [<code>batch_size</code>, <code>action_size</code>])</td>
  </tr>
  <tr>
    <td align="center">Tanh()</td>
  </tr>
  <tr>
    <td align="center">Output (, [<code>batch_size</code>, <code>actions_size</code>])</td>
  </tr>
</table>

<br/>
And the architecture of the Critic Network is as follows:
<table class="unchanged rich-diff-level-one">
  <tr>
    <td align="center" colspan="1">Input (, [<code>batch_size</code>, <code>state_size</code>])</td>
    <td align="center" colspan="1"></td>
  </tr>
  <tr>
    <td align="center" colspan="1">Linear ([<code>batch_size</code>, <code>state_size</code>],<br/>[<code>batch_size</code>, 256])</td>
    <td align="center" colspan="1"></td>
  </tr>
  <tr>
    <td align="center" colspan="1">LeakyReLU(neg_slope=0.01)</td>
    <td align="center" colspan="1">Input (, [<code>batch_size</code>, <code>action_size</code>])</td>
  </tr>
  <tr>
    <td align="center" colspan="2">Concatenate ({[<code>batch_size</code>, 256], [<code>batch_size</code>, <code>action_size</code>]}, [<code>batch_size</code>, 256 + <code>action_size</code>])</td>
  </tr>
  <tr>
    <td align="center" colspan="2">Linear ([<code>batch_size</code>, 256 + <code>action_size</code>], [<code>batch_size</code>, 128])</td>
  </tr>
  <tr>
    <td align="center" colspan="2">LeakyReLU(neg_slope=0.01)</td>
  </tr>
  <tr>
    <td align="center" colspan="2">Linear ([<code>batch_size</code>, 128], [<code>batch_size</code>, 128])</td>
  </tr>
  <tr>
    <td align="center" colspan="2">LeakyReLU(neg_slope=0.01)</td>
  </tr>
  <tr>
    <td align="center" colspan="2">Linear ([<code>batch_size</code>, 128], [<code>batch_size</code>, <code>n_atoms</code> <b>or</b> 1])</td>
  </tr>
  <tr>
    <td align="center" colspan="2">Output (, [<code>batch_size</code>, <code>n_atoms</code>] <b>or</b> 1)</td>
  </tr>
</table>

## Training & Evaluation

I've trained four `Agent` instances with seeds `1-4` and found that they could achieve a environment-solving performance in **less than 30 episodes with 20 distributed agents:**

![training plot](images/train_plot.png)

But, of course, the condition for solving the environment is getting an average score of at least `+30` for all agents over **100** episodes, and 30 episodes of training does not meet the criterion.

Therefore, I evaluated the trained agents by running for additional 100 episodes without further training to ensure that my agents can, indeed, solve the environment.

![evaluation plot](images/eval_plot.png)

This plot proves that these agents did achieve the required performance to solve the environment.

The weights for the best agent—seed `4`—was saved in `pretrained.pth`.

_As a side note:_ again, interestingly, agents performed better with their noises, albeit just by a little bit. This trend appeared on the last project, Navigation, and I think these agents are utilizing the stochasticity added by the noise to counter the uncertainty of their states. However, this is still just my speculation and, thus, this would need further investigation to be verified and understood.

## Future Ideas

The current performance of the agent is already quite satisfying, but I have not tried any on-policy algorithm such as PPO or A2C. I would be interested in trying those out and see if I can get an even better performance.

Also, I want to try using Q-Prop, an algorithm that combines both off-policy learning and on-policy learning, and compare it with all others as well.

Additionally, I noticed that some others have used batch normalization in their models. I'm curious whether adding a batch normalization would improve my current model even further or not.
