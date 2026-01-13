import argparse

def PPOLagParser():
    """Argument parser for PPO + Lagrangian baseline.

    Style matches configs/cal.py so main.py can switch via --agent ppolag.

    Notes:
    - PPO is on-policy: collect rollout with current policy, then update.
    - This codebase often packs (reward, cost) into a 2-dim array; keep
      cost_limit semantics consistent with your environment wrapper.
    """

    parser = argparse.ArgumentParser(description='PPO-Lagrangian')

    # ---------------------Agent Config-----------------------
    parser.add_argument('--agent', default='ppolag', type=str)

    # ----------------------Env Config------------------------
    parser.add_argument('--env_name', default='Hopper-v3')
    # MuJoCo: 'Hopper-v3' 'HalfCheetah-v3' 'Ant-v3', 'Humanoid-v3'
    # Safety-Gym: 'Safexp-PointButton1-v0' 'Safexp-CarButton1-v0'
    # 'Safexp-PointButton2-v0' 'Safexp-CarButton2-v0' 'Safexp-PointPush1-v0'
    parser.add_argument('--safetygym', action='store_true', default=False)
    parser.add_argument('--constraint_type', default='velocity', help="['safetygym', 'velocity']")
    parser.add_argument('--epoch_length', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123456)

    # -------------------Experiment Config---------------------
    parser.add_argument('--cuda', default=True, action="store_true",
                        help='run on CUDA (default: True)')
    parser.add_argument('--cuda_num', default='0',
                        help='select the cuda number you want your program to run on')
    parser.add_argument('--use_tensorboard', action='store_true', default=False)
    parser.add_argument('--user_name', default='wenxuan-yuan')
    parser.add_argument('--n_training_threads', default=10)
    parser.add_argument('--experiment_name', default='exp')
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--num_eval_epochs', type=int, default=1)
    parser.add_argument('--save_parameters', action='store_true', default=False)
    parser.add_argument('--save_history', action='store_true', default=False)

    # ---------------------PPO Core Hyperparameters-------------------------
    # Discounting / GAE
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--safety_gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)
    parser.add_argument('--cost_gae_lambda', type=float, default=0.95)

    # PPO update
    parser.add_argument('--clip_ratio', type=float, default=0.2, help='PPO clipping epsilon')
    parser.add_argument('--ppo_epochs', type=int, default=10, help='number of SGD epochs per rollout')
    parser.add_argument('--minibatch_size', type=int, default=64, help='minibatch size for PPO update')
    parser.add_argument('--max_grad_norm', type=float, default=0.5, help='gradient clipping norm')

    # Loss coefficients
    parser.add_argument('--vf_coef', type=float, default=0.5, help='reward value loss coefficient')
    parser.add_argument('--cost_vf_coef', type=float, default=0.5, help='cost value loss coefficient')
    parser.add_argument('--ent_coef', type=float, default=0.0, help='entropy bonus coefficient')

    # Optimization
    parser.add_argument('--lr', type=float, default=3e-4, help='actor+critic learning rate')
    parser.add_argument('--adam_eps', type=float, default=1e-5, help='Adam epsilon')

    # Optional stabilizers
    parser.add_argument('--target_kl', type=float, default=None,
                        help='early stop PPO epochs if approx KL exceeds this (set None to disable)')
    parser.add_argument('--use_value_clip', action='store_true', default=False,
                        help='use clipped value function loss (like PPO2)')
    parser.add_argument('--normalize_adv', action='store_true', default=True,
                        help='normalize reward advantages in batch (default True)')
    parser.add_argument('--normalize_cost_adv', action='store_true', default=True,
                        help='normalize cost advantages in batch (default True)')

    # ---------------------Lagrangian (Constraint) Hyperparameters-------------------------
    # Constraint form: E[cost] <= cost_limit
    # Safty-Gym: 10, HalfCheetah: 151.989, Hopper: 82.748, Ant: 103.115, Humanoid: 20.140
    parser.add_argument('--cost_limit', type=float, default=10,
                        help='constraint threshold d (budget). Tune per env/task.')
    parser.add_argument('--lambda_init', type=float, default=1.0, help='initial Lagrange multiplier')
    parser.add_argument('--lambda_lr', type=float, default=5e-3, help='dual ascent step size')
    parser.add_argument('--lambda_max', type=float, default=80.0, help='cap for lambda to avoid explosion')
    parser.add_argument('--dual_update_every', type=int, default=1,
                        help='update lambda every N policy updates')

    # Scaling / interpretation helper
    parser.add_argument('--cost_is_per_step', action='store_true', default=False,
                        help='if True, treat cost_limit as per-step; else per-episode/epoch budget')

    # ---------------------Rollout / Data Collection-------------------------
    # PPO is on-policy: collect rollout then update.
    parser.add_argument('--steps_per_rollout', type=int, default=2048,
                        help='on-policy steps collected per update (per worker if vectorized)')
    parser.add_argument('--num_train_repeat', type=int, default=1,
                        help='(compat) update calls per epoch; usually 1 for PPO')

    # -------------------Network Hyperparameters---------------------
    parser.add_argument('--hidden_size', type=int, default=256)
    
    parser.add_argument('--random_seed_each_run', action='store_true', default=True)
    parser.add_argument('--debug_nan', action='store_true', default=False)

    return parser.parse_args()
