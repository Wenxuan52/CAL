import argparse


def OffPolicyPPOLagParser():
    parser = argparse.ArgumentParser(description='Replay PPO-Lagrangian')

    parser.add_argument('--agent', default='offpolicy_ppolag', type=str)

    # Env
    parser.add_argument('--env_name', default='Hopper-v3')
    parser.add_argument('--safetygym', action='store_true', default=False)
    parser.add_argument('--constraint_type', default='velocity', help="['safetygym', 'velocity']")
    parser.add_argument('--epoch_length', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123456)

    # Experiment
    parser.add_argument('--cuda', default=True, action='store_true')
    parser.add_argument('--cuda_num', default='0')
    parser.add_argument('--use_tensorboard', action='store_true', default=False)
    parser.add_argument('--user_name', default='wenxuan-yuan')
    parser.add_argument('--n_training_threads', default=10)
    parser.add_argument('--experiment_name', default='exp')
    parser.add_argument('--num_epoch', type=int, default=300)
    parser.add_argument('--num_eval_epochs', type=int, default=1)
    parser.add_argument('--save_parameters', action='store_true', default=False)
    parser.add_argument('--save_history', action='store_true', default=False)

    # Compatibility fields used by main.py prints
    parser.add_argument('--qc_ens_size', type=int, default=1)
    parser.add_argument('--M', type=int, default=1)

    # Core RL
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--safety_gamma', type=float, default=0.99)
    parser.add_argument('--gae_lambda', type=float, default=0.95)

    # Replay / collection
    parser.add_argument('--init_exploration_steps', type=int, default=5000)
    parser.add_argument('--train_every_n_steps', type=int, default=1)
    parser.add_argument('--num_train_repeat', type=int, default=1)
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5)
    parser.add_argument('--replay_size', type=int, default=100000)
    parser.add_argument('--min_pool_size', type=int, default=1000)
    parser.add_argument('--policy_train_batch_size', type=int, default=128)
    parser.add_argument('--recent_replay_size', type=int, default=50000)
    parser.add_argument('--trace_len', type=int, default=16)

    # PPO-Lag objective
    parser.add_argument('--clip_ratio', type=float, default=0.2)
    parser.add_argument('--ppo_epochs', type=int, default=2)
    parser.add_argument('--rho_bar', type=float, default=1.5)
    parser.add_argument('--c_bar', type=float, default=1.0)
    parser.add_argument('--ent_coef', type=float, default=0.0)
    parser.add_argument('--vf_coef', type=float, default=0.5)
    parser.add_argument('--cost_vf_coef', type=float, default=0.5)
    parser.add_argument('--max_grad_norm', type=float, default=0.5)

    # Optimization
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--vf_lr', type=float, default=3e-4)

    # Dual update
    parser.add_argument('--lambda_init', type=float, default=1.0)
    parser.add_argument('--lambda_lr', type=float, default=5e-4)
    parser.add_argument('--lambda_max', type=float, default=80.0)

    # Target value net
    parser.add_argument('--target_tau', type=float, default=0.01)
    parser.add_argument('--target_update_every', type=int, default=1)

    return parser.parse_args()
