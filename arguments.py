import argparse

def readParser():
    parser = argparse.ArgumentParser(description='CAL')
    # ---------------------Agent Config-----------------------
    parser.add_argument('--agent', default='cal', type=str,
                    choices=['cal', 'algd', 'dem'],
                    help="Select which agent to use: ['cal', 'algd']")

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

    # ---------------------Algorithm Config-------------------------
    parser.add_argument('--k', type=float, default=1.0)
    parser.add_argument('--qc_ens_size', type=int, default=4)
    parser.add_argument('--c', type=float, default=10)
    parser.add_argument('--num_train_repeat', type=int, default=10)

    parser.add_argument('--intrgt_max', action='store_true', default=False)
    parser.add_argument('--M', type=int, default=4, help='this number should be <= qc_ens_size')

    # -------------------Basic Hyperparameters---------------------
    parser.add_argument('--epsilon', default=1e-3)
    parser.add_argument('--init_exploration_steps', type=int, default=5000)
    parser.add_argument('--train_every_n_steps', type=int, default=1)
    parser.add_argument('--safety_gamma', type=float, default=0.99)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--tau', type=float, default=0.005)
    parser.add_argument('--alpha', type=float, default=0.2)
    parser.add_argument('--policy', default="Gaussian",
                        help='Policy Type: Gaussian | Deterministic (default: Gaussian)')
    parser.add_argument('--hidden_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=0.0003)
    parser.add_argument('--qc_lr', type=float, default=0.0003)
    parser.add_argument('--critic_target_update_frequency', type=int, default=2)
    parser.add_argument('--replay_size', type=int, default=1000000)
    parser.add_argument('--min_pool_size', type=int, default=1000)
    parser.add_argument('--max_train_repeat_per_step', type=int, default=5)
    parser.add_argument('--policy_train_batch_size', type=int, default=256)
    
    # ======== ALGD / Diffusion Policy ========
    parser.add_argument('--diffusion_K', type=int, default=30,
        help='Number of reverse diffusion steps K (default: 30)')

    parser.add_argument('--sigma_min', type=float, default=0.01,
        help='Minimum noise level for diffusion policy')
    parser.add_argument('--sigma_max', type=float, default=0.05,
        help='Maximum noise level for diffusion policy')

    parser.add_argument('--n_mc', type=int, default=4,
        help='Monte Carlo samples per score matching update')
    parser.add_argument('--beta', type=float, default=1.0,
        help='Temperature for importance weighting in score matching')

    parser.add_argument('--score_hidden_layers', type=int, default=3,
        help='Number of hidden layers in the diffusion score network')

    parser.add_argument('--t_dim', type=int, default=128,
        help='Time embedding feature dimension (emb_size) for the diffusion score network')

    parser.add_argument('--time_embedding', type=str, default='sinusoidal', choices=['sinusoidal', 'identity'],
        help='Embedding type used to encode diffusion timesteps')

    parser.add_argument('--diffusion_step_scale', type=float, default=1.0,
        help='Step scale for diffusion discretization (for stability)')

    parser.add_argument('--deterministic_eval', action='store_true', default=False,
        help='No noise sampling in reverse diffusion during evaluation')
    
    parser.add_argument('--algd_warmup_steps', type=int, default=0,
        help='Number of environment steps to follow the Gaussian CAL policy before switching to diffusion')

    return parser.parse_args()
