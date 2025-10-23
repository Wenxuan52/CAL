import argparse

def readParser():
    parser = argparse.ArgumentParser(description='CAL')
    # ---------------------Agent Config-----------------------
    parser.add_argument('--agent', default='cal', type=str,
                    choices=['cal', 'qsm', 'ssm', 'ssm_gauss'],
                    help="Select which agent to use: ['cal', 'qsm', 'ssm', 'ssm_gauss']")

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
    
    # ============================================================
    # QSM / SSM Specific Config
    # ============================================================
    parser.add_argument('--T', type=int, default=40,
                        help='Number of diffusion timesteps (for DDPM)')
    parser.add_argument('--M_q', type=float, default=1.0,
                        help='Scaling factor for Q-score matching')
    parser.add_argument('--ddpm_temperature', type=float, default=1.0,
                        help='Noise temperature in DDPM sampling')
    parser.add_argument('--beta_schedule', type=str, default='cosine',
                        choices=['cosine', 'vp', 'linear'],
                        help='Beta schedule type for diffusion process')
    parser.add_argument('--time_dim', type=int, default=16,
                        help='Dimension of Fourier time embeddings')

    # ============================================================
    # SSM-specific safety parameters
    # ============================================================
    parser.add_argument('--safe_threshold', type=float, default=0.0,
                        help='Safety value threshold: V_h(s) <= safe_threshold indicates safe states')
    parser.add_argument('--alpha_sm', type=float, default=1.0,
                        help='Score guidance weight for reward critic gradient in safe regions')
    parser.add_argument('--beta_sm', type=float, default=1.0,
                        help='Score guidance weight for safety critic gradient in unsafe regions')

    return parser.parse_args()

