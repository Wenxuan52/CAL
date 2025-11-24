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

    # ------------------- ALGD --------
    parser.add_argument('--algd_score_coef', type=float, default=0.1,
                        help='weight for score matching regularization in ALGD agent')
    parser.add_argument('--algd_score_sigma_min', type=float, default=0.1,
                        help='minimum diffusion noise level for ALGD score matching')
    parser.add_argument('--algd_score_sigma_max', type=float, default=1.0,
                        help='maximum diffusion noise level for ALGD score matching')
    parser.add_argument('--algd_diffusion_steps', type=int, default=5,
                        help='number of diffusion sampling steps for ALGD policy')
    parser.add_argument('--algd_langevin_scale', type=float, default=0.5,
                        help='step size scale for Langevin dynamics during ALGD diffusion sampling')

    return parser.parse_args()
