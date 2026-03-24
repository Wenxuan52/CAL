## Safety-Gym:

Safety-Gym: 

- 'Safexp-PointButton1-v0' # 250
- 'Safexp-CarButton1-v0' # 375
- 'Safexp-PointButton2-v0' # 500
- 'Safexp-PointPush1-v0' # 500
- 'Safexp-CarButton2-v0' # 750

MOJOCO: 

- 'HalfCheetah-v3' # 100
- 'Hopper-v3' # 150
- 'Ant-v3' # 200
- 'Humanoid-v3' # 500

python time.py --agent algd --env_name Hopper-v3 --experiment_name algd_time --num_epoch 10 --cuda

### CarButton1

python main.py --agent cal --env_name Humanoid-v3 --experiment_name humanoid_cal --num_epoch 500 --cuda --use_tensorboard --save_history --save_parameters

python main.py --agent algd --env_name Ant-v3 --experiment_name ant_algd_v6 --num_epoch 200 --cuda --use_tensorboard --save_history --save_parameters


python main.py --agent algd --env_name HalfCheetah-v3 --experiment_name ant_algd_v6_T10_scorebeta0.02_scoresigmascale0.3 --num_epoch 100 --cuda --use_tensorboard --save_history --save_parameters


python main.py --agent saclag --env_name Humanoid-v3 --experiment_name humanoid_saclag --num_epoch 500 --cuda --use_tensorboard --save_history --save_parameters

python main.py --agent sacauglag --env_name Humanoid-v3 --experiment_name humanoid_sacauglag --num_epoch 500 --cuda --use_tensorboard --save_history --save_parameters

python main.py --agent hjb --env_name Humanoid-v3 --experiment_name humanoid_hjb --num_epoch 500 --cuda --use_tensorboard --save_history --save_parameters


# on-policy

python main_on_policy.py --agent ppolag --env_name Safexp-PointButton1-v0 --experiment_name pointbutton1_ppolag_long --num_epoch 2500 --cuda --use_tensorboard --save_history --save_parameters

python time_on_policy.py --agent ppolag --env_name Hopper-v3 --experiment_name hopper_ppolag_long --num_epoch 2500 --cuda

# lambda comparison

python main.py --agent algd --env_name Safexp-CarButton1-v0 --experiment_name carbutton1_algd_auglambda --num_epoch 375 --cuda --use_tensorboard --save_history --save_parameters

## set use_aug_lag == False in algd_v5.py
python main.py --agent algd --env_name Safexp-CarButton1-v0 --experiment_name carbutton1_algd_lambda --num_epoch 375 --cuda --use_tensorboard --save_history --save_parameters







# MC ablation

- 'Safexp-PointButton1-v0' # 250
- 'Safexp-PointButton2-v0' # 500
- 'Safexp-PointPush1-v0' # 500
- 'Safexp-CarButton2-v0' # 750

## set score_mc_samples = 2, 4, 6, 8, 16 in algd_v5.py

python main.py --agent algd --env_name Safexp-PointButton2-v0 --experiment_name pointbutton2_algd_ablationMC4 --num_epoch 500 --cuda --use_tensorboard --save_history --save_parameters


# ensamble ablation

- 'Safexp-PointButton1-v0' # 250
- 'Safexp-PointButton2-v0' # 500
- 'Safexp-PointPush1-v0' # 500
- 'Safexp-CarButton2-v0' # 750

## set qc_ens_size & M = 2, 4, 6, 8, 16 in /configs/cal.py

python main.py --agent algd --env_name Safexp-PointPush1-v0 --experiment_name pointpush1_algd_ablationENS16 --num_epoch 500 --cuda --use_tensorboard --save_history --save_parameters

python main.py --agent algd --env_name Safexp-PointButton2-v0 --experiment_name pointbutton2_algd_ablationENS16 --num_epoch 500 --cuda --use_tensorboard --save_history --save_parameters

# rho ablation

- 'Safexp-PointButton1-v0' # 250
- 'Hopper-v3' # 150

## set self.rho = getattr(args, "rho", 1.0) = 0.1, 0.5, 2.0, 4.0 in algd_v5.py

python main.py --agent algd --env_name Hopper-v3 --experiment_name hopper_algd_ablationRHO4.0 --num_epoch 150 --cuda --use_tensorboard --save_history --save_parameters
