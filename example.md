## Test

python main.py --env_name Safexp-CarButton1-v0 --experiment_name carbutton1_test --num_epoch 5

## Safety-Gym:

### CarButton1

python main.py --agent cal --env_name Safexp-CarButton1-v0 --experiment_name carbutton1_cal --num_epoch 375 --cuda --use_tensorboard --save_history --save_parameters

python main.py --agent algd --env_name Safexp-CarButton1-v0 --experiment_name carbutton1_algd_new --num_epoch 375 --cuda --use_tensorboard --save_history --save_parameters

### CarButton2

python main.py --env_name Safexp-CarButton2-v0 --experiment_name carbutton2_test --num_epoch 750 --cuda --use_tensorboard --save_parameters --save_history

### PointButton1

python main.py --env_name Safexp-PointButton1-v0 --experiment_name pointbutton1_test --num_epoch 375 --cuda --use_tensorboard --save_parameters --save_history

## MuJoCo:

### Hopper

python main.py --env_name Hopper-v3 --experiment_name hopper_test --num_epoch 150 --cuda --use_tensorboard --save_parameters --init_exploration_steps 15000 --save_history

### HalfCheetah

python main.py --env_name HalfCheetah-v3 --experiment_name halfcheetah_test --num_epoch 150 --cuda --use_tensorboard --save_parameters --init_exploration_steps 15000 --save_history

### Ant

python main.py --env_name Ant-v3 --experiment_name ant_test --num_epoch 200 --cuda --use_tensorboard --save_parameters --init_exploration_steps 15000 --save_history