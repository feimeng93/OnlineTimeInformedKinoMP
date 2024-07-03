#!/bin/bash


export PYTHONPATH="/home/robot/mf/OnlineTimeInformedKinoMP:$PYTHONPATH"

export CUDA_VISIBLE_DEVICES="1"
cd /home/robot/mf/OnlineTimeInformedKinoMP/Deepkoopman/train/
# python DIKU.py --env "DampingPendulum" --suffix "DampingPendulum0221_1"  --layer_depth 3 --encode_dim 20 --N_aff 1  --N_changed_elements 1 --gamma 0.9 &
# python DIKU.py --env "Linear2D" --suffix "Linear2D0221_1" --layer_depth 3 --encode_dim 20 --N_aff 1 --N_changed_elements 1  --gamma 0.9  & 
# python DIKU.py --env "Nonlinear3D" --suffix "Nonlinear3D0221_1" --layer_depth 3 --encode_dim 20 --N_aff 1 --N_changed_elements 1 --gamma 0.9   &
# python DIKU.py --env "TwoLinkRobot" --suffix "TwoLinkRobot0221_1"  --layer_depth 3 --encode_dim 20 --N_aff 1 --N_changed_elements 1 --gamma 0.9  &
# python DIKU.py --env "CartPole" --suffix "CartPole0221_1"  --layer_depth 3 --encode_dim 20 --N_aff 1 --N_changed_elements 1 --gamma 0.9  &
python DIKU_quad.py --env "PlanarQuadrotor" --suffix "Quad0615"  --dkiu 1 --N_aff 1 --N_changed_elements 1 --gamma 0.9 &


# python DKU.py --env "DampingPendulum" --suffix "DampingPendulum0221_1" --trained_suffix "DampingPendulum0221_1" &
# python DKU.py --env "Linear2D" --suffix "Linear2D0221_1"  --trained_suffix "Linear2D0221_1"  &
# python DKU.py --env "Nonlinear3D" --suffix "Nonlinear3D0221_1"  --trained_suffix "Nonlinear3D0221_1" &
# python DKU.py --env "TwoLinkRobot" --suffix "TwoLinkRobot0221_1"  --trained_suffix "TwoLinkRobot0221_1" &
# python DKU.py --env "CartPole" --suffix "CartPole0221_1"  --trained_suffix "CartPole0221_1" &
# python DIKU_quad.py --env "PlanarQuadrotor" --suffix "Quad0221_1" --trained_suffix "Quad0221_1" --dkiu 0  &

