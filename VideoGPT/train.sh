!python3 train_vqvae.py --data_path "./data04" --sequence_length 8 --gpus 1
!python3 train_videogpt.py --data_path "./data04" --resolution 64 --gpus 1 --max_steps 36000 --vqvae "vqvae.ckpt" --sequence_length 8 --n_cond_frames 4