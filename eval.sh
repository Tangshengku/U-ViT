# the evaluation setting
export num_processes=1  # the number of gpus you have, e.g., 2
export eval_script=eval.py  # the evaluation script, one of <eval.py|eval_ldm.py|eval_ldm_discrete.py|eval_t2i_discrete.py>
                     # eval.py: for models trained with train.py (i.e., pixel space models)
                     # eval_ldm.py: for models trained with train_ldm.py (i.e., latent space models with continuous timesteps)
                     # eval_ldm_discrete.py: for models trained with train_ldm_discrete.py (i.e., latent space models with discrete timesteps)
                     # eval_t2i_discrete.py: for models trained with train_t2i_discrete.py (i.e., text-to-image models on latent space)
export config=configs/cifar10_uvit_small.py  # the training configuration

# launch evaluation
accelerate launch   --num_processes $num_processes --mixed_precision fp16 $eval_script --config=$config --nnet_path=/home/dongk/dkgroup/tsk/projects/U-ViT/workdir/cifar10_uvit_small/default/ckpts/320000.ckpt/nnet_ema.pth
