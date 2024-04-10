python ../train.py \
       --optimizer=KFAC \
       --KFAC_damping=1e-10 \
       --KFAC_ema_factor=0.5 \
       --KFAC_initialize_to_identity \
       --num_steps=10_000
