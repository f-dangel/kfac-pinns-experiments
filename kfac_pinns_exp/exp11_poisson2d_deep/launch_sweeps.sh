# Launch all sweeps using the sbatch command

# launch each script with different qos,
# e.g. sbatch SGD.sh --qos=m4
for optim in \
    SGD \
        Adam \
        LBFGS HessianFree \
        ENGD_diagonal \
        ENGD_full \
        ENGD_per_layer \
        KFAC KFAC_empirical \
        KFAC_forward_only \
    ; do
    for qos in \
        m4 \
            m3 \
            m2 \
            m \
            normal\
        ; do
        echo sbatch $optim.sh --qos=$qos
        sbatch $optim.sh --qos=$qos
    done
done
