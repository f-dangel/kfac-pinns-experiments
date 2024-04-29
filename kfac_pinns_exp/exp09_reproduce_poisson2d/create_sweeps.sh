# Generate sweeps for all yaml config files
# NOTE: This is usually only necessary once
python yaml_to_sh.py sweeps/SGD.yaml sweeps/SGD.sh --qos=m5
python yaml_to_sh.py sweeps/Adam.yaml sweeps/Adam.sh --qos=m5
python yaml_to_sh.py sweeps/LBFGS.yaml sweeps/LBFGS.sh --qos=m5
python yaml_to_sh.py sweeps/ENGD_diagonal.yaml sweeps/ENGD_diagonal.sh --qos=m5
python yaml_to_sh.py sweeps/ENGD_full.yaml sweeps/ENGD_full.sh --qos=m4
python yaml_to_sh.py sweeps/ENGD_per_layer.yaml sweeps/ENGD_per_layer.sh --qos=m4
python yaml_to_sh.py sweeps/HessianFree.yaml sweeps/HessianFree.sh --qos=m4
python yaml_to_sh.py sweeps/KFAC.yaml sweeps/KFAC.sh --qos=m5
python yaml_to_sh.py sweeps/KFAC_empirical.yaml sweeps/KFAC_empirical.sh --qos=m5
python yaml_to_sh.py sweeps/KFAC_forward_only.yaml sweeps/KFAC_forward_only.sh --qos=m5
