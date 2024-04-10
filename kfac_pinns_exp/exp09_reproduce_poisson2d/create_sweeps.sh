# Generate sweeps for all yaml config files
# NOTE: This is usually only necessary once
python yaml_to_sh.py sweeps/SGD.yaml sweeps/SGD.sh
python yaml_to_sh.py sweeps/Adam.yaml sweeps/Adam.sh
python yaml_to_sh.py sweeps/LBFGS.yaml sweeps/LBFGS.sh
python yaml_to_sh.py sweeps/ENGD_diagonal.yaml sweeps/ENGD_diagonal.sh
python yaml_to_sh.py sweeps/ENGD_full.yaml sweeps/ENGD_full.sh
python yaml_to_sh.py sweeps/ENGD_per_layer.yaml sweeps/ENGD_per_layer.sh
python yaml_to_sh.py sweeps/HessianFree.yaml sweeps/HessianFree.sh
python yaml_to_sh.py sweeps/KFAC.yaml sweeps/KFAC.sh
python yaml_to_sh.py sweeps/KFAC_empirical.yaml sweeps/KFAC_empirical.sh
python yaml_to_sh.py sweeps/KFAC_forward_only.yaml sweeps/KFAC_forward_only.sh
