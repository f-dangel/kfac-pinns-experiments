# Generate sweeps for all yaml config files
# NOTE: This is usually only necessary once
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/SGD.yaml sweeps/SGD.sh --qos=m4
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/Adam.yaml sweeps/Adam.sh --qos=m4
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/LBFGS.yaml sweeps/LBFGS.sh --qos=m4
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/HessianFree.yaml sweeps/HessianFree.sh --qos=m4
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/KFAC.yaml sweeps/KFAC.sh --qos=m4
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/KFAC_empirical.yaml sweeps/KFAC_empirical.sh --qos=m4
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/KFAC_forward_only.yaml sweeps/KFAC_forward_only.sh --qos=m4
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/KFAC_auto.yaml sweeps/KFAC_auto.sh --qos=m4
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/KFAC_empirical_auto.yaml sweeps/KFAC_empirical_auto.sh --qos=m4
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/KFAC_forward_only_auto.yaml sweeps/KFAC_forward_only_auto.sh --qos=m4
