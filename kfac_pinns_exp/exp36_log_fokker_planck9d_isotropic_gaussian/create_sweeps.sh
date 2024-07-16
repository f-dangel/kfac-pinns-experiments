# Generate sweeps for all yaml config files
# NOTE: This is usually only necessary once
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/SGD.yaml sweeps/SGD.sh --qos=m4 --array_max_active=1
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/Adam.yaml sweeps/Adam.sh --qos=m4 --array_max_active=1
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/LBFGS.yaml sweeps/LBFGS.sh --qos=m4 --array_max_active=1
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/ENGD_diagonal.yaml sweeps/ENGD_diagonal.sh --qos=m4 --array_max_active=1
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/HessianFree.yaml sweeps/HessianFree.sh --qos=m4 --array_max_active=1
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/KFAC.yaml sweeps/KFAC.sh --qos=m4 --array_max_active=1
python ../exp09_reproduce_poisson2d/yaml_to_sh.py sweeps/KFAC_auto.yaml sweeps/KFAC_auto.sh --qos=m4 --array_max_active=1
