"""Speed comparison between HessianFree optimizer w/o caching."""

from time import time

from kfac_pinns_exp import train
from kfac_pinns_exp.utils import run_verbose


def test_speed_HessianFree_wo_caching():
    """Make sure that training with cached Gramian-vector products is faster."""
    basic_cmd = [
        "python",
        train.__file__,
        "--num_steps=5",
        "--dim_Omega=5",
        "--N_Omega=1000",
        "--N_dOmega=500",
        "--model=mlp-tanh-64-64-48-48",
        "--equation=poisson",
        "--boundary_condition=cos_sum",
    ]
    max_cg_iters = 250
    damping = 1.0

    # run HessianFree without caching
    cmd = basic_cmd + [
        "--optimizer=HessianFree",
        f"--HessianFree_cg_max_iter={max_cg_iters}",
        f"--HessianFree_damping={damping}",
        "--HessianFree_curvature_opt=ggn",
    ]
    start = time()
    run_verbose(cmd)
    t_uncached = time() - start
    print(f"Run time without caching: {t_uncached:.2f} s")

    # run HessianFree with caching
    cmd = basic_cmd + [
        "--optimizer=HessianFreeCached",
        f"--HessianFreeCached_cg_max_iter={max_cg_iters}",
        f"--HessianFreeCached_damping={damping}",
    ]
    start = time()
    run_verbose(cmd)
    t_cached = time() - start
    print(f"Run time with caching: {t_cached:.2f} s")

    assert t_cached < t_uncached
