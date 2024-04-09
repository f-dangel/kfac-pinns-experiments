# Debugging KFAC

This is a log containing debugging notes about our KFAC implementation.

I have added the following option to our implementation:

- Adding `--KFAC_USE_EXACT_INTERIOR_GRAMIAN` will use the exact Gramian of the interior loss.
  The KFAC approximation of the boundary loss will be expanded into its matrix representation, then added to the interior Gramian before adding damping and inverting.

## Ground truth with ENGD

Let's run `ENGD`, which uses the exact Gramians for both terms, as ground truth.
Note that we set relatively small number of logging points to keep the output readable.
Also, note that we set `N_dOmega=1` (explanation further below).

Running the following command
```bash
python train.py \
       --optimizer=ENGD \
       --num_steps=100 \
       --N_dOmega=1 \
       --max_logs=10 \
       --ENGD_damping=1e-8 \
       --ENGD_approximation=per_layer
```
produces the output
```bash
Step: 000000/000100, Loss: 48.674902441834504, L2 Error: 4.5677061750417485, Interior: 48.6736741716, Boundary: 0.0012282702, Time: 0.2s
Step: 000001/000100, Loss: 28.60346582209855, L2 Error: 4.820931632757762, Interior: 24.1607256045, Boundary: 4.4427402176, Time: 0.3s
Step: 000003/000100, Loss: 26.910669554087818, L2 Error: 4.603365945212342, Interior: 21.3692427377, Boundary: 5.5414268164, Time: 0.5s
Step: 000005/000100, Loss: 26.308644508109786, L2 Error: 4.0974485291749385, Interior: 21.0601124912, Boundary: 5.2485320169, Time: 0.7s
Step: 000009/000100, Loss: 14.190799106231687, L2 Error: 7.156053785929508, Interior: 10.3055133998, Boundary: 3.8852857064, Time: 1.0s
Step: 000017/000100, Loss: 0.004915141616701139, L2 Error: 10.604138284083414, Interior: 0.0025822916, Boundary: 0.0023328500, Time: 1.7s
Step: 000031/000100, Loss: 1.5052706127296104e-06, L2 Error: 10.62001122160855, Interior: 0.0000015033, Boundary: 0.0000000019, Time: 3.0s
Step: 000055/000100, Loss: 3.2388268299520715e-08, L2 Error: 10.624714530479517, Interior: 0.0000000323, Boundary: 0.0000000001, Time: 5.1s
Step: 000098/000100, Loss: 4.698426408125667e-09, L2 Error: 10.64832072286645, Interior: 0.0000000047, Boundary: 0.0000000000, Time: 8.9s
```
The loss reduces quite dramatically within 100 steps.

## KFAC, exact Gramian of interior loss

Next, we will look at KFAC and only approximate the boundary Gramian with KFAC.
Because we set `N_dOmega=1`, KFAC should equal the boundary Gramian.
Therefore, we expect this algorithm to perform as well as `ENGD` (the only difference is that `KFAC` uses `torch.linalg.inverse` for applying the pre-conditioner).

Let's run this scenario with
```bash
python ../train.py \
       --optimizer=KFAC \
       --num_steps=100 \
       --N_dOmega=1 \
       --max_logs=10 \
       --KFAC_ema_factor=0.0 \
       --KFAC_damping=1e-8 \
       --KFAC_USE_EXACT_INTERIOR_GRAMIAN
```
which produces
```bash
Step: 000000/000100, Loss: 48.674902441834504, L2 Error: 0.40062723331221134, Interior: 48.6736741716, Boundary: 0.0012282702, Time: 0.2s
Step: 000001/000100, Loss: 48.674902441834504, L2 Error: 0.40062723331221134, Interior: 48.6736741716, Boundary: 0.0012282702, Time: 0.2s
Step: 000003/000100, Loss: 48.674902441834504, L2 Error: 0.40062723331221134, Interior: 48.6736741716, Boundary: 0.0012282702, Time: 0.4s
Step: 000005/000100, Loss: 48.674902441834504, L2 Error: 0.40062723331221134, Interior: 48.6736741716, Boundary: 0.0012282702, Time: 0.5s
Step: 000009/000100, Loss: 48.674902441834504, L2 Error: 0.40062723331221134, Interior: 48.6736741716, Boundary: 0.0012282702, Time: 0.7s
Step: 000017/000100, Loss: 48.674902441834504, L2 Error: 0.40062723331221134, Interior: 48.6736741716, Boundary: 0.0012282702, Time: 1.2s
Step: 000031/000100, Loss: 48.674902441834504, L2 Error: 0.40062723331221134, Interior: 48.6736741716, Boundary: 0.0012282702, Time: 2.1s
Step: 000055/000100, Loss: 48.674902441834504, L2 Error: 0.40062723331221134, Interior: 48.6736741716, Boundary: 0.0012282702, Time: 3.6s
Step: 000098/000100, Loss: 48.674902441834504, L2 Error: 0.40062723331221134, Interior: 48.6736741716, Boundary: 0.0012282702, Time: 6.1s
```
This indicates that the line search gets stuck at the very first step.
Maybe we have a sign problem with the gradient.
