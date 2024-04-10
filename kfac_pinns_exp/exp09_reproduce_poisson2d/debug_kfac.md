# Debugging KFAC

This is a log containing debugging notes about our KFAC implementation.

I have added the following options to our implementation:

- Adding `--KFAC_USE_EXACT_INTERIOR_GRAMIAN` will use the exact Gramian of the interior loss.
- Adding `--KFAC_USE_EXACT_BOUNDARY_GRAMIAN` will use the exact Gramian of the boundary loss.

If at least one Gramian is exact, the pre-conditioner will be expanded as matrix, then damped and inverted.

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

## KFAC, exact Gramian of interior loss and boundary loss

This is a sanity check to make sure the internal numerics used by `KFAC` versus `ENGD` to apply the pre-conditioner are not causing issues.
`ENGD` solves a least-squares problem, while `KFAC` relies on matrix inversion, which is known to be more unstable, even though we use `float64`.

Let's run this with
```bash
python train.py \
       --optimizer=KFAC \
       --num_steps=100 \
       --N_dOmega=1 \
       --max_logs=10 \
       --KFAC_ema_factor=0.0 \
       --KFAC_damping=1e-8 \
       --KFAC_USE_EXACT_INTERIOR_GRAMIAN \
       --KFAC_USE_EXACT_BOUNDARY_GRAMIAN
```
whose output is
```bash
Step: 000000/000100, Loss: 48.674902441834504, L2 Error: 0.9229613855862834, Interior: 48.6736741716, Boundary: 0.0012282702, Time: 0.2s
Step: 000001/000100, Loss: 46.48477464583893, L2 Error: 2.4444005179425146, Interior: 46.3539288848, Boundary: 0.1308457610, Time: 0.2s
Step: 000003/000100, Loss: 44.699882785333514, L2 Error: 2.4444005179425146, Interior: 42.6094125700, Boundary: 2.0904702154, Time: 0.4s
Step: 000005/000100, Loss: 44.699882785333514, L2 Error: 2.4444005179425146, Interior: 42.6094125700, Boundary: 2.0904702154, Time: 0.5s
Step: 000009/000100, Loss: 44.699882785333514, L2 Error: 2.4444005179425146, Interior: 42.6094125700, Boundary: 2.0904702154, Time: 0.7s
Step: 000017/000100, Loss: 44.699882785333514, L2 Error: 2.4444005179425146, Interior: 42.6094125700, Boundary: 2.0904702154, Time: 1.1s
Step: 000031/000100, Loss: 44.699882785333514, L2 Error: 2.4444005179425146, Interior: 42.6094125700, Boundary: 2.0904702154, Time: 1.8s
Step: 000055/000100, Loss: 44.699882785333514, L2 Error: 2.4444005179425146, Interior: 42.6094125700, Boundary: 2.0904702154, Time: 3.1s
Step: 000098/000100, Loss: 44.699882785333514, L2 Error: 2.4444005179425146, Interior: 42.6094125700, Boundary: 2.0904702154, Time: 5.3s
```
The line search seems to get stuck.
There are three possible explanations for this: (i) matrix inversion versus linear system solve matters, (ii) there is still a bug in my
implementation of the debugging flags, (iii) the line search gets stuck for other reasons that are unknown so far.

Re (i):
- Switching to `lstsq` did not help.

Re (ii):
- I verified that the Gramians used in KFAC and ENGD match at step 0.
- I verified that the gradients used in KFAC and ENGD match at step 0.
- I found that the update directions used by KFAC and ENGD do not match at step 0
- I checked that the damped Gramians used by KFAC and ENGD match at step 0. Note that I had to convert the ENGD Gramians to the KFAC basis, which might contain a bug.
- FOUND A BUG. While comparing the flattened gradients between ENGD and KFAC, I noticed that the bias entries were off. I was accidentally using `layer.bias.data` rather than `layer.bias.grad.data`
