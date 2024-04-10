# Debugging KFAC

This is a log containing debugging notes about our KFAC implementation.

I have added the following options to our implementation:

- Adding `--KFAC_USE_EXACT_INTERIOR_GRAMIAN` will use the exact Gramian of the interior loss.
- Adding `--KFAC_USE_EXACT_BOUNDARY_GRAMIAN` will use the exact Gramian of the boundary loss.

If at least one Gramian is exact, the pre-conditioner will be expanded as matrix, then damped and inverted.

## Sanity check 1: ENGD versus KFAC with exact interior Gramian (batch size 1)
### ENGD

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

### SOLVED: Bug 1

While comparing the flattened gradients between ENGD and KFAC, I noticed that the bias entries were off. I was accidentally using `layer.bias.data` rather than `layer.bias.grad.data`

### KFAC

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
Step: 000000/000100, Loss: 48.674902441834504, L2 Error: 4.567706181081753, Interior: 48.6736741716, Boundary: 0.0012282702, Time: 0.3s
Step: 000001/000100, Loss: 28.6034656795989, L2 Error: 4.82093188994386, Interior: 24.1607254668, Boundary: 4.4427402128, Time: 0.4s
Step: 000003/000100, Loss: 26.910669230270617, L2 Error: 4.603366078496967, Interior: 21.3692423117, Boundary: 5.5414269186, Time: 0.5s
Step: 000005/000100, Loss: 26.30864412210545, L2 Error: 4.097448578928954, Interior: 21.0601121820, Boundary: 5.2485319401, Time: 0.6s
Step: 000009/000100, Loss: 14.190886106062786, L2 Error: 7.156063448212089, Interior: 10.3055702178, Boundary: 3.8853158883, Time: 0.9s
Step: 000017/000100, Loss: 0.004919849857819333, L2 Error: 10.604007676171113, Interior: 0.0025856131, Boundary: 0.0023342368, Time: 1.3s
Step: 000031/000100, Loss: 1.5120130734887077e-06, L2 Error: 10.619862074856167, Interior: 0.0000015101, Boundary: 0.0000000020, Time: 2.2s
Step: 000055/000100, Loss: 3.226573209460433e-08, L2 Error: 10.624884648717433, Interior: 0.0000000322, Boundary: 0.0000000001, Time: 3.7s
Step: 000098/000100, Loss: 4.66872915569577e-09, L2 Error: 10.649207597087296, Interior: 0.0000000047, Boundary: 0.0000000000, Time: 6.2s
```
This looks very much like what we saw with `ENGD` above.

## Sanity check 2: ENGD versus KFAC with exact interior Gramian (batch size >1)

Our next sanity check will be to increase `N_dOmega`.
This will break the KFAC approximation for the boundary loss, and we want to make sure this does not harm optimization.

### ENGD
Let's run `ENGD` with
```bash
python train.py \
       --optimizer=ENGD \
       --num_steps=100 \
       --max_logs=10 \
       --ENGD_damping=1e-8 \
       --ENGD_approximation=per_layer
```
which gives the output
```bash
Step: 000000/000100, Loss: 48.68664253228789, L2 Error: 0.3227345981744154, Interior: 48.6736741716, Boundary: 0.0129683607, Time: 0.2s
Step: 000001/000100, Loss: 48.43198478560529, L2 Error: 0.31693732540007885, Interior: 48.3639551422, Boundary: 0.0680296434, Time: 0.3s
Step: 000003/000100, Loss: 36.85774549976797, L2 Error: 1.0286745776758233, Interior: 36.4241502822, Boundary: 0.4335952176, Time: 0.5s
Step: 000005/000100, Loss: 7.518396128963724, L2 Error: 0.6090377227011867, Interior: 6.8339783183, Boundary: 0.6844178107, Time: 0.7s
Step: 000009/000100, Loss: 2.004746441630896, L2 Error: 0.6426475133266946, Interior: 1.6703217411, Boundary: 0.3344247005, Time: 1.1s
Step: 000017/000100, Loss: 0.07780191820061236, L2 Error: 0.13026408311833373, Interior: 0.0702724212, Boundary: 0.0075294970, Time: 1.9s
Step: 000031/000100, Loss: 7.598455321964844e-08, L2 Error: 8.219681289261738e-05, Interior: 0.0000000669, Boundary: 0.0000000091, Time: 3.2s
Step: 000055/000100, Loss: 1.0847790502254288e-09, L2 Error: 1.4922185340150647e-05, Interior: 0.0000000009, Boundary: 0.0000000002, Time: 5.4s
Step: 000098/000100, Loss: 1.4894556114857534e-10, L2 Error: 5.1193052197733816e-06, Interior: 0.0000000001, Boundary: 0.0000000000, Time: 9.5s
```

### KFAC

Let's see if we can also get decent optimization using KFAC for the boundary Gramian:
```bash
python train.py \
       --optimizer=KFAC \
       --num_steps=100 \
       --max_logs=10 \
       --KFAC_ema_factor=0.0 \
       --KFAC_damping=1e-8
```
which yields
```bash
Step: 000000/000100, Loss: 48.674902441834504, L2 Error: 4.5677061853337255, Interior: 48.6736741716, Boundary: 0.0012282702, Time: 0.2s
Step: 000001/000100, Loss: 28.60346572130631, L2 Error: 4.820932102230395, Interior: 24.1607255059, Boundary: 4.4427402154, Time: 0.2s
Step: 000003/000100, Loss: 26.91066987346395, L2 Error: 4.6033664161121095, Interior: 21.3692430158, Boundary: 5.5414268577, Time: 0.4s
Step: 000005/000100, Loss: 26.308645663394696, L2 Error: 4.097448090758403, Interior: 21.0601136688, Boundary: 5.2485319945, Time: 0.5s
Step: 000009/000100, Loss: 14.190862649223519, L2 Error: 7.1560566442943205, Interior: 10.3055612673, Boundary: 3.8853013819, Time: 0.7s
Step: 000017/000100, Loss: 0.0049158047001465165, L2 Error: 10.604031011858325, Interior: 0.0025825426, Boundary: 0.0023332621, Time: 1.2s
Step: 000031/000100, Loss: 1.5108648204745067e-06, L2 Error: 10.619860080697004, Interior: 0.0000015089, Boundary: 0.0000000020, Time: 2.0s
Step: 000055/000100, Loss: 3.22951970460023e-08, L2 Error: 10.624994977431925, Interior: 0.0000000322, Boundary: 0.0000000001, Time: 3.4s
Step: 000098/000100, Loss: 4.674758653102768e-09, L2 Error: 10.649351063742465, Interior: 0.0000000047, Boundary: 0.0000000000, Time: 5.8s
```

NOTE: I played around with adding a `sqrt(batch_size)` in the backpropagated error for the boundary KFAC.
This did not change the performance at all.
This is weird because I would expect it to change at least some trailing digits, but they remained identical.

```bash
python train.py \
       --optimizer=KFAC \
       --num_steps=100 \
       --max_logs=10 \
       --KFAC_ema_factor=0.95 \
       --KFAC_damping=1e-8 \
       --KFAC_ggn_type=empirical
```
