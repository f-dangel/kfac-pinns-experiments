# Update plots of all Bayesian search experiments

for DIR in exp14_poisson_100d_weinan \
               exp26_poisson5d_mlp_tanh_256_bayes \
               exp31_heat4d_mlp_tanh_256_bayes \
               exp32_poisson10d_mlp_tanh_256_bayes \
               exp34_poisson100d_mlp_tanh_product_bayes; do
    echo "Updating plots in $DIR"
    cd $DIR
    python plot.py &
    cd -
done

wait

# group plots recycle the downloaded data from the individual experiments, they
# must thus be updated after to be up-to-date
DIR=exp33_poisson_bayes_groupplot
echo "Updating group plot in $DIR"
cd $DIR
python plot.py
cd -
