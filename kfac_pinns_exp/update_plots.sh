# Update all plots
echo "Updating non-Bayesian plots"

# Update individual plots
for DIR in ./exp09_reproduce_poisson2d/ \
                                      ./exp10_reproduce_poisson5d/ \
                                      ./exp11_poisson2d_deep/ \
                                      ./exp12_poisson5d_deep/ \
                                      ./exp13_reproduce_heat1d/ \
                                      ./exp15_poisson2d_deepwide/ \
                                      ./exp16_poisson5d_deepwide/ \
                                      ./exp19_poisson5d_mlp_tanh_256/ \
                                      ./exp20_poisson2d_mlp_tanh_256/ \
                                      ./exp21_poisson_10d/ \
                                      ./exp22_heat1d_mlp_tanh_64/ \
                                      ./exp23_heat1d_mlp_tanh_256/ \
                                      ./exp27_heat4d_small/ \
                                      ./exp28_heat4d_medium/ \
                                      ./exp29_heat4d_big/; do
    echo "Updating plots in $DIR"
    cd $DIR
    python plot.py &
    cd -
done

wait

for DIR in ./exp17_groupplot_poisson2d/ \
                                      ./exp18_groupplot_poisson5d/ \
                                      ./exp24_heat1d_groupplot/ \
                                      ./exp30_heat4d_groupplot/; do
    echo "Updating group plot in $DIR"
    cd $DIR
    python plot.py &
    cd -

echo "Updating Bayesian plots"
bash update_bayes_plots.sh
