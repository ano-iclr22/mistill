# Reproducibility

Describes the step necessary to reproduce the results of the paper.

1) Change to the folder `docker` and run `docker build -f Dockerfile_torch -t pytorch-image`.
2) Update the path in the shell script `run_container.sh` to the location where you cloned the repository.
3) Execute the script `run_container.sh`.
4) In the docker container, change directory to `/opt/project`.
5) Run `python3 plot_results.py`.

This will start an interactive session in which the simulation results with the
trained models are reproduced and the plots from the paper recreated. The interactive
program asks you to enter one of three possible TE policies. Enter them in the order
`hula`, `lcp`, `wcmp`, `hula` (`hula` corresponds to the `MinMax` policy).

After the program finishes (which can take some time), the images will be located
in the `img/gs` and `img/sparsemax` folder. The simulation results are written to
the corresponding folders in `data/results/`.


# Neural Network Model
The final NN model of the publication can be found in `models/stateful.py`. Its the
class `StatefulModel`.


# Training Method
The training of the models can be inspected in the file `training/stateful.py`. At
the bottom is the definition of the search space.

# Attention scores
The data frame with the attention scores is located in `data/results/gs/k8-hula-b64/scores` and `data/results/sparsemax/k8-hula-b64/scores`.
The scores can be plotted with the function `plot_converged_attention_scores` in the module `present.py`. Here, the argument `path_to_df` must
be set to one of the HDF files contained in the above folders.

To create the plots, open a python console, import the `present` module and call the function `plot_converged_attention_scores`.

# HLSA content
The HLSA content can be visualized with the script `eval/eval_hlsas.py`. By executing the
command `python3 -m eval.eval_hlsas` in the folder `/opt/project` the HLSA messages for the
TE policies will be generated.
