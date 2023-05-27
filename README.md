# Midpoint-Mixup-Multi-View

Here we outline how to recreate the main experiments from the paper.

# Modified Image Classification Benchmarks 

To generate the plots in our paper, one can look at `training/run_all_experiments_cat_channel.sh` to see examples of how to run the training script. If slurm is available, the aforementioned script can be run directly after adding an appropriate slurm header to `training/run_training_v2.py`.

# Simulating Theory Directly

The code necessary to simulate a near-exact replica of our theoretical setting is available in `data_utils.py` and `models/two_layer_cnn.py`. Unfortunately, one can check quickly that when using `models/two_layer_cnn.py` gradients become very small for choices of alpha close to the settings in our paper (anything > 2 becomes small quickly).
