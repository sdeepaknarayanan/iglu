## Instructions to replicate main results

This directory contains the code for different experimental settings, and dataset specific architectures that are used for IGLU. 

### Sub Directory Information

The repository is subdivided into the following key directories:

- ```inductive``` - This directory contains the runner scripts for the main experiments (inductive setting) of the paper in a dataset specific manner.
- ```transductive``` - This directory contains the runner scripts for the transductive experiments reported in the paper in a dataset specific manner.
- ```gat``` - This directory contains the runner scripts for the experiments that use the Graph Attention Network architecture in a dataset specific manner. IGLU updates here are conducted in a full batch manner.

### Execution of the Scripts

- We use scripts with arguments for the main experiments (Table 1), which take multiple arguments as inputs.
    - ```batchsize``` - Batch Size for each SGD update
    - ```learning_rate``` - Initial learning rate for the layer-wise updates
    - ```data_path``` - Path to the dataset
    - ```gpuid``` - GPU ID
- For example, to execute the inductive experiment on the Reddit Dataset, we recommend executing the following command within the ```inductive``` directory.
	- ```python train_reddit.py --gpuid 0 --batchsize 10000 --learning_rate 1e-2 --data_path <path to data>```
- The learning rate and batch size hyperparameters are very important for IGLU's performance. The default values for these hyperparameters provided inside each runner script by us are the **optimal** hyperparameters for IGLU. 
- The results for other experiments such as transductive setting, scalability with products and GAT can be replicated using standalone scripts with predefined hyperparameter settings. These scripts do not take any arguments. Kindly change the datapath for these scripts.
- As training proceeds, all of the scripts output metrics of interest such as accuracy and time for optimization.

**NB** - For Products, the implementation has been modified to make IGLU scale to millions of nodes. 