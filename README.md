# GeGATVulDet

This repository is a python implementation of a graph attention network vulnerability detection method that generalizes global features of smart contracts.

## Requirements

### Required Packages
* **python** 3.7.0
* **TensorFlow** 2.0
* **numpy** 1.18.2
* **sklearn** 0.20.2


Run the following script to install the required packages.
```shell
pip install --upgrade pip
pip install tensorflow==2.0
pip install numpy==1.18.2
pip install scikit-learn==0.20.2
```


## Graph generator
The contract graph and its feature are extracted by the automatic graph extractor in the `contract_graph_generator` directory 


## Running Project
* To run program, please use this command: python3 VulDetector.py.
* Also, you can set specific hyperparameters, and all the hyperparameters can be found in `parser.py`.
