# GPSCVulDetector

This repo is a python implementation of a graph attention network vulnerability detection method that generalizes global features of smart contracts.

## Requirements

### Required Packages
* **python** 3+
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


## Graph extractorr
The contract graph and its feature are extracted by the automatic graph extractor in the `graph_extractor_example` directory 
 


Notably, you can also use the features extracted in [AMEVulDetector](https://github.com/Messi-Q/AMEVulDetector).

If any question, please email to messi.qp711@gmail.com.


## Running Project
* To run program, please use this command: python3 GPSCVulDetector.py.
* Also, you can set specific hyperparameters, and all the hyperparameters can be found in `parser.py`.

Examples:
```shell
python3 GPSCVulDetector.py
python3 GPSCVulDetector.py --model CGE --lr 0.002 --dropout 0.2 --epochs 100 --batch_size 32
```

## References
1. Smart Contract Vulnerability Detection Using Graph Neural Networks. IJCAI 2020. [GNNSCVulDetector](https://github.com/Messi-Q/GNNSCVulDetector).
```
@inproceedings{ijcai2020-454,
  title     = {Smart Contract Vulnerability Detection using Graph Neural Network},
  author    = {Zhuang, Yuan and Liu, Zhenguang and Qian, Peng and Liu, Qi and Wang, Xiang and He, Qinming},
  booktitle = {Proceedings of the Twenty-Ninth International Joint Conference on
               Artificial Intelligence, {IJCAI-20}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization}, 
  pages     = {3283--3290},
  year      = {2020},
}

```
2. Towards Automated Reentrancy Detection for Smart Contracts Based on Sequential Models. IEEE Access. [ReChecker](https://github.com/Messi-Q/ReChecker).
```
@article{qian2020towards,
  title={Towards Automated Reentrancy Detection for Smart Contracts Based on Sequential Models},
  author={Qian, Peng and Liu, Zhenguang and He, Qinming and Zimmermann, Roger and Wang, Xun},
  journal={IEEE Access},
  year={2020},
  publisher={IEEE}
}
```
