# AlertPrioritization

Corresponding code to the paper "Finding Needles in a Moving Haystack: Prioritizing Alerts with Adversarial Reinforcement Learning
" at AAAI 2020 ([PDF](https://liang-tong.me/publication/tong2020finding/tong2020finding.pdf)).

## Installation
The following libraries are required.
* Python 3.6
* Tensorflow 1.13
* Scikit-learn
* Matplotlib
* Scipy
* Pandas
* imbalanced-learn 

## Datasets
[Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud) and [CICIDS2017 Dataset](https://www.unb.ca/cic/datasets/ids-2017.html).

## Trained models
All the trained neural network models are available [here](https://www.dropbox.com/s/ajbmpqmbhya5ehu/converge.zip?dl=0). Please download and unfold the models, then put the ```converge``` folder under ```model```.

## Usage

### Reproduce the experiments
It is higly recommended to use a desktop with more than 20 cpu cores for reporoduction. 
Once the trained models are downloaded, run the following command to reprodue the experiments:
```
cd src
python evaluate.py [dataset] [defense] [def_budget] [estimate_adv_budget] [attack] [actual_adv_budget] [n_experiment]
```
Here, ```dataset``` can be either ```ids``` or ```fraud```, corresponding to the case study on network intrusion detection or fraud detection. 
For network intursion detection, ```defense``` can be one of ```rl```, ```uniform``` and ```suricata```;
For fraud detection, ```defense``` can be one of ```rl```, ```uniform```, ```gain``` and ```rio```.
```attack``` can be one of the following: ```rl```, ```uniform``` and ```greedy```, as described in the paper.
```n_experiment``` is the number of random seeds used for reinforcement learning.
In our experiments, it was set to be 20.

Note that the value of the defense and attack budgets should be the values displayed in Figure 2, 3, 4, 5 of the paper.
When using ```fraud``` as the case study, please revise the following value in ```project.conf``` as follows:
```
h_def = 16
h_adv = 32
```
For ```ids```, using the following in ```project.conf```:
```
h_def = 32
h_adv = 64
```

### Train models from scatch
Please use the following commands for training the models of our approach:
```
python double_oracle.py [dataset] [def_budget] [adv_budget] [n_experiment]
```

## Contributors
* Liang Tong (liangtong@wustl.edu)
* Aron Laszka (alaszka@uh.edu)

## Citation

```
@inproceedings{tong2020finding,
  title={Finding Needles in a Moving Haystack: Prioritizing Alerts with Adversarial Reinforcement Learning},
  author={Tong, Liang and Lazska, Aron and Yan, Chao and Zhang, Ning and Vorobeychik, Yevgeniy},
  booktitle={34th AAAI Conference on Artificial Intelligence},
  year={2020}
}
```
