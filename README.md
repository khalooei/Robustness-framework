# Robustness Framework

<div align="center">

<img alt="Robustness Framework" src="https://raw.githubusercontent.com/khalooei/robustness-framework/main/img-robustness-framework.jpg" width="800px" style="max-width: 100%;">

<br/>
<br/>

**A robustness framework for baseline of standard and adversarial machine learning research projects**

[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/robustness-framework)](https://img.shields.io/pypi/pyversions/robustness-framework)
[![PyPI Status](https://badge.fury.io/py/robustness-framework.svg)](https://badge.fury.io/py/robustness-framework)
[![Conda](https://img.shields.io/conda/v/conda-forge/robustness-framework?label=conda&color=success)](https://anaconda.org/conda-forge/robustness-framework)
[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)  

</div>

The robustness framework is based on top-tier machine learning libraries based on Pytorch. Additionally, it allows for embedding different model architectures and training processes in addition to fixing all research issues.
This framework integrates the following libraries:
 * Pytorch-Lightning
 * Hydra
 * torchattacks

The following architectures are covered in this framework and other networks will be added as needed:
 * MKToyNet
 * LeNet
 * DenseNet
 * ResNet
 * WideResNet

The logging system of this framework is also customizable and resolves all your ideas. As a result, we can take advantage of the efficiency of the following libraries:
 * TorchMetrics 
 * Loggings
 * Neptune
 * Comet
 * MLFlow
 * ...

## Installation
To install this interesting framework for standard and adversarial machine learning, follow the steps below, and don't waste your time developing an efficient baseline.
For installing robustness framework, we have two approaches:

### 1- Manual installation
```
git clone https://github.com/khalooei/robustness-framework.git
pip install -r requirements.txt
```

### 2- Automatic installation
```
pip install robustness-framework
```


## Usage
You can just follow the `main.py` file as a main anchor of this framework. You can define your own configurations in `configs` directory as we defined `training_mnist.yaml` and `training_cifar10.yaml` configuration. 
You can run this framework for running on CIFAR10 dataset as below:
 ```
   python main.py +configs=training_cifar10
 ```
If you want to change the default `training_cifar10` configurations, you can pass them as below. Below is an example of adversarial training using a PGD attack.
```
   python main.py +config=training_cifar10 training_params.type="AT" adversarial_training_params.name="PGD"
```


[TOBE COMPLETED]


## Acknowledgements
Thanks to the people behind Pytorch, Lightning, torchattacks, hydra, and MLOps libraries whose work inspired this repository. Furthermore, I would like to thank my supervisors [Prof. Mohammad Mehdi Homayounpour](https://scholar.google.com/citations?user=1PVbtE4AAAAJ&hl=en) and [Dr. Maryam Amirmazlagnani](https://scholar.google.com/citations?user=gxbTUfEAAAAJ&hl=en) for their efforts and guidance.


## Citation
If you use this package, please cite the following BibTex ([SemanticScholar](https://www.semanticscholar.org/author/Mohammad-Khalooei/35915175), [GoogleScholar](https://scholar.google.com/citations?user=2HFVUn4AAAAJ&hl=en)):
```
@article{mkhalooei2023robustnessframework,
  title={Robustness Framework: A pytorch repository for baselines of standard and adversarial machine learning research projects},
  author={Mohammad Khalooei and Mohammad Mehdi Homayounpour and Maryam Amirmazlaghani},
  url = {https://github.com/khalooei/robustness-framework},
  year={2023}
}
```
