# AMAR 2.0: Ask Me Any Rating 

Code for the paper "Deep Content-based Recommender Systems Exploiting Recurrent Neural Networks and Linked Open Data. UMAP (Adjunct Publication) 2018: 239-244".

## Description
In this work we propose *AMAR 2.0* as an extension of *Ask Me Any Rating (AMAR)*, a content-based recommender system based on deep neural networks which is able to produce top-N recommendations leveraging user and item embeddings which are learnt from textual information describing the items. 

A comprehensive experimental evaluation conducted on state of-the-art datasets such as *MovieLens 1M* and *DBbook* showed a significant improvement over all the baselines taken into account and a slight improvement respect to the previous version of this framework.

## Improvements
The main improvement of the second version of AMAR lies on the architecture of the framework and on the learnt inputs. The former presents a new set of layers - that are BiRNN and AutoEncoders - stacked together that were exploited to better learn from the data. Considering the input data, new features are used in the process of learning to improve the performances. In the specific case, genres, directors, properties, wiki_categories were employed.

The code implements the model indicated in the paper "Deep Content-based Recommender Systems Exploiting Recurrent Neural Networks and Linked Open Data. UMAP (Adjunct Publication) 2018: 239-244", however results may slightly vary due to some differences in terms of structure and used parameters:
* the module dedicated to users learning was not employed
* the tested optimization method is rmsprop
* the tested number of epochs is 20  

## Requirements

- Lua
- Torch
  - nn
  - rnn
  - cudnn
  - cunn
  - cutorch
  - optim
  - pl
  - cjson
  - lfs

## Usage

1. Retrieve item descriptions and save them in ".txt" files. Each file should be named as the item identifier in the used dataset (e.g., item 1 has a description file named 1.txt)
2. Create JSON configuration file for the training file that you want to use and start the training using *Torch* specifying the configuration filename with the `-config` parameter
3. Evaluate the trained model using the `run_amar_experiments.lua` program specifying the configuration filename with the `-config` parameter

## Configuration files
Training and evaluation configuration files are in JSON format and are composed by specific fields. They are used in order to modify model parameters and to specify the supplementary files used to train the models or to evaluate the models.

For instance, the training configuration file for the `train_amar_autoenc.lua` file is composed by the following fields:
 - items: path of item descriptions
 - genres: filename of item genres (optional)
 - directors: filename of item directors (optional)
 - properties: filename of item properties (optional)
 - wiki_categories: filename of item wiki categories (optional)
 - models_mapping: dictionary which associates training sets to models
 - optim_method: optimization method identifier used in optim package
 - training_params: parameters of the optimization method
 - batch_size: number of training examples in a batch
 - num_epochs: number of training epochs
 - save_after: save model each save_after epochs
 
We report in the following code snippet a real configuration script that we used for our model:
```json
{
  "items": "../datasets/ml1m/content/simple",
  "models_mapping": {
    "../datasets/ml1m/ratings/u1.base": "../resources/ml1m/amar/autoenc50/base/u1.model",
    "../datasets/ml1m/ratings/u2.base": "../resources/ml1m/amar/autoenc50/base/u2.model",
    "../datasets/ml1m/ratings/u3.base": "../resources/ml1m/amar/autoenc50/base/u3.model",
    "../datasets/ml1m/ratings/u4.base": "../resources/ml1m/amar/autoenc50/base/u4.model",
    "../datasets/ml1m/ratings/u5.base": "../resources/ml1m/amar/autoenc50/base/u5.model"
  },
  "optim_method": "rmsprop",
  "training_params": {
    "learningRate": 1e-3,
    "alpha": 0.9
  },
  "batch_size": 32,
  "num_epochs": 20,
  "save_after": 5
}
```

In addition, the evaluation configuration file for the `run_amar_experiments.lua` file is in JSON format and is composed by the following fields:
 - items: path of items descriptions
 - genres: filename of item genres (optional)
 - directors: filename of item directors (optional)
 - properties: filename of item properties (optional)
 - wiki_categories: filename of item wiki categories (optional)
 - models_mapping: dictionary which associates test files to models
 - predictions: generated predictions filename
 - batch_size: number of examples in a batch
 - topn: list of cutoff values
 
We report in the following code snippet a real configuration script that we used for our model:
```json
{
  "items": "../datasets/ml1m/content/simple",
  "models_mapping": {
    "../datasets/ml1m/ratings/u1.test": "../resources/ml1m/amar/autoenc50/base/u1.model",
    "../datasets/ml1m/ratings/u2.test": "../resources/ml1m/amar/autoenc50/base/u2.model",
    "../datasets/ml1m/ratings/u3.test": "../resources/ml1m/amar/autoenc50/base/u3.model",
    "../datasets/ml1m/ratings/u4.test": "../resources/ml1m/amar/autoenc50/base/u4.model",
    "../datasets/ml1m/ratings/u5.test": "../resources/ml1m/amar/autoenc50/base/u5.model"
  },
  "predictions": "../experiments/results/ml1m/amar/autoenc50/base/predictions_%d_%d.txt",
  "batch_size": 32,
  "topn": [5]
}
```
