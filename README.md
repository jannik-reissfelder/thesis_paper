# Predicting Microbial Community Composition from Plant Phylogeny

## Description
This repository builds upon the source code of my master's thesis project titled "Predicting the Microbial Community Composition from Plant Phylogeny" which can be found [here](https://github.com/jannik-reissfelder/thesis).
The goal of this work is to extend the previous logic with furhter analysis and data to make the initial work ready for publication.

The overall prediction aim remains the same: utilizing the plant phylogenetic information to predict microbial community compositions for 60 different plant species. This repository provides all necessary tools to preprocess data, train the predictive models, and evaluate their performance against the ground truth data.
In addition to using the distance matrix, other strategies like using the aligned sequences directly, or using a k-mers approach have also been conducted. However, these logics are mainly discarded as they did worse the results.

Another important difference to the previous version of the paper is that this contains a seperate logic to modeling and predicting the microbial composition using  Gaussian Process Regression (GPR) which is a probabilistic regression method.

## Installation
To set up this project, clone the repository and install the required dependencies.

```bash
git clone https://github.com/jannik-reissfelder/thesis_paper.git
cd thesis_paper
pip install -r requirements.txt
```

## Usage
To run the main pipeline that processes the data and trains the model, execute the main.py script:

```bash
python main.py
```

Please note that in the default configuration, the script will use ``random_forest`` as the algorithm for training. To change the algorithm, modify the ``ALGO_NAME`` variable in the main.py script. The available algorithms are: ``random_forest``, ``knn``, ``linear_regression``, ``elastic_net``. 
Similarly, per default no augmentation is used. To use augmentation, set the ``AUGMENTATION`` variable to ``True`` in the main.py script.

To apply ``gaussian_process`` please utilize the ``GPU_runner_Gaussian.ipynb`` notebook that implememented the GPR on GPUs.


This script initializes the preprocessing of the input data, followed by the training of models using the Preprocessor and TrainerClass.

To evaluate the models, run the evaluation.py script after the models have been trained and predictions have been made:

```bash
python evaluation.py
```

This will assess the model predictions using three main metrics and store the results per species.


## Features
- **Data Preprocessing**: Standardizes and prepares plant phylogenetic data and microbial data for model training.
- **Model Training**: Trains models to predict microbial communities based on plant phylogeny. A suit of algorithms is available, including Random Forest, K-Nearest Neighbors
- **Evaluation**: Compares predictions with actual microbial communities and quantifies model performance for three metrics: Bray-Curtis Distance, Jensen-Shannon Divergence, and Bhattacharyya Distance.




## Credits
This project was developed by Jannik Reißfelder. Special thanks to ATB Leibniz-Institute Potsdam, Department of Data Science.

## Contact
For support or collaboration, please contact me.
