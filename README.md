# Delta-ML Excited States
This repository contains the database and code for **Improving the accuracy of inexpensive excited state energy calculations through machine learning for virtual screening of organic molecules** by *M del Cueto*, *A Coxson*, *O Omar*  and *A Troisi*

---

## Prerequisites

The necessary packages (with the tested versions with Python 3.8.10) are specified in the file requirements.txt. These packages can be installed with pip:

```
pip3 install -r requirements.txt
```
---

## Contents

- **FFNN.py**: program used to perform the predictions with a feed-forward neural network, as discussed in the manuscript

- **KRR.py**: program used to perform the predictions with kernel ridge regression, as discussed in the supporting information

- **database**: directory containing the csv files with the train dataset and the test set (see *Database* below)

- **reproduce_Fig3**: directory containing the necessary scripts to reproduce the main results presented in the article in Figure 3 (see *Usage and examples* below)

---

## Database

- **train_data.csv**: due to the large size of the training dataset (10506 molecules with their labels, SMILES, Morgan fingerprint and HOMO radial distribution function), it has been split and compressed in five parts: *train_data_01-05.zip*. We offer a bash script to merge these files in a single csv file: *train_data.csv*

- **test_data.csv**: this file contains the test dataset (524 molecules)

---

## Usage and examples

- To reproduce the main results, while at the main directory, one can simply do:

```
cd database
bash merge_train_data.sh
cd ..
python FFNN.py
```

- For convenience, the resulting output files, *results_train.csv* and *results_test.csv*, have been provided in the folder **reproduce_Fig3**, which also contains the program **plot_Fig3.py**, which reads the output data and plots it in the format used in Figure 3 of the manuscript. To generate this figure, simply run:

```
python plot_Fig3.py
```

This will generate a .png file with the results: *Fig3.png*

---

## License
**Authors**: [Marcos del Cueto](https://github.com/marcosdelcueto), [Adam Coxson](https://github.com/AdamCoxson), Ömer Omar and Alessandro Troisi

Licensed under the [MIT License](LICENSE)   
