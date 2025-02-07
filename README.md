<p align="center">
  <a href=""><img src="https://i.imgur.com/Iu7CvC1.png" alt="PRAIG-logo" width="100"></a>
</p>

<h1 align="center">CRNN+CTC for Optical Music Recognition</h1>

<h3 align="center">Training and prediction code</h3>

<p align="center">
  <img src="https://img.shields.io/badge/python-orange" alt="Gitter">
  <img src="https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white" alt="Lightning">
  <img src="https://img.shields.io/static/v1?label=License&message=MIT&color=blue" alt="License">
</p>


<p align="center">
  <a href="#setup">Setup</a> •
  <a href="#train">Train</a> •
  <a href="#predict">Predict</a> •
  <a href="#license">License</a>
</p>

## Setup

### Install requirements.txt
```
# inside a python environment
pip install -r requirements.txt
```

### Folder structure inside data.

You should prepare inside `data/` the following files and folders.

#### Splits
```
data/splits/<ds_name>/train_<fold_number>.dat
data/splits/<ds_name>/val_<fold_number>.dat
data/splits/<ds_name>/test_<fold_number>.dat
````
Each of these files contains lines where is each line is:
```data/gt/<ds_name>/my_dataset_file_without_extension```

Try to keep this format and the file names as it is prepare to autoload the files given the fold number.

#### Dataset
```
# annotations
data/gt/<ds_name>/your_files.txt
# images
data/jpg/<ds_name>/your_files.jpg 
````
We use `.txt` with a sequence of symbols but if you use another structure just change the dataset class. To change the annotation reading (e.g. tokenization), change [encoding_convertions.py](https://github.com/OMR-PRAIG-UA-ES/CRNN-CTC/blob/main/utils/encoding_convertions.py).

Please mantain the structure where annotations are under /gt and images under /jpg.


## Train
```
bash train_example.sh
```
Important, if you do any modification to the tokenization or data, remove the `.json` files that represent the vocabs in `data/vocabs`.

## Predict
Prepare your samples as in the `example/` folder.
```
bash predict_example.sh
```
This script will generate a predictions.json file with all the images and its corresponding prediction made by the model.


## License
This work is under a [MIT](LICENSE) license.
