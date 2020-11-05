# fake-news-detection-code

Fake News detection using MLP (Multilayer perceptron) and LSTM (Long short-term memory) Artificial Neural Networks

## Install dependencies

To install the dependencies, in the project folder just type

`pip3 install -r requirements.txt`

## Datasets

This project has two datasets

### Dataset text

The `dataset_text.csv` is the dataset with news in text format and is available at [fake-new-dataset](https://github.com/mathiasarturschulz/fake-news-dataset)

### Dataset Converted

The `dataset_converted.csv` is the dataset with the news converted to numerical representation

Neural network models work only with numbers, this was the reason for the conversion

The conversion is performed through the Gensim library using the Doc2Vec model

To create the dataset, in the project folder type:

`python3 script_convert.py`

**Time to generate the CSV:**

Runtime: XXXX minutes

## Detection

The detection of false and true news is carried out using script `script_detection.py`

To create the detection, in the project folder type:

`python3 script_detection.py`

**Time to generate the detection:**

Runtime: XXXX minutes

The script also creates the images in the `grapihcs` folder, the images are graphics of the quality detection metrics of the model detection
