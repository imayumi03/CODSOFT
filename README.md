# Movie Genre Prediction from Plot Summaries

This repository contains the code for a machine learning project aimed at predicting movie genres based on their plot summaries. The project uses text classification techniques to train models on movie plot summaries and predict genres, employing various machine learning classifiers.

## Project Overview

The goal of this project is to demonstrate the application of text classification methods in natural language processing. We use TF-IDF vectorization to transform text data into a usable format for machine learning models and apply three different classifiers to predict movie genres:
- Naive Bayes
- Logistic Regression
- Support Vector Machine (SVM)

## Repository Structure

- `train_data.txt`: Text file containing the training dataset with plot summaries and genres.
- `test_data.txt`: Text file containing the test dataset with plot summaries. Genres are initially marked as 'Unknown' and are predicted by the models.
- `test.py`: Python script for training models and predicting genres.
- `Naive_Bayes_predicted_test_data.csv`: Output file with genres predicted using the Naive Bayes classifier.
- `Logistic_Regression_predicted_test_data.csv`: Output file with genres predicted using Logistic Regression.
- `SVM_predicted_test_data.csv`: Output file with genres predicted using the SVM classifier.

## How to Run

### Prerequisites

Ensure you have Python installed on your system, along with the following libraries:
- pandas
- scikit-learn

You can install these packages using pip:

```bash
pip install pandas scikit-learn
```

### Execution

Run the script `movie_genre_prediction.py` to train the models and predict genres. This script will output CSV files for each classifier showing the predicted genres for the test dataset.

```bash
python movie_genre_prediction.py
```

## Classifiers Used

- **Naive Bayes**: A simple probabilistic classifier based on applying Bayes' theorem with strong (naive) independence assumptions between the features.
- **Logistic Regression**: A linear model for classification rather than regression. Logistic regression measures the relationship between the categorical dependent variable and one or more independent variables by estimating probabilities using a logistic function.
- **Support Vector Machine (SVM)**: A powerful classifier that works well on a wide range of classification problems, even those with complex boundaries.

## Results

Each classifier's performance can be evaluated by reviewing the generated CSV files. These files contain the original plot summaries from the `test_data.txt` along with the predicted genres, allowing for a straightforward assessment of each model's accuracy and effectiveness.

## Contributing

Contributions to this project are welcome. You can contribute in several ways:
- Enhancing the algorithm
- Adding more classifiers
- Improving the feature engineering process

For major changes, please open an issue first to discuss what you would like to change.

