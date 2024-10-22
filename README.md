# Machine Learning Lab Experiments
#### Live at ::: mluvceai.netlify.app
This repository contains solutions for the Machine Learning Lab experiments. Each experiment demonstrates the implementation of various algorithms used in machine learning.

## Lab Experiments

1. ### Find-S Algorithm
   Implement and demonstrate the **Find-S algorithm** for finding the most specific hypothesis based on a given set of training data samples. The training data is read from a `.CSV` file.

   **Link to program**: [Find-S Algorithm](Programs/first-find-s.py) -----------[Input CSV](Programs/dataset-1.csv)

2. ### Candidate Elimination Algorithm
   For a given set of training data stored in a `.CSV` file, implement and demonstrate the **Candidate-Elimination algorithm** to output a description of the set of all hypotheses consistent with the training examples.

   **Link to program**: [Candidate Elimination Algorithm](Programs/candidate.py) -----------[Input CSV](Programs/dataset-1.csv)

3. ### Decision Tree (ID3 Algorithm)
   Write a program to demonstrate the working of a decision tree based on the **ID3 algorithm**. Use an appropriate dataset to build the decision tree and apply this knowledge to classify a new sample.

   **Link to program**: [Decision Tree (ID3 Algorithm)](Programs/ID3.py) -----------[Input CSV](Programs/ID3.csv)

4. ### Artificial Neural Network (ANN)
   Build an **Artificial Neural Network** by implementing the backpropagation algorithm and test it using appropriate datasets.

   **Link to program**: [Artificial Neural Network (ANN)](Programs/ANN.py)

5. ### Naive Bayes Classifier
   Write a program to implement the **Naive Bayesian classifier** for a sample training dataset stored as a `.CSV` file. Compute the accuracy of the classifier, considering a few test datasets. Use built-in Java classes/API or Python to implement the Naive Bayesian Classifier model.

   **Link to program**: [Naive Bayesian Classifier](Programs/NaiveBayes.py)--------[Input CSV](Programs/NB.csv)
6. ### Bayesian Network
   Write a program to construct a **Bayesian Network** considering medical data. This program should calculate the accuracy, precision, and recall for your dataset.

   **Link to program**: [Bayesian Network](Programs/bayes-network.py)--------[Input CSV](Programs/network.csv)
### 7. **K-Means Clustering**
   - **Description**: This program applies the K-Means algorithm to cluster data points from a given CSV file into groups.
   - **Link to the program**: [K-Means Clustering Program](Programs/k-means-cluster.py)
   - **ML Library**: Python ML libraries such as `sklearn` are used for implementation.

### 8. **K-Nearest Neighbour (K-NN)**
   - **Description**: This program implements the K-Nearest Neighbour algorithm to classify data points from a dataset, printing both correct and incorrect predictions.
   - **Link to the program**: [K-Nearest Neighbour Program](Programs/k_nearest_neighbour.py)
   - **ML Library**: Python ML libraries such as `sklearn` are used for implementation.

### 9. **Locally Weighted Regression**
   - **Description**: This program implements non-parametric Locally Weighted Regression to fit data points to a curve, with appropriate graph plotting.
   - **Link to the program**: [Locally Weighted Regression Program](Programs/regression.py)
   - **ML Library**: Implemented using Python libraries like `numpy` and `matplotlib` for regression and graphing.

## Getting Started

To run these programs, you need:

- Python 3.x or Java (for Naive Bayes if using Java)
- A `.CSV` dataset for each experiment.

## Instructions

- Download or clone the repository.
- Navigate to the folder corresponding to the specific experiment.
- Follow the instructions in each folder's `README.md` file to set up and run the program.

## Dataset

Make sure you have a properly formatted dataset file for each experiment. The structure of the dataset is essential for the success of the algorithm implementations.

## License

This project is licensed under the MIT License.
