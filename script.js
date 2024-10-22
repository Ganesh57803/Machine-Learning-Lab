function openPopup(program) {
    const popup = document.getElementById("popup");
    const title = document.getElementById("popup-title");
    const description = document.getElementById("popup-description");
    const link = document.getElementById("popup-link");
    const inputCsv = document.getElementById("popup-input-csv");

    switch (program) {
        case 'findS':
            title.textContent = "Find-S Algorithm";
            description.textContent = "Implement and demonstrate the Find-S algorithm for finding the most specific hypothesis based on a given set of training data samples. The training data is read from a .CSV file.";
            link.href = "Programs/first-find-s.py";
            inputCsv.textContent = "Programs/dataset-1.csv";
            break;
        case 'candidateElimination':
            title.textContent = "Candidate Elimination Algorithm";
            description.textContent = "Implement and demonstrate the Candidate-Elimination algorithm to output a description of the set of all hypotheses consistent with the training examples.";
            link.href = "Programs/candidate.py";
            inputCsv.textContent = "Programs/dataset-1.csv";
            break;
        case 'id3':
            title.textContent = "Decision Tree (ID3 Algorithm)";
            description.textContent = "Write a program to demonstrate the working of a decision tree based on the ID3 algorithm. Use an appropriate dataset to build the decision tree and apply this knowledge to classify a new sample.";
            link.href = "Programs/ID3.py";
            inputCsv.textContent = "Programs/ID3.csv";
            break;
        case 'ann':
            title.textContent = "Artificial Neural Network (ANN)";
            description.textContent = "Build an Artificial Neural Network by implementing the backpropagation algorithm and test it using appropriate datasets.";
            link.href = "Programs/ANN.py";
            inputCsv.textContent = "No specific CSV file.";
            break;
        case 'naiveBayes':
            title.textContent = "Naive Bayes Classifier";
            description.textContent = "Implement the Naive Bayesian classifier for a sample training dataset stored as a .CSV file. Compute the accuracy of the classifier, considering a few test datasets.";
            link.href = "Programs/NaiveBayes.py";
            inputCsv.textContent = "Programs/NB.csv";
            break;
        case 'bayesianNetwork':
            title.textContent = "Bayesian Network";
            description.textContent = "Construct a Bayesian Network considering medical data. This program should calculate the accuracy, precision, and recall for your dataset.";
            link.href = "Programs/bayes-network.py";
            inputCsv.textContent = "Programs/network.csv";
            break;
        case 'kMeans':
            title.textContent = "K-Means Clustering";
            description.textContent = "This program applies the K-Means algorithm to cluster data points from a given CSV file into groups.";
            link.href = "Programs/k-means-cluster.py";
            inputCsv.textContent = "No specific CSV file.";
            break;
        case 'knn':
            title.textContent = "K-Nearest Neighbour (K-NN)";
            description.textContent = "This program implements the K-Nearest Neighbour algorithm to classify data points from a dataset.";
            link.href = "Programs/k_nearest_neighbour.py";
            inputCsv.textContent = "No specific CSV file.";
            break;
        case 'locallyWeightedRegression':
            title.textContent = "Locally Weighted Regression";
            description.textContent = "This program implements non-parametric Locally Weighted Regression to fit data points to a curve.";
            link.href = "Programs/regression.py";
            inputCsv.textContent = "No specific CSV file.";
            break;
    }
    
    popup.style.display = "block";
}

function closePopup() {
    const popup = document.getElementById("popup");
    popup.style.display = "none";
}
