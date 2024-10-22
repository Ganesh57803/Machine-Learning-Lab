import pandas as pd
import numpy as np
import warnings

# Ignore warnings
warnings.filterwarnings('ignore')

# Load the dataset
heart_disease = pd.read_csv('network.csv')
print('Columns in dataset:')
for col in heart_disease.columns:
    print(col)

# Define the Bayesian model structure
from pgmpy.models import BayesianModel
from pgmpy.estimators import MaximumLikelihoodEstimator as MLE

model = BayesianModel([
    ('age', 'trestbps'),
    ('age', 'fbs'),
    ('sex', 'trestbps'),
    ('exang', 'trestbps'),
    ('trestbps', 'heartdisease'),
    ('fbs', 'heartdisease'),
    ('heartdisease', 'restecg'),
    ('heartdisease', 'thalach'),
    ('heartdisease', 'chol')
])

# Fit the model using Maximum Likelihood Estimation
model.fit(heart_disease, estimator=MLE)

# Print the conditional probability distribution for 'sex'
print(model.get_cpds('sex'))

# Perform inference
from pgmpy.inference import VariableElimination

HeartDisease_infer = VariableElimination(model)

# Query the model
q = HeartDisease_infer.query(variables=['heartdisease'], evidence={'age': 29, 'sex': 0, 'fbs': 1})
print(q)
