import random
import csv

# Define attributes
attributes = [['Sunny', 'Rainy'],
              ['Warm', 'Cold'],
              ['Normal', 'High'],
              ['Strong', 'Weak'],
              ['Warm', 'Cool'],
              ['Same', 'Change']]

num_attributes = len(attributes)

print("\nThe most general hypothesis: ['?', '?', '?', '?', '?', '?']\n")
print("The most specific hypothesis: ['0', '0', '0', '0', '0', '0']\n")

# Read the training data set from the CSV file
a = []
print("\nThe Given Training Data Set:\n")
with open("dataset-1.csv", 'r') as csvFile:
    reader = csv.reader(csvFile)
    for row in reader:
        a.append(row)
        print(row)

# Initialize hypothesis
print("\nThe initial value of hypothesis: ")
hypothesis = ['0'] * num_attributes
print(hypothesis)

# Set the hypothesis to the first positive example
for j in range(num_attributes):
    hypothesis[j] = a[0][j]

print("\nFind S: Finding a Maximally Specific Hypothesis\n")
# Iterate through the dataset to find the most specific hypothesis
for i in range(len(a)):
    if a[i][num_attributes] == 'Yes':  # Check if the instance is a positive example
        for j in range(num_attributes):
            if a[i][j] != hypothesis[j]:  # If the attribute does not match, replace with '?'
                hypothesis[j] = '?'
            # else: (This line is unnecessary, as we don't need to change anything if they match)
    
    print("For Training Example No :{0}, the hypothesis is: ".format(i), hypothesis)

print("\nThe Maximally Specific Hypothesis for the given Training Examples:\n")
print(hypothesis)
