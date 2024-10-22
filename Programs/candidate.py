import csv

# Open the CSV file and read data
with open("dataset-1.csv") as f:
    csv_file = csv.reader(f)
    data = list(csv_file)

# Initialize specific hypothesis to the first positive example
specific = data[1][:-1]

# Initialize general hypothesis to the most general form
general = [['?' for _ in range(len(specific))] for _ in range(len(specific))]

# Iterate over the data set
for i in range(1, len(data)):  # Skip the header row if it exists
    if data[i][-1] == "Yes":  # If the instance is positive
        for j in range(len(specific)):
            if data[i][j] != specific[j]:  # If attribute value doesn't match
                specific[j] = "?"  # Generalize the specific hypothesis
                general[j][j] = "?"  # Specialize the general hypothesis

    elif data[i][-1] == "No":  # If the instance is negative
        for j in range(len(specific)):
            if data[i][j] != specific[j]:  # If attribute value differs from specific
                general[j][j] = specific[j]  # Specialize the general hypothesis
            else:
                general[j][j] = '?'  # Otherwise, keep it general

    print("\nStep " + str(i) + " of Candidate Elimination Algorithm")
    print("Specific Hypothesis:", specific)
    print("General Hypothesis:", general)

# Clean up general hypothesis by removing overly specific hypotheses
gh = []  # gh = general Hypothesis
for g in general:
    if any(val != '?' for val in g):  # Add only non-trivial hypotheses
        gh.append(g)

print("\nFinal Specific Hypothesis:\n", specific)
print("\nFinal General Hypothesis:\n", gh)
