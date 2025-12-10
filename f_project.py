# from csv import reader


# def load_file(filename):
#     dataset = list()
#     with open(filename, 'r') as file:
#         csv_reader = reader(file)
#         for row in csv_reader:
#             if not row:
#                 continue
#             dataset.append(row)
#     return dataset
#
# filename = 'dataset.csv'
# dataset = load_file(filename)
# print(dataset)
#
# def separate_by_class(dataset):
# 	separated = {}
# 	for i in range(1,len(dataset)):
# 		vector = dataset[i]
# 		class_value = vector[-1]
# 		if (class_value not in separated):
# 			separated[class_value] = list()
# 		separated[class_value].append(vector)
# 	return separated
#
# def mean(numbers):
# 	return sum(numbers)/float(len(numbers))
# print(separate_by_class(dataset))
#


from csv import reader
from math import sqrt
from math import exp
from math import pi


# Load a CSV file
def load_csv(filename):
    dataset = list()
    with open(filename, 'r') as file:
        csv_reader = reader(file)
        for row in csv_reader:
            if not row:
                continue
            dataset.append(row)
    return dataset


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column])


# Split the dataset by class values, returns a dictionary
def separate_by_class(dataset):
    separated = dict()
    for i in range(len(dataset)):
        vector = dataset[i]
        class_value = vector[-1]
        if (class_value not in separated):
            separated[class_value] = list()
        separated[class_value].append(vector)
    return separated


# Calculate the mean of a list of numbers
def mean(numbers):
    return sum(numbers) / float(len(numbers))


# Calculate the standard deviation of a list of numbers
def stdev(numbers):
    avg = mean(numbers)
    variance = sum([(x - avg) ** 2 for x in numbers]) / float(len(numbers) - 1)
    return sqrt(variance)


# Calculate the mean, stdev and count for each column in a dataset
def summarize_dataset(dataset):
    summaries = [(mean(column), stdev(column), len(column)) for column in zip(*dataset)]
    del (summaries[-1])
    return summaries


# Split dataset by class then calculate statistics for each row
def summarize_by_class(dataset):
    separated = separate_by_class(dataset)
    summaries = dict()
    for class_value, rows in separated.items():
        summaries[class_value] = summarize_dataset(rows)
    return summaries


# Calculate the Gaussian probability distribution function for x
def calculate_probability(x, mean, stdev):
    return (1 / (sqrt(2 * pi) * stdev)) * exp(-((x - mean) ** 2 / (2 * stdev ** 2)))


# Calculate the probabilities of predicting each class for a given row
def calculate_class_probabilities(summaries, row):
    total_rows = sum([summaries[label][0][2] for label in summaries])
    probabilities = dict()
    for class_value, class_summaries in summaries.items():
        probabilities[class_value] = summaries[class_value][0][2] / float(total_rows)
        for i in range(len(class_summaries)):
            mean, stdev, _ = class_summaries[i]
            probabilities[class_value] *= calculate_probability(row[i], mean, stdev)
    return probabilities


# Predict the class for a given row
def predict(summaries, row):
    probabilities = calculate_class_probabilities(summaries, row)
    best_label, best_prob = None, -1
    for class_value, probability in probabilities.items():
        if best_label is None or probability > best_prob:
            best_prob = probability
            best_label = class_value
    return best_label


# Make a prediction with Naive Bayes on Iris Dataset
filename = 'dataset_t.csv'
dataset = load_csv(filename)
del dataset[0]
print(dataset)
for i in range(len(dataset[0])):
    str_column_to_float(dataset, i)
# fit model
model = summarize_by_class(dataset)
# define a new record
# row = [90, 5.5]
row = [90, 5.5]
# predict the label
label = predict(model, row)
print('Data=%s, Predicted: %s' % (row, label))
