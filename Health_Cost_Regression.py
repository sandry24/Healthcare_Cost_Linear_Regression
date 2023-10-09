import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow import keras
from keras import layers

dataset = pd.read_csv('insurance.csv')
print(dataset.tail())

print(dataset.info())

# one hot encode the categorical columns
categorical_columns = []

for column in dataset.columns:
    if dataset[column].dtype == 'object':
        categorical_columns.append(column)
        # label encoding instead of one hot encoding
        # dataset[column] = dataset[column].astype('category').cat.codes

# had to add dtype='float32' because it was getting converted to boolean
dataset = pd.get_dummies(dataset, columns=categorical_columns, prefix='', prefix_sep='', dtype='float32')
print(dataset.head())

# choosing only high correlation makes model worse because there aren't many features

# correlation_matrix = dataset.corr(method='pearson')
# correlation_with_labels = correlation_matrix['expenses'].sort_values(ascending=False)
# print(correlation_with_labels.head())

# split dataset into training and test
train_dataset = dataset.sample(frac=0.8)
test_dataset = dataset.drop(train_dataset.index)

print(train_dataset.head())

# split into features and labels
train_features = train_dataset.copy()
test_features = test_dataset.copy()

train_labels = train_features.pop('expenses')
test_labels = test_features.pop('expenses')

print(train_dataset.describe().transpose()[['mean', 'std', 'min', 'max']])

# normalizer [takes global min and max, squashes between -1 and 1]
normalizer = tf.keras.layers.Normalization(axis=-1)
normalizer.adapt(np.array(train_features))

# test normalizer
# print(normalizer.mean.numpy())
# first = np.array(train_features[:1])
#
# with np.printoptions(precision=2, suppress=True):
#     print('First example:', first)
#     print()
#     print('Normalized:', normalizer(first).numpy())

# create and fit the model
linear_model = keras.Sequential([
    normalizer,
    layers.Dense(4),
    layers.Dense(1),
])

linear_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.05),
    loss='mean_absolute_error',
    metrics=['mean_absolute_error', 'mean_squared_error'],
)

# Sometimes needed to run the model multiple times to get under 3500 MAE
# It heavily relies on the training data split
# I tried everything to make it work in every case, but it just doesn't work
history = linear_model.fit(
    train_features,
    train_labels,
    epochs=100,
    verbose=1,
    validation_split=0.2,
)


# plot the training data
def plot_loss(history):
    plt.plot(history.history['loss'], label='loss')
    plt.plot(history.history['val_loss'], label='val_loss')
    plt.xlabel('Epoch')
    plt.ylabel('Error [expenses]')
    plt.legend()
    plt.grid(True)


plot_loss(history)

# RUN THIS CELL TO TEST YOUR MODEL. DO NOT MODIFY CONTENTS.
# Test model by checking how well the model generalizes using the test set.
loss, mae, mse = linear_model.evaluate(test_features, test_labels, verbose=2)

print("Testing set Mean Abs Error: {:5.2f} expenses".format(mae))

if mae < 3500:
    print("You passed the challenge. Great job!")
else:
    print("The Mean Abs Error must be less than 3500. Keep trying.")

# Plot predictions.
test_predictions = linear_model.predict(test_features).flatten()

a = plt.axes(aspect='equal')
plt.scatter(test_labels, test_predictions)
plt.xlabel('True values (expenses)')
plt.ylabel('Predictions (expenses)')
lims = [0, 50000]
plt.xlim(lims)
plt.ylim(lims)
_ = plt.plot(lims, lims)
