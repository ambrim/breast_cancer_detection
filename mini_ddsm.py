import pandas as pd
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout,  Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

"""# Data Importing"""

# Commented out IPython magic to ensure Python compatibility.
# %matplotlib inline
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# cores = mp.cpu_count()

# plt.rcParams.update({'font.size': 10})
# plt.rcParams['figure.figsize'] = (8, 6)

# print('Cores:', cores)
# print('Device:', device)
# print('Day: ', datetime.now())

# For JPEG
# name_dataset = "/content/drive/Shareddrives/COS429 Final Project/MINI-DDSM-Complete-JPEG-8"
name_dataset = "Mini-DDSM Data/MINI-DDSM-Complete-JPEG-8"
def create_df(name_dataset):
    is_cancer = {'Benign': 1, 'Cancer': 1, 'Normal': 0}
    data = {'patient_id': [], 'image_id':[],'Age':[], 'AgeIndex': [], 'Density': [], 'cancer':[], 'view': [], 'laterality': [], 'Path': [], 'Path_local': [], 'Label': []}
    # extract the path and view
    paths = glob(f"{name_dataset}/*/*")
    for head in tqdm(paths, desc="Loading MINI-DDSM dataset"):
        path = glob(f"{head}/*")
        path_ics = [x for x in path if "ics" in x][0]
        path_img = [x for x in path if ("jpg" in x and 'MASK' not in x.upper())]
        if len(path_img) >= 1:
            # get information from file *.jpg
            for txt in path_img:
                view = txt.split('.')[-2].split('_')[1]
                laterality = txt.split('.')[-2].split('_')[0]
                data['view'].append(view)
                data['laterality'].append('L' if laterality=='LEFT' else 'R')
                data['Path'].append(txt)
                data['Path_local'].append(txt)
                if 'Normal' in txt:
                  data['Label'].append(0)
                if 'Benign' in txt:
                  data['Label'].append(1)
                if 'Cancer' in txt:
                  data['Label'].append(2)
                data['cancer'].append(is_cancer[head.split('/')[-2]])
                data['patient_id'].append(txt.split('/')[-2])
                 # get information from file *.ics
                f = open(path_ics, "r")
                ics_text = f.read().strip().split("\n")
                for txt in ics_text:
                    if txt.split()[0].upper() == 'FILENAME':
                        data['image_id'].append(txt.split()[1] if len(txt.split()) > 1 else 'NaN')
                    if txt.split()[0].upper() == 'PATIENT_AGE':
                        data['Age'].append(txt.split()[1] if len(txt.split()) > 1 else 'NaN')
                        age_index = int(data['Age'][-1]) // 10 if len(txt.split()) > 1 else 'NaN'
                        data['AgeIndex'].append(age_index)
                    if txt.split()[0].upper() == 'DENSITY':
                        data['Density'].append(txt.split()[1] if len(txt.split()) > 1 else 'NaN')
    return pd.DataFrame(data)
s = time.time()
df = create_df(name_dataset)
e = time.time()
print(f'Time running: {e-s}')

print(df.head())

# Create training and testing data splits
# Load the image filenames and labels from a dataframe
df = df[df['Age'] != 'NaN']
filenames = df['Path'].values
labels = df['Label'].values
ages = df['AgeIndex'].astype(str).values

# One hot encode the labels
labels = to_categorical(labels, num_classes=3)
# Create training and testing splits
(train_X, test_X, train_Y, test_Y) = train_test_split(filenames, labels, test_size= 0.20, random_state= 4342, stratify=ages)

# Functions for loading in data
# Define a generator to load the images one at a time from the directory
def image_generator(filenames, labels, batch_size):
    while True:
        # Get the next batch of images and labels
        indices = np.random.randint(0, len(filenames), batch_size)
        batch_filenames = filenames[indices]
        batch_labels = labels[indices]
        x = []
        y = []
        for i in range(batch_size):
            # Load the image and resize it to the desired shape
            image = load_img(batch_filenames[i], target_size=(224, 224))
            image = img_to_array(image) / 255.0
            x.append(image)
            # Load the corresponding label
            label = batch_labels[i]
            y.append(label)
        x = np.array(x)
        y = np.array(y)
        # Yield the batch of images and labels
        yield x, y

# Get Accuracies by Age
def get_accuracies_by_age(data):
  num_bins = 11
  bin_ranges = np.linspace(0, 100, num_bins)
  total_amount = np.zeros(num_bins - 1)
  correct_amount = np.zeros(num_bins - 1)
  for i in range(len(data)):
    label = data['Label'][i]
    prediction = data['Prediction'][i]
    age = float(data['Age'][i])
    age_index = -1
    for j in range(len(bin_ranges) - 1):
      if age >= bin_ranges[j] and age < bin_ranges[j + 1]:
        age_index = j
        break
    total_amount[age_index] += 1
    if label == prediction:
      correct_amount[age_index] += 1
  for i in range(len(total_amount)):
     if total_amount[i] == 0:
        total_amount[i] == 1
  return correct_amount / total_amount

# Get Accuracies by Density
def get_accuracies_by_density(data):
  bin_ranges = np.arange(int(min(data['Density'])), int(max(data['Density'])) + 1)
  total_amount = np.zeros(len(bin_ranges))
  correct_amount = np.zeros(len(bin_ranges))
  for i in range(len(data)):
    label = data['Label'][i]
    prediction = data['Prediction'][i]
    density = int(data['Density'][i])
    for j in range(len(bin_ranges)):
      if density == bin_ranges[j]:
        total_amount[j] += 1
        if label == prediction:
          correct_amount[j] += 1
        break
  return correct_amount / total_amount

"""# Basic CNN"""
# hyper-parameters
batch_size = 16
# 3 categories of images
num_classes = 3
# number of training epochs
epochs = 50
# Steps per epoch
steps_per_epoch = 200

# Define the model
model = Sequential()
# Add the first convolutional layer with 32 filters, a 3x3 kernel size, and ReLU activation
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)))
# Add the first pooling layer with a 2x2 pool size
model.add(MaxPooling2D((2, 2)))
# Add the second convolutional layer with 64 filters, a 3x3 kernel size, and ReLU activation
model.add(Conv2D(64, (3, 3), activation='relu'))
# Add the second pooling layer with a 2x2 pool size
model.add(MaxPooling2D((2, 2)))
# Flatten the output from the convolutional layers
model.add(Flatten())
# Add a fully connected layer with 64 units and ReLU activation
model.add(Dense(64, activation='relu'))
# Add the output layer with 3 units and softmax activation
model.add(Dense(num_classes, activation='softmax'))
# Set other training features
optimizer = Adam(learning_rate=1e-5)
loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']
# Compile and train model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.fit(image_generator(train_X, train_Y, batch_size), epochs=epochs, steps_per_epoch=steps_per_epoch)

model.save("model_cnn")

# Get all age/density values for the test set
test_age = []
test_density = []
for i in range(len(test_X)):
  index = df.index[df['Path'] == test_X[i]].tolist()[0]
  test_age.append(df['Age'][index])
  test_density.append(df['Density'][index])

# Evaluate model on test set
steps_test = len(test_X)
test_generator = image_generator(test_X, test_Y, 1)
predictions = model.predict(test_generator, steps=steps_test)

# Get predicted class labels
predicted_labels = np.argmax(predictions, axis=1)
test_Y_labels = np.argmax(test_Y, axis=1)
# Create dataframe with predicted labels and actual labels
df_cnn = pd.DataFrame({'Path': test_X, 'Label': test_Y_labels, 'Prediction': predicted_labels, 'Age': test_age, 'Density': test_density})
# Print the dataframe
print(get_accuracies_by_age(df_cnn))

print(get_accuracies_by_density(df_cnn))

df_cnn.to_csv('df_cnn.csv')

# Evaluate model on test set
steps_test = len(test_X)
test_generator = image_generator(test_X, test_Y, 1)
eval_metrics = model.evaluate(test_generator, steps=steps_test)

# Print the evaluation metrics
print('Test loss:', eval_metrics[0])
print('Test accuracy:', eval_metrics[1])

# # Functions for loading in data but not randomly
# # Define a generator to load the images one at a time from the directory
# def image_generator_full(filenames, labels, batch_size):
#     while True:
#         # Get the next batch of images and labels
#         current_index = 0
#         for i in range(len(df)):
#           if df['Read'][i]:
#             continue
#           elif df['Path'] in test_X:
#             current_index = i
#             df['Read'][i] = True
#             break
#         x = []
#         y = []
#         # Load the image and resize it to the desired shape
#         image = load_img(filenames[current_index], target_size=(224, 224))
#         image = img_to_array(image) / 255.0
#         x.append(image)
#         # Load the corresponding label
#         label = labels[current_index]
#         y.append(label)
#         x = np.array(x)
#         y = np.array(y)
#         # Yield the batch of images and labels
#         yield x, y

# # Get predictions on whole dataset
# df['Read'] = False
# steps_full = len(filenames)
# full_generator = image_generator_full(filenames, labels, 1)
# predictions = model.predict(full_generator, steps=steps_full)
# df['Read'] = False

# df_cnn = df[['Path', 'Label', 'Age', 'Density']]
# predictions_indices = [np.argmax(row) for row in predictions]
# df_cnn['Prediction'] = predictions_indices 
# df_cnn.to_csv('df_cnn.csv')

print(get_accuracies_by_age(df_cnn))

print(get_accuracies_by_density(df_cnn))

"""# VGG-16"""
# Define the number of classes and the input shape of the images
num_classes = 3
input_shape = (224, 224, 3)

# Define the batch size and the number of training steps per epoch
batch_size = 16
steps_per_epoch = 200
epochs = 50

# Get the VGG16 model with pre-trained weights
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
# Freeze the layers of the VGG16 model
for layer in vgg_model.layers:
    layer.trainable = False
transfer_layer = vgg_model.get_layer('block5_pool')
conv_model = Model(inputs=vgg_model.input, outputs=transfer_layer.output)
# Start a new Keras Sequential model.
model = Sequential()
# Add the convolutional part of the VGG16 model from above.
model.add(conv_model)
# Flatten the output of the VGG16 model because it is from a
# convolutional layer.
model.add(Flatten())
# Add a dense (aka. fully-connected) layer.
# This is for combining features that the VGG16 model has
# recognized in the image.
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
model.add(Dense(1024, activation='relu'))
# Add a dropout-layer which may prevent overfitting and
# improve generalization ability to unseen data e.g. the test-set.
model.add(Dropout(0.5))
# Add the final layer for the actual classification.
model.add(Dense(num_classes, activation='softmax'))
# Set other training features
optimizer = Adam(learning_rate=1e-5)
loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']
# Compile and train model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.fit(image_generator(train_X, train_Y, batch_size), epochs=epochs, steps_per_epoch=steps_per_epoch)

model.save("model_vgg")

# Get all age/density values for the test set
test_age = []
test_density = []
for i in range(len(test_X)):
  index = df.index[df['Path'] == test_X[i]].tolist()[0]
  test_age.append(df['Age'][index])
  test_density.append(df['Density'][index])

# Evaluate model on test set
steps_test = len(test_X)
test_generator = image_generator(test_X, test_Y, 1)
predictions = model.predict(test_generator, steps=steps_test)

# Get predicted class labels
predicted_labels = np.argmax(predictions, axis=1)
test_Y_labels = np.argmax(test_Y, axis=1)

# Create dataframe with predicted labels and actual labels
df_vgg = pd.DataFrame({'Path': test_X, 'Label': test_Y_labels, 'Prediction': predicted_labels, 'Age': test_age, 'Density': test_density})
# Print the dataframe
print(df_vgg)
df_vgg.to_csv('df_vgg.csv')

print(get_accuracies_by_age(df_vgg))
print(get_accuracies_by_density(df_vgg))

# Evaluate model on test set
steps_test = len(test_X)
test_generator = image_generator(test_X, test_Y, 1)
eval_metrics = model.evaluate(test_generator, steps=steps_test)

# Print the evaluation metrics
print('Test loss:', eval_metrics[0])
print('Test accuracy:', eval_metrics[1])

# # Functions for loading in data but not randomly
# # Define a generator to load the images one at a time from the directory
# def image_generator_full(filenames, labels, batch_size):
#     while True:
#         # Get the next batch of images and labels
#         current_index = 0
#         for i in range(len(df)):
#           if df['Read'][i]:
#             continue
#           else:
#             current_index = i
#             df['Read'][i] = True
#             break
#         x = []
#         y = []
#         # Load the image and resize it to the desired shape
#         image = load_img(filenames[current_index], target_size=(224, 224))
#         image = img_to_array(image) / 255.0
#         x.append(image)
#         # Load the corresponding label
#         label = labels[current_index]
#         y.append(label)
#         x = np.array(x)
#         y = np.array(y)
#         # Yield the batch of images and labels
#         yield x, y

# # Get predictions on whole dataset
# df['Read'] = False
# steps_full = len(filenames)
# full_generator = image_generator_full(filenames, labels, 1)
# predictions = model.predict(full_generator, steps=steps_full)
# df['Read'] = False

# df_vgg = df[['Path', 'Label', 'Age', 'Density']]
# predictions_indices = [np.argmax(row) for row in predictions]
# df_vgg['Prediction'] = predictions_indices 
# df_vgg.to_csv('df_vgg.csv')

# print(get_accuracies_by_age(df_vgg))

# print(get_accuracies_by_density(df_vgg))

# Define the number of classes and the input shape of the images
num_classes = 3
input_shape = (224, 224, 3)

# Define the batch size and the number of training steps per epoch
batch_size = 16
steps_per_epoch = 200
epochs = 50

# Get the VGG16 model with pre-trained weights
vgg_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers of the VGG16 model
for layer in vgg_model.layers:
    layer.trainable = False

# Create a new Sequential model and add the VGG16 model to it
model = Sequential()
model.add(vgg_model)

# Add some additional layers for classification
model.add(Conv2D(512, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# Set other training features
optimizer = Adam(learning_rate=1e-5)
loss = 'categorical_crossentropy'
metrics = ['categorical_accuracy']

# Compile and train model
model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
model.fit(image_generator(train_X, train_Y, batch_size), epochs=epochs, steps_per_epoch=steps_per_epoch)
model.save('model_vgg_2')