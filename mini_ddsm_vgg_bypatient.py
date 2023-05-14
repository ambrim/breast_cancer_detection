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
import matplotlib.pyplot as plt

"""# Data Importing"""
# If model already trained, set this to True
model_saved = False
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

from sklearn.model_selection import train_test_split
import numpy as np

# Load the image filenames and labels from a dataframe
df = df[df['Age'] != 'NaN']
df = df.reset_index()
filenames = df['Path'].values
labels = df['Label'].values
ages = df['AgeIndex'].astype(str).values
patient = df['image_id'].values

# One hot encode the labels
labels = to_categorical(labels, num_classes=3)

# Group the image filenames and labels by patient names
patient_data = {}
for i in range(len(patient)):
    if patient[i] in patient_data:
        patient_data[patient[i]]['filenames'].append(filenames[i])
        patient_data[patient[i]]['labels'].append(labels[i])
    else:
        patient_data[patient[i]] = {'filenames': [filenames[i]], 'labels': [labels[i]]}

# Split the patient groups into training and test sets
patient_names = list(patient_data.keys())
train_patients, test_patients = train_test_split(patient_names, test_size=0.2, random_state=4342)

# Combine the image filenames and labels for each patient group
train_filenames = []
train_labels = []
test_filenames = []
test_labels = []
for patient_name in train_patients:
    train_filenames += patient_data[patient_name]['filenames']
    train_labels += patient_data[patient_name]['labels']
for patient_name in test_patients:
    test_filenames += patient_data[patient_name]['filenames']
    test_labels += patient_data[patient_name]['labels']

# Convert the labels to arrays
train_labels = np.array(train_labels)
test_labels = np.array(test_labels)

# Create training and testing splits
(train_X, test_X, train_Y, test_Y) = train_filenames, test_filenames, train_labels, test_labels


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

# Define a generator to load the images one at a time from the directory
def image_generator_full(filenames, labels, df, batch_size = 1):
    while True:
        # Get the next batch of images and labels
        current_index = 0
        for i in range(len(df)):
          if df['Read'][i]:
            continue
          elif df['Path'][i] in filenames:
            current_index = i
            df.loc[i, "Read"] = True
            break
        x = []
        y = []
        # Load the image and resize it to the desired shape
        image = load_img(df['Path'][current_index], target_size=(224, 224))
        image = img_to_array(image) / 255.0
        x.append(image)
        # Load the corresponding label
        label = df['Label'][current_index]
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
  patient_predictions = {}
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
    patient_id = data['PatientID'][i]
    if patient_id not in patient_predictions:
      patient_predictions[patient_id] = []
    patient_predictions[patient_id].append(prediction)
  for patient_id, predictions in patient_predictions.items():
    majority_prediction = max(set(predictions), key=predictions.count)
    for i in range(len(data)):
      if data['PatientID'][i] == patient_id:
        if data['Prediction'][i] == majority_prediction:
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
  patient_predictions = {}
  for i in range(len(data)):
    label = data['Label'][i]
    prediction = data['Prediction'][i]
    density = int(data['Density'][i])
    for j in range(len(bin_ranges)):
      if density == bin_ranges[j]:
        total_amount[j] += 1
        patient_id = data['PatientID'][i]
        if patient_id not in patient_predictions:
          patient_predictions[patient_id] = []
        patient_predictions[patient_id].append(prediction)
        if label == prediction:
          correct_amount[j] += 1
        break
  for patient_id, predictions in patient_predictions.items():
    majority_prediction = max(set(predictions), key=predictions.count)
    for i in range(len(data)):
      if data['PatientID'][i] == patient_id:
        if data['Prediction'][i] == majority_prediction:
          correct_amount[j] += 1
  return correct_amount / total_amount

if not model_saved:
    """# VGG-16"""
    num_classes = 3
    input_shape = (224, 224, 3)

    # Define the batch size and the number of training steps per epoch
    batch_size = 20
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
    # model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    # Set other training features
    optimizer = Adam(learning_rate=1e-5)
    loss = 'categorical_crossentropy'
    metrics = ['categorical_accuracy']

    # Compile and train model
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    train_generator = image_generator(train_X, train_Y, batch_size)
    validation_generator = image_generator(train_X, train_Y, batch_size)
    history = model.fit(train_generator, validation_data = validation_generator, validation_steps = batch_size, epochs=epochs, steps_per_epoch=steps_per_epoch)
    plt.plot(range(len(history.history['loss'])) ,history.history['loss'])
    plt.plot(range(len(history.history['val_loss'])) ,history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VGG16 Loss Per Epoch')
    plt.savefig('bypatient_loss.png')
    plt.clf()
    plt.plot(range(len(history.history['categorical_accuracy'])) ,history.history['categorical_accuracy'])
    plt.plot(range(len(history.history['val_categorical_accuracy'])) ,history.history['val_categorical_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('VGG16 Accuracy Per Epoch')
    plt.savefig('bypatient_accuracy.png')
    plt.clf()
    model.save('model_bypatient')
    # Get all age/density values for the test set
    test_age = []
    test_density = []
    for i in range(len(test_X)):
        index = df.index[df['Path'] == test_X[i]].tolist()[0]
        test_age.append(df['Age'][index])
        test_density.append(df['Density'][index])

    # Evaluate model on test set
    df['Read'] = False
    steps_test = len(test_X)
    test_generator = image_generator_full(test_X, test_Y, df, 1)
    predictions = model.predict(test_generator, steps=steps_test)
    # df['Read'] = False

    # Get predicted class labels
    predicted_labels = np.argmax(predictions, axis=1)
    test_Y_labels = np.argmax(test_Y, axis=1)
    # selecting rows based on condition
    df_read = df[df['Read']]
    df_read = df_read.reset_index()

    # Create dataframe with predicted labels and actual labels
    df_vgg_2 = pd.DataFrame({'Path': df_read['Path'], 'Label': df_read['Label'], 'Prediction': predicted_labels, 'Age': df_read['Age'], 'Density': df_read['Density']})
    # Print the dataframe
    print(df_vgg_2)
    df_vgg_2.to_csv('df_bypatient.csv')

    print(get_accuracies_by_age(df_vgg_2))
    print(get_accuracies_by_density(df_vgg_2))

    # Evaluate model on test set
    steps_test = len(test_X)
    test_generator = image_generator(test_X, test_Y, 1)
    eval_metrics = model.evaluate(test_generator, steps=steps_test)

    # Print the evaluation metrics
    print('Test loss:', eval_metrics[0])
    print('Test accuracy:', eval_metrics[1])
else:
    model = tf.keras.models.load_model("model_bypatient")
    df_vgg = pd.read_csv('df_bypatient.csv')
    print("Test Accuracy: " + str(sum(df_vgg['Label'] == df_vgg['Prediction']) / len(df_vgg)))
    print(get_accuracies_by_age(df_vgg))
    print(get_accuracies_by_density(df_vgg))