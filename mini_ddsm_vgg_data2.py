import pandas as pd
import numpy as np
from glob import glob
from tqdm.notebook import tqdm
import time
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import Dense, Flatten, Dropout,  Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


"""# Data Importing"""
# If model already trained, set this to True
model_saved = True
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
df = df.reset_index(drop=True)
filenames = df['Path'].values
labels = df['Label'].values
ages = df['AgeIndex'].astype(str).values

# Create training and testing splits
X = df.copy()
y = df['Label'] # get the label column
# One hot encode the labels
y = to_categorical(labels, num_classes=3)
(train_X, test_X, train_Y, test_Y) = train_test_split(X, y, test_size= 0.20, random_state= 4342, stratify=ages)
# Functions for loading in data
# Define a generator to load the images one at a time from the directory and apply data augmentation
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
          elif df['Path'].iloc[i] in filenames.values:
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

if not model_saved:
    """# VGG-16"""
    num_classes = 3
    input_shape = (224, 224, 3)

    # Define the batch size and the number of training steps per epoch
    batch_size = 20
    steps_per_epoch = 200
    epochs = 70
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
    # Create an ImageDataGenerator with the desired augmentation parameters
    datagen = ImageDataGenerator(
        rotation_range=10,
        zoom_range=0.01,
        width_shift_range=0.05,
        height_shift_range=0.05,
        shear_range=0.01,
        horizontal_flip=True,
        vertical_flip=False,
        fill_mode="nearest")
    datagen_val = ImageDataGenerator()
    train_X['Label'] = train_X['Label'].astype(str)
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_X,
        x_col='Path',
        y_col='Label',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')
    val_generator = datagen_val.flow_from_dataframe(
        dataframe=train_X,
        x_col='Path',
        y_col='Label',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical')
    # train_generator = image_generator(train_X, batch_size)
    # validation_generator = image_generator(train_X, batch_size)
    history = model.fit(train_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, validation_data = val_generator, validation_steps = batch_size)
    plt.plot(range(len(history.history['loss'])) ,history.history['loss'])
    plt.plot(range(len(history.history['val_loss'])) ,history.history['val_loss'])
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('VGG16 Loss Per Epoch')
    plt.savefig('vgg_data2_loss.png')
    plt.clf()
    plt.plot(range(len(history.history['categorical_accuracy'])) ,history.history['categorical_accuracy'])
    plt.plot(range(len(history.history['val_categorical_accuracy'])) ,history.history['val_categorical_accuracy'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('VGG16 Accuracy Per Epoch')
    plt.savefig('vgg_data2_accuracy.png')
    plt.clf()
    model.save('model_vgg_data2')
    # Get all age/density values for the test set
    # Get all age/density values for the test set
    test_age = []
    test_density = []
    for i in range(len(test_X)):
        index = df.index[df['Path'] == test_X['Path'].iloc[i]].tolist()[0]
        test_age.append(df['Age'][index])
        test_density.append(df['Density'][index])

    # Evaluate model on test set
    df['Read'] = False
    steps_test = len(test_X)
    test_generator = image_generator_full(test_X['Path'], test_Y, df, 1)
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
    df_vgg_2.to_csv('df_vgg_data2.csv')

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
    model = tf.keras.models.load_model("model_vgg_data2")
    # test_age = []
    # test_density = []
    # predictions = []
    # # Create an ImageDataGenerator object with the desired augmentation parameters
    # test_gen = ImageDataGenerator(
    #     rotation_range=10,
    #     zoom_range=0.01,
    #     width_shift_range=0.05,
    #     height_shift_range=0.05,
    #     shear_range=0.01,
    #     horizontal_flip=True,
    #     vertical_flip=False,
    #     fill_mode="nearest")

    # # Save the augmented image to file
    # for i in range(len(test_X)):
    #     index = df.index[df['Path'] == test_X['Path'].iloc[i]].tolist()[0]
    #     test_age.append(df['Age'][index])
    #     test_density.append(df['Density'][index])
    #      # Load the image and convert it to a numpy array
    #     img = load_img(test_X['Path'].iloc[i], target_size=(224, 224))
    #     img_array = img_to_array(img)
    #     # Reshape the array to have batch dimension of 1
    #     img_array = np.expand_dims(img_array, axis=0)
    #     # Create an instance of the ImageDataGenerator with the desired data augmentation parameters
    #     datagen = ImageDataGenerator(rotation_range=10,
    #         zoom_range=0.01,
    #         width_shift_range=0.05,
    #         height_shift_range=0.05,
    #         shear_range=0.01,
    #         horizontal_flip=True,
    #         vertical_flip=False,
    #         fill_mode="nearest")
    #     # Apply the data augmentation on the image
    #     x_augmented = datagen.flow(img_array, batch_size=1).next()
    #     # Apply the data augmentation to the image
    #     prediction = model.predict(x_augmented)
    #     predictions.append(np.argmax(prediction, axis=1)[0])

    # # Evaluate model on test set
    # # df['Read'] = False
    # # steps_test = len(test_X)
    # # test_generator = image_generator_full(test_X['Path'], test_Y, df, 1)
    # # predictions = model.predict(test_generator, steps=steps_test)
    # # df['Read'] = False
    # print(predictions)
    # # Get predicted class labels
    # # predicted_labels = np.argmax(predictions, axis=1)
    # test_Y_labels = np.argmax(test_Y, axis=1)
    # # # selecting rows based on condition
    # # df_read = df[df['Read']]
    # # df_read = df_read.reset_index()
    # # print(predicted_labels[0])
    # # print(test_Y_labels[0])
    # # print(test_X['Path'].iloc[0])
    # # print(test_age[0])
    # # print(test_density[0])
    # # Create dataframe with predicted labels and actual labels
    # df_vgg_2 = pd.DataFrame({'Path': test_X['Path'], 'Label': test_Y_labels, 'Prediction': predictions, 'Age': test_age, 'Density': test_density})
    # df_vgg_2 = df_vgg_2.reset_index(drop = True)
    # # Print the dataframe
    # print(df_vgg_2)
    # df_vgg_2.to_csv('df_vgg_uniform_2.csv')

    # print(get_accuracies_by_age(df_vgg_2))
    # print(get_accuracies_by_density(df_vgg_2))

    # # Evaluate model on test set
    # steps_test = len(test_X)
    # test_generator = image_generator(test_X, test_Y, 1)
    # eval_metrics = model.evaluate(test_generator, steps=steps_test)

    # # Print the evaluation metrics
    # print('Test loss:', eval_metrics[0])
    # print('Test accuracy:', eval_metrics[1])
    df_vgg_2 = pd.read_csv('df_vgg_uniform_2.csv')
    print("Test Accuracy: " + str(sum(df_vgg_2['Label'] == df_vgg_2['Prediction']) / len(df_vgg_2)))
    print(get_accuracies_by_age(df_vgg_2))
    print(get_accuracies_by_density(df_vgg_2))