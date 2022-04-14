# -*- coding: utf-8 -*-
"""
Created on Thu Apr 14 13:17:23 2022

@author: wang
"""

import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
import pathlib

# Read dataset and split to train & validation dataset
file_path=r'C:\Users\wang\Desktop\shrdc\DeepLearning\exercise\project\AI05 Project Concrete Crack Images for Classification\dataset'
data_dir = pathlib.Path(file_path)
SEED=12345
IMG_SIZE = (160,160)
BATCH_SIZE = 64
train_dataset = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                         validation_split=0.3,
                                                         subset='training',
                                                         seed=SEED,
                                                         shuffle=True,
                                                         image_size=IMG_SIZE,
                                                         batch_size=BATCH_SIZE)
val_dataset = tf.keras.utils.image_dataset_from_directory(data_dir,
                                                         validation_split=0.3,
                                                         subset='validation',
                                                         seed=SEED,
                                                         shuffle=True,
                                                         image_size=IMG_SIZE,
                                                         batch_size=BATCH_SIZE)

#%%
#visualize some images from train dataset

class_names = train_dataset.class_names

plt.figure(figsize=(10,10))
for images, labels in train_dataset.take(1):
  for i in range(64):
    ax = plt.subplot(8, 8, i + 1)
    plt.imshow(images[i].numpy().astype("uint8"))
    plt.title(class_names[labels[i]])
    plt.axis("off")
   

#%%
#from validation set, split further into validation-test set
val_batches = tf.data.experimental.cardinality(val_dataset)
test_dataset = val_dataset.take(val_batches // 5)
validation_dataset = val_dataset.skip(val_batches // 5)

#%%
#create prefetch dataset
AUTOTUNE = tf.data.AUTOTUNE
train_dataset_pf = train_dataset.prefetch(buffer_size=AUTOTUNE)
validation_dataset_pf = validation_dataset.prefetch(buffer_size=AUTOTUNE)
test_dataset_pf = test_dataset.prefetch(buffer_size=AUTOTUNE)

#%%
#create data augmentation model
#Data augmentation can be used to address both the requirements, 
#the diversity of the training data, and the amount of data. 
#Is a set of techniques to artificially increase the amount of data by generating 
#new data points from existing data which includes making small changes to data or 
#using deep learning models to generate new data points

data_augmentation = tf.keras.Sequential()
data_augmentation.add(tf.keras.layers.RandomFlip('horizontal'))
data_augmentation.add(tf.keras.layers.RandomRotation(0.2))

#%%
#display examples of image augmentation

for image, labels in train_dataset_pf.take(1):
    first_image = image[0]
    plt.figure(figsize=(10, 10))
    for i in range(9):
        plt.subplot(3, 3, i + 1)
        augmented_image = data_augmentation(tf.expand_dims(first_image, 0))
        plt.imshow(augmented_image[0] / 255.0)
        plt.axis('off')

#data preparation completed

#%%
# By applying transfer learning, use pretrained model object to rescale input 
# use model MobileNetV2 for this project.
# Define a layer that preprocess inputs for the transfer learning model
preprocess_input = tf.keras.applications.mobilenet_v2.preprocess_input

#3.2. Create base model with MobileNetV2
IMG_SHAPE = IMG_SIZE + (3,)
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,
                                               weights='imagenet')

#freeze the base model and show the model summary
base_model.trainable = False
base_model.summary()

#%%
#create our own classification layer with global average pooling and dense layer
class_names = train_dataset.class_names    
global_avg_pool = tf.keras.layers.GlobalAveragePooling2D()
output_dense = tf.keras.layers.Dense(len(class_names),activation='softmax')

#use functional api to construct the entire model 
#(augmentation + preprocess input + NN)

inputs = tf.keras.Input(shape = IMG_SHAPE)
x = data_augmentation(inputs)
x = preprocess_input(x)
x = base_model(x, training=False)
x = global_avg_pool(x)
outputs = output_dense(x)

model = tf.keras.Model(inputs, outputs)
model.summary()
tf.keras.utils.plot_model(model, to_file='model_plot.png', 
                          show_shapes=True, show_layer_names=True)
#%%
#compile model
adam = tf.keras.optimizers.Adam(learning_rate= 0.0001)
loss = tf.keras.losses.SparseCategoricalCrossentropy()

model.compile(optimizer=adam,loss=loss,metrics=['accuracy'])

#%%
#Perform training
import datetime
EPOCHS = 10
base_log_path = r'C:\Users\wang\Desktop\shrdc\DeepLearning\exercise\Tensorboard\concrete_log'
log_path = os.path.join(base_log_path, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
tb_callback = tf.keras.callbacks.TensorBoard(log_dir=log_path)
es_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss',patience=5,verbose=2)
history = model.fit(train_dataset_pf,validation_data=validation_dataset_pf,epochs=EPOCHS,callbacks=[tb_callback,es_callback])

#%%
#evaluate with tset dataset
test_loss,test_accuracy = model.evaluate(test_dataset_pf)

print('------------------------------------Test Result---------------------')
print(f'Loss={test_loss}')
print(f'Accuracy = {test_accuracy}')

#%%
#deploy model to make prediction
image_batch, label_batch = test_dataset_pf.as_numpy_iterator().next()
predictions = model.predict_on_batch(image_batch)
class_predictions = np.argmax(predictions,axis=1)

#%%
#plot the prediction
plt.figure(figsize=(10,10))

for i in range(9):
    axs = plt.subplot(3,3,i+1)
    plt.imshow(image_batch[i].astype('uint8'))
    current_prediction = class_names[class_predictions[i]]
    current_label = class_names[label_batch[i]]
    plt.title(f"Prediction: {current_prediction}, Actual: {current_label}")
    plt.axis('off')
plt.tight_layout()
save_path = r"C:\Users\wang\Desktop\shrdc\DeepLearning\exercise\project\AI05 Project Concrete Crack Images for Classification\img"
plt.savefig(os.path.join(save_path,"result.png"),bbox_inches='tight')

#%%
# Making the Confusion Matrix:

import seaborn as sns
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(label_batch,class_predictions)
#sns.heatmap(cm, annot = True)

ax = sns.heatmap(cm, annot=True, cmap='Blues')

ax.set_title('Confusion Matrix with labels\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values ');

## Ticket labels - List must be in alphabetical order
ax.xaxis.set_ticklabels(['Negative','Positive'])
ax.yaxis.set_ticklabels(['Negative','Positive'])
save_path = r"C:\Users\wang\Desktop\shrdc\DeepLearning\exercise\project\AI05 Project Concrete Crack Images for Classification\img"
plt.savefig(os.path.join(save_path,"confusion_matrix.png"),bbox_inches='tight')
