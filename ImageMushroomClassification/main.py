import os
import pickle
import numpy as np
import pandas as pd

#encoding imports
import PIL.Image
from sklearn.preprocessing import LabelEncoder
from ast import literal_eval
#model training imports
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
#evaluation imports
from sklearn.metrics import accuracy_score, confusion_matrix
#plotting imports
import seaborn as sns
from matplotlib import pyplot as plt

def load_images(folder):
    print("loading images...")
    #initialize list of images and labels
    images = []
    labels = []

    #loop through all folders (names of folders are the label)
    for label in os.listdir(folder):
        label_path = os.path.join(folder, label)
        if os.path.isdir(label_path):
            #loop through images of mushrooms in each folder
            for filename in os.listdir(label_path):
                img_path = os.path.join(label_path, filename)
                #append the image path and labels to list
                images.append(img_path)
                labels.append(label)

    #create dataframe from the lists and return
    return pd.DataFrame({'image': images, 'label' : labels})

def process_images(image_paths, target_size=(128, 128)):
    print("processing images...")
    images_encoded = []
    #open each image
    for img_path in image_paths:
        #resize to (128, 128)
        img = PIL.Image.open(img_path).resize(target_size)
        #normalize pixel values to [0, 1]
        img_array = np.array(img) / 255.0  
        #flatten to one dimensional array to be able to be used
        images_encoded.append(img_array.flatten())

    return images_encoded                

def main(): 

    #init encoder
    le = LabelEncoder()

    if os.path.exists('ImageMushroomClassification/encoded_data.pkl'):
        with open('ImageMushroomClassification/encoded_data.pkl', 'rb') as file:
            data = pickle.load(file)

        print('encoded data loaded.')

    else:
        # build the dataframe of images and labels
        data = load_images('ImageMushroomClassification\data\data')

        #encode images (turn image to array of pixels and flatten to one dimensional array)
        data['images_encoded'] = process_images(data['image'])

        #encode labels
        data['label_encoded'] = le.fit_transform(data['label'])

        #save dataframe to format suitable for encoded images
        with open('ImageMushroomClassification/encoded_data.pkl', 'wb') as file:
            pickle.dump(data, file)

        print('encoded data saved.')

    #reduce dataset for experimenting
    data = data.sample(frac=0.005)

    print("the processed data:")
    print(data)

    #set up "x" and "y" columns
    #convert encoded images to 2d numpy array for training
    x = np.stack(data['images_encoded'].values)
    y = data['label_encoded']

    #TODO split data into training and testing

    #TODO choose a model (start with logistic regression)

    #TODO fit the model

    #TODO generate predictions

    #TODO evaluate predictions

    #TODO plot confusion matrix


#running main
if __name__ == "__main__":
    main()