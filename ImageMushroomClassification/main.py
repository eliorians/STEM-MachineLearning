import os
import numpy as np
import pandas as pd
import time 
#preprocessing imports
import pickle
import PIL.Image
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from ast import literal_eval
#model training imports
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
#evaluation imports
from sklearn.metrics import accuracy_score, confusion_matrix
#plotting imports
import seaborn as sns
from matplotlib import pyplot as plt
import plotly.express as px

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

def process_images(image_paths, target_size):
    #TODO experiment with different image processing (different size, not flattening?)
    print("processing images...")
    images_encoded = []
    #open each image
    for img_path in image_paths:
        #resize image to given target size
        img = PIL.Image.open(img_path).resize(target_size)
        #normalize pixel values to [0, 1]
        img_array = np.array(img) / 255.0  
        #flatten to one dimensional array to be stored
        images_encoded.append(img_array.flatten())

    return images_encoded            


def main(): 

    #init encoder
    le = LabelEncoder()

    image_size = (128, 128)

    if os.path.exists('ImageMushroomClassification/encoded_data.pkl'):
        with open('ImageMushroomClassification/encoded_data.pkl', 'rb') as file:
            data = pickle.load(file)

        with open('ImageMushroomClassification/label_encoder.pkl', 'rb') as le_file:
            le = pickle.load(le_file)

        print('Encoded data and LabelEncoder loaded.')

    else:
        # build the dataframe of images and labels
        data = load_images('ImageMushroomClassification/data/data')

        #encode images (turn image to array of pixels and flatten to one dimensional array)
        data['images_encoded'] = process_images(data['image'], image_size)

        #encode labels
        data['label_encoded'] = le.fit_transform(data['label'])

        #save dataframe to format suitable for encoded images
        with open('ImageMushroomClassification/encoded_data.pkl', 'wb') as file:
            pickle.dump(data, file)
        
        #save label encoder
        with open('ImageMushroomClassification/label_encoder.pkl', 'wb') as le_file:
            pickle.dump(le, le_file)

        print('Encoded data and LabelEncoder saved.')

    #reduce dataset for experimenting
    #data = data.sample(frac=0.005)

    print("the processed data:")
    print(data)

    #start timing
    start = time.time()

    #set up "x" and "y" columns
    #convert encoded images to 2d numpy array for training
    x = np.stack(data['images_encoded'].values)
    y = data['label_encoded']

    #split data into training and testing
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2) 

    #choose a model (start with logistic regression)    
    model = LogisticRegression(max_iter= 250)
    #model = SVC(kernel='linear', max_iter= 250)
    #model = RandomForestClassifier(n_estimators=250, max_depth=10)
    #model = KNeighborsClassifier(n_neighbors=5) 
    
    #fit the model
    model.fit(x_train, y_train)

    #generate predictions
    y_predictions = model.predict(x_test)

    #evaluate predictions
    accuracy = accuracy_score(y_test, y_predictions)
    accuracyPercent = str(accuracy * 100)
    print(f"accuracy for {model}: {accuracyPercent}%")

    #end timing here
    end = time.time()
    total_time = (end - start)
    print("Time spent: (seconds): " + str(total_time))
    
    # --- LOGGING --- #

    currDate = time.strftime("%Y-%m-%d_%H-%M", time.gmtime())
    with open(f'ImageMushroomClassification/eval.txt', 'a') as f:
        f.write(f"Evaluation Date: {currDate}\n")
        f.write(f"Total Time (seconds): {total_time} seconds\n")
        f.write(f"Model: {model}\n")
        f.write(f"Image Size: {image_size}\n")
        f.write(f"Accuracy: {accuracyPercent}%\n")
        f.write("\n")

    # --- PLOTTING --- #
        
    #remap labels labels
    unique_labels = np.unique(np.concatenate([y_test, y_predictions]))
    class_labels = le.inverse_transform(unique_labels)
   
    #create a df of confusion matrix
    conf_matrix = confusion_matrix(y_test, y_predictions)
    np.fill_diagonal(conf_matrix, 0)
    conf_df = pd.DataFrame(conf_matrix)

    #create the heatmap
    fig = px.imshow(
        conf_df,
        labels=dict(x="Predicted Label", y="True Label", color="Count"),
        x=class_labels,
        y=class_labels,
        color_continuous_scale="Blues",
        aspect="auto"
    )

    #update layout
    fig.update_layout(
        title="Confusion Matrix",
        xaxis_title="Predicted Label",
        yaxis_title="True Label"
    )

    #save plot with unique name
    fig.write_html(f'ImageMushroomClassification\plots\{model}_{currDate}.html')

    #display the plot
    fig.show()

#running main
if __name__ == "__main__":
    main()