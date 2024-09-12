#Import statements used for machine learning
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


def main(): 
    data = pd.read_csv("mushrooms.csv")
    #print(data)

    # Convert categorical data to numerical data using one-hot encoding
    data_encoded = pd.get_dummies(data)

    # Separate features and target
    x = data_encoded.drop(columns=["class_e", "class_p"])  # 'class' column is now split into 'class_e' and 'class_p'
    y = data_encoded["class_p"]  # Class 'p' as target (poisonous or not) (1 or 0)   

    #Training the model (x, y, 20% of the data)
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=.2) 
    
    #Making the model
    model = LogisticRegression (max_iter = 100)
    
    #Training the Model
    model.fit(x_train, y_train)

    #Predicting
    y_predictions = model.predict(x_test)
    
    #Evaluate
    accuracy = accuracy_score(y_test, y_predictions)
    accuracyPercent = str(accuracy * 100)
    print(f"accuracy for {model}: {accuracyPercent}%")


#Running main
if __name__ == "__main__":
    main()
    