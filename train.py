'''
Created on Feb 20, 2018

@author: Evan Kountouris
'''

import pandas
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn import metrics
import matplotlib.pyplot as plt
import seaborn

if __name__ == '__main__':
    
    
    
    #open csv, into data frame
    csv1 = "flat_records_wthHeader.csv"
    df = pandas.read_csv(csv1) 
    
    # display stats
#     ax1 = df.groupby("education_level").size().plot(kind="barh",figsize=(8,5))
    ax2 = df.groupby("occupation").size().plot(kind="barh",figsize=(6,4))
#     ax3 = df.groupby("race").size().plot(kind="barh",figsize=(8,5))
    g = seaborn.factorplot("over_50k", col="marital_status", col_wrap=4,data=df,kind="count", size=3, aspect=0.8)
    plt.show()
    

    #delete columns to be ignored from analysis
    del df["id"]
    del df["education_level"]
    
    #split into training and test data (majority training)
    train, test = train_test_split(df, test_size=0.20)
    
    #create X vectors and Y outputs for training and testing
    y_train = train.pop("over_50k")
    y_test = test.pop("over_50k")
    x_train = pandas.get_dummies(train)
    x_test = pandas.get_dummies(test)
    
    # ensure any missing headers are included in the test data
    if not x_train.columns.equals(x_test.columns):
        missing = set(x_train.columns) - set(x_test.columns)
        for c in missing:
            x_test[c] = 0
        x_test = x_test[x_train.columns]
    
    # Build Decision tree on training data, to depth of 4 to combat overfitting
    model = DecisionTreeClassifier(max_depth=3)
    model.fit(x_train, y_train)  
    
    # the tree can be visualised by using graphviz, transforming the dot file into an image
    # the dot file is created within this code, but must be transformed to image outside
    export_graphviz(model,feature_names=x_train.columns)
    
    # accuracy can be measured by predicting the test data set labels using our built model
    accuracy = metrics.mean_absolute_error(y_test,model.predict(x_test))     
    print("Decision tree is created.")
    print("Accuracy of model applied to test data:",accuracy)             
    
    
    
    
    
    
    
    
    
    
    
    
    