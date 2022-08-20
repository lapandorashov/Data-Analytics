#Coding Cuties: Joy Cheng, Emily Kim, Pandora Shou
#task: predict if the wine is qualified for market or not
#method: decision tree

#dataset source: http://archive.ics.uci.edu/ml/datasets/Wine+Quality

import pandas as pd
import numpy as np

import seaborn as sns
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import cross_val_score,train_test_split
from sklearn import metrics
from sklearn import tree


def main():
    #step 1: import the dataset
    df = pd.read_csv('winequality-red.csv',sep = "\;")
    print(df.info())
    #the dataset contains 1599 bottles of red wine and their attributes
    #we need to build a classification model to predict which are qualified for the market
    #UNDER THE ASSUMPTION that we don't know "quality" when we use model in the future

    #step 2: marking all the data as features except from initial label ("quality")
    features = list(df)[:-1]
    #will contain 11 features (all float)

    #step 3: build the target attribute named "qualified"
    #a binomial attribute that indicates if the wine is qualified
    #in our case, if quality > 5, the wine is qualified, so qualified = True
    #this is our new label/target attribute
    df['"qualified"'] = (df['"quality"'] > 5)


    #step 4: exploratory analysis with summary visualizations
    #color manipulation
    #set the color theme beforehand to coordinate the style of visualizations
    t_color = '#68BB59'
    f_color = '#E3242B'
    colors = {True:t_color,False:f_color,}
    cmap = 'RdYlGn'

    #1: correlation matrix
    correlationMatrix(df,cmap)

    #2: pie chart of ratio of qualified VS. not qualified wine
    pie(df,'"qualified"',colors)
    #since the pie is almost equal, no need for over-sampling

    #3: distribution plots of each feature and compare by "qualified"
    viz(df,'"qualified"',features,colors)


    #step 4: parameter optimization
    #testing from max_depth=1 to 10
    paramOptimize (df,10,features,'"qualified"',t_color)
    #result: accuracy is highest when max dapth is 6

    #step 5 & 6: build the decision tree using max_depth = 6, and evaluate the performance
    decisionTree (df,6,features,'"qualified"',['not qualified','qualified'])





def correlationMatrix(df,colors):
    #df: dataframe
    #colors: the color pattern for visualization

    #calculate the correlation between each feature
    corr = df.corr()
    #plotting the correlation matrix heatmap
    fig = plt.figure()
    plt.matshow(corr,cmap = colors,fignum = fig.number)
    plt.xticks(range(len(corr.columns)),corr.columns,rotation = 'vertical')
    plt.yticks(range(len(corr.columns)),corr.columns)
    plt.show()





def pie(df,target,colors):
    #plot the pie chart of target attribute, compare the size of two classes
    #df: dataframe
    #target: the target attribute
    #colors: the color pattern for the visualization

    fig = px.pie(df, names=target,
                color = target,
                color_discrete_map=colors)
    fig.show()





def viz(df,target,predictors,colors):
    #df: dataframe
    #target: the target attribute
    #predictors: all the features for prediction
    #colors: color pattern of the visualizations

    #plot the distribution of every feature
    for predictor in predictors:
        #compare across label classes (2 lines in one chart)
        ax = sns.displot(data = df,x = predictor, kind ='kde',
                        hue=target, palette=colors,
                        legend = False)
        plt.show()





def paramOptimize (df,max_depth,features,label,color):
    # optimize paramters of decision tree
    #df: dataframe
    #max_depth: the maximum depth that we want to try
    #features: the features we use to predict label class
    #label: the target attribute

    #set a range of paramters for testing
    complexity_values = range(1,max_depth)
    accuracies = []

    #here, we use nested cross validation to optimize the parameters
    #split into training set and test set first
    x_tn, x_test, y_tn, y_test = train_test_split(df[features],df[label],test_size=0.2)

    #calculate the average accuracy under each level of complexity
    for complexity_value in complexity_values:
        #build the decision tree
        clf = tree.DecisionTreeClassifier(criterion="entropy", max_depth=complexity_value)
        #nested cross validation
        scores = cross_val_score(clf, x_tn, y_tn, cv=10)
        accuracy = scores.mean()
        accuracies.append(accuracy)

    # level of complexity VS. accuracy plot
    plt.plot(complexity_values, accuracies, color = color)
    plt.xlabel("Maximum Depth")
    plt.ylabel("Average Accuracy")
    plt.show()





def decisionTree (df,max_dep,features,label,label_names,ftsize=7):
    #model building - decision tree & performance evaluation
    #df: dataframe
    #max_dep: maximum depth for the decision tree
    #features: features that we use for prediction
    #label: the target attribute
    #label_names: the classes of target attribute
    #ftsize: the size of font in tree plot


    #separate data into training set and test set at 0.2 ratio
    x_train, x_test, y_train, y_test = train_test_split(df[features],df[label],test_size=0.2)

    #build the tree model
    clf = tree.DecisionTreeClassifier(max_depth=max_dep)
    clf = clf.fit(x_train, y_train)

    #plot the decision tree model
    dt = tree.plot_tree(clf,
                    feature_names=features,
                    class_names = label_names,
                    filled=True, impurity=False,
                    fontsize=ftsize)
    fig = plt.gcf()
    fig.set_size_inches(10, 10)
    plt.show()

    #produce a png of decision tree
    fig.savefig('tree.png')

    #produce a textual decision tree and export to a text file
    text_representation = tree.export_text(clf, feature_names = features)
    f = open('treetext.txt','w')
    f.write(text_representation)
    f.close()

    # apply the model to the test dataset and get predictions
    y_predict = clf.predict(x_test)

    # evaluate performance using confusion matrix
    cmtx = pd.DataFrame(metrics.confusion_matrix(y_test, y_predict, labels=[True, False]),
                        index=['true:yes', 'true:no'],
                        columns=['pred:yes', 'pred:no'])
    print(cmtx)
    tp = cmtx.iloc[0,0]
    fn = cmtx.iloc[0,1]
    fp = cmtx.iloc[1,0]
    tn = cmtx.iloc[1,1]

    #other performance measures
    print("Precision: %0.2f" % (tp/(tp+fp)))
    print("Recall: %0.2f" % (tp/(tp+fn)))

    # accuracy if using cross validation
    scores = cross_val_score(clf, x_train, y_train, cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))





if __name__ == "__main__":main()
