import pandas as pd
import numpy as np
import sklearn
import nltk
from sklearn.metrics import classification_report
from models_w_functions import *
from sklearn.feature_extraction.text import TfidfVectorizer
# in order to run this code, you need NLTK and sklearn
# specifically within NLTK, VADER's SentimentIntensityAnalyzer and PorterStemmer are needed

pd.options.mode.chained_assignment = None


# create a processed csv file which has one row per product with the appropriate features
# for the training set
#createProcessedFile(True)

# create the processed csv file with one row per product for the test set
#createProcessedFile(False)

# read training data
training = pd.read_csv('Groceries_Processed_Training_Data_Vader.csv')
del training['Unnamed: 0']

Y = training['Awesome?']
X = training[['ProductID', 'Reviews', 'Summaries', 'Number of Reviews', 'compound25Text', 'compound50Text', 'compound75Text', 'pos25Text', 'pos50Text', 'pos75Text', 'neg25Text', 'neg50Text', 'neg75Text', 'neu25Text', 'neu50Text', 'neu75Text', 'compound25Summary', 'compound50Summary', 'compound75Summary', 'pos25Summary', 'pos50Summary', 'pos75Summary', 'neg25Summary', 'neg50Summary', 'neg75Summary', 'neu25Summary', 'neu50Summary', 'neu75Summary']]

# feature scaling
scale_factor = X['Number of Reviews'].max()
X['Number of Reviews'] = X['Number of Reviews'] / scale_factor

# create bag of words TF-IDF vectorizers for review bodies and summaries
# ngrams (1,2) was found to be best during hyperparam optimization for both the summaries and bodies
review_body_vectorizer = get_vectorizer('Reviews', X, (1, 2))
review_summary_vectorizer = get_vectorizer('Summaries', X, (1, 2))

#  split X and y into test and test sets
from sklearn.model_selection import train_test_split
# two training sets, one to train the inner models, one to train the final SVM which takes the inner model results
# X inner train is used to train all of the inner models (RF, ADAboost, NB, GBC)
# X outer train is for the final SVM that takes the inner model predictions
X_inner_train, X_outer_train, y_inner_train, y_outer_train = train_test_split(X, Y, test_size=0.4)

# process the training and cross validation sets into bag of words format

# process the TFIDF features to train the inner models (RF, NB, GBC, Adaboost)
processed_bodies_inner_train = process_TFIDF_bow(review_body_vectorizer, X_inner_train['Reviews'])
processed_summaries_inner_train = process_TFIDF_bow(review_summary_vectorizer, X_inner_train['Summaries'])

#process the bag of word TFIDF features for the final SVM
processed_bodies_outer_train = process_TFIDF_bow(review_body_vectorizer, X_outer_train['Reviews'])
processed_summaries_outer_train = process_TFIDF_bow(review_summary_vectorizer, X_outer_train['Summaries'])


print("done getting features")

# a dictionary of the inner models we have
# keys are the names of the models, values are the sklearn model classes
models = {}


## RANDOM FOREST

# create RF model based on TFIDF bag of words for combined summaries of each product
RF_summaries = get_trained_RandomForest_summaries(processed_summaries_inner_train, y_inner_train)
# add the model to the dictionary of inner_models
models['RFsummaries'] = RF_summaries

# create RF model based on bag of words for combined reviewTexts of each product
RF_bodies = get_trained_RandomForest_bodies(processed_bodies_inner_train, y_inner_train)
# add the model to the dictionary of inner_models
models['RFbodies'] = RF_bodies

print('rf')
## ADAboost Model

# create ADAboost models for the bodies and summaries, add them to the models dictionary
ADA_bodies = get_trained_AdaBoost_bodies(processed_bodies_inner_train, y_inner_train)
models['ADAbodies'] = ADA_bodies
ADA_summaries = get_trained_AdaBoost_summaries(processed_summaries_inner_train, y_inner_train)
models['ADAsummaries'] = ADA_summaries

print('ada')
## Create and Train Multinomial NaiveBayes Model

# create the models and add them to the models dictionary
NB_summaries = get_trained_MultinomialNB(processed_summaries_inner_train, y_inner_train)
models['NBsummaries'] = NB_summaries
NB_bodies = get_trained_MultinomialNB(processed_bodies_inner_train, y_inner_train)
models['NBbodies'] = NB_bodies

print('nb')
## Create and train Gradient Boosting classifier model

GBC_summaries = get_trained_GBC_summaries(processed_summaries_inner_train, y_inner_train)
models["GB_summaries"] = GBC_summaries
GBC_bodies = get_trained_GBC_bodies(processed_bodies_inner_train, y_inner_train)
models['GB_bodies'] = GBC_bodies

#### START EXTRA CREDIT ####
print('gbc')
## Create and train MLP classifier

MLPC_summaries = get_trained_MLPClassifier(processed_summaries_inner_train, y_inner_train)
MLPC_bodies = get_trained_MLPClassifier(processed_bodies_inner_train, y_inner_train)

print('MLPClassifier')
## Create and train DecisionTreeClassifier

DTC_summaries = get_trained_DecisionTreeClassifier(processed_summaries_inner_train, y_inner_train)
DTC_bodies = get_trained_DecisionTreeClassifier(processed_bodies_inner_train, y_inner_train)

print('DecisionTreeClassifier')

# Create output string to hold 10foldcv results
outputString = ""

# read in testing features (processed to a per product basis)
test = pd.read_csv('Groceries_Processed_Test_Data.csv')
# feature scaling number of reviews
test['Number of Reviews'] = test['Number of Reviews']/scale_factor
del test['Unnamed: 0']

# process bag of words
processed_test_bodies = process_TFIDF_bow(review_body_vectorizer, test['Reviews'])
processed_test_summaries = process_TFIDF_bow(review_summary_vectorizer, test['Summaries'])

# Loop through final SVM training 4 times, once with no new classifiers, once with just MLPClassifier, once with just DecisionTreeClassifier, and once with both
for i in range(4):
    labels = ['MLPC_summaries','MLPC_bodies','DTC_summaries','DTC_bodies']
    
    # remove new models from models dict if they are in models dict (pop doesn't throw a keyerror)
    for l in labels:
        models.pop(l, None)
    
    # add MLPClassifier if second or fourth iteration
    if i == 1 or i == 3:
        models[labels[0]] = MLPC_summaries
        models[labels[1]] = MLPC_bodies

    # add DecisionTreeClassifier if third or fourth iteration
    if i == 2 or i == 3:
        models[labels[2]] = DTC_summaries
        models[labels[3]] = DTC_bodies

    print("starting SVM phase " + str(i))

    ## Outer SVM

    #process training features
    SVM_training_features = get_SVM_features(models, processed_summaries_outer_train, processed_bodies_outer_train, X_outer_train)
    SVM_training_features["NumberReviews"] = X_outer_train['Number of Reviews'].values
    # SVM_training_features.to_csv("SVM_training_features_q3.csv", index=False)
    # y_outer_train.to_csv("SVM_train_Y_q3.csv", index=False)

    from sklearn import svm
    # check result metrics of 10 fold cv
    cv_predictions = tenFoldCV_Predict(svm.SVC(kernel='rbf'), SVM_training_features, y_outer_train)
    printOptions = ["No New Classifiers", "With MLPClassifier", "With DecisionTreeClassifier", "With MLPClassifier & DecisionTreeClassifier"]
    outputString += printOptions[i] + "\n\n" + str(classification_report(cv_predictions, y_outer_train)) + "\n\n\n\n"

    # get model trained on all of training set to make predictions on unlabeled test set
    final_SVM = get_trained_SVM(SVM_training_features, y_outer_train)

    # get all final SVM features, add model predictions
    SVM_testing_features = get_SVM_features(models, processed_test_summaries, processed_test_bodies, test)
    SVM_testing_features['Number of Reviews'] = test['Number of Reviews'].values

    # get trained final SVM
    final_SVM = get_trained_SVM(SVM_training_features, y_outer_train)
    final_predictions = test[['ProductID']]

    # make predictions
    final_predictions['Awesome?'] = final_SVM.predict(SVM_testing_features)

    # output to csv, include proper file name for given iteration
    fileOptions = ["No_New_Classifiers", "MLPClassifier", "DecisionTreeClassifier", "MLPClassifier_And_DecisionTreeClassifier"]
    final_predictions.to_csv('Deliverable4_Test_Set_Predictions_%s.csv' % fileOptions[i], index=False)

# Output 10foldCV results to txt file
f1ResultsFile = open('10FoldCV_Results.txt', 'w')
f1ResultsFile.write(outputString)
f1ResultsFile.close()
#### END EXTRA CREDIT ####