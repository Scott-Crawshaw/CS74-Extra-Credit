import pandas as pd
import numpy as np
import sklearn
import nltk

pd.options.mode.chained_assignment = None

from sklearn.feature_extraction.text import TfidfVectorizer

## PREPROCESSING HELPER FUNCTIONS:


# uses VADER's SentimentIntensityAnalyzer to add a four dimensional sentiment score to each review
def addVaderFeatures(panda, unprocessed_text, columnSuffix):
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    analyzer = SentimentIntensityAnalyzer()
    panda['compound' + columnSuffix] = [analyzer.polarity_scores(str(x))['compound'] for x in unprocessed_text]
    panda['neg' + columnSuffix] = [analyzer.polarity_scores(str(x))['neg'] for x in unprocessed_text]
    panda['neu' + columnSuffix] = [analyzer.polarity_scores(str(x))['neu'] for x in unprocessed_text]
    panda['pos' + columnSuffix] = [analyzer.polarity_scores(str(x))['pos'] for x in unprocessed_text]


# Generate csv file with appropriate data for models
# creates a csv with features on a review basis for the training or testing data
def createProcessedFile(isTraining):
    SUMMARY_SUFFIX = "Summary"
    TEXT_SUFFIX = "Text"
    fileChoice = "test"
    if isTraining:
        fileChoice = "training"

    pd.options.mode.chained_assignment = None
    # load data set
    inputFile = pd.read_json('Grocery_and_Gourmet_Food_Reviews_%s.json' % fileChoice, lines=True)
    print('Input File uploaded')
    # create new panda with only each productID
    processedReviews = pd.DataFrame({'ProductID': inputFile['asin']})
    # drop duplicate rows with the same ProductID
    processedReviews.drop_duplicates(subset=['ProductID'], keep='first', inplace=True)

    # get sentiment score for each review's summary and reviewText
    addVaderFeatures(inputFile, inputFile['summary'], SUMMARY_SUFFIX)
    addVaderFeatures(inputFile, inputFile['reviewText'], TEXT_SUFFIX)
    print('Vader features added')

    # keep count to print completeness
    count = 0
    onePer = int(len(processedReviews) / 100)

    # loop over each product ID to generate final CSV
    for value in processedReviews['ProductID']:
        count += 1
        # print percent completeness
        if onePer != 0 and count % onePer == 0:
            print(str(count / onePer) + "%")
        # strings with all of the reviews/summaries for the same product combined
        reviewList = ""
        summList = ""
        # get panda with all of the rows with the same productID
        sameID = inputFile.loc[inputFile['asin'] == value]
        try:
            # combine all of the reviews for the same product into one long string
            processedReviews.loc[(processedReviews['ProductID'] == value), "Reviews"] = " ".join(
                sameID['reviewText'].tolist())

        except TypeError:
            for review in sameID['reviewText']:
                reviewList += " " + str(review)

            processedReviews.loc[(processedReviews['ProductID'] == value), "Reviews"] = reviewList

        try:
            # combine all of the summaries for the same product into one long string
            processedReviews.loc[(processedReviews['ProductID'] == value), "Summaries"] = " ".join(
                sameID['summary'].tolist())

        except TypeError:
            for summary in sameID['summary']:
                summList += " " + str(summary)

            processedReviews.loc[(processedReviews['ProductID'] == value), "Summaries"] = summList

        # only include 'Awesome?' column if using training data
        if isTraining:
            # check if product is awesome, create y column of 1's and 0's
            meanScore = sameID['overall'].mean()
            if meanScore > 4.4:
                product_class = 1
            else:
                product_class = 0
            processedReviews.loc[(processedReviews['ProductID'] == value), "Awesome?"] = product_class

        processedReviews.loc[(processedReviews['ProductID'] == value), "Number of Reviews"] = sameID.shape[0]
        processedReviews.loc[(processedReviews['ProductID'] == value), "Proportion of Verified Reviewers"] = sameID[
                                                                                                                 'verified'].sum() / \
                                                                                                             sameID.shape[
                                                                                                                 0]

        # add 25th, 50th, and 75th percentile of each sentiment score generated by vader
        suff = [SUMMARY_SUFFIX, TEXT_SUFFIX]
        for s in suff:
            processedReviews.loc[(processedReviews['ProductID'] == value), "compound25" + s] = sameID[
                'compound' + s].quantile(.25)
            processedReviews.loc[(processedReviews['ProductID'] == value), "compound50" + s] = sameID[
                'compound' + s].quantile(.50)
            processedReviews.loc[(processedReviews['ProductID'] == value), "compound75" + s] = sameID[
                'compound' + s].quantile(.75)
            processedReviews.loc[(processedReviews['ProductID'] == value), "neg25" + s] = sameID['neg' + s].quantile(
                .25)
            processedReviews.loc[(processedReviews['ProductID'] == value), "neg50" + s] = sameID['neg' + s].quantile(
                .50)
            processedReviews.loc[(processedReviews['ProductID'] == value), "neg75" + s] = sameID['neg' + s].quantile(
                .75)
            processedReviews.loc[(processedReviews['ProductID'] == value), "neu25" + s] = sameID['neu' + s].quantile(
                .25)
            processedReviews.loc[(processedReviews['ProductID'] == value), "neu50" + s] = sameID['neu' + s].quantile(
                .50)
            processedReviews.loc[(processedReviews['ProductID'] == value), "neu75" + s] = sameID['neu' + s].quantile(
                .75)
            processedReviews.loc[(processedReviews['ProductID'] == value), "pos25" + s] = sameID['pos' + s].quantile(
                .25)
            processedReviews.loc[(processedReviews['ProductID'] == value), "pos50" + s] = sameID['pos' + s].quantile(
                .50)
            processedReviews.loc[(processedReviews['ProductID'] == value), "pos75" + s] = sameID['pos' + s].quantile(
                .75)

    # write to CSV
    fileChoice = 'T' + fileChoice[1:]
    processedReviews.to_csv('Groceries_Processed_%s_Data.csv' % fileChoice)
    print("Wrote to CSV")



# stems and tokenizes a string using the open source NLTK library
# supports the TFIDF vectorizer, helping with bag of words creation
def tokenize(text):
    from nltk.stem.porter import PorterStemmer
    tokens = nltk.word_tokenize(text)
    stems = []
    for item in tokens:
        stems.append(PorterStemmer().stem(item))
    return stems

# returns a sci_kit learn TFIDF vectorizer to convert text to TFIDF bag of words
def get_vectorizer(column, X, ngram_range, tokenizer = False):
    if tokenizer:
        vectorizer = TfidfVectorizer(max_features=4000, stop_words='english', ngram_range=ngram_range, tokenizer=tokenize)
    else:
        vectorizer = TfidfVectorizer(max_features=4000, ngram_range=ngram_range)
    vectorizer.fit(X[column].apply(lambda x: np.str_(x)))
    return vectorizer

# returns bag of words for a Panda column of text entries
def process_TFIDF_bow(vectorizer, unprocessed_column):
    result = vectorizer.transform(unprocessed_column.apply(lambda x: np.str_(x)))
    return result.toarray()


## INNER MODELS HELPER FUNCTIONS
# Our approach works by training a bunch of models on an inner training set
# Then we use the predicted probabilities of Awesome for those models as features in our final model

# returns a trained RandomForest model, not hyperparameter-optimized
def get_trained_RandomForest(training_X, training_Y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(n_estimators=100, random_state=3)
    model.fit(training_X, training_Y)
    return model

# returns the hyperparameter-optimized random forest model for review bodies
def get_trained_RandomForest_bodies(training_X, training_Y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=3, bootstrap=False, max_depth=50, max_features='auto', min_samples_leaf=1,
                                   min_samples_split=2, n_estimators=500)
    model.fit(training_X, training_Y)
    return model

# returns the hyperparameter-optimized randomforest model for the summaries
def get_trained_RandomForest_summaries(training_X, training_Y):
    from sklearn.ensemble import RandomForestClassifier
    model = RandomForestClassifier(random_state=3, bootstrap=True, max_depth=None, max_features='auto',
                                   min_samples_leaf=2, min_samples_split=10, n_estimators=100)
    model.fit(training_X, training_Y)
    return model

# hyperparameter optimization for RandomForest bodies, used to find the parameters for the two functions above
def get_RandomForest_optimized_parameters(training_X, training_Y):
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(RandomForestClassifier(random_state=3), param_grid={
        'n_estimators': (100, 200, 500),
        'max_features': ['auto', 'sqrt', 'log2'],
        'bootstrap': [True, False],
        'max_depth': [None, 5, 10, 50, 100],
        'min_samples_split': (2, 5, 10),
        'min_samples_leaf': (1, 2, 5, 10)
    }, n_jobs=-1)
    model.fit(training_X, training_Y)
    print("Best parameters for RF bodies model: ")
    print(str(model.best_params_) + "\n")

# hyperparameter optimization for AdaBoost model
# hyperparameters found to work best are used
# in get_trained_AdaBoost_bodies and get_trained_AdaBoost_summaries
def get_AdaBoost_optimized_parameters(training_X, training_Y):
    from sklearn.ensemble import AdaBoostClassifier
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(AdaBoostClassifier(random_state=3), param_grid={
        'n_estimators': (50, 100, 200),
        'learning_rate': (0.1, 0.5, 1)
    })
    model.fit(training_X, training_Y)
    print("Best parameters for Adaboost summaries model: ")
    print(str(model.best_params_) + "\n")

# returns a trained AdaBoost model
def get_trained_AdaBoost(training_X, training_Y):
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(n_estimators=100, random_state=3)
    model.fit(training_X, training_Y)
    return model

# returns a trained AdaBoost model, optimized with the best hyperparameters found for bodies
def get_trained_AdaBoost_bodies(training_X, training_Y):
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(random_state=3, learning_rate=0.1, n_estimators=200)
    model.fit(training_X, training_Y)
    return model

# optimized with the best hyperparameters found for summaries
def get_trained_AdaBoost_summaries(training_X, training_Y):
    from sklearn.ensemble import AdaBoostClassifier
    model = AdaBoostClassifier(random_state=3, learning_rate=0.5, n_estimators=50)
    model.fit(training_X, training_Y)
    return model

# returns a trained Multinomial Naive Bayes model
def get_trained_MultinomialNB(training_X, training_Y):
    from sklearn.naive_bayes import MultinomialNB
    model = MultinomialNB()
    model.fit(training_X, training_Y)
    return model

#### START EXTRA CREDIT ####
# returns a trained MLP Classifier
def get_trained_MLPClassifier(training_X, training_Y):
    from sklearn.neural_network import MLPClassifier
    model = MLPClassifier(random_state=3)
    model.fit(training_X, training_Y)
    return model

# returns a trained DecisionTreeClassifier
def get_trained_DecisionTreeClassifier(training_X, training_Y):
    from sklearn.tree import DecisionTreeClassifier
    model = DecisionTreeClassifier(random_state=3)
    model.fit(training_X, training_Y)
    return model
#### END EXTRA CREDIT ####

# returns a trained Gradient Boosting Classifier Model
def get_trained_GBC(training_X, training_Y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(n_estimators=100, random_state=3)
    model.fit(training_X, training_Y)
    return model

# returns a trained Gradient Boosting Classifier Model with optimized hyperparameters for the bodies
def get_trained_GBC_bodies(training_X, training_Y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(random_state=3, learning_rate=0.1, min_samples_split=2, n_estimators=500)
    model.fit(training_X, training_Y)
    return model

# returns a trained GBC model with optimized hyperparameters for the summaries
def get_trained_GBC_summaries(training_X, training_Y):
    from sklearn.ensemble import GradientBoostingClassifier
    model = GradientBoostingClassifier(random_state=3, learning_rate=0.1, min_samples_split=10, n_estimators=200)
    model.fit(training_X, training_Y)
    return model

# hyperparameter optimization for GBC summaries
def get_GBC_optimized_parameters(training_X, training_Y):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(GradientBoostingClassifier(random_state=3), param_grid={
            'learning_rate': (0.1, 0.2, 0.5),
            'n_estimators': (100, 200, 500),
            'min_samples_split': (2, 5, 10)
        })
    model.fit(training_X, training_Y)
    print("Best parameters for GBC summaries model: ")
    print(str(model.best_params_) + "\n")

# prints the metrics for a model's performance on a test set
def test_model(model, testX, testY):
    from sklearn.metrics import classification_report
    predictions = model.predict_proba(testX)[:, 1]
    print(classification_report(np.round(predictions), testY))



## FINAL SVM HELPER FUNCTIONS

# get_SVM_features combines all of the features for the final SVM into a panda
# takes:
# a dictionary of models
# TFIDF bag of words features for the review summaries
# TFIDF bag of words features for the review bodies,
# the Vader quantile Scores for each product
# returns a panda of all of the features for the final SVM
def get_SVM_features(models, processed_summaries, processed_bodies, vader_scores):
    result = pd.DataFrame()
    # for each model
    for model_name in models.keys():
        # if the model is on the review bodies
        if model_name[-6:] == "bodies":
            # make predictions on the body features
            result[model_name] = models[model_name].predict_proba(processed_bodies)[:, 1]
        # else if the model is on the summaries
        else:
            # make predictions on the summary features
            result[model_name] = models[model_name].predict_proba(processed_summaries)[:, 1]

    vader_score_col_names = ['compound25', 'compound50', 'compound75', 'pos25', 'pos50', 'pos75', 'neg25', 'neg50', 'neg75', 'neu25', 'neu50', 'neu75']
    # add the vader scores
    for name in vader_score_col_names:
        # one of each for the summaries and the reviewtexts
        temp = name + 'Text'
        result[temp] = vader_scores[temp].values
        temp = name + 'Summary'
        result[temp] = vader_scores[temp].values
    return result

# returns trained SVM model
def get_trained_SVM(processed_SVM_training_features, y_train):
    from sklearn import svm
    model = svm.SVC(kernel = 'rbf')
    model.fit(processed_SVM_training_features, y_train)
    return model

# hyperparameter optimization for SVM
def optimized_SVM_parameters(processed_SVM_training_features, y_train):
    from sklearn import svm
    from sklearn.model_selection import GridSearchCV
    model = GridSearchCV(svm.SVC(), param_grid={
        'C': (0.1, 1, 2),
        'kernel': ['poly', 'rbf'],
        'shrinking': [True, False]
    })
    model.fit(processed_SVM_training_features, y_train)
    print("Best parameters for SVC model: ")
    print(str(model.best_params_))

# returns the best subset of features for the final SVM
def get_best_feature_subset(X, Y):
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import classification_report
    from sklearn.metrics import f1_score
    from sklearn.feature_selection import RFE
    from sklearn.linear_model import LogisticRegression
    trainX, testX, trainY, testY = train_test_split(X, Y, test_size=0.1)

    best_f1 = 0
    best_model = None
    estimator = LogisticRegression()
    for i in range(5, 33):
        rfe = RFE(estimator, i)
        rfe.fit(trainX.values, trainY.values)

        predictions = rfe.predict(testX)

        f1 = f1_score(predictions, testY, average='macro')
        print(i)
        print(f1)
        if f1 > best_f1:
            best_f1 = f1
            best_model = rfe
    print("The subset of features for the best performing model are:")
    result = []
    for i, chosen in enumerate(best_model.support_.tolist()):
        if chosen:
            result.append(trainX.columns.values[i])
    print(result)
    print(classification_report(best_model.predict(testX), testY))
    return result


## TESTING HELPER FUNCTIONS

# returns 10 f1s for each testing set after tenFold cross validation
def tenFoldCVgetF1(untrained_model, X, y):
    from sklearn.model_selection import cross_val_score
    return cross_val_score(untrained_model, X, y, scoring="f1", cv = 10)

# makes predictions for a whole feature set with 10 fold cv
def tenFoldCV_Predict(untrained_model, X, y):
    from sklearn.model_selection import cross_val_predict
    return cross_val_predict(untrained_model, X, y, cv=10)



