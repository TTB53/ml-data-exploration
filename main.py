'''
Anthony Thomas-Bell
V1.0 July 23rd 2019
Updated: December 7th 2021

This is the main file that is going to run the Cannabis Classification problem. We are trying to identify (Target)
what strain(hybrid, sativa, indica) a users is looking up by the effects/flavors desired and give a likelihood of
it being a certain type of cannabis.

'''
import pickle

from sklearn.model_selection import train_test_split
import data_visualization_transfromer as viztransfomer
import ml_models as mlm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn
import math


import numpy as np
# Classes that help with Visualization and Models
dvt = viztransfomer.DataVizTransformer()
ml = mlm.MlModel()

# Loading the working data set and getting some basic statistics about it.
df = pd.read_csv('cannibas-project-processed_sept2019.csv', encoding="unicode_escape")
df.to_json('C:\\Users\\TTB53\\cannafx_beta_api\\cfx_api_app\\knn_data_orig_sept2019.json')
dvt.data_information(data=df, target_variable="Type", heading="Raw Data Exploration")

# Swapping Target Column to the end of the dataframe so it can be stripped without referencing name
cols = list(df.columns.values)
cols.pop(cols.index('Type'))
df = df[cols+['Type']]

# Creating binary dataframe of minority classes since hybrids make up majority of these classes.
df_binary = df[df['Type'] != "hybrid"]

# Confusion Matrix Labels
binary_type_labels = ["Sativa", "Indica"]
type_labels = ["Sativa", "Indica", "Hybrid"]
effect_labels = ["Aroused", "Creative", "Dry",
                 "Energetic", "Euphoric", "Focused",
                 "Giggly", "Happy", "Hungry", "Mouth",
                 "None", "Relaxed", "Sleepy", "Talkative",
                 "Tingly", "Uplifted"]

# Bar Plot showing the values by All Types
type_values = df['Type'].value_counts()
plt.figure()
sn.barplot(type_values.index, type_values.values, alpha=.4, palette='rocket')
plt.title("Count of Strains by Type")
plt.xlabel("Type")
plt.ylabel("Count of Type")

# Adding Numbers above bar
for i, bar in enumerate(plt.axes().patches):
    h = bar.get_height()
    plt.axes().text(
        i,  # bar index (x coordinate of text)
        h + 100,  # y coordinate of text
        '{}'.format(int(h)),  # y label
        ha='center',
        va='center',
        fontweight='bold',
        size=10)
plt.show()

# Bar Plot showing hte values by Binary Type (Indica and Sativa)
type_values = df_binary['Type'].value_counts()
plt.figure()
sn.barplot(type_values.index, type_values.values, alpha=.4, palette='rocket')
plt.title("Count of Strains by Type")
plt.xlabel("Type")
plt.ylabel("Count of Type")

# Adding Numbers above bar
for i, bar in enumerate(plt.axes().patches):
    h = bar.get_height()
    plt.axes().text(
        i,  # bar index (x coordinate of text)
        h + 100,  # y coordinate of text
        '{}'.format(int(h)),  # y label
        ha='center',
        va='center',
        fontweight='bold',
        size=10)
plt.show()


# Creating the training and testing dataset split from df_binary, and factorizing the Type Category of the data set
X_train, X_test, y_train, y_test = train_test_split(df_binary, df_binary['Type'].factorize()[0], random_state=0, test_size=.1)

if "%" in X_train or "%" in y_train:
    print("% in training data")

# Initial Shape of the Data set
print("Shape of X_Train:{} and X Test:{}".format(X_train.shape, X_test.shape))
print("Shape of y_train:{} and y_test:{}".format(y_train.shape, y_test.shape))

# Data Exploration on the new datasets creating counts for each feature for plotting purposes
dvt.data_information(X_train, "Type", "X Training Data Exploration")
X_train = dvt.data_explorer_plot_preprocessed(X_train, "Type")
X_test = dvt.data_explorer_plot_preprocessed(X_test, "Type")

# Creating copies of the original training and test datasets before paring them down this is only being kept for
# debugging and testing purposes.
X_train_orig = X_train.copy()
X_test_orig = X_test.copy()

# Getting only the effects, selecting the columns which are  alphabetized
# these are the base dataframes to be used for training and testing
effects_x_train_df = X_train[effect_labels]
print("Effects Train Column Names: {}".format(effects_x_train_df.columns))

# Checking for empty rows being pulled into our dataframe.
# for val, row in effects_x_train_df['Aroused'].iteritems():
#     if "%" in row:
#         print("% in training data found at {}".format(val))
#         effects_x_train_df.drop(index=val, inplace=True)


effects_x_test_df = X_test[effect_labels]

# Filling any nan or missing values with zeros
train_effects_count = effects_x_train_df.apply(pd.value_counts).fillna(0)
test_effects_count = effects_x_test_df.apply(pd.value_counts).fillna(0)

# Getting the positive (1) instances and negative instance counts by Effect and separating them out for plotting.
train_effects_zero = train_effects_count.iloc[0]  # neg
train_effects_one = train_effects_count.iloc[1]  # pos

# Transpose effects_counts to be column names as index and count values as columns
# train_effects_count_melt = pd.melt(train_effects_count)
# plt.figure()
# sn.barplot(train_effects_count['variable'], train_effects_count['value'], alpha=.4)
#
# # Adding Numbers above bar
# for i, bar in enumerate(plt.axes().patches):
#     h = bar.get_height()
#     plt.axes().text(
#         i,  # bar index (x coordinate of text)
#         h + 75,  # y coordinate of text
#         '{}'.format(int(h)),  # y label
#         ha='center',
#         va='center',
#         fontweight='bold',
#         size=10)
#
# plt.show()
# test_effects_count = pd.melt(test_effects_count)

# Shape of the training and testing data set
print("Training Data Set Shape: {}\nTesting Data Set Shape: {}\n".format(train_effects_count.shape,
                                                                         test_effects_count.shape))

# Plotting the positive values of the training effects data
plt.figure()
sn.barplot(train_effects_one.index, train_effects_one.values, alpha=.4, palette='rocket')

# Adding Numbers above bar
for i, bar in enumerate(plt.axes().patches):
    h = bar.get_height()
    plt.axes().text(
        i,  # bar index (x coordinate of text)
        h + 75,  # y coordinate of text
        '{}'.format(int(h)),  # y label
        ha='center',
        va='center',
        fontweight='bold',
        size=10)

plt.title("Count of Effects - Positive Instances")
plt.xlabel("Effect")
plt.ylabel("Instance Count of Effect")
plt.show()


# Returns the ADASYN and SMOTE oversampling of minority of class transformed training data.
# X_train_res, X_train_res_ada, y_train_res, y_train_res_ada = ml.balance_dataset(effects_x_train_df, y_train)

# Creating the Prediction that every model is going to try and predict the type of strain.
# This is a randomly chosen and can be replaced with additional holdout set data

# Sativas
#   Alaskan Thunder Fuck - Effects: Happy, Euphoric, Uplifted, Energetic, Relaxed, Dry, Mouth
#   Strawberry Mango Haze - Effects: Happy, Euphoric, Uplifted, Focused, Dry, Mouth, Paranoid, Anxious
#   Tangie Ghost Train - Effects: Uplifted, Energetic, Euphoric, Focused, Happy
# Indicas
#   Afghan Big Bud - Effects: Euphoric, Happy, Talkative, Relaxed, Sleepy, Dry, Mouth, Paranoid
#   Black-Label Kush - Effects: Euphoric, Relaxed, Sleepy, Happy, Uplifted, Dry, Mouth, Paranoid
#   California Hash Plant- Effects: Sleepy, Hungry, Relaxed
# Hybrids
#   Afghani Bullrider Effects: Uplifted, Relaxed, Happy, Euphoric, Dry, Mouth, Paranoid
#   Mr. Nice Effects: Relaxed, Happy, Hungry, Sleepy, Dry, Mouth, Anxious
#   Hawaiian Purple Kush Effects: Relaxed, Happy, Hungry, Sleepy, Dry, Mouth, Anxious

# Predictor Effect Order
# ["Aroused", "Creative", "Dry", "Energetic", "Euphoric", "Focused","Giggly", "Happy", "Hungry", "Mouth",
# "None", "Relaxed", "Sleepy", "Talkative","Tingly", "Uplifted"]
predictions_list = []
pred_selections = [0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1, 0, 0, 0, 1]  # Alaskan Thunder Fuck - Sativa
predictions_list.append(pred_selections)

pred_indica = [0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1]  # Black Label - Indica
predictions_list.append(pred_indica)

pred_hybrid = [0, 0, 1, 1, 0, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 0]  # Hawaiian Purple Kush
predictions_list.append(pred_hybrid)

effect_select_dict = dict(zip(effect_labels, pred_selections))


# **************************** MODEL TRAINING BELOW ********************************************************************

# Training a multi label classifier
classifier, y_score = ml.multilabel_classifier(effects_x_train_df, y_train, effects_x_test_df, y_test, estimator=None)
classifier_predictions = classifier.predict(effects_x_test_df)
ml.plot_and_confusion_matrix(classifier, classifier_predictions, binary_type_labels, effects_x_test_df, y_test, "Multilabel Classifier")
ml.model_metrics(y_test, classifier_predictions, "Multi Label Classifier")
predicted = classifier.predict([pred_selections])
print("Based on your selections the predicted class Multi Label Classifier chose is {}".format(binary_type_labels[predicted.item(0)]))
print("The actual class is {} and the name of the strain is {}".format("Sativa", "Alaskan Thunder Fuck"))


# training a KNN classifier
knn, knn_predictions = ml.k_nearest_neighbors(neighbors=int(math.sqrt(effects_x_train_df.size)), X_train=effects_x_train_df,
                                              y_train=y_train, X_test=effects_x_test_df)
ml.plot_and_confusion_matrix(knn, knn_predictions, ["Sativa", "Indica"], effects_x_test_df, y_test,
                             "KNN Confusion Matrix")
ml.model_metrics(y_test, knn_predictions, "KNN")

for i, strain in enumerate(predictions_list):
    neighbors = knn.kneighbors([strain], n_neighbors=10, return_distance=False)
    if i == 0:
        type = "Sativa"
    elif i == 1:
        type = "Indica"
    else:
        type = "Hybrid"

    print("The closest neighbors to your prediction have the following indicies in your test set {}".format(neighbors))
    neighbors = neighbors.flatten().tolist()
    print("The Strain Names and Type are Listed Below")
    for indicie in neighbors:
        print("Strain: {}, Type: {}".format(X_train_orig['Strain'].iloc[indicie], X_train_orig['Type'].iloc[indicie]))

    predicted = knn.predict([strain])
    print("Based on your selections for the {} type the predicted class KNN chose is {}".format(type, binary_type_labels[predicted.item(0)]))
    # print("The actual class is {} and the name of the strain is {}".format("Sativa", "Alaskan Thunder Fuck"))

#TODO Combine X Training and X Testing dataframes and save as csv's, then convert to json
X_train.to_json('C:\\Users\\TTB53\\cannafx_beta_api\\cfx_api_app\\knn_data_sept2019.json')

# X_test.to_json('C:\\Users\\TTB53\\cannafx_beta_api\\knn_test_sept2019.file')

# Persisting the KNN Model, and training data to disk
import joblib
joblib.dump(knn, 'C:\\Users\\TTB53\\cannafx_beta_api\\cfx_api_app\\knn_model_sept2019.file')
# pickle.dump(knn, 'knn_model_sept2019')


# training a decision tree
dtree, dtree_predictions = ml.decision_tree(3, effects_x_train_df, effects_x_test_df, y_train, y_test)
ml.plot_and_confusion_matrix(dtree, dtree_predictions, binary_type_labels, effects_x_test_df, y_test, "Decision Tree")
ml.model_metrics(y_test, dtree_predictions, "Decision Tree Classifier")
predicted = dtree.predict([pred_selections])
print("Based on your selections the predicted class Decision Tree Classifier chose is {}".format(
    binary_type_labels[predicted.item(0)]))

# training a logistic regression model
log_reg, log_reg_predictions = ml.logistic_regression(x=effects_x_train_df, y=y_train, x_test=effects_x_test_df,
                                                      y_test=y_test)
ml.plot_and_confusion_matrix(log_reg, log_reg_predictions, binary_type_labels, effects_x_test_df, y_test,
                             "Logistic Regression")
ml.model_metrics(y_test, log_reg_predictions, "Logistic Regression Classifier")
predicted = log_reg.predict([pred_selections])
print("Based on your selections the predicted class Log Reg chose is {}".format(binary_type_labels[predicted.item(0)]))
print("The actual class is {} and the name of the strain is {}".format("Sativa", "Alaskan Thunder Fuck"))

# training a multinomial logistic regression model
multi_log_reg, multi_log_predictions = ml.logistic_regression(x=effects_x_train_df, y=y_train,
                                                              x_test=effects_x_test_df, y_test=y_test, multi=True)
ml.plot_and_confusion_matrix(multi_log_reg, multi_log_predictions, binary_type_labels, effects_x_test_df, y_test,
                             "Multinomial Logistic Regression")
ml.model_metrics(y_test, multi_log_predictions, "Multinomial Logistic Regression")
predicted = multi_log_reg.predict([pred_selections])
print("Based on your selections the predicted class Log Reg chose is {}".format(binary_type_labels[predicted.item(0)]))
print("The actual class is {} and the name of the strain is {}".format("Sativa", "Alaskan Thunder Fuck"))

# Training svm classifier
svm, svm_predictions = ml.svm_classifier(X_train=effects_x_train_df, X_test=effects_x_test_df,
                                         y_train=y_train, kernal='rbf', c=1.0)
ml.plot_and_confusion_matrix(svm, svm_predictions, binary_type_labels, effects_x_test_df, y_test, "SVM")
ml.model_metrics(y_test, svm_predictions, "Support Vector Machines")
predicted = svm.predict([pred_selections])
print("Based on your selections the predicted class SVM chose is {}".format(binary_type_labels[predicted.item(0)]))
print("The actual class is {} and the name of the strain is {}".format("Sativa", "Alaskan Thunder Fuck"))

# Creating a Naive Bayes Gaussian Classifier
gnb, gnb_predictions = ml.nb_guassian_classifer(effects_x_train_df, y_train, effects_x_test_df)
print("Naive Bayes Gaussian Accuracy: {}".format(gnb.score(effects_x_test_df, y_test)))
ml.plot_and_confusion_matrix(gnb, gnb_predictions, binary_type_labels, effects_x_test_df, y_test, "Naive Bayes Guassian")
ml.model_metrics(y_test, gnb_predictions, "Naive Bayes Gaussian Classifier")
predicted = gnb.predict([pred_selections])
print("Based on your selections the predicted class GNB chose is {}".format(binary_type_labels[predicted.item(0)]))
print("The actual class is {} and the name of the strain is {}".format("Sativa", "Alaskan Thunder Fuck"))

