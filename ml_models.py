'''

Anthony Thomas-Bell
V1.0 July 23rd 2019
Updated: December 7th 2

Class that is going to run the scikit learn models for the cannabis classification problem.

'''


class MlModel:
    # TODO Insert INIT function that sets the X_Train, X_Test, y_train, y_test datasets
    # def __init__(self, x_train, x_test, y_train, y_test):
    #     self.X_TRAIN = x_train
    #     self.X_TEST = x_test
    #     self.Y_TRAIN = y_train
    #     self.Y_TEST = x_test

    # Balance the classes in the target column utilizing SMOTE or ADASYN oversampling techniques
    def balance_dataset(self, x_data, y_data, target_col=None, sampling='SMOTE'):
        from imblearn.over_sampling import SMOTE, ADASYN
        # TODO Get the Highest class value, and set every other class to be sampled with that amount in dictionary
        # Since the training dataset is skewed we are going to use SMOTE and ADASYN to artificially inflate
        # our minority classes for this is for training purposes only, it will not effect the testing data.
        # need to adjust balance dict to account for target_col variable that is passed in
        balance_dict = {0: len(y_data), 1: len(y_data)}

        # Sampling Algorithms
        sm = SMOTE(random_state=42, sampling_strategy=balance_dict)
        ada = ADASYN(random_state=42, sampling_strategy=balance_dict)
        # try:

        # Resampled training data
        X_res, y_res = sm.fit_resample(x_data, y_data)
        X_res_ada, y_res_ada = ada.fit_resample(x_data, y_data)

        # except:
        # print("Exception Occurred")

        # print(X_res, y_res)
        return X_res, X_res_ada, y_res, y_res_ada

    # Output the Models Metric information such as the Accuracy, Precision, Recall, F-1 Score, and AUC value
    def model_metrics(self, y_test, predictions, model_name="ML Model", plot=True):
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_curve, roc_auc_score
        from matplotlib import pyplot as plt

        print("==================================================================")
        print(model_name + "\n")
        print("==================================================================")
        print("{} has an accuracy of {}".format(model_name, accuracy_score(y_test, predictions)))
        print("==================================================================")
        print("Precision, Recall, F-Score, Weight")
        print("Base Information {}".format(precision_recall_fscore_support(y_test, predictions)))
        print("Macro {}".format(
            precision_recall_fscore_support(y_test, predictions, average='macro')))
        print("Micro {}".format(
            precision_recall_fscore_support(y_test, predictions, average='micro')))
        print("Weighted {}".format(
            precision_recall_fscore_support(y_test, predictions, average='weighted')))
        print("==================================================================")
        auc = roc_auc_score(y_test, predictions)
        print("{} has a AUC(area under the curve) of : {}".format(model_name, auc))
        print("==================================================================")

        if plot:
            # Plotting the ROC Curve
            fpr, tpr, thresholds = roc_curve(y_test, predictions)
            plt.plot(1, 1)
            plt.plot(fpr, tpr, color='green', label='ROC', marker='.')
            plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('{} ROC Curve'.format(model_name))
            plt.legend()
            plt.figlegend(["AUC = {0:.4f}".format(auc)])
            plt.show()

    # Plot and show the confusion matrix
    def plot_and_confusion_matrix(self, model, data_predictions, type_labels, X_test, y_test, model_name, n_classes=2):
        from matplotlib import pyplot as plt
        import seaborn as sn
        from sklearn.metrics import confusion_matrix

        modelcm = confusion_matrix(y_test, data_predictions)
        print("{} confusion matrix numbers \n{}".format(model_name, modelcm))
        print("Type : {}".format(type(modelcm)))

        # Getting the y_score based on the selected model name, fix redundant elif
        if model_name.__contains__("Decision Tree"):
            y_score = model.decision_path(X_test)
        elif model_name.__contains__("KNN"):
            y_score = model.score(X_test, y_test)
        else:
            y_score = model.score(X_test, y_test)

        plt.title(model_name)
        sn.heatmap(modelcm, annot=True, xticklabels=type_labels, yticklabels=type_labels, fmt=".2f")
        plt.ylabel("Truth")
        plt.xlabel("Predicted")
        plt.show()

    def multilabel_classifier(self, X_train, y_train, X_test, y_test, estimator=None):
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.svm import SVC

        if estimator:
            classifer = OneVsRestClassifier(estimator).fit(X_train, y_train)
        else:
            classifer = OneVsRestClassifier(SVC(kernel='linear')).fit(X_train, y_train)

        # print("Multilabel Classifier Score {}".format(classifer.score(X_test, y_test)))
        y_score = classifer.decision_function(X_test)

        return classifer, y_score

    def k_nearest_neighbors(self, neighbors, X_train, y_train, X_test):
        from sklearn.neighbors import KNeighborsClassifier
        # knn model
        try:
            knn = KNeighborsClassifier(n_neighbors=neighbors, p=2).fit(X_train, y_train)
        except ValueError:
            print("KNN was not created and threw a Value Error")

        knn_predictions = knn.predict(X_test)

        return knn, knn_predictions

    def decision_tree(self, max_depth, X_train, X_test, y_train, y_test):
        from sklearn.tree import DecisionTreeClassifier
        dtree_model = DecisionTreeClassifier(max_depth=max_depth).fit(X_train, y_train)
        # print(dtree_model.score(X_test, y_test))
        dtree_predictions = dtree_model.predict(X_test)

        return dtree_model, dtree_predictions

    def svm_classifier(self, X_train, X_test, y_train, kernal='rbf', c=1.0):
        from sklearn.svm import SVC
        svm_model_linear = SVC(kernel=kernal, C=c, probability=True)
        svm_model_linear.fit(X_train, y_train)
        svm_predictions = svm_model_linear.predict(X_test)
        log_svm_predictions = svm_model_linear.predict_log_proba(X_test)
        # print(log_svm_predictions)

        return svm_model_linear, svm_predictions

    def logistic_regression(self, x, y, x_test, y_test, multi=False):
        from sklearn.linear_model import LogisticRegression

        if multi:
            log_regres_model = LogisticRegression(multi_class="multinomial", solver="newton-cg")
        else:
            log_regres_model = LogisticRegression()

        predicted_labels = None

        if x is not None and y is not None:

            log_regres_model.fit(X=x, y=y)
            r_sq = log_regres_model.score(x, y)
            print("The R Squared for this model is : {}".format(r_sq))

            if x_test is not None:
                predicted_labels = log_regres_model.predict(X=x_test)
                # print(predicted_labels)
                # print(y_test)
                #
                # test_df = pd.DataFrame(data={"predicted": predicted_labels, "actual": y_test})
                # print(test_df)

        return log_regres_model, predicted_labels

    def nb_guassian_classifer(self, X_train, y_train, X_test):
        from sklearn.naive_bayes import GaussianNB

        gnb = GaussianNB()
        gnb_predictions = gnb.fit(X_train, y_train).predict(X_test)
        # print(gnb_predictions)

        return gnb, gnb_predictions

    def csv_to_json(self, csvFilePath, jsonFilePath):
        import csv
        import json

        data = {}

        # Reading and converting csvFile into JSON using the StrainID as the id
        with open(csvFilePath) as csvFile:
            for rows in csv.DictReader(csvFile):
                id = rows['StrainID']
                data[id] = rows
        with open(jsonFilePath, 'w') as jsonFile:
            jsonFile.write(json.dumps(data, indent=4))

        return jsonFile
