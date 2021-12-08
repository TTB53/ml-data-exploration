'''
Anthony Thomas-Bell
V1.0 July 23rd 2019
Updated: December 7th 2021


This is the class that is going to help with data visualization and data transformation

'''
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sb

# Class for Exploratory Data Analysis, Plotting and Charting, anything pertaining to visualization of data
class DataVizTransformer:
    # TODO Insert init function that could be used to set defaults like data
    # __init__(data):
    #     self.data = data

    # Helper Functions to give information about a pandas dataframe.
    def data_information(self, data, target_variable, heading=None):

        if heading:
            print("==================================================================")
            print(heading)
            print("==================================================================")

        print("Data First 5 Rows")
        print(data.head(5))
        print("\n")
        # print("Information about the dataset:\n{} \n".format(data.info()))
        print("Shape of the dataset :\n{} \n".format(data.shape))
        print("Null Values by Features in dataset :\n{} \n".format(data.isnull().sum()))
        # print("Amount of memory consumed by dataset:\n{} \n".format(data.memory_usage()))
        print("Count by {} :\n{} \n".format(target_variable, data.groupby(target_variable).size()))

    def data_explorer_plot(self, data, target=None):
        counts = []
        type_stats = None

        typeCategories = list(data.Type.values)
        # counts = data.Type.values.count()
        # data_zip = zip(categories, counts)
        # type_stats = pd.DataFrame(data_zip, columns=[target, target+"_Count"])

        type_stats = data.groupby(typeCategories).size()
        print(type_stats)
        plt.title("Count by Type")
        plt.ylabel("Number of Occurances")
        plt.xlabel(" Strain Types ")
        plt.bar(type_stats)
        plt.show()
        # type_stats.rename_axis([target], axis='columns')

        e1 = list(data.Effect_1.values)

        e2 = list(data.Effect_2.values)

        e3 = list(data.Effect_3.values)

        e4 = list(data.Effect_4.values)

        e5 = list(data.Effect_5.values)


        # effectLists = e1+e2+e3+e4+e5
        # effectLists = [x for x in set(effectLists) if type(x) == str and x is not None]
        # print(effectLists)

        effect_1_stats = data.groupby(e1).size()
        effect_1_stats.name = "Effect 1"

        effect_2_stats = data.groupby(e2).size()
        effect_2_stats.name = "Effect 2"

        effect_3_stats = data.groupby(e3).size()
        effect_3_stats.name = "Effect 3"

        effect_4_stats = data.groupby(e4).size()
        effect_4_stats.name = "Effect 4"

        effect_5_stats = data.groupby(e5).size()
        effect_5_stats.name = "Effect 5"

        #Creates the effects_stats dataframe
        effects_stats = pd.concat([effect_1_stats, effect_2_stats, effect_3_stats, effect_4_stats, effect_5_stats],
                                  axis=1)

        print(effects_stats)
        plt.title("Count by Effect 1-5")
        plt.ylabel("Number of Occurences")
        plt.xlabel(" Strain Effect Types ")
        effects_stats.plot.bar()
        ax1 = effects_stats.plot(kind="scatter", x="Effect 1", y="Effect 2", c=["Green", "Blue"])
        effects_stats.plot(kind="scatter", x="Effect 3", y="Effect 4", c=["purple", "yellow"], ax=ax1)
        plt.show()

    def data_explorer_plot_preprocessed(self, data, target=None):
        features = data.drop(columns=["Description", "Rating", "Effect_1", "Effect_2", "Effect_3", "Effect_4",
                                     "Effect_5", "Flavor_1", "Flavor_2", "Flavor_3", "Flavor_4"], inplace=True)
        features = data.copy()

        if target:
            features[target + '_id'] = data[target].factorize()[0]
        print(features)

        return features

        # if type_stats is not None:
        #     #Plotting data
        #     type_stats.plot(type_stats, type_stats.index, data['Rating'], "bar")

            # Creating and showing the data as bar chart
            # data.plot.bar(type_stats, x=x, y=y)

            # Creating and showing the data as a scatter plot
            # data.plot.scatter(type_stats, x=x, y=y)

        # else:
        #
        #     print("data not plotted.")

    # Takes a pandas dataframe and converts non numerical columns to a numerical representation
    def categorical_to_numeric(self, data, fn=None):

        data_orig = data.copy(deep=True)
        # data_orig.to_csv(fn)

        for column in data.columns:
            if data[column].dtype.name.__contains__('object') and not (column.__contains__('Effects_Combined')
                                                                       or column.__contains__('Flavor_Combined')):
                if column.__contains__('Description'):
                    print("Vectorizing the descriptions")
                else:
                    labels = data[column].astype('category').cat.categories.tolist()
                    replace_map_comp = {column: {k: v for k, v in zip(labels, list(range(1, len(labels)+1)))}}
                    data.replace(replace_map_comp, inplace=True)

        data.fillna(0, inplace=True)  # Replacing all the nan with 0.

        return data






