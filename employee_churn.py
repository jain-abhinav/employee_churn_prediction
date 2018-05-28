import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn import tree
from sklearn import metrics as mt
from sklearn.model_selection import train_test_split
import graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from sklearn.neighbors import DistanceMetric
import operator
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import bokeh.plotting as bp
from bokeh.models import HoverTool
from bokeh.plotting import show
import seaborn as sns
import pyclust

def int_error_testing(any_input):
    try:
        any_input = int(any_input)
    except:
        any_input = "invalid"
    return any_input

def visualize_cluster(tsne_k_algos, k_algos_clusters, emp_data_refactored, important_columns, number_clusters, clustering_type):
    colormap = np.array(["#6d8dca", "#69de53", "#723bca", "#c3e14c", "#c84dc9", "#68af4e", "#6e6cd5",
    "#e3be38", "#4e2d7c", "#5fdfa8", "#d34690", "#3f6d31", "#d44427", "#7fcdd8", "#cb4053", "#5e9981",
    "#803a62", "#9b9e39", "#c88cca", "#e1c37b", "#34223b", "#bdd8a3", "#6e3326", "#cfbdce", "#d07d3c",
    "#52697d", "#7d6d33", "#d27c88", "#36422b", "#b68f79"])
    plot_k_algos = bp.figure(plot_width=700, plot_height=600, title="{} Clustering of Employees".format(clustering_type),
        tools="pan,wheel_zoom,box_zoom,reset,hover,previewsave",
        x_axis_type=None, y_axis_type=None, min_border=1)
    k_algos_df = pd.DataFrame(tsne_k_algos, columns=['x', 'y'])
    k_algos_df['colors'] = colormap[k_algos_clusters]
    k_algos_df['cluster'] = k_algos_clusters
    k_algos_df['CurrentEmployee'] = emp_data_refactored['CurrentEmployee']

    for names in important_columns:
        k_algos_df[names] = emp_data_refactored[names]

    #Plotting Clusters
    plot_k_algos.scatter(x='x', y='y',
                        color="colors",
                        source=k_algos_df)
    hover = plot_k_algos.select(dict(type=HoverTool))
    hover.tooltips={names: "@{}".format(names) for names in k_algos_df.columns[3:]}
    #show(plot_k_algos)

    #Creating Crosstabs with Clusters
    print("All plots are being saved in your local directory.")
    for names in k_algos_df.columns[3:]:
        if names != "cluster":
            #print(*[emp_data_all[names].value_counts() for names in col_names])
            crosstab_emp = pd.crosstab(k_algos_df[names], k_algos_df["cluster"])
            stacked = crosstab_emp.stack().reset_index().rename(columns={0:'value'})
            plot = sns.barplot(x=stacked[names], y=stacked.value, hue=stacked["cluster"])
            plt.setp(plot.get_xticklabels(), rotation=30, horizontalalignment='right')
            fig = plot.get_figure()
            plt.tight_layout()
            fig.savefig("{}_Cluster_{}_{}.png".format(clustering_type, number_clusters, names))
            plt.close()
  

def k_medoids(g_distance, number_clusters):
    kmedoids = pyclust.KMedoids(n_clusters=number_clusters, n_trials=50, random_state = 0).fit_predict(g_distance)
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
    tsne_kmedoids = tsne_model.fit_transform(g_distance)
    return tsne_kmedoids, kmedoids

def gower_distance(X):
    """
    Distance metrics used for:
    Nominal variables: Dice distance (https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient)
    Numeric variables: Manhattan distance normalized by the range of the variable (https://en.wikipedia.org/wiki/Taxicab_geometry)
    """
    individual_variable_distances = []

    for i in range(X.shape[1]):
        feature = X.iloc[:,[i]]
        if feature.dtypes[0] == np.object:
            feature_dist = DistanceMetric.get_metric('dice').pairwise(pd.get_dummies(feature))
        else:
            feature_dist = DistanceMetric.get_metric('manhattan').pairwise(feature) / np.ptp(feature.values)
            
        individual_variable_distances.append(feature_dist.mean(0))

    return np.array(individual_variable_distances).T

def k_means(emp_data, number_clusters):
    kmeans = KMeans(n_clusters=number_clusters, random_state=0).fit(emp_data)
    kmeans_distances = kmeans.transform(emp_data)
    tsne_model = TSNE(n_components=2, verbose=1, random_state=0)
    tsne_kmeans = tsne_model.fit_transform(kmeans_distances)
    return tsne_kmeans, kmeans

def mention_number_clusters():
    while True:
        choice = input("\nInput Number of Clusters:")
        if choice == "":
            break
        choice = int_error_testing(choice)
        if choice == "invalid":
            print("\nInappropriate Input.")
        else:
            return choice

def clustering(emp_data, emp_data_original, emp_data_refactored, important_columns):
    while True:
        emp_data = emp_data[important_columns]
        emp_data_original = emp_data_original[important_columns]
        choice = input("\nInput 1 - K-Means:\nInput 2 - K-Medoids:\n")
        if choice == "":
            break
        choice = int_error_testing(choice)
        if choice == "invalid":
            print("\nInappropriate Input.")
        elif choice == 1:
            number_clusters = mention_number_clusters()
            tsne_kmeans, kmeans = k_means(emp_data, number_clusters)
            visualize_cluster(tsne_kmeans, kmeans.labels_, emp_data_refactored, important_columns, number_clusters, "K-Means")
        elif choice == 2:
            number_clusters = mention_number_clusters()
            g_distance = gower_distance(emp_data_original)
            tsne_kmedoids, kmedoids = k_medoids(g_distance, number_clusters)
            visualize_cluster(tsne_kmedoids, kmedoids, emp_data_refactored, important_columns, number_clusters, "K-Medoids")
        else:
            print("\nInappropriate Input.")

def tree_model(clf, X_train, X_test, y_train, y_test, tree_type):
    clf = clf.fit(X_train, y_train)
    y_predicted = pd.Series(clf.predict(X_test))
    #Decision Tree Output
    print("Accuracy: ", clf.score(X_test, y_test))
    print("Confusion Matrix:\n", mt.confusion_matrix(y_test, y_predicted))
    print("Mean Squared Error: ", mt.mean_squared_error(y_test, y_predicted))
    print("Number of Features: ", clf.n_features_)
    feature_importance = {X_train.columns[i]: clf.feature_importances_[i] for i in range(len(X_train.columns))}
    print("Feature Importance:", *sorted(feature_importance.items(), key=operator.itemgetter(1)), sep = "\n")    
    #Visualization
    if tree_type != "Random Forests":
        feature_name = X_train.columns
        target_name = "CurrentEmployee"
        tree_data = tree.export_graphviz(clf, out_file = None, feature_names = feature_name, class_names = target_name, filled = True, rounded = True, special_characters = True)
        graph = graphviz.Source(tree_data)
        graph.format = "png"
        graph.render("{} Employee".format(tree_type))
    
    
def decision_tree(emp_data):
    while True:
        X = emp_data.drop(["CurrentEmployee"], axis = 1)
        y = emp_data["CurrentEmployee"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        choice = input("\nInput 1 - Un-pruned Tree:\nInput 2 - Constrained Tree:\nInput 3 - Random Forests:\n")
        if choice == "":
            break
        choice = int_error_testing(choice)
        if choice == "invalid":
            print("\nInappropriate Input.")
        elif choice == 1:
            clf = tree.DecisionTreeClassifier(criterion = "entropy") #"gini"
            tree_model(clf, X_train, X_test, y_train, y_test, "Unpruned") 
        elif choice ==2:
            clf = tree.DecisionTreeClassifier(max_leaf_nodes = 10, min_samples_leaf = 20, max_depth= 5, criterion = "entropy")
            tree_model(clf, X_train, X_test, y_train, y_test, "Constrained") 
        elif choice == 3:
            clf = RandomForestClassifier(n_estimators = 500, random_state = 42, criterion = "entropy")
            tree_model(clf, X_train, X_test, y_train, y_test, "Random Forests")
        else:
            print("\nInappropriate Input.")

def learning_models(emp_data, emp_data_original, emp_data_refactored, important_columns):
    while True:
        choice = input("\nInput 1 - Decision Trees:\nInput 2 - Clustering:\n")
        if choice == "":
            break
        choice = int_error_testing(choice)
        if choice == "invalid":
            print("\nInappropriate Input.")
        elif choice == 1:
            decision_tree(emp_data) 
        elif choice == 2:
            clustering(emp_data, emp_data_original, emp_data_refactored, important_columns)
        else:
            print("\nInappropriate Input.")

def cross_tab(emp_data_refactored):
    print("All plots are being saved in your local directory.")
    col_names = list(emp_data_refactored.columns)
    for names in col_names:    
        if names != "CurrentEmployee":
            crosstab_emp = pd.crosstab(emp_data_refactored[names], emp_data_refactored["CurrentEmployee"])
            stacked = crosstab_emp.stack().reset_index().rename(columns={0:'value'})
            plot = sns.barplot(x=stacked[names], y=stacked.value, hue=stacked["CurrentEmployee"])
            plt.setp(plot.get_xticklabels(), rotation=30, horizontalalignment='right')
            fig = plot.get_figure()
            plt.tight_layout()
            fig.savefig("CrossTab_{}.png".format(names))
#            plt.show()
            plt.close()    

def box_plot(emp_data):
    emp_data = emp_data.drop(["CurrentEmployee", "BusinessTravel", "Department", "EducationField", "Gender", "JobRole", "MaritalStatus", "OverTime"], axis = 1)
    fig = emp_data.plot(kind = "box", subplots = True, layout = (4,6), sharex = False, sharey = False)
#    fig.savefig("Employee_boxplot.png")
    plt.show()

def correl_coef(emp_data):
    emp_data = emp_data.drop(["Department", "EducationField", "JobRole"], axis = 1)
    correlations = emp_data.corr()
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(correlations, vmin=-1, vmax=1)
    fig.colorbar(cax)
    ticks = np.arange(0,28,1)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels(emp_data.columns, rotation='vertical')
    ax.set_yticklabels(emp_data.columns)
    fig = fig.tight_layout()
    plt.show()

def descriptive_analysis(emp_data, emp_data_refactored):
    while True:
        choice = input("\nInput 1 - Correlation Coefficients:\nInput 2 - BoxPlots:\nInput 3 - Value Counts/Crosstabs:\n")
        if choice == "":
            break
        choice = int_error_testing(choice)
        if choice == "invalid":
            print("\nInappropriate Input.")
        elif choice == 1:
            correl_coef(emp_data)
        elif choice ==2:
            box_plot(emp_data) 
        elif choice == 3:
            cross_tab(emp_data_refactored)
        else:
            print("\nInappropriate Input.")

def analysis_options(emp_data, emp_data_original, emp_data_refactored, important_columns):
    while True:
        choice = input("\nInput 1 - Descriptive Analysis:\nInput 2 - Learning Models:\n")
        if choice == "":
            break
        choice = int_error_testing(choice)
        if choice == "invalid":
            print("\nInappropriate Input.")
        elif choice == 1:
            descriptive_analysis(emp_data, emp_data_refactored)
        elif choice == 2:
            learning_models(emp_data, emp_data_original, emp_data_refactored, important_columns)
        else:
            print("\nInappropriate Input.")

def refactor(emp_data):
    emp_data["PercentSalaryHike"] = pd.cut(emp_data["PercentSalaryHike"], bins=[0,15,20,25], include_lowest=True)
    emp_data["YearsAtCompany"] = pd.cut(emp_data["YearsAtCompany"], bins=[0,2,5,10,20,40], include_lowest=True)
    emp_data["YearsInCurrentRole"] = pd.cut(emp_data["YearsInCurrentRole"], bins=[0, 2,5,10,20], include_lowest=True)
    emp_data["YearsSinceLastPromotion"] = pd.cut(emp_data["YearsSinceLastPromotion"], bins=[0, 2,5,10,20], include_lowest=True)
    emp_data["YearsWithCurrManager"] = pd.cut(emp_data["YearsWithCurrManager"], bins=[0,2,5,10,20], include_lowest=True)
    emp_data["DailyRate"] = pd.cut(emp_data["DailyRate"], bins=[0,250,500,1000,1500], include_lowest=True)
    emp_data["Age"] = pd.cut(emp_data["Age"], bins=[0,20,30,40,50, 60], include_lowest=True)
    emp_data["HourlyRate"] = pd.cut(emp_data["HourlyRate"], bins=[0,40,60,80,100], include_lowest=True)
    emp_data["MonthlyIncome"] = pd.cut(emp_data["MonthlyIncome"], bins=[0,2000,5000,10000,15000, 20000], include_lowest=True)
    emp_data["MonthlyRate"] = pd.cut(emp_data["MonthlyRate"], bins=[0,5000,10000,15000,20000, 25000, 30000], include_lowest=True)
    emp_data["NumCompaniesWorked"] = pd.cut(emp_data["NumCompaniesWorked"], bins=[0,2,5,10], include_lowest=True)
    emp_data["TotalWorkingYears"] = pd.cut(emp_data["TotalWorkingYears"], bins=[0,2,5,10,20,40], include_lowest=True)
    emp_data["DistanceFromHome"] = pd.cut(emp_data["DistanceFromHome"], bins=[0,5,10,15,20, 25, 30], include_lowest=True)
    return emp_data

def data_processing(emp_data, categorical_columns):
    for names in categorical_columns:  
        le = preprocessing.LabelEncoder()
        le.fit(emp_data[names])
        emp_data[names] = le.transform(emp_data[names])
    return emp_data

def main():
    emp_data = pd.read_csv("employee_data.csv", header = 0)
    emp_data = emp_data.drop(["EmployeeNumber", "Over18", "EmployeeCount", "StandardHours"], axis = 1)      #Dropping columns with same values or employee count
    numeric_columns = ["Age", "DailyRate", "DistanceFromHome", "HourlyRate", "MonthlyIncome", "MonthlyRate", "PercentSalaryHike", "TotalWorkingYears", "YearsAtCompany", "YearsInCurrentRole", "YearsSinceLastPromotion", "YearsWithCurrManager"]     #rest all columns are categorical
    categorical_columns = list(emp_data.drop(numeric_columns, axis = 1).columns)
    for names in categorical_columns:#category_columns_list:    
        emp_data[names] = emp_data[names].astype('object')
    emp_data_original = emp_data.copy(deep = True)
    emp_data_bucketed = emp_data.copy(deep = True)
    emp_data_refactored = refactor(emp_data_bucketed)
    emp_data = data_processing(emp_data, categorical_columns)
    important_columns = ["Age", "BusinessTravel", "Department", "YearsAtCompany", "JobLevel", "JobRole", "MaritalStatus", "OverTime", "TotalWorkingYears", "EducationField", "EnvironmentSatisfaction", "StockOptionLevel"]
    analysis_options(emp_data, emp_data_original, emp_data_refactored, important_columns)
    
if __name__ == "__main__":
    main()
