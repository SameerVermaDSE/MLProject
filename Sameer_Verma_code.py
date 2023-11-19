import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix,f1_score,make_scorer
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import GridSearchCV , train_test_split
from collections import Counter
from sklearn.metrics import roc_curve, auc
from sklearn.ensemble import VotingClassifier
import warnings

warnings.filterwarnings("ignore")


# ******************  User Defined Function ************* #

def pca(dataset,num_comp):
  pca = PCA(n_components=num_comp)
  df= pca.fit_transform(dataset)
  return df

def replace_missing(data, permissible_values):
    processed_data = data.applymap(lambda x: x if x in permissible_values else pd.NA)
    return processed_data

def plot_roc_curve(y_test, y_pred):
    # Compute ROC curve and ROC area for the positive class
    #label_mapping = {'e': 1, 'p': 0}

    #Y_test1 = [label_mapping[label] for label in y_test]
    #Y_pred1 = [label_mapping[label] for label in y_pred]
    #fpr, tpr, _ = roc_curve(Y_test1, Y_pred1, pos_label=1)
    fpr, tpr, _ = roc_curve(y_test, y_pred, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = {:.2f})'.format(roc_auc))
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.show()

    print("AUC :",roc_auc)

def plot_explained_variance(data, max_components=None):
    pca = PCA(n_components=max_components)
    pca.fit(data)

    # Plotting explained variance ratio and cumulative explained variance
    plt.figure(figsize=(10, 6))

    # Explained variance ratio
    plt.subplot(1, 2, 1)
    plt.plot(pca.explained_variance_ratio_, marker='o', linestyle='-', color='b')
    plt.title('Explained Variance Ratio')
    plt.xlabel('Number of Components')
    plt.ylabel('Explained Variance Ratio')
    plt.grid(True)

    # Cumulative explained variance
    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(pca.explained_variance_ratio_), marker='o', linestyle='-', color='b')
    plt.title('Cumulative Explained Variance')
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


def process_dataset(data, permissible_values):
    processed_data = data.applymap(lambda x: x if x in permissible_values else pd.NA)
    stats = processed_data.describe()
    missing_values = processed_data.isna().sum().to_frame(name='Missing Values')

    return stats, missing_values

def statistics(Dataframe , target):
  labs=target.to_frame()
  stat_data, missing_value_data = process_dataset(Dataframe, permit)
  stat_label, missing_value_label = process_dataset(labs, permit_2)

  print("\n Missing Values in the dataset :")
  print("\n",missing_value_data)
  print("\n Stats of the dataset :")
  print("\n",stat_data)
  print("\n Missing Values in the labels :")
  print("\n", missing_value_label)
  print("\n Stats of the Labels :")
  print("\n",stat_label)


def Test_model(model,Dataset,target):
    reverse_mapping = {1: 'e', 0 : 'p'}
    Dataset = pd.get_dummies(Dataset)
    Dataset = pca(Dataset,5)

    final = model.predict(Dataset)

    final=pd.Series(final)
    final=final.map(reverse_mapping)
    final=pd.Series(final)
    final=final["0"]
    result=classification_report(target,final)
    mat=confusion_matrix(target,final)

    print('\n *************** Final Report ***************  \n')
    print(result)
    print('\n *************** Final Confusion ***************  \n')
    print(mat)
    print('\n ****************** ROC Curve ********************** \n')
    plot_roc_curve(target, final)


permit=['a','b','c','d','e','f','g','h','k','l','m','n','o','p','r','s','t','u','v','w','x','y','z','?'] # only feature values that are allowed according to the data description.
#permit=['a','b','c','d','e','f','g','h','k','l','m','n','o','p','r','s','t','u','v','w','x','y','z']
permit_2=['e','p']
label_mapping = {'e': 1, 'p': 0}
reverse_mapping = {1: 'e', 0 : 'p'}

# ************** LOADING TRAINING DATA ******************* #
label_headers = [ "index","class"]

Train_x = pd.read_csv('mushroom_trn_data.csv')
Train_y = pd.read_csv('mushroom_trn_class_labels.csv',names=label_headers )
train_y = Train_y["class"]

class_counts_train = Counter(train_y)
print(class_counts_train)


# ********** Checking for missing values **************** #
print("*********************** Statistic of Missing Value in the Train dataset and Labels ***************************************")

statistics(Train_x ,train_y)
#Train_x=replace_missing(Train_x,permit)
#Train_x['stalk-root'].fillna(Train_x['stalk-root'].mode()[0], inplace=True)
#print(Train_x['stalk-root'].head(10))

Train_x=Train_x.drop("veil-type",axis=1)    # only one unique value

# *************** One Hot Encoding ********************** #

train_x= pd.get_dummies(Train_x)


# ************** Dimensionality Reduction **************** #
plot_explained_variance(train_x)
train_x=pca(train_x,5)
train_y=train_y.map(label_mapping)
#train_x=pca(train_x,10)  #(tried all the possible values for n_components as suggestedby the elbow method)
#train_x=pca(train_x,15)  #(Did'nt include this in the pipeline as it was taking too long to run)
#train_x=pca(train_x,20)
#train_x=pca(train_x,25)
#train_x=pca(train_x,30)
#train_x=pca(train_x,35)
#train_x=pca(train_x,40)
#train_x=pca(train_x,4)
#train_x=pca(train_x,3)
#train_x=pca(train_x,6)
#train_x=pca(train_x,7)
#train_x=pca(train_x,8)
#train_x=pca(train_x,9)



# *************** Data Splitting ************************* #
X_train, X_test, Y_train, Y_test = train_test_split(train_x, train_y, test_size=0.2, random_state=42)





models = [
    ('NaiveBayes', GaussianNB()),
    ('LogisticRegression', LogisticRegression()),
    ('SVM', SVC()),
    ('KNN', KNeighborsClassifier()),
    ('AdaBoost', AdaBoostClassifier()),
    ('RandomForest', RandomForestClassifier()),
]

param_grid = {

    'NaiveBayes': {
        'model__var_smoothing': np.logspace(0,-9, num=100),
    },
    'LogisticRegression': {
        'model__C': [0.001, 0.01, 0.1, 1, 10, 100],
        'model__penalty': ['l1', 'l2',],
        #'model__solver' : ['lbfgs'],              #lbfgs not compatible with l1
        #'model__penalty': ['l2','none']
        'model__solver': ['liblinear','saga',],

    },
    'SVM': {
        'model__C': [0.1, 1, 10],
        'model__kernel': ['linear', 'rbf', 'poly', 'sigmoid'],
        'model__gamma': ['scale', 'auto'],
    },
    'KNN': {
        'model__n_neighbors': [3, 5, 7, 10],
        #'model__n_neighbors' : [10,20,30,40,50,60,70,80,90,100],
        'model__weights': ['uniform', 'distance'],
        'model__p': [1, 2],
        "model__algorithm":['auto', 'ball_tree', 'kd_tree', 'brute'] ,
    },
    'AdaBoost': {
        'model__n_estimators': [50, 100, 200],
        'model__learning_rate': [0.01, 0.1, 1.0],
    },
     'RandomForest': {
        'model__n_estimators': [ 100,],
        #'model__n_estimators' : [10,50,100,150,200]
        #'model__max_depth' : [10,20,30,40,50,60,70,80,90,100],
        'model__criterion' : [ 'log_loss',],
        #'model__criterion' : ['log_loss','gini','entropy'],
        'model__max_depth': [ 30, None],
        'model__min_samples_split': [ 5, 10],
        'model__min_samples_leaf': [1, 4],
        'model__max_features': [ 'log2','sqrt', None],
    }
}

k_fold_value = 10
top_models = []

best_model = None
best_score = -1


for name, model in models:
    pipeline = Pipeline([

        ('model', model)
    ])

    scorer = make_scorer(f1_score,pos_label=1)
    grid_search = GridSearchCV(pipeline, param_grid[name], cv=k_fold_value , scoring=scorer)
    grid_search.fit(X_train, Y_train)


    print('\n *************** Classification Report for', name, '***************  \n')
    print('\n')
    print(f"{name}: Best Parameters - {grid_search.best_params_}, Best Score - {grid_search.best_score_}")
    predicted = grid_search.best_estimator_.predict(X_test)
    print('\n')
    print(classification_report(Y_test, predicted))
    print('\n *************** Confusion Matrix for', name, '***************  \n')
    print(confusion_matrix(Y_test, predicted))
    print("\n")
    plot_roc_curve(Y_test, predicted)
    top_models.append((name, best_model, grid_search.best_score_))

    if grid_search.best_score_ > best_score:
        best_model = grid_search.best_estimator_
        best_score = grid_search.best_score_

if best_model:
    print('\n *************** Best Model ***************  \n')
    print(best_model.named_steps['model'])
    print("\n")
top_models.sort(key=lambda x: x[2], reverse=True)

# Display the top 3 models
for i, (name, model, score) in enumerate(top_models[:3], 1):
    print(f"Top {i} Model - {name}: F1 Score - {score}")
    print(f"Best Parameters - {model.named_steps['model'].get_params()}")
    print("\n")

# Now you can access the top 3 models from the 'top_models' list
top_model1 = top_models[0][1]  # Access the best model from the list
top_model2 = top_models[1][1]  # Access the second-best model from the list
top_model3 = top_models[2][1]


top_models = [
    ('model1', top_model1),
    ('model2', top_model2),
    ('model3', top_model3)
]


voting_classifier = VotingClassifier(estimators=top_models, voting='hard')


voting_classifier.fit(X_train, Y_train)


predictions = voting_classifier.predict(X_test)

print(classification_report(Y_test, predictions))
print(confusion_matrix(Y_test, predictions))
plot_roc_curve(Y_test,predictions)

# ********************************** Testing ********************************************** #
Test_data=pd.read_csv('mushroom_tst_data.csv')
#Test=replace_missing(Test,permit)
#Test['stalk-root'].fillna(Test['stalk-root'].mode()[0], inplace=True)
Test_data=pd.get_dummies(Test_data)
Test_data=pca(Test_data,5)
Final_labels=voting_classifier.predict(Test_data)
Final_labels=pd.Series(Final_labels)
Final_labels=Final_labels.map(reverse_mapping)
Final_labels=pd.Series(Final_labels)
Final_labels.to_csv("predicted_Labels.csv")




# *********************** Checking for missing values in the test dataset and labels **********************************************#


#Test=pd.read_csv("/content/real_test.csv")
#Label=pd.read_csv('/content/real_labels.csv' )
#Labels= Label['class']
#class_counts = Counter(Labels)
#print(class_counts)
#print("********************* Statistics of Missing Values in Test dataset and Labels *************************************")
#print("\n")
#statistics(Test,Labels)

# ************************* Evaluation on Test dataset ************************** #
#print("/n")

#Test_model(voting_classifier ,Test ,Labels)
