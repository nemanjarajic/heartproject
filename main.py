import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from mlxtend.plotting import plot_confusion_matrix
from colorama import Fore, Back, Style 

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, precision_score, f1_score
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

dataset = pd.read_csv('heart_records.csv')
dataset.head()

dataset.isnull().sum()
dataset['DEATH_EVENT'].unique()

print('Death', round(dataset['DEATH_EVENT'].value_counts()[1] / len(dataset) * 100, 2), '% of the dataset')
print('No Death', round(dataset['DEATH_EVENT'].value_counts()[0] / len(dataset) * 100, 2), '% of the dataset')

colors = ['#0101df', '#df0101']

sns.countplot('DEATH_EVENT', data=dataset, palette=colors)
plt.title('Death Event Distribution \n (0: No Death || 1: Deatah)', fontsize=14)

dataset.describe()

dataset.info()
dataset.shape
dataset.columns

corrmatrix = dataset.corr()
plt.figure(figsize=(25, 16))
hm = sns.heatmap(corrmatrix, annot = True, linewidths=.5, cmap='coolwarm_r')
hm.set_title(label='Heatmap of dataset', fontsize=20)
plt.show()

f, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))

sns.regplot(x='DEATH_EVENT', y='time', data=dataset, ax=axes[0])
axes[0].set_title('Time VS DEATH_EVENT Negative Correlation')

sns.regplot(x='DEATH_EVENT', y='serum_sodium', data=dataset, ax=axes[1])
axes[1].set_title('Serum Sodium VS DEATH_EVENT Negative Correlation')

sns.regplot(x='DEATH_EVENT', y='ejection_fraction', data=dataset, ax=axes[2])
axes[2].set_title('Ejection Fraction VS DEATH_EVENT Negative Correlation')
plt.show()

f, axes = plt.subplots(nrows=1, ncols=2, figsize=(20, 4))

sns.regplot(x='DEATH_EVENT', y='age', data=dataset, ax=axes[0])
axes[0].set_title('Age VS DEATH_EVENT Positive Correlation')

sns.regplot(x='DEATH_EVENT', y='serum_creatinine', data=dataset, ax=axes[1])
axes[1].set_title('Serum Creatinine VS DEATH_EVENT Positive Correlation')

plt.show()

colors = ['#0101df', '#df0101']

f, axes = plt.subplots(nrows=1, ncols=3, figsize=(20, 4))

# Negative Correlation with DEATH_EVENT (The lower our feature value the more likely it will be a death event)
sns.boxplot(x='DEATH_EVENT', y='time', data=dataset, palette=colors, ax=axes[0])
axes[0].set_title('Time VS DEATH_EVENT')

sns.boxplot(x='DEATH_EVENT', y='serum_sodium', data=dataset, palette=colors, ax=axes[1])
axes[1].set_title('Serum Sodium VS DEATH_EVENT')

sns.boxplot(x='DEATH_EVENT', y='ejection_fraction', data=dataset, palette=colors, ax=axes[2])
axes[2].set_title('Ejection Fraction VS DEATH_EVENT')

plt.show()

f, axes = plt.subplots(nrows=1, ncols = 2, figsize=(20, 4))

# Positive Correlations with DEATH_EVENT (The higher the feature the probability increases that it will be a death event)
sns.boxplot(x='DEATH_EVENT', y='age', data=dataset, palette=colors, ax=axes[0])
axes[0].set_title('Age VS DEATH_EVENT')

sns.boxplot(x='DEATH_EVENT', y='serum_creatinine', data=dataset, palette=colors, ax=axes[1])
axes[1].set_title('Serum Creatinine VS DEATH_EVENT')

plt.show()

from scipy.stats import norm

f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(20, 6))

sodium_death_dist = dataset['serum_sodium'].loc[dataset['DEATH_EVENT'] == 1].values
sns.distplot(sodium_death_dist, ax=ax1, fit=norm, color='#FB8861')
ax1.set_title('Serum Sodium Distribution \n (DEATH Event)', fontsize=14)

ejection_death_dist = dataset['ejection_fraction'].loc[dataset['DEATH_EVENT'] == 1].values
sns.distplot(ejection_death_dist, ax=ax2, fit=norm, color='#56F9BB')
ax2.set_title('Ejection Fraction Distribution \n (DEATH Event)', fontsize=14)

creatinine_death_dist = dataset['serum_creatinine'].loc[dataset['DEATH_EVENT'] == 1].values
sns.distplot(creatinine_death_dist, ax=ax3, fit=norm, color='#C5B3F9')
ax3.set_title('Serum Creatinine Distribution \n (DEATH Event)', fontsize=14)

plt.show()

sns.set(style='dark', palette='colorblind', color_codes=True)
x = dataset.loc[dataset['DEATH_EVENT'] == 1]['age']
plt.figure(figsize=(12, 8))
ax = sns.histplot(x, bins=7, kde=False, color='g')
ax.set_xlabel(xlabel='Patient\'s Age', fontsize=16)
ax.set_ylabel(ylabel='Number of Patients', fontsize=16)
ax.set_title(label='Histogram of Patient Age with DEATH_EVENT = 1', fontsize=20)
plt.show()

# Top Five Eldest Death Event
dataset_death.sort_values('age', ascending=False)[targeted_columns].head(5).reset_index(drop=True)

# Top Five Youngest Death Event
dataset_death.sort_values('age')[targeted_columns].head(5).reset_index(drop=True)


sns.boxplot(y=dataset.age)
sns.boxplot(y=dataset.creatinine_phosphokinase)
sns.boxplot(y=dataset.ejection_fraction)
sns.boxplot(y=dataset.platelets)
sns.boxplot(y=dataset.serum_creatinine)
sns.boxplot(y=dataset.serum_sodium)
sns.boxplot(y=dataset.time)

sns.pairplot(dataset, hue="DEATH_EVENT", size=3, diag_kind="kde")

# Removing ejection outliers from death
ejection_death = dataset['ejection_fraction'].loc[dataset['DEATH_EVENT'] == 1].values
q25, q75 = np.percentile(ejection_death, 25), np.percentile(ejection_death, 75)
ejection_iqr = q75 - q25
ejection_cut_off = ejection_iqr * 1.5
ejection_lower, ejection_upper = q25 - ejection_cut_off, q75 + ejection_cut_off
print('Ejection Fraction Cut Off: {}'.format(ejection_cut_off))
print('Ejection Fraction Lower: {}'.format(ejection_lower))
print('Ejection Fraction Upper: {}'.format(ejection_upper))

outliers = [x for x in ejection_death if x < ejection_lower or x > ejection_upper]
print('Feature Ejection Fraction Outliers for Death Case: {}'.format(len(outliers)))
print('Ejection Fraction Outliers: {}'.format(outliers))
dataset = dataset.drop(dataset[(dataset['ejection_fraction'] > ejection_upper) | (dataset['ejection_fraction'] < ejection_lower)].index)
print('---' * 10)


# Removing serum creatinine outliers from death
creatinine_death = dataset['serum_creatinine'].loc[dataset['DEATH_EVENT'] == 1].values
q25, q75 = np.percentile(creatinine_death, 25), np.percentile(creatinine_death, 75)
creatinine_iqr = q75 - q25
creatinine_cut_off = creatinine_iqr * 1.5
creatinine_lower, creatinine_upper = q25 - creatinine_cut_off, q75 + creatinine_cut_off
print('Serum Creatinine Cut Off: {}'.format(creatinine_cut_off))
print('Serum Creatinine Lower: {}'.format(creatinine_lower))
print('Serum Creatinine Upper: {}'.format(creatinine_upper))

outliers = [x for x in creatinine_death if x < creatinine_lower or x > creatinine_upper]
print('Feature Serum Creatinine Outliers for Death Case: {}'.format(len(outliers)))
print('Serum Creatinine Outliers: {}'.format(outliers))
dataset = dataset.drop(dataset[(dataset['serum_creatinine'] > creatinine_upper) | (dataset['serum_creatinine'] < creatinine_lower)].index)
print('---' * 10)

colors = ['#B3F9C5', '#f9c5b3']
f, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 6))

# Boxplots with outliers removed
# Feature Ejection Fraction
sns.boxplot(x='DEATH_EVENT', y='ejection_fraction', data=dataset, ax=ax1, palette=colors)
ax1.set_title('Ejection Fraction Feature \n (Reduction of outliers)')

sns.boxplot(x='DEATH_EVENT', y='serum_creatinine', data=dataset, ax=ax2, palette=colors)
ax2.set_title('Serum Creatinine Feature \n (Reduction of outliers)')

X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

X.head()
y.head()

from sklearn.ensemble import ExtraTreesClassifier
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize']=15,6 
sns.set_style("darkgrid")

model = ExtraTreesClassifier()
model.fit(X,y)

print(model.feature_importances_) 
feat_importances = pd.Series(model.feature_importances_, index=X.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()

pca = PCA()
sc = StandardScaler()
X = sc.fit_transform(X)
pca.fit_transform(X)

explained_variance_ratio = pca.explained_variance_ratio_
explained_variance_ratio = [x * 100 for x in explained_variance_ratio]

plt.style.context('dark_background')
plt.figure(figsize=(6, 4))
    
plt.bar(range(12), explained_variance_ratio, alpha=0.5, align='center',
        label='individual explained variance')
plt.ylabel('Explained variance ratio (%)')
plt.xlabel('Principal components')
plt.legend(loc='best')
plt.tight_layout()
plt.show()

N = dataset.iloc[:, :-1]
pca = PCA(n_components=2)
x = pca.fit_transform(N)

plt.figure(figsize=(5, 5))
plt.scatter(x[:, 0], x[:, 1])
plt.show()

from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=2, random_state=0)
X_clustered = kmeans.fit_predict(N)

LABEL_COLOURED_MAP = {
    0: 'g',
    1: 'y'
}

label_color = [LABEL_COLOURED_MAP[l] for l in X_clustered]
plt.figure(figsize=(5, 5))
plt.scatter(x[:, 0], x[:, 1], c=label_color)
plt.show()

X = dataset.iloc[:, [4,7,11]]
y = dataset.iloc[:, -1]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

accuracy_list = []

logistic_classifier = LogisticRegression(random_state=0)
logistic_classifier.fit(X_train, y_train)

log_y_pred = logistic_classifier.predict(X_test)

cm = confusion_matrix(y_test, log_y_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Logistic Regression Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

log_ac = accuracy_score(y_test, log_y_pred)
print(Fore.GREEN + "Accuracy of Logistic Regression is : ", "{:.2f}%".format(100* log_ac))
accuracy_list.append(100*log_ac)

svm_classifier = SVC(kernel = 'linear', random_state = 0)
svm_classifier.fit(X_train, y_train)
svm_y_pred = svm_classifier.predict(X_test)

cm = confusion_matrix(y_test, svm_y_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("SVM Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

svm_ac = accuracy_score(y_test, svm_y_pred)
print(Fore.GREEN + "Accuracy of Support Vector Machine is : ", "{:.2f}%".format(100* svm_ac))
accuracy_list.append(100*svm_ac)

kernel_svm_classifier = SVC(kernel = 'rbf', random_state = 0)
kernel_svm_classifier.fit(X_train, y_train)

kernel_svm_y_pred = kernel_svm_classifier.predict(X_test)

cm = confusion_matrix(y_test, kernel_svm_y_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Kernel SVM Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

kernel_svm_ac = accuracy_score(y_test, kernel_svm_y_pred)
print(Fore.GREEN + "Accuracy of Kernel Support Vector Machine is : ", "{:.2f}%".format(100* kernel_svm_ac))
accuracy_list.append(100*kernel_svm_ac)

naive_classifier = GaussianNB()
naive_classifier.fit(X_train, y_train)

naive_y_pred = naive_classifier.predict(X_test)

cm = confusion_matrix(y_test, naive_y_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Naive Bayes - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

naive_ac = accuracy_score(y_test, naive_y_pred)
print(Fore.GREEN + "Accuracy of Naive Bayes is : ", "{:.2f}%".format(100* naive_ac))
accuracy_list.append(100*naive_ac)

knn_classifier = KNeighborsClassifier(n_neighbors = 6, metric = 'minkowski', p = 2)
knn_classifier.fit(X_train, y_train)
knn_y_pred = knn_classifier.predict(X_test)

cm = confusion_matrix(y_test, knn_y_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("KNN Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

knn_ac = accuracy_score(y_test, knn_y_pred)
print(Fore.GREEN + "Accuracy of KNN is : ", "{:.2f}%".format(100* knn_ac))
accuracy_list.append(100*knn_ac)

decision_classifier = DecisionTreeClassifier(criterion = 'entropy', random_state = 0)
decision_classifier.fit(X_train, y_train)

decision_y_pred = decision_classifier.predict(X_test)

cm = confusion_matrix(y_test, decision_y_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Decision Tree Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

decision_ac = accuracy_score(y_test, decision_y_pred)
print(Fore.GREEN + "Accuracy of Decision Tree is : ", "{:.2f}%".format(100* decision_ac))
accuracy_list.append(100*decision_ac)

forest_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest_classifier.fit(X_train, y_train)

forest_classifier = RandomForestClassifier(n_estimators = 10, criterion = 'entropy', random_state = 0)
forest_classifier.fit(X_train, y_train)

forest_y_pred = forest_classifier.predict(X_test)

cm = confusion_matrix(y_test, forest_y_pred)
plt.figure()
plot_confusion_matrix(cm, figsize=(12,8), hide_ticks=True, cmap=plt.cm.Blues)
plt.title("Random Forest Model - Confusion Matrix")
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=16)
plt.show()

forest_ac = accuracy_score(y_test, forest_y_pred)
print(Fore.GREEN + "Accuracy of Random Forest is : ", "{:.2f}%".format(100* forest_ac))
accuracy_list.append(100*forest_ac)

model_list = ['Logistic Regression', 'SVM', 'Kernel SVM', 'Naive Bayes', 'KNearest Neighbours', 'Decision Tree', 'Random Forest']

plt.rcParams['figure.figsize']=20,8
sns.set_style('darkgrid')
ax = sns.barplot(x=model_list, y=accuracy_list, palette = "husl", saturation =2.0)
plt.xlabel('Classifier Models', fontsize = 20 )
plt.ylabel('% of Accuracy', fontsize = 20)
plt.title('Accuracy of different Classifier Models', fontsize = 20)
plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 12)
for i in ax.patches:
    width, height = i.get_width(), i.get_height()
    x, y = i.get_xy() 
    ax.annotate(f'{round(height,2)}%', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()

from sklearn.model_selection import GridSearchCV

# Logistic Regression 
log_reg_params = {
                  'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000]
                }
grid_log_reg = GridSearchCV(LogisticRegression(), log_reg_params, cv=10)
grid_log_reg.fit(X_train, y_train)
# We automatically get the logistic regression with the best parameters.
log_reg = grid_log_reg.best_estimator_

# KNearest
knears_params = {
      "n_neighbors": list(range(1,100,1)), 
      'algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute']
      }
grid_knears = GridSearchCV(KNeighborsClassifier(), knears_params, cv=10)
grid_knears.fit(X_train, y_train)
# KNears best estimator
knears_neighbors = grid_knears.best_estimator_

# Support Vector Classifier
svc_params = {
              'C': [0.5, 0.7, 0.9, 1, 10, 100,500, 1000], 
              'kernel': ['rbf', 'poly', 'sigmoid', 'linear']
              }
grid_svc = GridSearchCV(SVC(), svc_params, cv=10)
grid_svc.fit(X_train, y_train)
# SVC best estimator
svc = grid_svc.best_estimator_

# DecisionTree Classifier
tree_params= {
                'criterion': ['gini','entropy'], 
                'max_features': ["auto","sqrt","log2"],
                'min_samples_leaf': range(1,100,1) , 
                'max_depth': range(1,50,1)
              }
grid_tree = GridSearchCV(DecisionTreeClassifier(), tree_params, cv=10)
grid_tree.fit(X_train, y_train)
# tree best estimator
tree_clf = grid_tree.best_estimator_

# Naive Bayes (There isn't a parameter to tune for naive bayes classifier)

# Random Forest
forest_params = {'n_estimators': range(10,100,10), 
              'max_features': ['log2', 'sqrt','auto'], 
              'criterion': ['entropy', 'gini'],
              'max_depth': [2, 3, 5, 10], 
              'min_samples_split': [2, 3, 5],
              'min_samples_leaf': range(10,100,10)
             }

forest_grid = GridSearchCV(RandomForestClassifier(), forest_params, cv=10)
forest_grid.fit(X_train, y_train)
random_forest = forest_grid.best_estimator_

k_fold_accuracy_list = []
k_fold_stdev_list = []
model_list.remove('Kernel SVM')

logistic_accuracies = cross_val_score(estimator= log_reg, X=X_train, y=y_train, cv=10)
k_fold_accuracy_list.append(logistic_accuracies.mean() * 100)
k_fold_stdev_list.append(logistic_accuracies.std() * 100)

svm_accuracies = cross_val_score(estimator= svc, X=X_train, y=y_train, cv=10)
k_fold_accuracy_list.append(svm_accuracies.mean() * 100)
k_fold_stdev_list.append(svm_accuracies.std() * 100)

naive_accuracies = cross_val_score(estimator= naive_classifier, X=X_train, y=y_train, cv=10)
k_fold_accuracy_list.append(naive_accuracies.mean() * 100)
k_fold_stdev_list.append(naive_accuracies.std() * 100)

knn_accuracies = cross_val_score(estimator= knears_neighbors, X=X_train, y=y_train, cv=10)
k_fold_accuracy_list.append(knn_accuracies.mean() * 100)
k_fold_stdev_list.append(knn_accuracies.std() * 100)

decision_accuracies = cross_val_score(estimator= tree_clf, X=X_train, y=y_train, cv=10)
k_fold_accuracy_list.append(decision_accuracies.mean() * 100)
k_fold_stdev_list.append(decision_accuracies.std() * 100)

forest_accuracies = cross_val_score(estimator= random_forest, X=X_train, y=y_train, cv=10)
k_fold_accuracy_list.append(forest_accuracies.mean() * 100)
k_fold_stdev_list.append(forest_accuracies.std() * 100)

plt.rcParams['figure.figsize']=20,8
sns.set_style('darkgrid')
ax = sns.barplot(x=model_list, y=k_fold_accuracy_list, palette = "husl", saturation =2.0)
plt.xlabel('Classifier Models', fontsize = 20 )
plt.ylabel('% of K-fold Accuracy', fontsize = 20)
plt.title('K-fold Accuracy of different Classifier Models', fontsize = 20)
plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 12)
for i in ax.patches:
    width, height = i.get_width(), i.get_height()
    x, y = i.get_xy() 
    ax.annotate(f'{round(height,2)}%', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()

plt.rcParams['figure.figsize']=20,8
sns.set_style('darkgrid')
ax = sns.barplot(x=model_list, y=k_fold_stdev_list, palette = "husl", saturation =2.0)
plt.xlabel('Classifier Models', fontsize = 20 )
plt.ylabel('% of K-fold Standard Deviation', fontsize = 20)
plt.title('K-fold Standard Deviation of different Classifier Models', fontsize = 20)
plt.xticks(fontsize = 12, horizontalalignment = 'center', rotation = 8)
plt.yticks(fontsize = 12)
for i in ax.patches:
    width, height = i.get_width(), i.get_height()
    x, y = i.get_xy() 
    ax.annotate(f'{round(height,2)}%', (x + width/2, y + height*1.02), ha='center', fontsize = 'x-large')
plt.show()

from sklearn.metrics import classification_report

print('Logistic Regression:')
y_pred_log_reg = log_reg.predict(X_test)
print(classification_report(y_test, y_pred_log_reg))

print('Support Vector Classifier:')
y_pred_svc = svc.predict(X_test)
print(classification_report(y_test, y_pred_svc))

print('Naive Bayes Classifier:')
y_pred_naive = naive_classifier.predict(X_test)
print(classification_report(y_test, y_pred_naive))

print('KNears Neighbors:')
y_pred_knear = knears_neighbors.predict(X_test)
print(classification_report(y_test, y_pred_knear))

print('Decision Tree Classifier:')
y_pred_tree = tree_clf.predict(X_test)
print(classification_report(y_test, y_pred_tree))

print('Random Forest Classifier:')
y_pred_forest = random_forest.predict(X_test)
print(classification_report(y_test, y_pred_forest))

from sklearn.model_selection import ShuffleSplit, learning_curve

def plot_learning_curve(estimator1, estimator2, estimator3, estimator4, estimator5, estimator6, X, y, ylim=None, cv=None, 
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
  f, ((ax1, ax2), (ax3, ax4), (ax5, ax6)) = plt.subplots(3, 2, figsize=(20, 20),  sharey=True)
  if ylim is not None:
    plt.ylim(*ylim)
  
  # First Estimator
  train_sizes, train_scores, test_scores =  learning_curve(estimator1, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)
  ax1.fill_between(
      train_sizes, 
      train_scores_mean - train_scores_std, 
      train_scores_mean + train_scores_std,
      alpha=0.1,
      color="#ff9124"
      )
  ax1.fill_between(
      train_sizes,
      test_scores_mean - test_scores_std,
      test_scores_mean + test_scores_std,
      alpha=0.1,
      color="#2492ff"
  )
  ax1.plot(
      train_sizes,
      train_scores_mean,
      'o-',
      color="#ff9124",
      label='Training Score'
      )
  ax1.plot(
      train_sizes,
      test_scores_mean,
      'o-',
      color="#2492ff",
      label='Cross-validation_score'
  )
  ax1.set_title('Logistic Regression Learning Curve', fontsize=14)
  ax1.set_xlabel('Train size(m)')
  ax1.set_ylabel('Score')
  ax1.grid(True)
  ax1.legend(loc='best')

  # Second Estimator
  train_sizes, train_scores, test_scores =  learning_curve(estimator2, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)
  ax2.fill_between(
      train_sizes, 
      train_scores_mean - train_scores_std, 
      train_scores_mean + train_scores_std,
      alpha=0.1,
      color="#ff9124"
      )
  ax2.fill_between(
      train_sizes,
      test_scores_mean - test_scores_std,
      test_scores_mean + test_scores_std,
      alpha=0.1,
      color="#2492ff"
  )
  ax2.plot(
      train_sizes,
      train_scores_mean,
      'o-',
      color="#ff9124",
      label='Training Score'
      )
  ax2.plot(
      train_sizes,
      test_scores_mean,
      'o-',
      color="#2492ff",
      label='Cross-validation_score'
  )
  ax2.set_title('K-nearest Neighbor Learning Curve', fontsize=14)
  ax2.set_xlabel('Train size(m)')
  ax2.set_ylabel('Score')
  ax2.grid(True)
  ax2.legend(loc='best')

  # Third Estimator
  train_sizes, train_scores, test_scores =  learning_curve(estimator3, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)
  ax3.fill_between(
      train_sizes, 
      train_scores_mean - train_scores_std, 
      train_scores_mean + train_scores_std,
      alpha=0.1,
      color="#ff9124"
      )
  ax3.fill_between(
      train_sizes,
      test_scores_mean - test_scores_std,
      test_scores_mean + test_scores_std,
      alpha=0.1,
      color="#2492ff"
  )
  ax3.plot(
      train_sizes,
      train_scores_mean,
      'o-',
      color="#ff9124",
      label='Training Score'
      )
  ax3.plot(
      train_sizes,
      test_scores_mean,
      'o-',
      color="#2492ff",
      label='Cross-validation_score'
  )
  ax3.set_title('SVC Learning Curve', fontsize=14)
  ax3.set_xlabel('Train size(m)')
  ax3.set_ylabel('Score')
  ax3.grid(True)
  ax3.legend(loc='best')

  # Fourth Estimator
  train_sizes, train_scores, test_scores =  learning_curve(estimator4, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)
  ax4.fill_between(
      train_sizes, 
      train_scores_mean - train_scores_std, 
      train_scores_mean + train_scores_std,
      alpha=0.1,
      color="#ff9124"
      )
  ax4.fill_between(
      train_sizes,
      test_scores_mean - test_scores_std,
      test_scores_mean + test_scores_std,
      alpha=0.1,
      color="#2492ff"
  )
  ax4.plot(
      train_sizes,
      train_scores_mean,
      'o-',
      color="#ff9124",
      label='Training Score'
      )
  ax4.plot(
      train_sizes,
      test_scores_mean,
      'o-',
      color="#2492ff",
      label='Cross-validation_score'
  )
  ax4.set_title('Decision Tree Learning Curve', fontsize=14)
  ax4.set_xlabel('Train size(m)')
  ax4.set_ylabel('Score')
  ax4.grid(True)
  ax4.legend(loc='best')

  # Fifth Estimator
  train_sizes, train_scores, test_scores =  learning_curve(estimator5, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)
  ax5.fill_between(
      train_sizes, 
      train_scores_mean - train_scores_std, 
      train_scores_mean + train_scores_std,
      alpha=0.1,
      color="#ff9124"
      )
  ax5.fill_between(
      train_sizes,
      test_scores_mean - test_scores_std,
      test_scores_mean + test_scores_std,
      alpha=0.1,
      color="#2492ff"
  )
  ax5.plot(
      train_sizes,
      train_scores_mean,
      'o-',
      color="#ff9124",
      label='Training Score'
      )
  ax5.plot(
      train_sizes,
      test_scores_mean,
      'o-',
      color="#2492ff",
      label='Cross-validation_score'
  )
  ax5.set_title('Naive Bayes Learning Curve', fontsize=14)
  ax5.set_xlabel('Train size(m)')
  ax5.set_ylabel('Score')
  ax5.grid(True)
  ax5.legend(loc='best')

  # Sixth Estimator
  train_sizes, train_scores, test_scores =  learning_curve(estimator6, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
  train_scores_mean = np.mean(train_scores, axis=1)
  train_scores_std = np.std(train_scores, axis=1)
  test_scores_mean = np.mean(test_scores, axis=1)
  test_scores_std = np.std(test_scores, axis=1)
  ax6.fill_between(
      train_sizes, 
      train_scores_mean - train_scores_std, 
      train_scores_mean + train_scores_std,
      alpha=0.1,
      color="#ff9124"
      )
  ax6.fill_between(
      train_sizes,
      test_scores_mean - test_scores_std,
      test_scores_mean + test_scores_std,
      alpha=0.1,
      color="#2492ff"
  )
  ax6.plot(
      train_sizes,
      train_scores_mean,
      'o-',
      color="#ff9124",
      label='Training Score'
      )
  ax6.plot(
      train_sizes,
      test_scores_mean,
      'o-',
      color="#2492ff",
      label='Cross-validation_score'
  )
  ax6.set_title('Random Forest Learning Curve', fontsize=14)
  ax6.set_xlabel('Train size(m)')
  ax6.set_ylabel('Score')
  ax6.grid(True)
  ax6.legend(loc='best')

cv = ShuffleSplit(n_splits=100, test_size=0.2, random_state=42)
plot_learning_curve(log_reg, knears_neighbors, svc, tree_clf, naive_classifier, random_forest, X_train, y_train, (0.6, 1.01), cv=cv, n_jobs=4)

from sklearn.metrics import roc_curve, roc_auc_score

log_fpr, log_tpr, log_threshold = roc_curve(y_test, y_pred_log_reg)
knear_fpr, knear_tpr, knear_threshold = roc_curve(y_test, y_pred_knear)
svc_fpr, svc_tpr, svc_threshold = roc_curve(y_test, y_pred_svc)
tree_fpr, tree_tpr, tree_threshold = roc_curve(y_test, y_pred_tree)
naive_fpr, naive_tpr, naive_threshold = roc_curve(y_test, y_pred_naive)
random_fpr, random_tpr, random_threshold = roc_curve(y_test, y_pred_forest)

def graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr, naive_fpr, naive_tpr, random_fpr,
                             random_tpr):
  plt.figure(figsize=(16, 8))
  plt.title('ROC Curve \n Top 6 Classifiers', fontsize=18)
  plt.plot(log_fpr, log_tpr, label='Logistic Regression Classifier Score: {:.4f}'.format(roc_auc_score(y_test, y_pred_log_reg)))
  plt.plot(knear_fpr, knear_tpr, label='KNears Neighbors Classifier Score: {:.4f}'.format(roc_auc_score(y_test, y_pred_knear)))
  plt.plot(svc_fpr, svc_tpr, label='Support Vector Classifier Score: {:.4f}'.format(roc_auc_score(y_test, y_pred_svc)))
  plt.plot(tree_fpr, tree_tpr, label='Decision Tree Classifier Score: {:.4f}'.format(roc_auc_score(y_test, y_pred_tree)))
  plt.plot(naive_fpr, naive_tpr, label='Naive Bayes Classifier Score: {:.4f}'.format(roc_auc_score(y_test, y_pred_naive)))
  plt.plot(random_fpr, random_tpr, label='Random Forest Classifier Score: {:.4f}'.format(roc_auc_score(y_test, y_pred_forest)))
  plt.plot([0, 1], [0, 1], 'k--')
  plt.axis([-0.01, 1, 0, 1])
  plt.xlabel('False Positive Rate', fontsize=16)
  plt.ylabel('True Positive Rate', fontsize=16)
  plt.annotate('Minimum ROC Score of 50% \n (This is the minimum score to get)', xy=(0.5, 0.5), xytext=(0.6, 0.3),
              arrowprops=dict(facecolor='#6E726D', shrink=0.05),
              )
  plt.legend()

graph_roc_curve_multiple(log_fpr, log_tpr, knear_fpr, knear_tpr, svc_fpr, svc_tpr, tree_fpr, tree_tpr, naive_fpr, naive_tpr, random_fpr,
                             random_tpr)
plt.show()

