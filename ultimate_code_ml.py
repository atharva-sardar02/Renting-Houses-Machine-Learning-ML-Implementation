from sklearn import tree
import pandas as pd
import warnings
import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics import mean_squared_error, confusion_matrix, precision_score, recall_score, f1_score, accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler, StandardScaler
import statsmodels.api as sm
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.tree import plot_tree, DecisionTreeClassifier
from statsmodels.stats.outliers_influence import variance_inflation_factor
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import cross_val_score
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.metrics import roc_curve, auc




warnings.filterwarnings("ignore")





def plot_multiclass_roc_curve(classifier, X_test, y_test):


    y_scores = classifier.predict_proba(X_test)

    plt.figure(figsize=(8, 8))

    for i in range(len(classifier.classes_)):
        fpr, tpr, _ = roc_curve(y_test == classifier.classes_[i], y_scores[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f'Class {classifier.classes_[i]} (AUC = {roc_auc:.2f})')

    plt.plot([0, 1], [0, 1], color='navy', linestyle='--', linewidth=2, label='Random', alpha=0.8)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve (Multiclass)')
    plt.legend(loc='lower right')
    plt.show()

"""Loading the Data"""

paths = "apartments_for_rent_classified_100K.csv"
df_initial_loaded = pd.read_csv(paths, encoding='ISO-8859-1', sep=';',low_memory=False)
# Selected features
selected_features = ['id','category', 'fee', 'has_photo', 'pets_allowed', 'price_type', 'bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude', 'price']

# DataFrame with selected features
df = df_initial_loaded[selected_features].iloc[np.random.randint(1, len(df_initial_loaded), 1000)]
print(df.head().to_string())
print("Length of DataFrame:",len(df))
df2=df

"""PHASE 1"""


#####
#   #
# # #
#
#      1

###################################################
######### Checking the missing values   ########
###################################################

print("Missing Observations:",df.isnull().sum())

# When we check the missing values we find that most of none values are with pets_allowed, so replaced it with "NotAllowed"
print("Replacing the Null of pets_allowed with NotAllowed:")
df['pets_allowed'].fillna('NotAllowed', inplace=True)
print("Missing Observations:",df.isnull().sum())
# Drop the null values
print("Dropping the Null values")
df.dropna(inplace=True)
print("Missing Observations:",df.isnull().sum())
###################################################
## Checking for the duplicates and removing them  ##
###################################################

duplicates = df.duplicated()
print("Duplicate Rows:")
print(df[duplicates])
df = df.drop_duplicates()
print(df)

# No Aggregation and the down sampling

###################################################
#########  Discretization & Binarization ########
###################################################

# One hot encoding for the fee feature:
onehot_encoder = OneHotEncoder(sparse=False, drop='first')
onehot_encoded_fee = onehot_encoder.fit_transform(df[['fee']])
df['fee'] = onehot_encoded_fee
print(df.head())
# Label encode 'category', 'has_photo', 'pets_allowed', 'price_type'
label_encoder = LabelEncoder()
df['category'] = label_encoder.fit_transform(df['category'])
df['has_photo'] = label_encoder.fit_transform(df['has_photo'])
df['pets_allowed'] = label_encoder.fit_transform(df['pets_allowed'])
df['price_type'] = label_encoder.fit_transform(df['price_type'])
print("\nLabel-Encoded Data:")
print(df.head())

###################################################
######### Variable Transformation  ########
###################################################

numerical_features = ['bathrooms', 'bedrooms', 'square_feet', 'latitude', 'longitude']
scaler = MinMaxScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])
scaler = StandardScaler()
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# We will not do differencing because this is not a time series data, in which case it is generally useful

###################################################
## Dimensionality reduction/feature selection #####
###################################################

X = df.drop('price', axis=1)
y = df['price']

# Random Forest
rf_model = RandomForestRegressor()
rf_model.fit(X, y)
feature_importances = pd.Series(rf_model.feature_importances_, index=X.columns)
selected_features_rf = feature_importances[feature_importances > 0.01].index.tolist()
print("Selected Features from Random Forest Analysis:")
print(selected_features_rf)

# PCA and Condition Number
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
pca = PCA()
X_pca = pca.fit_transform(X_scaled)
# Check the condition number before and after PCA
condition_number_before_pca = np.linalg.cond(X)
condition_number_after_pca = np.linalg.cond(X_pca)
feature_importances_pca = np.abs(pca.components_)
explained_variance_ratio = pca.explained_variance_ratio_
threshold_variance_ratio = 0.01  # Adjust the threshold
selected_features_pca = X.columns[explained_variance_ratio > threshold_variance_ratio]
print("Selected Features based on PCA:")
print(selected_features_pca)

# # Singular Value Decomposition (SVD) Analysis
feature_names = X.columns.tolist()
X_train, X_test = train_test_split(X, test_size=0.2, random_state=5805)
n_components = 5  # Number of components to retain
svd = TruncatedSVD(n_components=n_components)
X_train_svd = svd.fit_transform(X_train)
X_test_svd = svd.transform(X_test)
first_component = svd.components_[0]
feature_contributions = pd.DataFrame({
    'Feature': feature_names,
    'Contribution': first_component
})
feature_contributions = feature_contributions.reindex(feature_contributions['Contribution'].abs().sort_values(ascending=False).index)
print("Selected Features and Their Contributions in the SVD Analysis:")
print(feature_contributions)



# VIF for Collinearity Check
X_scaled_with_intercept = sm.add_constant(X_scaled)
vif_data = pd.DataFrame()
vif_data["Variable"] = X.columns
vif_data["VIF"] = [variance_inflation_factor(X_scaled_with_intercept, i) for i in range(1, X_scaled_with_intercept.shape[1])]
# Remove highly correlated features based on VIF
selected_features_vif = vif_data[vif_data["VIF"] < 5]["Variable"].tolist()
print("\nSelected Features from VIF:")
print(selected_features_vif)


####################################################
# We will use the features selected by Random Forest and use those features moving forward
###################################################
df = df[selected_features_rf + ['price']]
print("Feature Reduced Data:\n",df.head().to_string)


###################################################
########## Outlier detection  ########
###################################################

Q1 = df['price'].quantile(0.25)
Q3 = df['price'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
df_no_outliers = df[(df['price'] >= lower_bound) & (df['price'] <= upper_bound)]
print("Original Dataset Shape:", df.shape)
print("Dataset Shape After Outlier Removal:", df_no_outliers.shape)
df = df_no_outliers


###################################################
######### covariance matrix  ########
###################################################

covariance_matrix = np.cov(X_scaled, rowvar=False)
plt.figure(figsize=(10, 8))
sns.heatmap(covariance_matrix, annot=True, fmt=".4f", cmap="coolwarm", xticklabels=X.columns, yticklabels=X.columns)
plt.title('Sample Covariance Matrix Heatmap')
plt.show()


###################################################
# sample Pearson correlation coefficients matrix  #
###################################################

correlation_matrix = X.corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f", xticklabels=correlation_matrix.columns, yticklabels=correlation_matrix.columns)
plt.title('Sample Pearson Correlation Coefficients Matrix Heatmap')
plt.show()


# Checking whether the dataset is balanced
sns.histplot(df['price'], bins=3, kde=True)
plt.title('Distribution of Prices')
plt.xlabel('Price')
plt.show()
# It looks like the most of the data is in the middle range, but this is common on renting industry and hence will not balance the data.

"""PHASE 2"""

#####
#   #
# # #
#
#      2

X = df.drop('price', axis=1)
y = df['price']


X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(X, y, test_size=0.2, random_state=5805)


X_train_reg = sm.add_constant(X_train_reg)
X_test_reg = sm.add_constant(X_test_reg)

final_model = sm.OLS(y_train_reg, X_train_reg).fit()

# T-test analysis for each variable
t_test_results = final_model.t_test(np.eye(len(final_model.params)))
print("T-test analysis:")
print(t_test_results)

# F-test analysis
f_test_result = final_model.f_test(np.eye(len(final_model.params)))
print("\nF-test analysis:")
print(f_test_result)

# Prediction on train and test sets
y_train_pred = final_model.predict(X_train_reg)
y_test_pred = final_model.predict(X_test_reg)

# Plotting train, test, and predicted variables
plt.scatter(range(len(y_train_reg)), y_train_reg, label='Train Actual Prices')
plt.scatter(range(len(y_train_reg), len(y_train_reg) + len(y_test_reg)), y_test_reg, label='Test Actual Prices')
plt.scatter(range(len(y_train_reg)), y_train_pred, label='Train Predicted Prices')
plt.scatter(range(len(y_train_reg), len(y_train_reg) + len(y_test_reg)), y_test_pred, label='Test Predicted Prices')
plt.xlabel('Observation Index')
plt.ylabel('Price')
plt.legend()
plt.show()

# R-squared, adjusted R-square, AIC, BIC, and MSE for train set
r2_train = final_model.rsquared
adj_r2_train = final_model.rsquared_adj
aic_train = final_model.aic
bic_train = final_model.bic
mse_train = mean_squared_error(y_train_reg, y_train_pred)

# R-squared, adjusted R-square, AIC, BIC, and MSE for test set
r2_test = final_model.rsquared
adj_r2_test = final_model.rsquared_adj
aic_test = final_model.aic
bic_test = final_model.bic
mse_test = mean_squared_error(y_test_reg, y_test_pred)

plt.figure(figsize=(10, 6))
plt.scatter(y_train_reg, y_train_pred, label="Train Predicted Price", alpha=0.7)
plt.scatter(y_train_reg, y_train_reg, label="Train Price", alpha=0.7)
plt.scatter(y_test_reg, y_test_pred, label="Test Predicted Price", alpha=0.7)
plt.scatter(y_test_reg, y_test_reg, label="Test Price", alpha=0.7)
plt.xlabel("Test Data Price")
plt.ylabel("Predicted Data Price")
plt.legend()
plt.show()

print(f"\nTrain Set Metrics:")
print(f"R-squared: {r2_train}")
print(f"Adjusted R-squared: {adj_r2_train}")
print(f"AIC: {aic_train}")
print(f"BIC: {bic_train}")
print(f"MSE: {mse_train}")

print(f"\nTest Set Metrics:")
print(f"R-squared: {r2_test}")
print(f"Adjusted R-squared: {adj_r2_test}")
print(f"AIC: {aic_test}")
print(f"BIC: {bic_test}")
print(f"MSE: {mse_test}")

# Confidence interval analysis
conf_int = final_model.conf_int()
print("\nConfidence Intervals for Coefficients:")
print(conf_int)

# printing the equation
coefficients = final_model.params
linear_regression_equation = f"Price = {coefficients['const']:.2f}"

for feature, coefficient in coefficients.items():
    if feature != 'const':
        linear_regression_equation += f" + {coefficient:.2f} * {feature}"

def forward_selected(data, response):
    remaining = set(data.columns)
    remaining.remove(response)
    selected = []
    current_score, best_new_score = float('inf'), float('inf')
    while remaining and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in remaining:
            formula = "{} ~ {}".format(response, ' + '.join(selected + [candidate]))
            score = sm.formula.ols(formula, data).fit().rsquared_adj
            scores_with_candidates.append((score, candidate))
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        if current_score > best_new_score:
            remaining.remove(best_candidate)
            selected.append(best_candidate)
            current_score = best_new_score
    formula = "{} ~ {}".format(response, ' + '.join(selected))
    model = sm.formula.ols(formula, data).fit()
    return model




print("Linear Regression Equation:")
print(linear_regression_equation)


forward_selected_model = forward_selected(df, 'price')

print(forward_selected_model.summary())

# Adjusted R-square analysis
adj_r2_values = []
num_features = range(1, len(selected_features_rf) + 1)

for num in num_features:
    selected_features_adj_r2 = selected_features_rf[:num]
    formula_adj_r2 = "price ~ {}".format(' + '.join(selected_features_adj_r2))
    model_adj_r2 = sm.formula.ols(formula_adj_r2, df).fit()
    adj_r2_values.append(model_adj_r2.rsquared_adj)

plt.plot(num_features, adj_r2_values, marker='o')
plt.title('Adjusted R-square Analysis')
plt.xlabel('Number of Features')
plt.ylabel('Adjusted R-square')
plt.show()



"""PHASE 3"""

#####
#   #
# # #
#
#      3

###################################################
######## Creating Classification Feature   ########
###################################################



# df['price_category'], bin_edges = pd.qcut(df['price'], q=3, labels=[0,1,2], retbins=True) #30000 in each part
df['price_category'], bin_edges = pd.qcut(df['price'], q=3, labels=['Cheap','Affordable','Costly'], retbins=True) #30000 in each part



print(df.head())
print(len(df))
# length of each category
category_lengths = df['price_category'].value_counts()
print("\nLength of each category:")
print(category_lengths)

# Plot a bar plot for the distribution of price categories
plt.figure(figsize=(8, 6))
category_lengths.plot(kind='bar', color=['blue', 'green', 'red'], edgecolor='black')
plt.title('Distribution of Price Categories')
plt.xlabel('Price Category')
plt.ylabel('Count')
plt.show()
# Print starting and ending values of each category
for i in range(len(bin_edges) - 1):
    start_value = bin_edges[i]
    end_value = bin_edges[i + 1]
    category = df['price_category'].cat.categories[i]
    print(f"{category} category: {start_value} to {end_value}")

label_encoder = LabelEncoder()
df['price_category'] = label_encoder.fit_transform(df['price_category'])


classification_table = pd.DataFrame(columns=['Model', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'Specificity', 'Cross Validation Mean Score'])


X = df.drop(['price', 'price_category'], axis=1)
# Exclude 'price' and 'price_category' from features
y = df['price_category']

# ###################################################
# #############     Decision Tree    ################
# ###################################################





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

dt_classifier = DecisionTreeClassifier(random_state=5805)

# hyperparameters
param_grid = {
    'criterion': ['gini', 'entropy'],
    'splitter': ['best', 'random'],
    'max_depth': [None, 10, 20, 30, 40, 50],
    'min_samples_split': [2, 5, 10],
    'max_features': [None, 'sqrt', 'log2'],
    'ccp_alpha': [0.0, 0.01, 0.1, 0.2, 0.5]
}

# Grid Search with 5-fold cross-validation
grid_search = GridSearchCV(dt_classifier, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)
dt_classifier.fit(X_train, y_train)
y_pred = dt_classifier.predict(X_test)
classification_rep = classification_report(y_test, y_pred)
plt.figure(figsize=(15, 10))
plot_tree(dt_classifier, feature_names=X.columns, class_names=np.unique(y).astype(str), filled=True, rounded=True)
plt.show()
scores = cross_val_score(dt_classifier, X_train, y_train, cv=5, scoring='accuracy')


classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Decision Tree',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)
plot_multiclass_roc_curve(dt_classifier, X_test, y_test)

# best hyperparameters
best_params = grid_search.best_params_
print("Best Hyperparameters:", best_params)

print("Prepruning Decision Tree:")

best_dt_classifier = DecisionTreeClassifier(**best_params, random_state=5805)
best_dt_classifier.fit(X_train, y_train)

y_pred = best_dt_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

print("Accuracy:", accuracy)
print("\nClassification Report:\n", classification_rep)
plt.figure(figsize=(15, 10))
plot_tree(best_dt_classifier, feature_names=X.columns, class_names=np.unique(y).astype(str), filled=True, rounded=True)
plt.show()

scores = cross_val_score(best_dt_classifier, X_train, y_train, cv=5, scoring='accuracy')
classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Prepruned Decision Tree',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)
plot_multiclass_roc_curve(best_dt_classifier, X_test, y_test)

# alpha for post pruning
print("Post pruning Decision Tree")
clf = DecisionTreeClassifier(random_state=5805)
path = clf.cost_complexity_pruning_path(X_train,y_train)
alphas = np.linspace(0, 0.005, 100)
#=============================
# Grid search for best alpha
#=============================
accuracy_train, accuracy_test = [],[]
for i in alphas:
    clf_post = DecisionTreeClassifier(ccp_alpha=i)
    clf_post.fit(X_train, y_train)
    y_train_pred = clf_post.predict(X_train)
    y_test_pred = clf_post.predict(X_test)
    accuracy_train.append(accuracy_score(y_train, y_train_pred))
    accuracy_test.append(accuracy_score(y_test, y_test_pred))
fig, ax = plt.subplots()
ax.set_xlabel('alpha')
ax.set_ylabel('accuracy')
ax.set_title("Accuracy vs alpha for training and testing sets")
ax.plot(alphas, accuracy_train, marker="o", label="train",
drawstyle="steps-post")
ax.plot(alphas, accuracy_test, marker="o", label="test",
drawstyle="steps-post")
ax.legend()
plt.grid()
plt.tight_layout()
plt.show()

# ========================
# So best alpha is 0.0001
#==========================
clf_post_final = DecisionTreeClassifier(random_state=5805, ccp_alpha=0.0001)
clf_post_final.fit(X_train, y_train)
y_train_pred = clf_post_final.predict(X_train)
y_test_pred = clf_post_final.predict(X_test)
print("Post pruning Decision Tree")
print(f'Train accuracy {accuracy_score(y_train, y_train_pred):.2f}')
print(f'Test accuracy {accuracy_score(y_test, y_test_pred):.2f}')
plt.figure(figsize=(16,8))
tree.plot_tree(clf_post_final, rounded=True, filled=True)
plt.show()
y_pred = clf_post_final.predict(X_test)
scores = cross_val_score(clf_post_final, X_train, y_train, cv=5, scoring='accuracy')
classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Postpruning Decision Tree',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)

plot_multiclass_roc_curve(clf_post_final, X_test, y_test)

###################################################
#############   Logistic Regression   ###############
###################################################


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)
Multiclassmodel = LogisticRegression(multi_class='ovr',random_state=5805)
Multiclassmodel.fit(X_train, y_train)
y_pred = Multiclassmodel.predict(X_test)
probs_y = Multiclassmodel.predict_proba(X_test)
cnf_matrix = metrics.confusion_matrix(y_test, y_pred)
print(cnf_matrix)
sns.heatmap(pd.DataFrame(cnf_matrix), annot=True, cmap="YlGnBu" ,fmt='g',
xticklabels=np.unique(y_test),
yticklabels=np.unique(y_test))
plt.title('Logistic Regression Confusion matrix', y=1.1)
plt.ylabel('Actual label')
plt.xlabel('Predicted label')
plt.tight_layout()
plt.show()
plt.show()
#===================================
# Calculating micro and macro precision
#========================================
# ==================
# Precision
#================
macro_averaged_precision = metrics.precision_score(y_test, y_pred,
average='macro')
micro_averaged_precision = metrics.precision_score(y_test
                                                   , y_pred,
average='micro')
print(f"Logistic Regression Macro-Averaged Precision score using sklearn "
f"library : {macro_averaged_precision:.2f}")
print(f"Logistic Regression Micro-Averaged Precision score using sklearn "
f"library : {micro_averaged_precision:.2f}")
# ==================
# Recall
#================
macro_averaged_recall = metrics.recall_score(y_test, y_pred,
average = 'macro')
micro_averaged_recall = metrics.recall_score(y_test, y_pred,
average = 'micro')
print(f"Logistic Regression Macro-averaged recall score using "
f"sklearn : {macro_averaged_recall:.2f}")
print(f"Logistic Regression Micro-averaged recall score using "
f"sklearn : {micro_averaged_recall:.2f}")
# ==================
# f1 score
#================
macro_averaged_f1 = metrics.f1_score(y_test, y_pred, average = 'macro')
print(f"Logistic Regression Macro-Averaged F1 score using sklearn library"
f" : {macro_averaged_f1:.2f}")
micro_averaged_f1 = metrics.f1_score(y_test, y_pred, average = 'micro')
print(f"Logistic Regression Micro-Averaged F1 score using sklearn library "
f": {micro_averaged_f1:.2f}")

scores = cross_val_score(Multiclassmodel, X_train, y_train, cv=5, scoring='accuracy')
classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Logistic Regression',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)
plot_multiclass_roc_curve(Multiclassmodel, X_test, y_test)

###################################################
#############   KNN   ###############
###################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)
error_rate = []
for i in range(1,100):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train,y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
plt.figure(figsize=(10,6))
plt.plot(range(1,100),error_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

# Find the optimum k
optimal_k = error_rate.index(min(error_rate)) + 1
print(f"Optimal K: {optimal_k}")

final_knn_classifier = KNeighborsClassifier(n_neighbors=optimal_k)
final_knn_classifier.fit(X_train, y_train)

y_pred = final_knn_classifier.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy with Optimal K: {accuracy:.2f}")

scores = cross_val_score(final_knn_classifier, X_train, y_train, cv=5, scoring='accuracy')
classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'KNN',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)
plot_multiclass_roc_curve(final_knn_classifier, X_test, y_test)

###################################################
#############   SVM  ##############################
###################################################





X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)

svc = SVC(probability=True,random_state=5805)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)

print('Performance of the baseline svm on the test set:')
print('Accuracy of the model:', accuracy_score(y_test, y_pred).round(2))
print('Confusion Matrix:\n', confusion_matrix(y_test, y_pred))
print('Precision Score:', precision_score(y_test, y_pred, average='weighted').round(2))
print('Recall Score:', recall_score(y_test, y_pred, average='weighted').round(2))
print('Specificity Score:', recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2))
print('F1 Score:', f1_score(y_test, y_pred, average='weighted').round(2))



scores = cross_val_score(svc, X_train, y_train, cv=5, scoring='accuracy')
classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'SVC',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)

plot_multiclass_roc_curve(svc, X_test, y_test)

svc = SVC(probability=True,random_state=5805)

parameters = [{'kernel': ['rbf', 'poly'],
                    'gamma': [0.1, 1],
                    'C': [0.1, 1]}]

clf = GridSearchCV(svc, parameters, cv=5, scoring='accuracy', n_jobs=-1, verbose=False)
clf.fit(X_train, y_train)

print("Best Estimator: \n", clf.best_estimator_)
print("Best Score: \n", clf.best_score_)
print("Best Paramters: \n", clf.best_params_)

best_svc = clf.best_estimator_
best_svc.fit(X_train, y_train)

y_pred = best_svc.predict(X_test)




scores = cross_val_score(best_svc, X_train, y_train, cv=5, scoring='accuracy')
classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Best SVC',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)
plot_multiclass_roc_curve(best_svc, X_test, y_test)


###################################################
#############   Naive Bayes  ######################
###################################################



nb_model = GaussianNB()

nb_model.fit(X_train, y_train)



y_pred = nb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

scores = cross_val_score(nb_model, X_train, y_train, cv=5, scoring='accuracy')
classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Naive Bayes',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)
plot_multiclass_roc_curve(nb_model, X_test, y_test)

###################################################
#############  Random Forest  #####################
###################################################


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)


rf_model = RandomForestClassifier(n_estimators=100, random_state=5805)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("Random Forest Classifier:")
print(f'Accuracy: {accuracy_score(y_test, y_pred_rf):.2f}')
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))
print("\n")

scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='accuracy')
classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Random Forest',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)
plot_multiclass_roc_curve(rf_model, X_test, y_test)

label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.transform(y_test)

# Stacking
# Create base classifiers
classifier1 = RandomForestClassifier(n_estimators=50, random_state=5805)
classifier2 = AdaBoostClassifier(n_estimators=50, random_state=5805)

meta_classifier = LogisticRegression()

# Create a stacking classifier
stacking_model = StackingClassifier(classifiers=[classifier1, classifier2],
                                    meta_classifier=meta_classifier)

stacking_model.fit(X_train, y_train)

y_pred_stacking_encoded = stacking_model.predict(X_test)

# Decoding predicted labels back to original categorical ones
y_pred = label_encoder.inverse_transform(y_pred_stacking_encoded)

# Evaluate Stacking
print("Stacking Classifier:")
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
print("\n")

scores = cross_val_score(stacking_model, X_train, y_train, cv=5, scoring='accuracy')
classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Stacking Random Forest',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)

# Boosting (AdaBoost)
adaboost_model = AdaBoostClassifier(n_estimators=50, random_state=5805)
adaboost_model.fit(X_train, y_train)
y_pred = adaboost_model.predict(X_test)

# Evaluate AdaBoost
print("AdaBoost Classifier:")
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

scores = cross_val_score(adaboost_model, X_train, y_train, cv=5, scoring='accuracy')
classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'Adaboost Model',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)
plot_multiclass_roc_curve(adaboost_model, X_test, y_test)

###################################################
#############  Neural Network  ####################
###################################################

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5805)


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


mlp_model = MLPClassifier(hidden_layer_sizes=(100, 50), max_iter=500, random_state=5805)
mlp_model.fit(X_train_scaled, y_train)


y_pred = mlp_model.predict(X_test_scaled)


print("Multi-layered Perceptron (MLP) Classifier:")
print(f'Accuracy: {accuracy_score(y_test, y_pred):.2f}')
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

scores = cross_val_score(mlp_model, X_train, y_train, cv=5, scoring='accuracy')
classification_table = pd.concat([classification_table, pd.DataFrame({'Model': 'MLP',
                                'Accuracy': accuracy_score(y_test, y_pred).round(2),
                                'Precision': precision_score(y_test, y_pred, average='weighted').round(2),
                                'Recall': recall_score(y_test, y_pred, average='weighted').round(2),
                                'F1 Score': f1_score(y_test, y_pred, average='weighted').round(2),
                                'Specificity': recall_score(y_test, y_pred, pos_label=0, average='weighted').round(2),
                                'Cross Validation Mean Score': scores.mean().round(2)}, index=[0])], axis=0)
plot_multiclass_roc_curve(mlp_model, X_test, y_test)

print("Classification Table:\n",classification_table)

"""PHASE 4"""

#####
#   #
# # #
#
#      4

# Silhouette analysis
X = df.drop(['price','price_category'], axis=1)




silhouette_scores = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=5805)
    kmeans.fit(X)
    labels = kmeans.labels_
    silhouette_avg = silhouette_score(X, labels)
    silhouette_scores.append(silhouette_avg)

# Plotting Silhouette Scores
plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), silhouette_scores, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis for KMeans Clustering')
plt.show()



#Elbow Method


inertia_values = []

for k in range(2, 11):
    kmeans = KMeans(n_clusters=k, random_state=5805)
    kmeans.fit(X)
    inertia_values.append(kmeans.inertia_)

plt.figure(figsize=(8, 6))
plt.plot(range(2, 11), inertia_values, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Inertia')
plt.title('Elbow Method for Optimal K')
plt.show()


# So finally k=6
k = 6

# Apply KMeans++ initialization
kmeans_model = KMeans(n_clusters=k, init='k-means++', random_state=5805)
kmeans_labels = kmeans_model.fit_predict(X)

# Use PCA for dimensionality reduction to 2D
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)


plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, cmap='viridis', edgecolor='k')
plt.scatter(pca.transform(kmeans_model.cluster_centers_)[:, 0], pca.transform(kmeans_model.cluster_centers_)[:, 1], s=200, marker='X', c='red')
plt.title('K-Means Clustering with KMeans++ Initialization (k=6)')
plt.show()

# Access cluster centers and labels
cluster_centers = kmeans_model.cluster_centers_
labels = kmeans_model.labels_

############################################
######### Appriori Algorithm ###############
############################################




transactions_df = df[['bedrooms', 'price_category']]

# dataFrame to a transactional format
transactions = transactions_df.groupby('bedrooms')['price_category'].apply(list).reset_index(name='cat')

# binary dataset
binary_df = pd.get_dummies(transactions['cat'].apply(pd.Series).stack(), prefix='', prefix_sep='').groupby(level=0).max()


min_support = 0.1
frequent_itemsets = apriori(binary_df, min_support=min_support, use_colnames=True)

rules = association_rules(frequent_itemsets, metric='confidence', min_threshold=0.7)

print("Frequent Itemsets:")
print(frequent_itemsets)

print("\nAssociation Rules:")
print(rules)


