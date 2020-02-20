#coding out knn
def predict_price(new_listing):
    temp_df = dc_listings.copy()
    temp_df['distance'] = temp_df['accommodates'].apply(lambda x: np.abs(x - new_listing))
    temp_df = temp_df.sort_values('distance')
    nearest_neighbors = temp_df.price.iloc[0:5]
    predicted_price = nearest_neighbors.mean()
    return(predicted_price)

#writing out KFolds
def train_and_validate(df, folds):
    fold_rmses = []
    for fold in range(1,folds+1):
        # Train
        model = KNeighborsRegressor()
        train = df[df["fold"] != fold]
        test = df[df["fold"] == fold].copy()
        model.fit(train[["accommodates"]], train["price"])
        # Predict
        labels = model.predict(test[["accommodates"]])
        test["predicted_price"] = labels
        mse = mean_squared_error(test["price"], test["predicted_price"])
        rmse = mse**(1/2)
        fold_rmses.append(rmse)
    return(fold_rmses)

# coding rmse by hand
rmse = np.sqrt(np.mean(np.square(prediction - y_valid)))

#Scree Plot
per_var = np.round(pca.explained_variance_ratio_*100, decimals=1)
labels = ['PC' + str(x) for x in range(1, len(per_var)+1)]

plt.bar(x=range(1, len(per_var)+1), height = per_var, tick_label=labels)
plt.ylabel('Percent Variance Explained')
plt.xlabel('Principal Component')
plt.title('"V" column Scree Plot')

## performing random permutation
shuffled_index = np.random.permutation(df.index)
    rand_df = df.reindex(shuffled_index)

dc_listings = dc_listings.loc[np.random.permutation(len(dc_listings))]

def split_train_test(data, test_ratio):
	shuffled_indiced = np.random.permutation(len(data))
	test_set_size = int(len(data)*test_ratio)
	test_indices = shuffled_indiced[:test_set_size]
	train_indices = shuffled_indiced[test_set_size:]
	return data.iloc[train_indices], data.iloc[test_indices]

train_set, test_set = split_train_test(housing,0.2)

#statified shuffle split - important to have a sufficient number of instances in your dataset for each stratum,
#or else the estimate of your stratum's importance will be biased
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1,test_size=.2, random_state=42)
for train_index, test_index in split.split(housing,housing['income_cat']):
	strat_train_set = housing.loc[train_index]
	strat_test_set = housing.loc[test_index]

strat_test_set['income_cat'].value_counts/len(strat_test_set)


##full pipeline
##Transformer class to combine specific attributes
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, household_ix = 3,4,5,6

class CombinedAttribAdder(BaseEstimator,TransformerMixin):
	def __init__(self, add_bedrooms_per_room=True):
		self.add_bedrooms_per_room = add_bedrooms_per_room
	def fit(self,X,y=None):
		return self
	def transform(self,X,y=None):
		rooms_per_household = X[:,rooms_ix]/ X[:,household_ix]
		populations_per_household = X[:population_ix]/ X[:,household_ix]
		if self.add_bedrooms_per_room:
			bedrooms_per_room = X[:, bedrooms_ix]/X[:,rooms_ix]
			return np.c_[X,rooms_per_household,populations_per_household,bedrooms_per_room]

		else:
			return np.c_[X,rooms_per_household,populations_per_household]

attr_adder = CombinedAttribAdder(add_bedrooms_per_room=False)
housing_extra_attributes = attr_adder.transform(housing.values)

from sklearn.pipeline import pipeline
from sklearn import StandardScaler

num_pipeline = Pipeline([
	('imputer', SimpleImputer(strategy='median')),
	('attribs_adder', CombinedAttribAdder()),
	('std_scalar', StandardScaler()),
	])

housing_num = num_pipeline.fit_transform(housing_num)

from sklearn import ColumnTransformer

full_pipeline = ColumnTransformer([
	('num',num_pipeline,num_attribs),
	('cat',OneHotEncoder,cat_attribs),
	])

housing_prepared = full_pipeline.fit_transform(housing)

###encoding one column and returning a dataframe from it
train_enc = OneHotEncoder.fit_transform(np.asarray(train['primary_use']).reshape(1,-1))

import scipy.sparse
train_cat = pd.DataFrame(train_enc.toarray(),columns = OneHotEncoder.get_feature_names())

train_cat.head()

##model-prediction basics
import numpy as np
from sklearn.metrics import mean_squared_error

lr = LinearRegression()
lr.fit(train[['Gr Liv Area']], train['SalePrice'])

train_predictions = lr.predict(train[['Gr Liv Area']])
test_predictions = lr.predict(test[['Gr Liv Area']])

train['prediction'] = train_predictions
test['prediction'] = test_predictions

##derivative and gradient descent
def derivative(a1, xi_list, yi_list):
    len_data = len(xi_list)
    error = 0
    for i in range(0, len_data):
        error += xi_list[i]*(a1*xi_list[i] - yi_list[i])
    deriv = 2*error/len_data
    return deriv

def gradient_descent(xi_list, yi_list, max_iterations, alpha, a1_initial):
    a1_list = [a1_initial]

    for i in range(0, max_iterations):
        a1 = a1_list[i]
        deriv = derivative(a1, xi_list, yi_list)
        a1_new = a1 - alpha*deriv
        a1_list.append(a1_new)
    return(a1_list)

param_iterations = gradient_descent(train['Gr Liv Area'], train['SalePrice'], 20, .0000003, 150)
final_param = param_iterations[-1]

##Dummifying columns syntax
for col in text_cols:
    col_dummies = pd.get_dummies(train[col])
    train = pd.concat([train, col_dummies], axis=1)
    del train[col]

##Finding accuracy by hand
admissions['matches'] = (admissions['predicted_label']==admissions['actual_label'])

correct_predictions = admissions.loc[admissions['matches']==True]

correct_predictions.head()

accuracy = len(correct_predictions)/len(admissions)

#filtering rows and finding True P rate and True N rate
true_positive_filter = (admissions["predicted_label"] == 1) & (admissions["actual_label"] == 1)
true_positives = len(admissions[true_positive_filter])

true_negative_filter = (admissions["predicted_label"] == 0) & (admissions["actual_label"] == 0)
true_negatives = len(admissions[true_negative_filter])

##Sensitivity
false_positive_filter = (admissions["predicted_label"] == 1) & (admissions["actual_label"] == 0)
false_positives = len(admissions[false_positive_filter])

false_negative_filter = (admissions["predicted_label"] == 0) & (admissions["actual_label"] == 1)
false_negatives = len(admissions[false_negative_filter])

sensitivity = true_positives/(true_positives+false_negatives)

#dummifying with prefixes
dummy_cylinders = pd.get_dummies(cars["cylinders"], prefix="cyl")
cars = pd.concat([cars, dummy_cylinders], axis=1)

#random shuffling rows
shuffled_rows = np.random.permutation(cars.index) ## can also do len(cars) instead of cars.index
shuffled_cars = cars.iloc[shuffled_rows]

##train-test split based on percentage
highest_train_row = int(cars.shape[0] * .70)
train = shuffled_cars.iloc[0:highest_train_row]
test = shuffled_cars.iloc[highest_train_row:]

#training a logistic regression model for all unique values for a target
#all one vs. all
unique_origins = cars["origin"].unique()
unique_origins.sort()

models = {}

features = [c for c in train.columns if c.startswith("cyl") or c.startswith("year")]

for origin in unique_origins:
    model = LogisticRegression()
    
    X_train = train[features]
    y_train = train["origin"] == origin

    model.fit(X_train, y_train)
    models[origin] = model
    
   #returning the proabilities into a dataframe
   testing_probs = pd.DataFrame(columns=unique_origins)

for origin in unique_origins:
    # Select testing features.
    X_test = test[features]   
    # Compute probability of observation being in the origin.
    testing_probs[origin] = models[origin].predict_proba(X_test)[:,1]

    #classifying each observation from the highest probability of all from predict_proba using idxmax
    predicted_origins = testing_probs.idxmax(axis=1)
    #here's what is returned
    'pandas.core.series.Series'

0	1
1	1
2	1
3	1
4	1
5	1
6	3
7	1
8	1
9	3
10	1
11	3
12	1

##retrieving the mse and variance for each model
def train_and_test(cols):
    lm = LinearRegression()
    y_train = filtered_cars['mpg']
    
    lm.fit(filtered_cars[cols],y_train)
    predictions = lm.predict(filtered_cars[cols])
    
    mse = mean_squared_error(filtered_cars['mpg'],predictions)
    variance = np.var(predictions)
    return(mse,variance)

cyl_mse, cyl_var = train_and_test(['cylinders'])

weight_mse, weight_var = train_and_test(['weight'])

##Computing euclidean distance
from sklearn.metrics.pairwise import euclidean_distances

distance = euclidean_distances(votes.iloc[0,3:].values.reshape(1,-1), votes.iloc[2,3:].values.reshape(1,-1))

####BECAUSE YOU'RE NOT PREDICTING ANYTHING IN CLUSTERING, THERE IS NO ISSUE OF OVERFITTING

#Using pd.crosstab() to compare how many of each catgory are in each cluster
labels = kmeans_model.labels_
print(pd.crosstab(labels,votes['party']))

party   D  I   R
row_0           
0      41  2   0
1       3  0  54

#Selecting the outliers
democratic_outliers = votes[(labels == 1) & (votes["party"] == "D")]

#cubing the distance to essentuate extreme values of each cluster
extremism = (senator_distances ** 3).sum(axis=1)

##While regression and other supervised machine learning techniques work well when we have a clear metric
# we want to optimize for and lots of pre-labelled data, we need to instead use unsupervised machine 
#learning techniques to explore the structure within a data set that doesn't have a clear value to optimize.

##getting centroid coordinate in dict form. One coordinate is centroid for nba[ppg], the other for nba[atr]
def centroids_to_dict(centroids):
    dictionary = dict()
    # iterating counter we use to generate a cluster_id
    counter = 0

    # iterate a pandas data frame row-wise using .iterrows()
    for index, row in centroids.iterrows():
        coordinates = [row['ppg'], row['atr']]
        dictionary[counter] = coordinates
        counter += 1

    return dictionary

centroids_dict = centroids_to_dict(centroids)

##calculating euclidean distance
def calculate_distance(vec1, vec2):
    root_distance = 0
    for x in range(0, len(vec1)):
        difference = centroid[x] - player_values[x]
        squared_difference = difference**2
        root_distance += squared_difference
euclid_distance = math.sqrt(root_distance)
return euclid_distance

#converting a list of categoricals to numerical
cols = ['education','marital_status','occupation','relationship','race',
        'sex','native_country','high_income']
for name in income[cols]:
    col = pandas.Categorical(income[name])
    income[name] = col.codes

#BIAS-VARIANCE TRADE-OFF

# When scores drop it's due to underfitting
#Underfitting is what occurs when our model is too simple to explain the relationships between the variables.


#This is known as the bias-variance tradeoff. Imagine that we take a random sample of training data
# and create many models. If the models' predictions for the same row are far apart from each other,
# we have high variance. Imagine this time that we take a random sample of the training data and create
# many models. If the models' predictions for the same row are close together but far from the actual value,
# then we have high bias.

#High bias can cause underfitting -- if a model is consistently failing to predict the correct value,
# it may be that it's too simple to model the data faithfully.

#High variance can cause overfitting. If a model varies its predictions significantly based on small changes
# in the input data, then it's likely fitting itself to quirks in the training data,
# rather than making a generalizable model.


def train_and_validate(df, folds):
	fold_rmses = []
	for i in range(1,folds+1):
		model = KNNRegressor()

		dc_listings.loc[dc_listings.index[0:745], "fold"] = 1
		dc_listings.loc[dc_listings.index[745:1490], "fold"] = 2
		dc_listings.loc[dc_listings.index[1490:2234], "fold"] = 3
		dc_listings.loc[dc_listings.index[2234:2978], "fold"] = 4
		dc_listings.loc[dc_listings.index[2978:3723], "fold"] = 5

		train = train.loc[train['fold']!= i]
		test = train.loc[train['fold']==i].copy()

		model.fit(train['accommodates'], train['SalePrice'])
		labels = model.predict(test['accommodates'])
		test['price'] = labels
		mse = mean_squared_error(test["price"], test["predicted_price"])
		fold_rmses.append(sqrt(mse))


def KFold(new_listing):
	temp = train.copy()
	temp['distance'] = train['accommodates'].apply(lambda x: abs(x-new_listing))
	temp = temp.sort_values('distance')
	nearest_neighbors = temp.iloc[:5]
	prediced_prices = nearest_neighbors.mean()

def random_train_test_split(df, test_ratio):
	shuffled_index = np.random_permutation(len(df))
	test_length = int(len(data)*test_ratio)
	train_indices = shuffled_index[:test_length]
	test_indices = shuffled_index[test_length:]
	return df.iloc[train_indices], df.iloc[test_indices]

##getting model error (also, look at format to return 5 decimal places)
def get_error(X_train, y_train, X_test, y_test, model, show = True):
    model.fit(X_train, y_train)
    train_error = 1 - model.score(X_train, y_train)
    test_error  = 1 - model.score(X_test, y_test)
    if show:
        print("The training error is: %.5f" %train_error)
        print("The test error is: %.5f" %test_error)
    return [train_error, test_error]

 ##return feature importances from grid_search, zip them together, and sort them
 # feature importance sort
tree_final = grid_search_tree.best_estimator_
feature_importance = list(zip(oj.columns[1:], tree_final.feature_importances_))
dtype = [('feature', 'S10'), ('importance', 'float')]
feature_importance = np.array(feature_importance, dtype = dtype)
feature_sort = np.sort(feature_importance, order='importance')[::-1]
feature_sort

##returning a confusion matrix after fitting a model
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

model = RandomForestClassifier(random_state= 101).fit(X_Train,Y_Train)
predictionforest = model.predict(X_Test)
print(confusion_matrix(Y_Test,predictionforest))
print(classification_report(Y_Test,predictionforest))
acc1 = accuracy_score(Y_Test,predictionforest)


#PERTINENT HYPERPERAMETERS

#GAMMA (BOOSTING - which is based on weak learners)
# controls regularization (prevents overfitting) - the higher the value, the higher the organization
# regularization penalizes coefficients (XGB default is zero)

#Tune trick: Start with 0 and check CV error rate. If you see train error >>> test error, bring gamma
#into action. Higher the gamma, lower the difference in train and test CV. If you have no clue what value 
#to use, use gamma=5 and see the performance. Remember that gamma brings improvement when you want to use 
#shallow (low max_depth) trees.

#LAMBDA 
#It controls L2 regularization (equivalent to Ridge regression) on weights. It is used to avoid overfitting.

#ALPHA
#It controls L1 regularization (equivalent to Lasso regression) on weights. In addition to shrinkage, 
#enabling alpha also results in feature selection. Hence, it’s more useful on high dimensional data sets.


##FULL PROCESS OF TRAINING MODEL, GETTING ERRORS, GRID-SEARCHING, AND PLOTTING
# 1) get train test split
X_train, X_test, y_train, y_test = train_test_split(oj.drop('Purchase',1),oj['Purchase'], random_state=0, test_size=0.2)
# 2) function to return model errors
def get_error(X_train, y_train, X_test, y_test, model, show = True):
    model.fit(X_train, y_train)
    train_error = 1 - model.score(X_train, y_train)
    test_error  = 1 - model.score(X_test, y_test)
    if show:
        print("The training error is: %.5f" %train_error)
        print("The test error is: %.5f" %test_error)
    return [train_error, test_error]
#3)fit model and return errors
from sklearn.ensemble import RandomForestClassifier
randomForest = RandomForestClassifier()
randomForest.set_params(random_state=0)

randomForest.fit(X_train, y_train) 

print("The training error is: %.5f" % (1 - randomForest.score(X_train,y_train)))
print("The test     error is: %.5f" % (1 - randomForest.score(X_test, y_test)))
#4)parameters for grid search
grid_para_forest = {
    'criterion': ['gini', 'entropy'],
    'max_depth': range(1, 31),
    'n_estimators': range(10, 110, 10)
}
#5) grid search
grid_search_forest = ms.GridSearchCV(randomForest, grid_para_forest, scoring='accuracy', cv=5, n_jobs=-1)
%time grid_search_forest.fit(X_train, y_train)
#6)best parameters and score and errors from these parameters
grid_search_forest.best_perameters_
grid_search_forest.best_score_
	##to grab coef after fitting 
		lm.coef_

print("The training error is: %.5f" % (1 - grid_search_forest.best_estimator_.score(X_train, y_train)))
print("The test error is: %.5f" % (1 - grid_search_forest.best_estimator_.score(X_test, y_test)))
#7) get feat importances
features = list(zip(oj.columns, randomForest.feature_importances_))
dtype = [('feature', 'S10'), ('importance', 'float')]
features = np.asarray(features, dtype = dtype)
sorted_feat = np.sort(features, order='importance')
#8)plot feature importance on hbar
feat_names, feat_imps = zip(*list(sorted_feat))
plt.barh(range(len(feat_imps)), feat_imps, tick_label = feat_names)