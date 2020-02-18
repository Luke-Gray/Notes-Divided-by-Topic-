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

#computing entropy
import math

a = len(income.loc[income['high_income']==1])/len(income)#probability of high income
b = len(income.loc[income['high_income']!=1])/len(income)#probability of not high income

income_entropy = -(a*math.log(a,2)+b*math.log(b,2))

#We're computing information gain (IG) for a given target variable (T), as well as a given variable
# we want to split on (A). To compute it, we first calculate the entropy for T. Then, for each unique value 'v' 
# in the variable (A), we compute the number of rows in which (A) takes on the value 'v', and divide it by
# the total number of rows. Next, we multiply the results by the entropy of the rows where (A) is 'v'.
# We add all of these subset entropies together, then subtract from the overall entropy to get information gain.
#The higher the result is, the more we've lowered entropy.

#eg when doing the calculation above for income['age'], We end up with 0.17, 
#which means that we gain 0.17 bits of information by splitting our data set on the age variable.

###DECISION TREES 
#Data scientists commonly use a metric called entropy for this purpose. 
#Entropy refers to disorder. The more "mixed together" 1s and 0s are,
# the higher the entropy. A data set consisting entirely of 1s in the
# high_income column would have low entropy.
import math

a = len(income.loc[income['high_income']==1])/len(income)
b = len(income.loc[income['high_income']!=1])/len(income)

income_entropy = -(a*math.log(a,2)+b*math.log(b,2))

#### calulating information gain to see if need of another split
import numpy

def calc_entropy(column):
    """
    Calculate entropy given a pandas series, list, or numpy array.
    """
    # Compute the counts of each unique value in the column
    counts = numpy.bincount(column)
    # Divide by the total column length to get a probability
    probabilities = counts / len(column)
    
    # Initialize the entropy to 0
    entropy = 0
    # Loop through the probabilities, and add each one to the total entropy
    for prob in probabilities:
        if prob > 0:
            entropy += prob * math.log(prob, 2)
    
    return -entropy

# Verify that our function matches our answer from earlier
entropy = calc_entropy([1,1,0,0,1])
print(entropy)

information_gain = entropy - ((.8 * calc_entropy([1,1,0,0])) + (.2 * calc_entropy([1])))
print(information_gain)

income_entropy = calc_entropy(income["high_income"])

median_age = income["age"].median()

left_split = income[income["age"] <= median_age]
right_split = income[income["age"] > median_age]

age_information_gain = income_entropy - ((left_split.shape[0] / income.shape[0])
 * calc_entropy(left_split["high_income"]) + ((right_split.shape[0] / income.shape[0])
  * calc_entropy(right_split["high_income"])))


### information_gain function and appliance to income dataset and finding max information gain
def calc_information_gain(data, split_name, target_name):
    """
    Calculate information gain given a data set, column to split on, and target
    """
    # Calculate the original entropy
    original_entropy = calc_entropy(data[target_name])
    
    # Find the median of the column we're splitting
    column = data[split_name]
    median = column.median()
    
    # Make two subsets of the data, based on the median
    left_split = data[column <= median]
    right_split = data[column > median]
    
    # Loop through the splits and calculate the subset entropies
    to_subtract = 0
    for subset in [left_split, right_split]:
        prob = (subset.shape[0] / data.shape[0]) 
        to_subtract += prob * calc_entropy(subset[target_name])
    
    # Return information gain
    return original_entropy - to_subtract

# Verify that our answer is the same as on the last screen
print(calc_information_gain(income, "age", "high_income"))

columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]

information_gains = []
# Loop through and compute information gains
for col in columns:
    information_gain = calc_information_gain(income, col, "high_income")
    information_gains.append(information_gain)

# Find the name of the column with the highest gain
highest_gain_index = information_gains.index(max(information_gains))
highest_gain = columns[highest_gain_index]

###All of the steps for building a decision tree
def id3(data, target, columns)
    1 Create a node for the tree
    2 If all values of the target attribute are 1, Return the node, with label = 1
    3 If all values of the target attribute are 0, Return the node, with label = 0
    4 Using information gain, find A, the column that splits the data best
    5 Find the median value in column A
    6 Split column A into values below or equal to the median (0), and values above the median (1)
    7 For each possible value (0 or 1), vi, of A,
    8    Add a new tree branch below Root that corresponds to rows of data where A = vi
    9    Let Examples(vi) be the subset of examples that have the value vi for A
   10    Below this new branch add the subtree id3(data[A==vi], target, columns)
   11 Return Root

###function to find the best column to split on
def find_best_column(data, target_name, columns):
    information_gains = []
    # Loop through and compute information gains
    for col in columns:
        information_gain = calc_information_gain(data, col, "high_income")
        information_gains.append(information_gain)

    # Find the name of the column with the highest gain
    highest_gain_index = information_gains.index(max(information_gains))
    highest_gain = columns[highest_gain_index]
    return highest_gain


# A list of columns to potentially split income with
columns = ["age", "workclass", "education_num", "marital_status", "occupation", "relationship", "race", "sex", "hours_per_week", "native_country"]

income_split = find_best_column(income, 'high_income', columns) 


# Gini impurity is a penalty measure derived from the variance of a bonomial distribution.
#It captures the diversity of the label class distribution

# Information entropy is derived from information theory. Lower entropy means higher information gain at a given split.
# Information gained is defined as expected amount of information that would be needed to specify the class of
# a new observation, given that the example observation has reached that node.

#entire id3 method for splitting nodes with dictionary 

# Create a dictionary to hold the tree  
# It has to be outside of the function so we can access it later
tree = {}

# This list will let us number the nodes  
# It has to be a list so we can access it inside the function
nodes = []

def id3(data, target, columns, tree):
    unique_targets = pandas.unique(data[target])
    nodes.append(len(nodes) + 1)
    tree["number"] = nodes[-1]

    if len(unique_targets) == 1:
        if 0 in unique_targets:
            tree["label"] = 0
        elif 1 in unique_targets:
            tree["label"] = 1
        return
    
    best_column = find_best_column(data, target, columns)
    column_median = data[best_column].median()
    
    tree["column"] = best_column
    tree["median"] = column_median
    
    left_split = data[data[best_column] <= column_median]
    right_split = data[data[best_column] > column_median]
    split_dict = [["left", left_split], ["right", right_split]]
    
    for name, split in split_dict:
        tree[name] = {}
        id3(split, target, columns, tree[name])


id3(data, "high_income", ["age", "marital_status"], tree)

#### formatting your decision tree dictionary
def print_with_depth(string, depth):
    # Add space before a string
    prefix = "    " * depth
    # Print a string, and indent it appropriately
    print("{0}{1}".format(prefix, string))
    
    
def print_node(tree, depth):
    # Check for the presence of "label" in the tree
    if "label" in tree:
        # If found, then this is a leaf, so print it and return
        print_with_depth("Leaf: Label {0}".format(tree["label"]), depth)
        # This is critical -- without it, you'll get infinite recursion
        return
    # Print information about what the node is splitting on
    print_with_depth("{0} > {1}".format(tree["column"], tree["median"]), depth)
    
    # Create a list of tree branches
    branches = [tree["left"], tree["right"]]
        
    for branch in branches:
        print_node(branch, depth+1)

print_node(tree, 0)

##function to make predictions
def predict(tree, row):
    if "label" in tree:
        return tree["label"]
    
    column = tree["column"]
    median = tree["median"]
    
    if row[column]<=median:
        return predict(tree['left'], row)
    else:
        return predict(tree['right'], row)

print(predict(tree, data.iloc[0]))

##function to apply prediction to each row
def batch_predict(tree, df):
    return df.apply(lambda x:predict(tree,x),axis=1)


###overfitting, where you memorize the details of the training set,
# but can't generalize to new examples you're asked to make predictions on.

#AUC ranges from 0 to 1, so it's ideal for binary classification. 
#The higher the AUC, the more accurate our predictions.

#We can compute AUC with the roc_auc_score function from sklearn.metrics.

#This function takes in two parameters:
y_true: true labels
y_score: predicted labels
#It then calculates and returns the AUC value.

#Trees overfit when they have too much depth and make overly complex rules that match the training data,
# but aren't able to generalize well to new data. This may seem to be a strange principle at first,
# but the deeper a tree is, the worse it typically performs on new data.
#need to 'prune' the tree and turn some of the higher nodes into leaves

There are three main ways to combat overfitting:

(1) "Prune" the tree after we build it to remove unnecessary leaves.
(2) Use ensembling to blend the predictions of many trees.
(3) Restrict the depth of the tree while we're building it.

####tree model parameters
We can restrict tree depth by adding a few parameters when we initialize the DecisionTreeClassifier class:

max_depth - Globally restricts how deep the tree can go
min_samples_split - The minimum number of rows a node should have before it can be split; if this is set to 2,
	 for example, then nodes with 2 rows won't be split, and will become leaves instead
min_samples_leaf - The minimum number of rows a leaf must have
min_weight_fraction_leaf - The fraction of input rows a leaf must have
max_leaf_nodes - The maximum number of total leaves; this will cap the count of leaf nodes as the tree
	 is being built
Some of these parameters aren't compatible, however. For example, we can't use max_depth and max_leaf_nodes
	 together.

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

#DECISION TREES SUFFER...
#Decision trees typically suffer from high variance. The entire structure of a decision tree can change if we
# make a minor alteration to its training data. By restricting the depth of the tree,
# we increase the bias and decrease the variance. If we restrict the depth too much, 
#we increase bias to the point where it will underfit.

Let's go over the main advantages and disadvantages of using decision trees. 

(1)Easy to interpret
(2)Relatively fast to fit and make predictions
(3)Able to handle multiple types of data
(4)Able to pick up nonlinearities in data, and usually fairly accurate

(1)The main disadvantage of using decision trees is their tendency to overfit.

#The most powerful way to reduce decision tree overfitting is to create ensembles of trees. 
#The random forest algorithm is a popular choice for doing this. In cases where prediction accuracy
# is the most important consideration, random forests usually perform better.


##ENSEMBLING DECISION TREES BY HAND (RANDOM FOREST!!)
# Each "bag" will have 60% of the number of original rows
bag_proportion = .6

predictions = []
for i in range(tree_count):
    # We select 60% of the rows from train, sampling with replacement
    # We set a random state to ensure we'll be able to replicate our results
    # We set it to i instead of a fixed value so we don't get the same sample in every loop
    # That would make all of our trees the same
    bag = train.sample(frac=bag_proportion, replace=True, random_state=i)
    
    # Fit a decision tree model to the "bag"
    clf = DecisionTreeClassifier(random_state=1, min_samples_leaf=2)
    clf.fit(bag[columns], bag["high_income"])
    
    # Using the model, make predictions on the test data
    predictions.append(clf.predict_proba(test[columns])[:,1])
    
final = numpy.round(sum(predictions)/10)

print(roc_auc_score(test['high_income'],final))


### breakdown of random forest 
(1)First we pick the maximum number of features we want to evaluate each time we split the tree.
This has to be less than the total number of columns in the data.
(2)Every time we split, we pick a random sample of features from the data.
(3)Then we compute the information gain for each feature in our random sample, 
	and pick the one with the highest information gain to split on.

When we instantiate a RandomForestClassifier, we pass in an n_estimators parameter that indicates
 how many trees to build. While adding more trees usually improves accuracy, 
 it also increases the overall time the model takes to train.


#We're repeating the same process to select the optimal split that minimizes entropy for a node. 
#However, we'll only evaluate a constrained set of features that we select randomly.
# This introduces variation into the trees, and makes for more powerful ensembles.


##another way to split train and test
train = bike_rentals.sample(frac=.8)
test = bike_rentals.loc[~bike_rentals.index.isin(train.index)]

##ENTIRE RANDOM FOREST BY HAND SELECTING ONLY TWO FEATURES TO SPLIT ON PER NODE
# The dictionary to store our tree
tree = {}
nodes = []

# The function to find the column to split on
def find_best_column(data, target_name, columns):
    information_gains = []
    
    colees = numpy.random.choice(columns,2)
    
    for col in colees:
        information_gain = calc_information_gain(data, col, "high_income")
        information_gains.append(information_gain)

    # Find the name of the column with the highest gain
    highest_gain_index = information_gains.index(max(information_gains))
    highest_gain = colees[highest_gain_index]
    return highest_gain

# The function to construct an ID3 decision tree
def id3(data, target, columns, tree):
    unique_targets = pandas.unique(data[target])
    nodes.append(len(nodes) + 1)
    tree["number"] = nodes[-1]

    if len(unique_targets) == 1:
        if 0 in unique_targets:
            tree["label"] = 0
        elif 1 in unique_targets:
            tree["label"] = 1
        return
    
    best_column = find_best_column(data, target, columns)
    column_median = data[best_column].median()
    
    tree["column"] = best_column
    tree["median"] = column_median
    
    left_split = data[data[best_column] <= column_median]
    right_split = data[data[best_column] > column_median]
    split_dict = [["left", left_split], ["right", right_split]]
    
    for name, split in split_dict:
        tree[name] = {}
        id3(split, target, columns, tree[name])


# Run the ID3 algorithm on our data set and print the resulting tree
id3(data, "high_income", ["employment", "age", "marital_status"], tree)
print(tree)


#One of the major advantages of random forests over single decision trees is that they tend to overfit less.
# Although each individual decision tree in a random forest varies widely, the average of their predictions 
#is less sensitive to the input data than a single tree is. This is because while one tree can construct an
# incorrect and overfit model, the average of 100 or more trees will be more likely to hone in on the signal
# and ignore the noise.

#The main strengths of a random forest are:

(1)Very accurate predictions - Random forests achieve near state-of-the-art performance on many
	 machine learning tasks. Along with neural networks and gradient-boosted trees,
	  they're typically one of the top-performing algorithms.
(2)Resistance to overfitting - Due to their construction, random forests are fairly resistant to overfitting.
	 We still need to set and tweak parameters like max_depth though

# main weaknesses of random forst:

(1)They're difficult to interpret - Because we've averaging the results of many trees, 
	it can be hard to figure out why a random forest is making predictions the way it is.
(2)They take longer to create - Making two trees takes twice as long as making one,
 	making three takes three times as long, and so on. Fortunately, we can exploit multicore processors
  to parallelize tree construction. Scikit allows us to do this through the n_jobs parameter
   on RandomForestClassifier. We'll discuss parallelization in greater detail later on.

###encoding one column and returning a dataframe from it
train_enc = OneHotEncoder.fit_transform(np.asarray(train['primary_use']).reshape(1,-1))

import scipy.sparse
train_cat = pd.DataFrame(train_enc.toarray(),columns = OneHotEncoder.get_feature_names())

train_cat.head()


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

##unzip