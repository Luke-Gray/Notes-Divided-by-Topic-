##NOTES ON RANDOM FOREST
# On the other hand, Random Forest uses fully grown decision trees (low bias, high variance). It tackles 
#the error reduction task in the opposite way: by reducing variance. The trees are made uncorrelated to 
#maximize the decrease in variance, but the algorithm cannot reduce bias (which is slightly higher than the
#bias of an individual tree in the forest). Hence the need for large, unpruned trees, so that the bias is 
#initially as low as possible.

#One straight-forward way is to limit the maximum allowable tree depth. The common way for tree based
#algorithms to overfit is when they get too deep. Thus you can use the maximum depth parameter as the
#regularization parameter — making it smaller will reduce the overfitting and introduce bias, increasing
#it will do the opposite.

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

##PERAMETERS
#N_ESTIMATORS (only used in Random Forests) is the number of decision trees used in making the
#forest (default = 100). Generally speaking, the more uncorrelated trees in our forest, the closer 
#their individual errors get to averaging out. However, more does not mean better since this can have 
#an exponential effect on computation costs. After a certain point, there exists statistical evidence 
#of diminishing returns. 
#Bias-Variance Tradeoff: in theory, the more trees, the more overfit the
#model (low bias). However, when coupled with bagging, we need not worry.

#MAX_DEPTH is an integer that sets the maximum depth of the tree. The default is None, which means the 
#nodes are expanded until all the leaves are pure (i.e., all the data belongs to a single class) or until 
#all leaves contain less than the min_samples_split, which we will define next.
#Bias-Variance Tradeoff: increasing the max_depth leads to overfitting (low bias)

#MIN_SAMPLES_SPLIT is the minimum number of samples required to split an internal node.
#Bias-Variance Tradeoff: the higher the minimum, the more “clustered” the decision will be, 
#which could lead to underfitting (high bias).

#MIN_SAMPLES_LEAF defines the minimum number of samples needed at each leaf. The default input here is 1. 
#Bias-Variance Tradeoff: similar to min_samples_split, if you do not allow the model to split (say because
#your min_samples_lear parameter is set too high) your model could be over generalizing the training data 
#(high bias).

#GINI VS. ENTROPY
#GINI - the probability of incorrectly classifying a randomly chosen datapoint if it were labeled according
#to the class distribution of the dataset.

#ENTROPY
#Entropy is a measure of chaos in your data set. If a split in the dataset results in lower entropy, 
#then you have gained information (i.e., your data has become more decision useful) and the split is 
#worthy of the additional computational costs.