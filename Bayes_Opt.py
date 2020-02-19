##BAYESIAN OPTIMIZATION 

#Bayesian model-based optimization methods build a probability model of the objective function to 
#propose smarter choices for the next set of hyperparameters to evaluate.

#Each time the algorithm proposes a new set of candidate hyperparameters, it evaluates them with the 
#actual objective function and records the result in a pair (score, hyperparameters). These records 
#form the history. The algorithm builds l(x) and g(x) using the history to come up with a probability 
#model of the objective function that improves with each iteration.

#Because the algorithm is proposing better candidate hyperparameters for evaluation, the score on the 
#objective function improves much more rapidly than with random or grid search leading to fewer 
#overall evaluations of the objective function.

#Even though the algorithm spends more time selecting the next hyperparameters by maximizing the 
#Expected Improvement, this is much cheaper in terms of computational cost than evaluating the 
#objective function. 

#LEADS TO
#1)Reduced running time of hyperparameter tuning
#2)Better scores on the testing set

##THE SURROGATE FUNCTION
#The Tree-structured Parzen Estimator builds a model by applying Bayes rule. 
#p (x | y), which is the probability of the hyperparameters given the score on the objective function


#5 ASPECTS OF MODEL-BASED HYPERPERAMETER OPTIMIZATION

#1)A domain of hyperparameters over which to search
from hyperopt import hp
# Create the domain space
space = hp.uniform('x', -5, 6)


#2)An objective function which takes in hyperparameters and outputs a score that we want to minimize (or maximize)
"""Objective function to minimize""" #could be rmse or otherwise

# Create the polynomial object
f = np.poly1d([1, -2, -28, 28, 12, -26, 100])

# Return the value of the polynomial
return f(x) * 0.05


#3)The surrogate model of the objective function


#4)A criteria, called a selection function, for evaluating which hyperparameters to choose next from the 
#surrogate model
from hyperopt import rand, tpe

# Create the algorithms
tpe_algo = tpe.suggest
rand_algo = rand.suggest


#5)A history consisting of (score, hyperparameter) pairs used by the algorithm to update the surrogate model

from hyperopt import Trials

# Create two trials objects
tpe_trials = Trials()
rand_trials = Trials()

from hyperopt import fmin

# Run 2000 evals with the tpe algorithm
tpe_best = fmin(fn=objective, space=space, algo=tpe_algo, trials=tpe_trials, 
                max_evals=2000, rstate= np.random.RandomState(50))

print(tpe_best)

# Run 2000 evals with the random algorithm
rand_best = fmin(fn=objective, space=space, algo=rand_algo, trials=rand_trials, 
                 max_evals=2000, rstate= np.random.RandomState(50))	


# Print out information about losses
print('Minimum loss attained with TPE:    {:.4f}'.format(tpe_trials.best_trial['result']['loss']))
print('Minimum loss attained with random: {:.4f}'.format(rand_trials.best_trial['result']['loss']))
print('Actual minimum of f(x):            {:.4f}'.format(miny))

# Print out information about number of trials
print('\nNumber of trials needed to attain minimum with TPE:    {}'.format(tpe_trials.best_trial['misc']['idxs']['x'][0]))
print('Number of trials needed to attain minimum with random: {}'.format(rand_trials.best_trial['misc']['idxs']['x'][0]))

# Print out information about value of x
print('\nBest value of x from TPE:    {:.4f}'.format(tpe_best['x']))
print('Best value of x from random: {:.4f}'.format(rand_best['x']))
print('Actual best value of x:      {:.4f}'.format(minx))

best = fmin(fn=objective, space=space, algo=tpe_algo, max_evals=200)


tpe_results = pd.DataFrame({'loss': [x['loss'] for x in tpe_trials.results], 'iteration': tpe_trials.idxs_vals[0]['x'],
                            'x': tpe_trials.idxs_vals[1]['x']})
                            
tpe_results.head()

# Sort with best loss first
tpe_results = tpe_results.sort_values('loss', ascending = True).reset_index()

##one line optimization
:
# Just because you can do it in one line doesn't mean you should! 
best = fmin(fn = lambda x: np.poly1d([1, -2, -28, 28, 12, -26, 100])(x) * 0.05,
            space = hp.normal('x', 4.9, 0.5), algo=tpe.suggest, 
            max_evals = 2000)