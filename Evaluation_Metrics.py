#EVALUATION METRICS FOR CLASSIFICATION MODELS
##BINARY CLASSIFICATION PERFORMANCE METRICS

#ACCURACY - the number of correct classiffications vs all classifications
## accuracy paradox  - Sometimes it may be desirable to select a model with a lower accuracy because 
# it has a greater predictive power on the problem.
# in a problem where there is a large class imbalance, a model can predict the value of the majority 
# class for all predictions and achieve a high classification accuracy, the problem is that this model 
# is not useful in the problem domain. As we saw in our breast cancer example.

#PRECISION - True positives/(true positives + false positives)
# number of correct true positives vs total positives predicted
# a low precision indicates a high number of false positives

#RECALL(Sensitivity) - True Positives/(true positives + false negatives)
# number of positive predictions vs total positive class values

#F1 SCORE - 2*((precision*recall)/(precision+recall))
# conveys the balance between precision and recall

#SPECIFICITY - True Negatives/(true_negatives + false positives)
#Specificity is the proportion of actual 0’s that were correctly predicted.
#Maximizing specificity is more relevant in cases like spam detection, where you strictly don’t want genuine 
#messages (0’s) to end up in spam (1’s).

#COHEN'S KAPPA - (Observed Accuracy - Expected Accuracy) / (1 - Expected Accuracy)
#Kappa is similar to Accuracy score, but it takes into account the accuracy that would have happened anyway 
#through random predictions.
#Same as Quadratic Weighted Kappa in sklearn when weights are set to Quadratic

#KOLMOGOROV-SMIRNOV STATISTIC 
#used to make decisions like: How many customers to target for a marketing campaign? or How many customers 
#should we pay for to show ads etc.
	#Step 1: Once the prediction probability scores are obtained, the observations are sorted by decreasing 
	#order of probability scores. This way, you can expect the rows at the top to be classified as 1 while 
	#rows at the bottom to be 0's.
	#Step 2: All observations are then split into 10 equal sized buckets (bins).
	#Step 3: Then, KS statistic is the maximum difference between the cumulative percentage of responders 
	#or 1's (cumulative true positive rate) and cumulative percentage of non-responders or 0's (cumulative 
	#false positive rate).
#The significance of KS statistic is, it helps to understand, what portion of the population should be 
#targeted to get the highest response rate (1's).

#AUC CURVE -  the only metric that measures how well the model does for different values of prediction 
#probability cutoffs.
#This is a way of analyzing how the sensitivity and specificity perform for the full range of probability 
#cutoffs, that is from 0 to 1.

#CONCORDANCE - A pair is said to be concordant if the probability score of True 1 is greater than the 
#probability score of True 0.

#DISCORDANCE - A pair is said to be discordant if the probability score of True 0 is greater than the 
#probability score of True 1.
#e.g.

Patient No	True Class	Probability Score
P1			1			0.9
P2			0			0.42
P3			1			0.30
P4			1			0.80

P1-P2 => 0.9 > 0.42 => Concordant!
P3-P2 => 0.3 < 0.42 => Discordant!
P4-P2 => 0.8 > 0.42 => Concordant!

#Out of the 3 pairs, only 2 are concordant. So, the concordance is 2/3 = 0.66 and 
#discordance is 1 - 0.66 = 0.33.
#For a perfect model, this will be 100%. So, the higher the concordance, the better is the quality of the model.

#SOMERS-D STATISTIC -(#Concordant Pairs - #Discordant Pairs - #Ties) / Total Pairs
#statistic, like concordance itself, judges the efficacy of the model

#GINI COEFFICIENT - (2 * AUROC) - 1
#Gini Coefficient is an indicator of how well the model outperforms random predictions.


#EVALUATION METRICS FOR REGRESSION MODELS

#MEAN SQUARED ERROR (MSE) - measures the average squared error of our predictions.
#For each point, it calculates square difference between the predictions and the target and then average 
#those values.
#The higher the mse, the worse your model
#Advantage - Useful if we have unexpected values that we should care about.
#Disadvantage - If we make a single very bad prediction, the squaring will make the error even worse and it 
#may skew the metric towards overestimating the model’s badness. Particularly an issue if we have very 
#noisy data. Even a “perfect” model may have a high MSE in that situation, so it becomes hard to judge how 
#well the model is performing. On the other hand, if all the errors are small, or rather, smaller than 1, than 
#the opposite effect is felt: we may underestimate the model’s badness.
#Important to note: MSE is more often used than RMSE since there is more difference between predictions.

#ROOT MEAN SQAURED ERROR (RMSE) - square root of mse to make the scale of errors the same as the scale of targets.
#Similarities to MSE: they share the same minimizers since sqaure root is a non-decreasing function.
	#For example, if we have two sets of predictions, A and B, and say MSE of A is greater than MSE of B, 
	#then we can be sure that RMSE of A is greater RMSE of B.
	#Given this, you can still compare models using MSE instead of RMSE
#Differences to MSE: In the case of gradient based methods, RMSE and MSE have different flowing rate so
#they are not interchangeable in this case (would need to adjust parameters like learning rate to make similar).

#MEAN ABSOLUTE ERROR (MAE) - average of the absolute differences between the targets and the predictions.
#The MAE is a linear score which means that all the individual differences are weighted equally in the average.
	# For example, the difference between 10 and 0 will be twice the difference between 5 and 0. However, 
	#same is not true for RMSE. 	
#What is important about this metric is that it penalizes huge errors that not as that badly as MSE does. 
#Thus, it’s not that sensitive to outliers as mean square error.
	#MAE is widely used in finance, where $10 error is usually exactly two times worse than $5 error. On the 
	#other hand, MSE metric thinks that $10 error is four times worse than $5 error. MAE is easier to justify 
	#than RMSE.

#Questions to ask when considering MAE or MSE:
	#1) Do you have outliers in your data? -- use MAE
	#2) Are you sure they are outliers? -- use MAE
	#3) Or are they just unexpected values we should care about? -- use MSE

***#Applicable to the above three (for the purposes of establishing a constant baseline):
	#Note that if we want to have a constant prediction the best one will be the median value of the target 
	#values. It can be found by setting the derivative of our total error with respect to that constant to zero, 
	#and find it from this equation.

#R2 SCORE - 1 - (MSE(model)/MSE(baseline))
	#baseline model would be to always predict the average of all samples.
#Similar to MSE but is scale-free, will always range from 0-1.
#R² is the ratio between how good our model is vs how good is the naive mean model.