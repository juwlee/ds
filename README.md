# Data Science project

This repo has a data science project in python.


### Goal

*short answer*: predict the next economic data point
*long answer*
- walk through my analysis process
- show my thought process
- share the results
- gather feedback for future research direction



### Overview

What am I trying to do? It is important to set the goal clearly before the start line as it will keep us pointing in the right direction. I will walk you through the process of statistical problem solving. 

For the particular problem that we are trying to answer, we are going to use supervised learning, which means that all the data points are labeled. Each data point has input fields `x` and a label `y`. The task is to figure out the relationship between `x` and `y` so that when we are given `x` only, we can forecast its corresponding `y` value. Within supervised learning, there are 2 branches: *regression* and *classification*. *Regression* works on continuous `y` values whereas *classification* is designed for discrete `y`. Hence, I will use *regression* in this analysis.




### Data source

name | description
--- | ---
[Quandl](https://www.quandl.com) | economic/financial data





### Analytics pipeline architecture

##### evaluation criteria

When we have finished model training and prediction tasks, how are we going to evaluate the result? As we are analyzing quantitatively, let's evaluate the model statistically as well. It is analogous to `Test-Driven Development` in software engineering in the sense that how successful the modeling is defined before the beginning of development process.

Here are a few frequently used criteria for clustering:
- mean absolute error (MAE)
- mean squared error (MSE)
- R2

I will use mean squared error (MSE) to compare different models' performance.


##### data preparation




##### data normalization (feature scaling)

Many machine learning methods depends on iterative algorithm. First, we model the problem and define objective function that describes how `y` (dependent variable) is associated with `x` (independent variables). Iterative algorithm comes in the next to find solution to the loss function. *If ranges of feature values vary too widely, it will be hard or take a long time to converge.*

To avoid it, let's standardize feature value ranges. Just like any other steps, there are many different ways to pre-process data.
- Rescaling: rescale the range of features to `[0, 1]` or `[-1, 1]`.
- Standardization: zero mean and unit variance
- Unit length scaling: rescale the whole feature vector to have length 1

Here, I am going to use *standardization* which is basically subtracting mean from feature and divide with variance. The outcome will be that `all features will have 0 mean and variance 1.`



##### dimensionality reduction (feature selection)

When data size is small, it is not too hard to do number crunching. However, what if data volume is too big? Of course, distributed computing can help by increased computation power. But we may be wasting resources on not-so-meaningful data. Out of so many features, there can be statistically meaningful features that will help prediction greatly, but the others will not help the task and just take up memory space and processing time.

To cut down the unnecessary resource investment, we can do dimensionality reduction. It means trimming down feature set by selecting only features that are relevant to y's prediction. This step has a few benefits:
- shorten training time
- shorten prediction time
- requires less computation resources (CPU, memory)

Here, I will *pick up features one by one recursively and greedily* by scoring their relevancy to label. At every iteration, it will first rank features by relevancy scores. Once they are ranked, add the top score feature to model. It continues until a termination condition (cumulative variance, # of features, etc) is met.

Why *greedy recursive feature addition*? That is because:
- it takes O(n!) to evaluate all combinations of features and to find the optimal subset. This is not practical and will not scale.
- if we do [univariate feature scoring](http://scikit-learn.org/stable/modules/feature_selection.html#univariate-feature-selection) and , then it will has O(n) time complexity, which is fast and scalable. However, it will not capture dependency between different features. That is, multiple features may provide the same kind of information and they are basically redundant. 



##### model selection

Now, we have input data ready. The next step is to choose the model that can capture features and label relationship the best. However, the model with the highest accuracy is not always the best. There are other factors to consider. Let me list them out:

1. *accuracy*: of course, correct inference is crucial for predictive modeling. It is just that, sometimes, we may need to trade it off with other factors listed above for realistic reasons.
1. *speed*: some statistical model need more inference and training time depending on the complexity of the model. If there are two models that fit your data and a much faster model yields a reasonable trade-off on accuracy, it will make sense to choose the faster model.
1. *scalability* (scale-out): now many models are supported by distributed machine learning frameworks (e.g., [MLlib](http://spark.apache.org/mllib/), [mahout](http://mahout.apache.org/), [petuum](http://petuum.github.io/)). Each framework support different set of models and is built on top of different platforms. For instance, if you already have [Spark](http://spark.apache.org/) platform, you may want to consider [MLlib](http://spark.apache.org/mllib/) as it is one of Spark stack products.

 

##### model training

Data set is ready and you picked the model. Now it is time for training. Just like any other previous step, this part also has caveats: *smoothing, parameter optimization*

1. *smoothing*: many times, the data set that we use for training may not reflect the actual universe data set. For example, stock market data from 2007 ~ 2009 is heavily biased toward downside and does not show the same pattern as other bull market or side-walk time periods. Even if you have all data from the beginning through present time, the future may exhibit different behavior. *Smoothing* is used to shave off this bias and to generalize the data characteristics. I will use a very simple method of smoothing: held-out data. The idea is set aside a certain portion of data set and train the model with the rest of the data. Usually, this held-out portion is about 10~20% of the input data set.
1. *parameter optimization*: some statistical models have hyper parameters (e.g., `C` in support vector machine). The idea is to optimize a measure of the algorithm's performance on an independent data set. `cross validation` is one well-known method. Also, the aforementioned smoothing can be seen as a special case of cross validation where there are 2 subsets and one set is much bigger than the other.




##### model inference

Your model has been trained with observed data set (`x` and `y`). Now let's feed test data (`x`) for prediction. Once it spits out `y`, we can compare it to the actual `y` to see how well it predicted in the next step.




##### output evaluation

As outlined above, model performance can be evaluated with many different criteria. The re





### Notes and future direction

##### notes

As it has been shown, *statistical analysis process involves a lot of decision making*. Sometimes, one model performs better than another. In some other scenarios, the other one does a better job than the first. Also, it is deployed to production, you should think about the different environment from your development setup. If you have a data science API service, chances are, you would have an SLA within which you should return a response. Under the restrictions, you need to make a series of educated decisions among which there are `linear vs. non-linear`, `simple vs. complex`, etc. It also applies to the recent interest on *Deep Learning*. Yes, it is very powerful. If trained appropriately, it can be so smart to recognize objects in images, disambiguate word meanings, analyze text sentiment. However, depending on your DNN's size and complexity, it may not be so viable to use in production. Plus, while many of traditional machine learning frameworks are designed to use CPUs, they outperform on GPUs, meaning that you may not be able to exploit the same hardware as the job profiles are different.

So what can we do about it? *First, know the pros and cons of different approaches. Second, understand your requirements. Third, look at your data and figure out its characteristics.* I will wrap up with a quote from a classic chinese book, `Know the enemy and know yourself; in a hundred battles you will never be in peril.`


##### future direction