# RECLAIM: Reverse Engineering CLAssIfication Metrics

This is the source code for RECLAIM, currently under review. 

## What is RECLAIM?

RECLAIM is a methodology that allows reconstructing classification metrics (think accuracy, precision, recall, etc.) starting from what little information may be available about some classifier. 

For example, a paper proposes a new classifier, and shows its performance in terms of accuracy on some dataset. However, the dataset is particularly unbalanced and a better metric, you believe, would be the F1 score. 

If you now build your own model, how can you compare it to the published one, in terms of F1 score? Of course, if the authors of the original paper have made their code available, you can just re-run their experiments. But if they did not (which unfortunately happens all too often), you are stuck to having to compare your model in terms of accuracy, which we already established is not a great idea. 

But what if you don't have to? With RECLAIM, you can reconstruct upper and lower boundaries for other metrics, by providing what little information is already available to you!

## How does it work?

The paper explains this aspect in more detail. In short, we try to reconstruct the confusion matrix starting from the available information.

If you have "enough" information (at least 4 constraints -- e.g. number of records, number of positive samples, accuracy and F1 score), you can reconstruct the confusion matrix exactly. 

If you have less than 4 (for now, we support a minimum of 3), we solve an integer programming optimization problem to identify upper and lower boundaries for the confusion matrix.

Once we have the confusion matrix (or its boundaries), we can reconstruct all other metrics we are interested in!

## What are its limitations?
First of all, RECLAIM currently only supports binary classification problems, though we are already planning on extending it to other types of problems (multi-class classification, regression are on our list).

Unfortunately, even for binary problems, there are some situations where RECLAIM may not be particularly useful. While we discuss these situations in the paper, here is a summary:

1. if the available results are overly rounded (think ~ 2 significant figures)

2. if you do not have enough constraints (2 or less)

3. if you have some "unhelpful" combination of constraints (more on this in the paper but, for example, we show that the recall isn't all that useful)

In all these situations RECLAIM would work, but the reconstructed metrics would not be all that useful! We are already hard at work to figure out how to narrow down the boundaries, by introducing some assumption on the models and the data distributions. 

## How do I use RECLAIM?
The main function made available by RECLAIM is `find_confusion_matrix()`, which handles the various cases automatically. You just pass the information you have available and voila', you get you confusion matrix (or some boundaries).

Let's see a more practical scenario.

First, we make ourselves a dataset and a classifier.

```python
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split

# build dataset and split into train/test
X, y = make_classification(10000, n_classes=2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=.8, random_state=42)
# build model, train it and predict labels for the test set
clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
# print some information (i.e. info you would find in a paper)
print("# points", X_test.shape[0])
print("positive samples", y_test.sum())
print("Accuracy", accuracy_score(y_test, y_pred))
print("F1 score", f1_score(y_test, y_pred))
```

By running the above code we might get, for example, the following output:
```
# points 2000
positive samples 994
Accuracy 0.898
F1 score 0.896761133603239
```
So, if this is the information we have available, we can run the following.

```python
from reclaim import find_confusion_matrix

print(find_confusion_matrix(C=2000, N_P=994, A=0.898, Fb=0.896761133603239))
```

And the output is:
```
[886. 910.  96. 108.] # <== [ tp, tn, fp, fn ]
```
Voila'. Our confusion matrix, as promised (you can check, it's correct). 

Great, but what are the chances that we know accuracy and F1 score to that many significant figures? Very slim, obviously. A more realistic case would be the one where we know the accuracy to be 0.898 and the F1 score to be 0.8968. In this case, by running the same code as above:
```
[886.37209302 909.62790698  96.37209302 107.62790698]
```
Now the output is no longer integers. However, with a little post-processing (rounding all values to the nearest integer) we can still reconstruct the exact confusion matrix. We might get some situations where, even through rounding, we would not get the exact result, but rather some off-by-1 or 2 results. 

Okay, what about if we don't have that much information? Let's say we only have the accuracy. What can we do with it?

```python
print(find_confusion_matrix(C=2000, N_P=994, A=0.898))
```

Gets us the following boundaries:

```
(array([ 790., 1006.,    0.,  204.]), array([994., 802., 204.,   0.]))
```
Both results (you can check) satisfy the constraints we imposed. So what if we want to reconstruct the F1 score from these values? 

The first confusion matrix is associated with an F1 score of 0.8856502242152466, the second confusion matrix has an F1 score of 0.906934306569343. The true F1 score should therefore be somewhere in between (and indeed, if we check what we mentioned above, that's exactly what happens). So now we are not reconstructing the exact metrics, but rather an upper and a lower boundary. The narrower this gap between upper and lower bounds, the better our reconstruction. in this case, the width of the range is of ~ 0.02, which is fairly good. In the paper, we discuss how this gap width varies in different situations. 

Finally, if we know a single metric with finite precision (i.e. some rounding has been applied) we need to specify the range of values that that metric may assume. So, for example, if we know F1 score to be 0.8968, we will pass as the range of values (0.89675, 0.89685), i.e. all values that, if rounded, produce the desired F1 score. Note that we do this in the 3-constraints problem and not in the 4-constraints problem (we didn't do this earlier!). This is because in this case, we need to explore all possible values, whereas earlier we were just looking for the solution that was "closest" to the desired result. In the future, we will attempt to make this transparent to the user. Anyway,

```python
find_confusion_matrix(C=2000, N_P=994, Fb=(0.89675, 0.89685))
```
Produces:
```
(array([ 808., 1006.,    0.,  186.]), array([991., 781., 225.,   3.]))
```
In this case, the range for accuracy will be from 0.886 to 0.907 -- once again a range of ~ 0.02. 