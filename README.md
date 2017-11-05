# multinomial-naive-bayes
Implementation of a Multinomial Naive Bayes classifier used to classify student and faculty websites

## Description
The dataset used contains 1400 entries which consists of equal amounts of student and faculty websites (700 of each). The dataset is split into training and test set containing 1000 and 400 features vectors respectively. 

Each feature vector contains 1309 features, which correspond to 1309 words in the dictionary, and the features specify the frequency of the word appearing in the current website.

The classification is done through the computation of the maximum-a-posteriori (MAP) estimates for each classes given the feature vectors. 

The initial prediction accuracy of the test set is 85.75%

### Backward elimination
After the initial training of the model, the mutual information for each word (feature) is calculated; and each feature is removed from the dataset, starting from the least relevant ones, and then the model is retrained. This process aids in the determination of the most optimal set of features to include and operate on, in order to achieve the highest prediction accuracy.