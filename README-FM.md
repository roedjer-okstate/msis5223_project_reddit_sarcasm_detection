
# Data Splitting and Sub-Sampling


### Data Partition Sets and Ratio of the Partitions

Before developing the predictive models, we decided to split our datasets into three subsets - `Training`, `Validation`, and `Testing`. The reason for choosing to divide the dataset into three sections are as follows:
* We wanted to ensure that we have a separate dataset for final testing that the model has never been exposed to and hence reducing the risk of overfitting our models on the test set. It provides us an `unbiased estimate` of how our models are peforming truly `on unseen data`. 
* Having a separate validation set allowed us to train our models on the training set and `fine-tune the hyperparameters` of the model by evaluating on the validation set and trying to improve the performance of the models on the validation set. This helped us keep the final test set intact and not use it anywhere in the training process. 
* Having a separate validation set also allowed us to perform model selection based on the validation set rather than the training set which could run the risk of overfitting. 

We started off with around `74000` rows of data in our overall dataset. We used a `60% - 20% - 20%` split ratio for our `Training - Validation - Testing` sets. This resulted in `44313 rows in our Training Set` and `14771 rows each in the Validation and Testing sets`. We utilized a `stratified sampling approach` for performing this split and the stratification was done based on two columns - `Subreddit Name` and `Target Label indicating Sarcasm/Not-Sarcasm`. This allowed us to ensure that equal ratios of the 3 subreddits were distributed across the three partitions of the data and more importantly we had equal ratios of the target label across the three partitions of the data. The reason for choosing the particular split ratio of `60% - 20% - 20%` are as follows:
* Choosing a 60% split ratio for the training set allowed us to ensure that we had enough data for the more complex ensmeble learning techniques like Random Forest to learn the patterns in the dataset and not end up overfitting on the training data. A 60% split resulted in almost 45000 data points for our training set which would be a decent dataset size for these complex models to learn well and avoid overfitting. 
* We wanted to ensure that the Validation and Test sets were very similar to each other because ensuring the the validation set is similar to the test set will allow the model to generalize well on the unseen test set too. This is because we will be using the validation set for performing all the model tuning and making adjustments to the model hyperparameters. We will be using the test set only for the final evaluations and reporting the final results of the model performance. Having a validation set that is similar in size and characteristics to the test set would be good for fine tuning the model. Hence, we decided to use 20% split ratios for both these sets. 
* Finally, a 20% split ratio for the test set resulted in almost 15000 rows for our test set. This allowed our test set to be a reliable dataset for testing how well our models would generalize on unseen data. The relatively large dataset size of around 15000 rows allowed us to ensure that the performance evalutions were not just a matter of chance or due to fluke and was a reliable indicator of model performance. 

Sample code Snippet for performing the data partitions  
```python
_X_train, X_test, _Y_train, Y_test = train_test_split(
    df[df.columns.difference(['label'])],
    df[['id', 'label']],
    test_size=0.20,
    stratify=df[['label', 'subreddit']],
    random_state=42
)
```


# Select Modeling Techniques

The objective of our modeling exercise is to `predict whether a particular comment on reddit is a sarcastic comment or not`. Hence, this will be treated a `Binary Classsification Problem` where the target variable is a `1/0 flag` indicating `Sarcasm(1) and Not-Sarcasm(0)`

As we have mentioned earlier, for this purpose we will be using a combination of text-related variables derived from the comments and the parent comments along with some numeric and categorical features. Brief recap of the features we will be using are as follows: 
* **Vectorized Text Features**: We will use TF-IDF scores to represent the text in each of the reddit comments and parent comments. This process has been covered in detail in the data preparation section above. 
* **Numeric Features**: We will include numeric features such as the number of upvotes, downvotes, score, karma gathered by the user, and age of the account to capture user engagement and history.
* **Categorical Features**: We will include the subreddit from which the comment was picked to indicate the topic of discussion as certain subreddits could have a higher inclination towards sarcasm.
* **Sentiment Analysis Metrics**: We will also use scores indicating the `scale of emotions` like `disgust, fear, trust, and surprise`. This metrics are obtained from the sentiment analysis exercise that we have carried out and the process has been covered in detail in the sentiment analysis section above. 
* **Topic Labels**: Through our topic analysis exercise, we have extracted a few broad topics that is being talked about in the comments. This process has also been covered in detail in the topic analysis section above. We will use these topic analysis labels to indicate which topic the comment is discussing about like `casual conversation, relationships & personal, politics & current events, mixed topics`.

With the objective and features available in consideration we have decided to use the following models for our predictive modeling exercise. We will be using two types of models with two types of variations in the input data finally resulting in four different models:
1. `Logistic Regression with PCA Transformed Input Features`
2. `Logistic Regression without Transformations (Base Input Features Used)`
3. `Random Forest with PCA Transformed Input Features`
4. `Random Forest without Transformations (Base Input Features Used)`

<br>

**Reasons for using two different types of input features - PCA Transformed & Untransformed**  
Having TFIDF vectorized text features results in a very high dimensional input dataset which can multiple issues in the predictive modeling process. It can also cause the very popular issue that has been termed as the `Curse of the Dimensionality` issue in predictive modeling. As the number of features increases, the volume of the feature space grows exponentially, leading to a sparsity of data. This can make it difficult for the model to capture meaningful patterns in the data, and can result in poor performance. Additionally, with an increase in the number of features, the model can become too complex and can start to fit the noise in the data, resulting in overfitting. This can lead to poor generalization performance on new data. Finally, having large number of features also increase the computation power required for training the models.   

Hence, due to these reasons, we decided to use PCA Transformed Inputs as one of the feature input types for our models. It can have the following advantages: 
* It reduces the dimensionality of the feature space, which can improve model performance by decreasing the risk of overfitting and reducing computational complexity.
* By reducing the dimensionality, it may also make it easier to visualize and interpret the text features.
* It can help to remove redundant or unimportant information from the text features, which may improve the interpretability of the model.

However, this can also come with multiple issues like reducing the dimensionality of the data can result in loss of information which can cause the model the lost some predictive performance. Another major issue is that transformation reduces the interpretability of the model by a large amount and it makes it difficult to understand which variables have a higher impact since the transformed features are linear combinations of the original features. Hence, for this reason, we have also decided to build models based on just the base features without transformations and compare these two sets of modeling iterations. 

**Reason for using Random Forest (RF)**  
Random Forest is a tree-based model that is suitable for both classification and regression tasks, which means it can handle predicting binary outcomes, like sarcasm detection.   
* It is know to be very robust since it is an ensemble learning model which uses a combination of multiple low-depth decision trees built on different feature subsets and finally uses and averaging appraoch to combine the results of these decision trees. This averaging appraoch reduces the risk of overfitting and improves the model's accuracy and generalization. It also provides a feature importance ranking, which allows us to identify the most important features that contribute the most to the prediction of sarcasm.
* Random Forest is known for its ability to handle high-dimensional data, which is a characteristic of vectorized text features and the combination of other numeric and categorical features. With the use of tf-idf scores, we can capture the importance of words within the text data and incorporate them as features in the model.
* Random Forest can also handle missing values in the data, which is a common issue in real-world datasets. This makes the model more robust and capable of handling incomplete or noisy data.
* It is known to work very well with tabular datasets and since we have a combination of vectorized text features and multiple numeric and categorical features, Random Forest seems to be a good choice. 
* Random Forest is also a relatively simple ensemble model that is easier to interpret compared to other more complex ensemble learning techniques like Gradient Boosted Decision Trees. It is definitely easier to interpret compared to Neural Networks. Interpretation is an important requirement for us since we want to understand the major contributors that help in identifying sarcasm. It is also relatively easier to train that more complex ensemble learning models like Gradient Boosted Decision Trees and Neural Networks. which requires more computing power for the training process and can be an overkill for smaller datasets. 

Hence, all these reasons make Random Forest a sensible choice for the problem that we are trying to tackle.
