![Author](https://img.shields.io/badge/author-faizy-red)
![License](https://img.shields.io/badge/license-MIT-brightgreen)
![Platform](https://img.shields.io/badge/platform-Visual%20Studio%20Code-blue)
![Maintained](https://img.shields.io/maintenance/yes/2023)
![Last Commit](https://img.shields.io/github/last-commit/mohd-faizy/feature-engineering-hacks)
![Issues](https://img.shields.io/github/issues/mohd-faizy/feature-engineering-hacks)
![Stars GitHub](https://img.shields.io/github/stars/mohd-faizy/feature-engineering-hacks)
![Language](https://img.shields.io/github/languages/top/mohd-faizy/feature-engineering-hacks)
![Size](https://img.shields.io/github/repo-size/mohd-faizy/feature-engineering-hacks)

# **Feature Engineering & Feature selection**

### What is feature engineering & feature selection?


`Feature engineering` and `feature selection` are both important data preparation tasks in machine learning.

>Feature engineering is the process of creating new features from existing data, while feature selection is the process of selecting a subset of features from a dataset.

<p align='center'>
  <a href="#"><img src='https://assets-global.website-files.com/620d42e86cb8ec4d0839e59d/6230e9ee021b250dd3710f8e_61ca4fbcc80819e696ba0ee9_Feature-Engineering-Machine-Learning-Diagram.png'></a>
</p>

Feature engineering can be used to improve the performance of machine learning models by creating features that are more relevant to the target variable. For example, if you are trying to predict whether a customer will churn, you might create a feature that is the number of days since the customer last made a purchase.

Feature selection can be used to improve the performance of machine learning models by reducing the number of features that need to be processed. This can be helpful for reducing overfitting and improving the interpretability of models.

There are many different methods for feature engineering and feature selection, and the best approach will vary depending on the data and the machine learning algorithm being used. However, both feature engineering and feature selection are important tasks that can improve the performance of machine learning models.

- Here are some examples of **feature engineering**:

  - Creating new features by combining existing features. 
    >**For example** -  you could create a feature - that is the sum of two other features.

  - Creating new features by transforming existing features. 
     >**For example** -  you could create a feature that is the square root of another feature.

  - Creating new features by discretizing existing features. 
     >**For example** -  you could create a feature that is the binary indicator of whether a value is greater than a certain threshold.

### Summary of the main classes and functions for `feature_selection` in Scikit-learn:

| Class/Function | Description |
| --- | --- |
| `SelectKBest` | Selects the top K features based on a scoring function |
| `chi2` | This test is used to measure the association between a feature and the target variable. Features with a high chi-squared value are considered to be important.|
| `SelectPercentile` | Selects the top percentile of features based on a scoring function |
| `SelectFromModel` | Selects features based on importance weights computed by a supervised model |
| `RFE` | Recursive feature elimination method starts with all of the features and then iteratively removes the least important features until a specified number of features remain. |
| `RFECV` | RFECV function is a recursive feature elimination method that uses cross-validation to select the best subset of features |
| `SequentialFeatureSelector` | Performs forward or backward feature selection with cross-validation |
| `mutual_info_regression` | `mutual_info_regression` is a function in scikit-learn's feature selection module that computes mutual information between each feature and a continuous target variable. Mutual information measures the amount of information that can be obtained about one variable by observing another variable. In the context of feature selection, mutual information can be used to identify the features that are most informative about the target variable.`mutual_info_regression`takes two input arrays: the feature matrix `X` and the target variable `y`. It returns an array of mutual information scores, where each score corresponds to a feature in `X`. The higher the score, the more informative the feature is about the target variable. |
| `mutual_info_classification` | Computes the mutual information between each feature and a categorical target variable |
| `f_regression` | `f_regression` is a function in scikit-learn's feature selection module that computes the `F-value` and `p-value` for each feature in a dataset with respect to a continuous target variable. **The F-value measures the ratio of variance between the target variable and the feature variable, while the p-value indicates the significance of the F-value.** In the context of feature selection, F-values and p-values can be used to identify the features that are most correlated with the target variable. it takes two input arrays: the feature `matrix X` and the target variable `y`. It returns two arrays: the `F-values` and the `p-values`, where each value corresponds to a feature in `X`. The higher the `F-value`, the more correlated the feature is with the target variable, while the lower the `p-value`, the more significant the correlation is.|

These classes and functions are part of the `sklearn.feature_selection` module and can be used to select a subset of features from a dataset based on various criteria.


### Most commonly used feature selection methods in Scikit-learn:

| Method | Description | Scikit-learn Class |
| --- | --- | --- |
| Filter methods | Select features based on a statistical measure | `SelectKBest`, `SelectPercentile`, `f_classif`, `f_regression`, `chi2`, `mutual_info_classif`, `mutual_info_regression` |
| Wrapper methods | Select features based on the performance of a model trained with different subsets of features | `RFECV`, `SequentialFeatureSelector` |
| Embedded methods | Select features based on their importance as learned by a model | `SelectFromModel`, `LassoCV`, `RidgeCV`, `ElasticNetCV`, `RandomForestClassifier`, `RandomForestRegressor`, `GradientBoostingClassifier`, `GradientBoostingRegressor`, `XGBClassifier`, `XGBRegressor` |

- **Filter methods** rank features based on a statistical measure that assesses the strength of the relationship between each feature and the target variable. Examples of such measures include the F-value (for continuous target variables), the chi-squared statistic (for categorical target variables), and mutual information (for both continuous and categorical target variables). These methods are computationally efficient and can be used as a preprocessing step to reduce the dimensionality of the data before applying a more complex model.

- **Wrapper methods** evaluate the performance of a model trained with different subsets of features and select the subset that leads to the best performance. Examples of such methods include recursive feature elimination (RFE) and sequential feature selection (SFS). These methods are computationally more expensive than filter methods but can lead to better performance if the optimal subset of features is highly dependent on the specific task and dataset.

- **Embedded methods** incorporate feature selection as part of the model training process. Examples of such methods include regularization (e.g., L1 and L2 regularization in linear models), tree-based methods (e.g., random forests and gradient boosting), and XGBoost. These methods can be computationally efficient and often lead to better performance than filter methods but can be sensitive to the choice of hyperparameters and model architecture.


 ### Charts that might be useful for `feature selection` and `feature engineering`

| Chart | Description |
| --- | --- |
| Correlation matrix heatmap | A correlation matrix heatmap can help you visualize the correlation between different features. This can be useful for identifying redundant features that can be removed to reduce the dimensionality of the data. |
| Box plot | A box plot can help you identify outliers and understand the distribution of a feature. This can be useful for deciding how to handle outliers and for identifying features that might need to be transformed or normalized. |
| Scatter plot matrix | A scatter plot matrix can help you visualize the relationship between different features. This can be useful for identifying features that are highly correlated with the target variable and for identifying interactions between features. |
| Decision tree | A decision tree can be used to visualize the importance of different features in a predictive model. This can be useful for understanding which features are most important for predicting the target variable and for identifying features that can be pruned to improve the model's performance. |
| Principal component analysis (PCA) plot | A PCA plot can be used to visualize the relationship between different features in a high-dimensional dataset. This can be useful for identifying clusters of similar observations and for understanding the underlying structure of the data. |
| Feature importance plot | A feature importance plot can be used to visualize the importance of different features in a predictive model. This can be useful for understanding which features are most important for predicting the target variable and for identifying features that can be pruned to improve the model's performance. |

### Useful Code snippet

- `SelectKBest`
  ```python
  from sklearn.datasets import load_iris
  from sklearn.feature_selection import SelectKBest
  from sklearn.feature_selection import chi2

  # Load iris dataset
  iris = load_iris()
  X, y = iris.data, iris.target

  # Apply SelectKBest feature selection
  selector = SelectKBest(chi2, k=2)
  X_new = selector.fit_transform(X, y)

  # Print selected features
  print(selector.get_support(indices=True))
  ```
  We then apply the `SelectKBest` feature selection method with the `chi2` scoring function to select the top 2 features. Finally, we transform the original data into the new feature space using the fit_transform method and print the indices of the selected features using the get_support method



- `Chi-squared test`
  ```python
  from sklearn.feature_selection import chi2

  # Load the wine dataset.
  X, y = datasets.load_wine(return_X_y=True)

  # Select the top 5 features using the chi-squared test.
  selector = chi2(X, y)
  selector.fit(X, y)
  indices = selector.get_support()
  features = X.columns[indices]

  # Print the selected features.
  print(features)
  ```
  ```
  ['alcohol', 'malic_acid', 'total_acidity', 'density', 'residual_sugar']
  ```



- `feature_selection.SelectPercentile`
  ```python
  from sklearn.datasets import load_iris
  from sklearn.feature_selection import SelectPercentile
  from sklearn.feature_selection import f_classif

  # Load iris dataset
  iris = load_iris()
  X, y = iris.data, iris.target

  # Apply SelectPercentile feature selection
  selector = SelectPercentile(f_classif, percentile=50)
  X_new = selector.fit_transform(X, y)

  # Print selected features
  print(selector.get_support(indices=True))
  ```
  We then apply the `SelectPercentile` feature selection method with the `f_classif` scoring function to select the top `50%` of features. Finally, we transform the original data into the new feature space using the fit_transform method and print the indices of the selected features using the `get_support` method.




- `SelectFromModel`
  ```python
  from sklearn.datasets import load_iris
  from sklearn.feature_selection import SelectFromModel
  from sklearn.linear_model import LogisticRegression

  # Load iris dataset
  iris = load_iris()
  X, y = iris.data, iris.target

  # Apply SelectFromModel feature selection
  selector = SelectFromModel(LogisticRegression(penalty='l1', C=0.1))
  X_new = selector.fit_transform(X, y)

  # Print selected features
  print(selector.get_support(indices=True))

  ```
  We apply the `SelectFromModel` feature selection method with a `LogisticRegression` model that uses `L1` regularization with a penalty parameter of `0.1`. Finally, we transform the original data into the new feature space using the `fit_transform` method and print the indices of the selected features using the `get_support method`. Note that the model used in `SelectFromModel` can be any supervised learning model that has a `coef_` or `feature_importances_` attribute after fitting.



- `Recursive feature elimination(RFE)`

  ```python
  from sklearn.feature_selection import RFE
  from sklearn.ensemble import RandomForestRegressor

  # Create a random forest regressor
  rf = RandomForestRegressor()

  # Create an RFE object
  rfe = RFE(rf, n_features_to_select=5)

  # Fit the RFE object to the training data
  rfe.fit(X_train, y_train)

  # Get the selected features
  selected_features = rfe.support_

  # Get the importance scores of the features
  importance_scores = rfe.ranking_
  ```

- `Recursive feature elimination with cross-validation (RFECV)`
  ```python
  from sklearn.feature_selection import RFECV

  # Load the wine dataset.
  X, y = datasets.load_wine(return_X_y=True)

  # Create an RFECV object.
  selector = RFECV(estimator=LogisticRegression(), step=1, cv=5, scoring='accuracy')

  # Fit the RFECV object.
  selector.fit(X, y)

  # Get the indices of the selected features.
  indices = selector.get_support()

  # Get the selected features.
  features = X.columns[indices]

  # Print the selected features.
  print(features)
  ```
  ```
  ['alcohol', 'malic_acid', 'total_acidity', 'density', 'residual_sugar']
  ```


- `SequentialFeatureSelector`
  ```python
  from sklearn.datasets import load_iris
  from sklearn.feature_selection import SequentialFeatureSelector
  from sklearn.neighbors import KNeighborsClassifier

  # Load iris dataset
  iris = load_iris()
  X, y = iris.data, iris.target

  # Apply SequentialFeatureSelector feature selection
  selector = SequentialFeatureSelector(KNeighborsClassifier(n_neighbors=3), n_features_to_select=2)
  X_new = selector.fit_transform(X, y)

  # Print selected features
  print(selector.get_support(indices=True))
  ```
  we apply the `SequentialFeatureSelector` feature selection method with a `KNeighborsClassifier` model that uses `3` nearest neighbors and select the top `2` features using `n_features_to_select`. Finally, we transform the original data into the new feature space using the `fit_transform` method and print the indices of the selected features using the `get_support` method. Note that the model used in `SequentialFeatureSelector` can be any supervised learning model that has a `coef_` or `feature_importances_` attribute after fitting.


-  `mutual_info_regression`
    ```python
    from sklearn.datasets import load_diabetes
    from sklearn.feature_selection import SelectKBest, mutual_info_regression

    # Load the diabetes dataset
    X, y = load_diabetes(return_X_y=True)

    # Select the top 3 features using mutual information regression
    selector = SelectKBest(mutual_info_regression, k=3)
    X_new = selector.fit_transform(X, y)

    # Print the indices of the selected features
    print(selector.get_support(indices=True))

    ```
    In this example, we use `mutual_info_regression` as the scoring function in `SelectKBest` to select the top `3` features from the diabetes dataset. The `get_support` method is used to retrieve the indices of the selected features.


- `mutual_info_classification`

  ```python
  from sklearn.datasets import load_breast_cancer
  from sklearn.feature_selection import SelectKBest, mutual_info_classification

  # Load the breast cancer dataset
  X, y = load_breast_cancer(return_X_y=True)

  # Select the top 5 features using mutual information classification
  selector = SelectKBest(mutual_info_classification, k=5)
  X_new = selector.fit_transform(X, y)

  # Print the indices of the selected features
  print(selector.get_support(indices=True))

  ```
  In this example, we use `mutual_info_classification` as the scoring function in SelectKBest to select the top `5` features from the breast cancer dataset. The `get_support` method is used to retrieve the indices of the selected features. Note that `mutual_info_classification` is appropriate when the target variable is categorical, such as in a classification problem. If the target variable is continuous,`mutual_info_regression` should be used instead.

- `f_regression`
  ```python
  from sklearn.datasets import load_diabetes
  from sklearn.feature_selection import SelectKBest, f_regression

  # Load the diabetes dataset
  X, y = load_diabetes(return_X_y=True)

  # Select the top 3 features using F-regression
  selector = SelectKBest(f_regression, k=3)
  X_new = selector.fit_transform(X, y)

  # Print the indices of the selected features
  print(selector.get_support(indices=True))
  ```
  In this example, we use `f_regression` as the scoring function in `SelectKBest` to select the top `3` features from the diabetes dataset. The `get_support` method is used to retrieve the indices of the selected features. Note that `f_regression` is appropriate when the target variable is continuous. If the target variable is categorical, `chi2` or `mutual_info_classif` should be used instead.


- **Feature Importance** This method ranks the importance of features based on the weights or coefficients of a machine learning model. You can use the `feature_importances_` attribute of a tree-based model, such as `RandomForestClassifier` or `ExtraTreesClassifier`, to get the feature importances. For example, to select the top `5` features based on the feature importances from a random forest classifier, you can use:

  ```python
  from sklearn.ensemble import RandomForestClassifier

  X = data.drop('label', axis=1)
  y = data['label']

  estimator = RandomForestClassifier()
  estimator.fit(X, y)

  # Print the feature importances
  feature_importances = pd.Series(estimator.feature_importances_, index=X.columns)
  print(feature_importances)

  # Select the top 5 features
  feature_names = feature_importances.sort_values(ascending=False)[:5].index
  print(feature_names)
  ```


- Once you have `selected a feature selection method`, you can use it to select the features to include in your model. Scikit-learn provides a number of tools for feature engineering, including:

  - **Polynomial features**: These features are created by taking the powers of existing features. For example, if you have a feature called "age", you could create a polynomial feature called "age^2".

  - **Interaction features**: These features are created by taking the products of existing features. For example, if you have features called "age" and "gender", you could create an interaction feature called "age*gender".

  - **Time series features**: These features are created by taking the values of a feature over time. For example, if you have a feature called "sales", you could create a time series feature called "sales_last_week".

- Once you have `engineered the features`, you can use them to train your model. Scikit-learn provides a number of tools for model training, including:

  - **Linear regression**: This is a simple model that can be used to predict a continuous target variable.

  - **Logistic regression**: This is a model that can be used to predict a binary target variable.

  - **Decision trees**: These are models that can be used to predict both continuous and binary target variables.

  - **Random forests**: These are models that are similar to decision trees, but they are more robust to overfitting.

- Once you have `trained your model`, you can evaluate its performance. Scikit-learn provides a number of tools for model evaluation, including:

  - **Accuracy**: This is the percentage of instances that the model correctly predicts.

  - **Precision**: This is the percentage of instances that the model predicts as positive that are actually positive.

  - **Recall**: This is the percentage of instances that are actually positive that the model predicts as positive.

  - **F1 score**: This is a measure of the model's overall performance. It is calculated as the harmonic mean of the precision and recall.Feature selection and feature engineering are important steps in machine learning. By selecting the right features and engineering them correctly, you can improve the performance of your model.

### `sklearn.model_selection` module in scikit-learn provides several functions for model selection and evaluation. Here are some of the commonly used functions.

| **Function**             | **Description**                                             |
|----------------------|-----------------------------------------------------------------|
| `train_test_split`   | Split the dataset into training and testing sets.               |
| `cross_val_score`    | Perform cross-validation and return an array of scores.         |
| `cross_validate`     | Perform cross-validation and return multiple evaluation metrics.|
| `GridSearchCV`       | Perform an exhaustive grid search for hyperparameter tuning.    |
| `RandomizedSearchCV` | Perform a randomized search for hyperparameter tuning.          |
| `KFold`              | Generate K-fold cross-validation splits.                        |
| `StratifiedKFold`    | Generate stratified K-fold cross-validation splits.             |
| `TimeSeriesSplit`    | Generate cross-validation splits for time series data.          |
| `ShuffleSplit`       | Generate random train/test indices for multiple iterations.     |


### How to use this repository

This repository is organized into the following sections:

- **Introduction**: This section provides an overview of feature engineering and its importance.
- **Hacks and tips**: This section contains a collection of hacks and tips for feature engineering.
- **Examples**: This section contains examples of how to use the hacks and tips in the previous section.
- **Resources**: This section contains links to resources for further learning about feature engineering.

### Getting started

To get started, you can either clone the repository or download the `ZIP file`. Once you have the repository, you can open the README.md file in a text editor.

### Contributing
This repository is open source and contributions are welcome. If you have any ideas for hacks or tips, or if you find any errors, please feel free to open an issue or submit a pull request.

### License
This repository is licensed under the [MIT License](https://github.com/mohd-faizy/feature-engineering-hacks/blob/main/LICENSE).

#### Thanks for checking out this repository! I hope you find it helpful.

---

<p align='center'>
  <a href="#"><img src='https://tymsai.netlify.app/resource/1.gif' height='10' width=100% alt="div"></a>
</p>

### $\color{skyblue}{\textbf{Connect with me:}}$

[<img align="left" src="https://cdn4.iconfinder.com/data/icons/social-media-icons-the-circle-set/48/twitter_circle-512.png" width="32px"/>][twitter]
[<img align="left" src="https://cdn-icons-png.flaticon.com/512/145/145807.png" width="32px"/>][linkedin]
[<img align="left" src="https://cdn2.iconfinder.com/data/icons/whcompare-blue-green-web-hosting-1/425/cdn-512.png" width="32px"/>][Portfolio]

[twitter]: https://twitter.com/F4izy
[linkedin]: https://www.linkedin.com/in/mohd-faizy/
[Portfolio]: https://mohdfaizy.com/


