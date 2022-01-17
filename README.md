# Project-Classification-Titanic_Machine_Learning

Data for passengers on the Titanic is provided and a machine learning classification model is written to predict the survival of a subset of the passengers
- Performed exploratory data analysis (EDA) on passenger data to find trends and inform feature engineering
- Employed hypothesis testing validate the statistical significance of engineered features
- Examined the performance of Logistic Regression, K-Neighbors, Decision Tree, and Random Forest Classifier models
- Used GridSearchCV in tandem with ShuffleSplit to optimize models and compare their performance against 30 derived training and validation data sets to increase accuracy while reducing over-fitting
- Created an aggregate classification model which averaged the survival probabilities determined by the previous models to create a new survival prediction for each passenger


# [Classification Project (Kaggle):  "Titanic - Machine Learning from Disaster"]
A classification model was developed to determine the outcome (survival or death) of passengers on the Titanic based on personal information such as the passenger gender, age, class, and other categorical and numerical variables.  Training and test data sets were provided by Kaggle.  The training data set was provided with passenger outcomes to train and fit various classification/machine learning algorithms.  The test data was provided without the passenger outcomes, and the developed model was used to predict their fates.<br>
### Major Accomplishments:
- Performed exploratory data analysis (EDA) on passenger data to find trends and inform feature engineering<br><br>
![](/Images/Classification_Titanic/Correlation_Matrix_small.png)<br><br>

- Employed hypothesis testing validate the statistical significance of engineered features<br><br>
![](/Images/Classification_Titanic/Age_Distro_Swarm_small.png)
![](/Images/Classification_Titanic/Survival_Ratio_vs_Cumulative_Age_Group.png)<br><br>
- Examined the performance of Logistic Regression, K-Neighbors, Decision Tree, and Random Forest Classifier models
- Used sklearn GridSearchCV to optimize models to increase model accuracy
- Sklearn ShuffleSplit was employed to generate training and validation sets reduce overfitting by simulating the effects of unseen data.  (Below, the "Test" column is actually the average performance for each model against the generated validation sets)<br><br>
![](/Images/Classification_Titanic/Model_Comparison_Table.png)<br>

### Key Outcomes:
- A random forest classifier model was chosen with a predicted accuracy of about 82% based on validation data
- The chosen random forest model predicted the test data with a 77.3% accuracy
- Next steps and model refinements are proposed in the code to improve this closer to the 82% prediction accuracy achieved on the validation data sets
