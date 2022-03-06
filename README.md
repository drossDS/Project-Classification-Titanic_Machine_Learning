<div align="center">
    <center><h1>Titanic Survival Model</h1></center>
</div>

<div align="center">
    <center><b><i>Implementing Classification Techniques to Predict Passenger Survival</i></b></center>
</div>

For the Kaggle competition, "Titanic - Machine Learning from Disaster," a classification model was developed to determine the outcome (survival or death) of passengers on the Titanic based on personal information such as the passenger gender, age, class, and other categorical and numerical variables.  Datasets were provided by Kaggle and included a training dataset with passenger outcomes model fitting and a test dataset on which to run the model and submit the results for the competition.<br>
### Major Accomplishments:
- Performed exploratory data analysis (EDA) on passenger data to find trends and inform feature engineering<br><br>

  ![](/Images/Classification_Titanic/Correlation_Matrix_small.png)<br><br>

- Employed hypothesis testing to validate the statistical significance of engineered features<br><br>
  ![](/Images/Classification_Titanic/Age_Distro_Swarm_small.png)
  ![](/Images/Classification_Titanic/Survival_Ratio_vs_Cumulative_Age_Group.png)<br><br>
- Examined the performance of Logistic Regression, K-Neighbors, Decision Tree, and Random Forest Classifier models
- Used sklearn GridSearchCV to optimize models to increase model accuracy
- Generated training and validation sets using sklearn ShuffleSplit to simulate the effects of unseen data and reduce overfitting<br><br>

![](Images/Classification_Titanic/Model_Comparison_Table.png)<br>

### Key Outcomes:
- A random forest classifier model was chosen with a predicted accuracy of about 82% based on validation data
- The chosen random forest model predicted the test data with a 77.3% accuracy
  - Next steps and model refinements are proposed in the code to improve this closer to the 82% prediction accuracy achieved on the validation datasets
