# Titanic Survival Model
***Implementing Classification Techniques to Predict Passenger Survival***

For the Kaggle competition, "Titanic - Machine Learning from Disaster," a classification model was developed to determine the outcome (survival or death) of passengers on the Titanic based on personal information such as the passenger gender, age, class, and other categorical and numerical variables.  Datasets were provided by Kaggle and included a training dataset with passenger outcomes model fitting and a test dataset on which to run the model and submit the results for the competition.<br>

### Major Accomplishments:
- Performed exploratory data analysis (EDA) on passenger data to find trends and inform feature engineering
- Employed hypothesis testing to validate the statistical significance of engineered features
- Examined the performance of Logistic Regression, K-Neighbors, Decision Tree, and Random Forest Classifier models
- Used sklearn GridSearchCV to optimize models to increase model accuracy
- Generated training and validation sets using sklearn ShuffleSplit to simulate the effects of unseen data and reduce overfitting<br><br>

### Key Outcomes:
- A random forest classifier model was chosen with a predicted accuracy of about 82% based on validation data
- The chosen random forest model predicted the test data with a 77.3% accuracy
  - Next steps and model refinements are proposed in the code to improve this closer to the 82% prediction accuracy achieved on the validation datasets

---

# Supplementary Details to Project Overview



## Feature Engineering
- Presence of age data
- Presence of Cabin data
- "Young" age data


## 


## Modeling
Four types of models were run on the provided data:
- Logistic Regression
- K-Neighbors
- Decision Tree
- Random Forrest

### Model Optimization
The general modeling strategy for optimizing each model type was executed as follows:
- Run a baseline model with default parameters to esablish an initial performance benchmark
- Perform a "Coarse Parameter Optimization" with gridsearchCV to run the model on every cobination of a wide range of hyperparameter values
- Perform a "Fine Tuning" optimization where the hyperparameter ranges are much smaller and centered around the best performing hyperparameter combinations from the coast optimization step
- The best performing hyperparameters from the fine tuning step were then used to run the final model for the paricular model type and output a final prediction of the validation data
- Repeat the above steps on all model types

### Model Evaluation with ShuffleSplit

