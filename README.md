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


## Inital Data Processing:

Data were provided with the following features for each pasenger in the inital dataset.  The table *******


| Provided Feature | Used/Dropped | Ratioonale |
| --- | :-: | ---|
| Passenger ID | Dropped | Arbitrary value unimportant to analysis |
| Survived | Used | Necessary to know outcome for each passenger when traiing models |
| Passenger Class (Pclass) | Used | Proved to be a significant indicator of survival |
| Passenger Name | Dropped | Creating a model to analyze passenger names and correlate with surivival was outside of hte scope of this project |
| Sex | Used | Proved to be a significant indicator of survival |
| Age | Used | Some features could be extracted from age data, and values imputed in the place of null values to improve model performance |
| Number of Siblings or Spouses Aboard (SibSp) | Used | When applicable, this information correlated with survival |
| Number of Parents or Children Aboard (Parch) | Used | When applicable, this information correlated with survival |
| Ticket | Dropped | Cabin data, and class were thought to be better indiciators of survival than potentially arbitrary ticket data.  Analysis of ticket data was outside of the scope of this project |
| Fare | Dropped | This information was thought to be the function of the Cabin and Pclass data which would better indicators of survival |
| Cabin | Used | Correlation data showed that having a cabin assignment was one of the features most positively corerlated with survival |
| Location of Embarkment | Used | Passengers from certain locations had higher chances of survival than others |


 




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

Initially, a logisitic regression model was run with default values

### Model Optimization
The general modeling strategy for optimizing each model type was executed as follows:
- Run a baseline model with default parameters to esablish an initial performance benchmark
- Perform a "Coarse Parameter Optimization" with gridsearchCV to run the model on every cobination of a wide range of hyperparameter values
- Perform a "Fine Tuning" optimization where the hyperparameter ranges are much smaller and centered around the best performing hyperparameter combinations from the coast optimization step
- The best performing hyperparameters from the fine tuning step were then used to run the final model for the paricular model type and output a final prediction of the validation data
- Repeat the above steps on all model types

### Model Evaluation with ShuffleSplit

