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


## Inital Data, Exploratory Analysis, and Feature Engineering:

A training dataset was provided for 891 passengers aboard the Titanic with the following features for each pasenger in the inital dataset.  The table below provides the each feature along with information on whether or not it was used in the model, and the rationale behind that decision as later informed by exploratory analysis and feature engineering.

| Provided Feature | Used/Dropped | Ratioonale |
| --- | --- | ---|
| Passenger ID | Dropped | Arbitrary value irrelevant to survival |
| Survived | Used | Necessary to know outcome for each passenger when traiing models |
| Passenger Class (Pclass) | Used | Proved to be a significant indicator of survival |
| Passenger Name | Dropped | Thought to be irrelevant to survival |
| Passenger Sex | Used | Proved to be a significant indicator of survival |
| Passenger Age | Used | Some features extracted from age data were found to be useful |
| No. Siblings/Spouses Aboard (SibSp) | Used | When applicable, this information correlated with survival |
| No. Parents/Children Aboard (Parch) | Used | When applicable, this information correlated with survival |
| Ticket | Dropped | Thought to be arbitrarty data.  Cabin data/class thought to be better survival |
| Fare | Dropped | Thought to be function of cabin and class data which were better survival indicators |
| Cabin | Used | Correlation data showed cabin assignment positively correlated with survival |
| Embarkment Location | Used | Passengers from certain locations had higher chances of survival than others |

Note that while it may have been possible to analyze the ticket, name, and fare data to engineer features that would correlate to survival outcomes, these activities weren ot chosen to be within the scope of this project, instead opting for simpler apporaches.  A future iteration of this project should include these feature engineer activities.


### Exploratory Data Analysis and Feature Engineering
The following obsevations were made of the provided features with supporting plots below:
- ***Sex***: Men had a much higher mortailty rate than women
- ***Passenger Class***: Chances of survival decreased with increasing class with the majority of first class passengers surviving and the majorioty of third class passengers perishing.  Due to this trend, it was ultiimately decided that the pasenger class could be treated as a numerical variable despite being categorical in nature
- ***Parch (parents and children) and SibSp (Siblings and spouses)***: From the plots below it can be seen that passengers with no parents, children, siblings, or spouses tended to represent the maojority of the two categories and had much higher mortailty rates.  With this, it was decided that this data would be useful.  While these features were treated as numerical variables in this project, a future iteration should probably treat them as cetagorical as both Parch and SibSp are actually a combination of two distinctly separate features

******* Show the 4 plots above here

The following features were created from the provided data:
- ***Presence of Cabin data***:  While only 204 of the 891 passengers were have recorded cabin assignments, being assigned a cabin was asssociated with a much higher survival rate.  Thus a binary "Cabin_data" feature was introduced to the training data
- ***Passenger Age***: Only 714 of the 891 passengers had recorded age data. This particular ffeature was used in different ways for certain models:
  - A binary feature (Age_data) was introduced to indicate the presence of age data or a lack thereof
  - When examining age data, it could be seen that younger passengers appeared to have a higher chance or survival. Analysis was performed to investigate how young a passenger needed to be in order to have a disticntly higher chance of survival. To do this, the ratio of passengers from a cumulative age group (ages 0 to n) that survived to those from that same age group that perished was plotted.  It could be seen that age nine was the age which all younger passengers appeared to be twice as likely to survive.  From this, a binary feature named "Young" was introduced for passengers whose recorded age was under nine years old.  Passengers with ages recorded to be younger than 9 were ecnoded as a 1, while older passengers and passngers without recorded age data were encoded as 0.  To verify the statistical significance of this finding, a probability mass fucntion was used to determine the likelyhood that 38 out of the 62 passengers below age nine would survive when the death rate of all titanic passengers was 68 percent assuming that all passengers had an equal chance or survival.  With the chances of this happening being 1.5/1,000,000 it was determined that the null hypothesis that passengers of all ages having an equal chance of survival could be rejected and that there was a significant correlation between a young age and survival

***********Show some plots here

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

