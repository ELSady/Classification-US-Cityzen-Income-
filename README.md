## Data Science Project Classification : United States Cityzen Income Classification Overview
* Build a machine learning model to classify United States citizen's income falls into below 50K or above 50K a year.
* Uniterd States records of over 30.000 cityzen.
* Feature Engineered several features for dataset to be easier to read and be inplemented by machine learning model.
* Data Exploration helps giving insight the distribution of `Income` category based on `Education, Occupation, Workclass`. and how much cityzen having an income based on them `Marital-Status` or `Relationship`.
* Using `Tree-based algorithm's feature importance` to determine which features contributes the most in terms of customer churn.
* Optimized Tree-based algorithm to output higher score Accuracy, F1, Precision, Recall

### Code and Resources Used
* **Packages** : pandas, numpy, matplotlib, seaborn, sci-kit learn, shap, yellowbrick, lightgbm.

### Data Cleaning
* Features / column name were rewritten and renamed as some all features have the blank space as first letter. Apply for all features.
* Missing value were non-existence, however there are features with _anomaly_ value. These features include `Workclass, Occupation and Native-country`
* Cleaning process begins first with replacing those _anomaly_ values to `np.Nan`.
* Replaced _anomaly_ with `np.Nan` then further replaced with either value with the most frequent occuring in respective feature.
* Cross checking dataset with ASSERT statement to ensure dataset had been cleaned.

### Exploratory Data Highlight
Frequency Distribution of Income per Education

![alt text](https://github.com/ELSady/Classification-US-Cityzen-Income-/blob/main/index1.png)

* The income distribution per `Education` basis, can be seen many cityzen earn below 50 K a year. WIth High-school graudate dominates followed by bachelors gradute and college student who have a part time job aside studying for the top 3 in this category On the other non minority side, interestingly some bachelors and high schoolgradute are able to earn above 50 K a year even on board college students are too. Perhaps the so called `side hustle` is indeed a thing in this modern age.

Frequency Distribution of Income Category per Occupation and Workclass

![alt text](https://github.com/ELSady/Classification-US-Cityzen-Income-/blob/main/index2.png)

* Similar to that of per `Education` basis distribution, here we can also see that below 50 K earners are still dominating the distribution. With cityzen who worked as Clerical administration, Craft-repair and other work related services have the highest count of workers amongst other occupation in this category. In contrast to previous statement, for above 50 K earners are dominated by Exective managers and Proffesors / Specialist occupied cityzen. 

* Distributon per `WOrkclass` basis, clearly seen for both categories, Private business / company dominates the chart. US cityzens are mostly entrepreneurs.

Frequency Distribution of Income Category per Marital-status and Relationship

![alt text](https://github.com/ELSady/Classification-US-Cityzen-Income-/blob/main/index3.png)

* Per `Marital-status` basis, the income distribution as for below 50 K earner dominates by either Married or non married cityzen. Well its no surprise, but interesting thing is here, the divorced couple counts has is almost as many as other status counts. While for above 50 K earners, married cityzen having the hihgest count of distribution. The disparity count is very very clear.

* As for per `Relationship` basis, one thing in common for both caegories, Husbands are  te family's backbone which is natural. Putting aside that fact, below 50 K earners are dominated by singled status cityzen. While for above 50 K earners, refer to the facts said in previous senetence.

### Data Preparation Before Model Implementation
* Imbalanced target feature `Income category` checking. This is to ensure which scoring parameter is best used for dataset.
* Train and Test splitting with a proportion of 75% Train and 25% Test.
* Standardizing for numerical data using Robust scaler and Onehot encoder for categorical features.

### Model Building 
* Tree-based algorithm model were used as they can plot features importances and gives weight to features and determines which ones are contribute leading up to customer churn.
* 3 Model used are `Random Forest, Stochastic Gradient Boosting, LightGBM.`

### Model Performance
* Random Forest
```
*****************Train*******************
Train Accuracy 0.9999590499590499
Train Precision 1.0
Train Recall 0.9998303071440693
Train F1 Score 0.9999151463725074
****************************************
******************Test*******************
Test Accuracy 0.8584940425009213
Test Precision 0.7406287787182587
Test Recall 0.6288501026694046
Test F1 Score 0.6801776790671848
*****************************************
```

* Stochastic Gradient Boosting
```
*****************Train*******************
Train Accuracy 0.8685094185094185
Train Precision 0.7891332470892626
Train Recall 0.621075852706601
Train F1 Score 0.6950906846453329
****************************************
******************Test*******************
Test Accuracy 0.8672153298120624
Test Precision 0.7787781350482315
Test Recall 0.6216632443531828
Test F1 Score 0.6914073651156152
*****************************************
```

* LightGBM
```
*****************Train*******************
Train Accuracy 0.8891482391482392
Train Precision 0.8144492696407422
Train Recall 0.7001527235703376
Train F1 Score 0.7529884113514006
****************************************
******************Test*******************
Test Accuracy 0.8779019776440241
Test Precision 0.7846062052505967
Test Recall 0.6750513347022588
Test F1 Score 0.7257174392935983
*****************************************
```
* Of all 3 tested models, `Lightgbm` has the edge ove the other 2 models. Indicated by its higher number of prediction scores of `Accuracy, F1, Precision and Recall`. This model will be used as base to build the optimized model later on.
* The Optimized model were build upon this LightGBM model. Optimization are done to reduce basically non performing features, features which contribute less or even dont contributes at all to prediction scores. The value of these can bee seen using model integrated's feature importance.

### Best Model Confusion Matrix and Classification Reports
* Confusion Matrix

![alt text](https://github.com/ELSady/Classification-US-Cityzen-Income-/blob/main/index.png)

* Classification Reports 
```
              precision    recall  f1-score   support

           0       0.90      0.94      0.92      6193
           1       0.78      0.68      0.73      1948

    accuracy                           0.88      8141
   macro avg       0.84      0.81      0.82      8141
weighted avg       0.87      0.88      0.87      8141
```



