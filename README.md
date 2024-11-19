# Jupyter_Notebooks

**AdClick_Gender_Classification.ipynb**
This notebooks  contains 10 functions which are :

1. **load_and_preprocess_data(file_path)** : It takes one argument file_path and return preprocessed data. The preprocessing involves :

   -- removing any characters like spaces from column name and replacing it with underscore ('_').

   -- Making sure 'Age' is an absolute value integer value.

   -- Dealing with missing values with median or mean depending upon whether the data is skewed or not. If skewness is more than equal to 0.5 the missing values are filled with median otherwise we use mean.

   -- Encode categorical variables.

   -- For categorical data, replace missing values with the most frequent value

2. **split_data(data, features, target)** : It takes three arguments and uses the 'train_test_split' function of SKlearn to split the data into training and test sets.

3. **build_pipeline(model_name)** : Take one argument and create a pipeline. It tells the code to normalize the values before training the model which user wants to use.

4. **train_and_evaluate_pipeline(pipeline, X_train, X_test, y_train, y_test)** : Takes in 5 arguments and train and evaluate the model. It returns back predicted                                                                                                                                    values along with their probabilities and also return all the metrics accuracy, precision, recall, F1-score, roc_auc.

5. **plot_confusion_matrix(y_test, y_pred, model_name,target_name, title="Confusion Matrix")**: Plot and saves the confusion matrix using seaborn before and after hyper-                                                                                                                                                     parameter tuning.

6. **plot_roc_curve(y_test, y_proba,model_name,target_name, title="ROC Curve")**: Plot and saves the ROC curve using matplotlib before and after hyper-parameter tuning.

7. **tune_hyperparameters(X_train, y_train, model_name)** : Perform hyperparameter tuning using GridSearchCV.

8. **compare_metrics(before_metrics, after_metrics, model_name)** : Returns a bar-plot comparing the values of metrics after and before hyper-parameter tuning.

9. **save_metrics_to_file(before_metrics, after_metrics, model_name, target_name, directory='./')** : Save metrics to a file with the model name in the filename

10.  **save_model(pipeline, model_name, target_name, directory='./')** : Save model as .pkl files.

**Model inputs**
 -For now it takes two models as input namely Naive Bayes Classifier and SVM.
 -The code can be modified to add different models.
 
**Traget Values**
  - Female (0) and Male (1)
  - Didn't clicked on Ad (0) and Clicked on Ad (1)

**Features Values used**
  - Daily Time Spent on site
  - Age
  - Area Income
  - Daily internet usage

 **Model files**

   - Generate separate models for each Target values
   - Since we have 2 target values here and 2 models options, a total of 4 model files are generated

## Inference_file.ipynb
   - Calls the 4 model file and generate 4 model inference reports in .json files.
