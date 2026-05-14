# Credit-Card-Fraud-Detection-using-Machine-Learning-and-SMOTE
Built a Machine Learning model to detect fraudulent credit card transactions using Logistic Regression. Applied SMOTE to handle class imbalance and evaluated model performance using precision, recall and F1-score.

## Objective
The objective of this project is to build a Machine Learning model that can identify fraudulent credit card transactions.<br>
Since fraud datasets are usually highly imbalanced, the project also focuses on improving fraud detection performance using SMOTE (Synthetic Minority Oversampling Technique).

## Dataset

The dataset consists of credit card transaction records with multiple anonymized features.
Target Column → Class<br>
0 represents Normal Transaction,

1 represents Fraud Transaction,

The dataset is highly imbalanced, with very few fraud transactions compared to normal ones.

## Tools & Technologies Used
-Python <br>
-Pandas <br>
-NumPy <br>
-Matplotlib <br>
-Seaborn <br>
-Scikit-learn <br>
-Imbalanced-learn (SMOTE)

 ## Exploratory Data Analysis:
 
Exploratory Data Analysis was performed to understand the distribution of transactions.<br>
Visualizations such as count plots, histograms, and correlation heatmaps were used.<br>
The analysis showed that the dataset is highly imbalanced, which can affect model performance.

## Model Building Without SMOTE

-A Logistic Regression model was first trained on the original imbalanced dataset.<br>
-The model achieved high accuracy in predicting normal transactions but failed to detect many fraud cases.<br>
-This indicates that the model was biased towards the majority class.

## Model Building With SMOTE

-To handle class imbalance, SMOTE was applied only on the training dataset to generate synthetic fraud samples.<br>
-The Logistic Regression model was retrained on the balanced dataset, which improved the model’s ability to detect fraud transactions.<br>
-After applying SMOTE, the model became significantly more effective in identifying fraudulent transactions.

## Model Evaluation

-Model performance was evaluated using:<br>
-Confusion Matrix <br>
-Precision  <br>
-Recall <br>
-F1 Score <br>
-Accuracy alone was not considered reliable due to class imbalance. <br>
In fraud detection problems, Recall is considered more important because failing to identify fraudulent transactions can lead to major financial losses.

## Result Comparison

After applying SMOTE, the model showed improved recall for the fraud class, meaning it was able to detect more fraudulent transactions.<br>
Although overall accuracy slightly changed, the model became more useful for real-world fraud detection.

 ## Conclusion
This project demonstrates the importance of handling imbalanced datasets in fraud detection problems.<br>
By applying SMOTE and evaluating the model using appropriate metrics, the fraud detection capability was improved.<br>
The project also highlights the complete Machine Learning workflow including preprocessing, visualization, imbalance handling, model training, and evaluation.
