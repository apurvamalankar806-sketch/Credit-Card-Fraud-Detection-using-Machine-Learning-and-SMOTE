# Credit-Card-Fraud-Detection-using-Machine-Learning-and-SMOTE
Built a Machine Learning model to detect fraudulent credit card transactions using Logistic Regression. Applied SMOTE to handle class imbalance and evaluated model performance using precision, recall and F1-score.

## Objective
The objective of this project is to build a Machine Learning model that can identify fraudulent credit card transactions.
Since fraud datasets are usually highly imbalanced, the project also focuses on improving fraud detection performance using SMOTE (Synthetic Minority Oversampling Technique).

## Dataset

The dataset consists of credit card transaction records with multiple anonymized features.
Target Column → Class
0 represents Normal Transaction
1 represents Fraud Transaction
The dataset is highly imbalanced, with very few fraud transactions compared to normal ones.

## Tools & Technologies Used
Python
Pandas
NumPy
Matplotlib
Seaborn
Scikit-learn
Imbalanced-learn (SMOTE)

 ## Exploratory Data Analysis:
 
Exploratory Data Analysis was performed to understand the distribution of transactions.
Visualizations such as count plots, histograms, and correlation heatmaps were used.
The analysis showed that the dataset is highly imbalanced, which can affect model performance.

## Model Building Without SMOTE
A Logistic Regression model was first trained on the original imbalanced dataset.
The model achieved high accuracy in predicting normal transactions but failed to detect many fraud cases.
This indicates that the model was biased towards the majority class.

## Model Building With SMOTE

To handle class imbalance, SMOTE was applied only on the training dataset to generate synthetic fraud samples.
The Logistic Regression model was retrained on the balanced dataset, which improved the model’s ability to detect fraud transactions.

## Model Evaluation

Model performance was evaluated using:
Confusion Matrix
Precision
Recall
F1 Score
Accuracy alone was not considered reliable due to class imbalance.
Recall was given more importance since missing fraud transactions can lead to financial losses.

## Result Comparison

After applying SMOTE, the model showed improved recall for the fraud class, meaning it was able to detect more fraudulent transactions.
Although overall accuracy slightly changed, the model became more useful for real-world fraud detection.

 ## Conclusion
This project demonstrates the importance of handling imbalanced datasets in fraud detection problems.
By applying SMOTE and evaluating the model using appropriate metrics, the fraud detection capability was improved.
The project also highlights the complete Machine Learning workflow including preprocessing, visualization, imbalance handling, model training, and evaluation.
