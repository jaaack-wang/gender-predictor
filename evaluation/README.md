Currently, Navie Bayes model and Multi-class Logistic Regression model are provided. Both models were trained on the first names and full names. Please note that, 
when calculating the prediction accuracies, both models do not know whether the inputted names are first names or full names or even fixed. 

The detailed comparisons of both models trained on first names and full names can be found in these two files: [naive_bayes.xlsx](https://github.com/jaaack-wang/gender-predicator/blob/main/evaluation/naive_bayes.xlsx) 
and [logistic_regression.xlsx](https://github.com/jaaack-wang/gender-predicator/blob/main/evaluation/logistic_regression.xlsx). The process of the 
prediction calculation can be found in [Full_Name_Models](https://github.com/jaaack-wang/gender-predicator/tree/main/Full_Name_Models) folder.

## ONE. Naive Bayes
Findings: 
- The first-names-trained bayes model outperform the the full-names-trained one by around 2% in accuracy when full names dataset are given. However, 
the full-names-trained model has 1%~5% higher prediction accuracies than the first-names-trained model for both first names dataset and mixed dataset. 

<p align="center">
<img width='700' height='400' src="https://github.com/jaaack-wang/gender-predicator/blob/main/evaluation/bayes_model_comp.png">
</p>

- For both types of models, Laplace smoothing method constantly works slightly better (about 0.6%) than the Good Turing smoothing method as a means to deal with unseen characters, as can be 
seen in [naive_bayes.xlsx](https://github.com/jaaack-wang/gender-predicator/blob/main/evaluation/naive_bayes.xlsx). 
- For both types of models, when gender undefined examples are filtered out, both models will have about 2% improvement of accuracy. See 
[naive_bayes.xlsx](https://github.com/jaaack-wang/gender-predicator/blob/main/evaluation/naive_bayes.xlsx) for details.

## TWO. Multi-class Logistic Regression
Findings: 
- The first-names-trained bayes model outperform the the full-names-trained one by around 2% in accuracy when full names dataset are given. However, 
the full-names-trained model has 4%~6% higher prediction accuracies than the first-names-trained model for both first names dataset and mixed dataset. 


<p align="center">
<img width='700' height='400' src="https://github.com/jaaack-wang/gender-predicator/blob/main/evaluation/lr_model_comp.png">
</p>

- For both types of models, when gender undefined examples are filtered out, both models will have about 1%~3% improvement of accuracy. See 
[logistic_regression.xlsx](https://github.com/jaaack-wang/gender-predicator/blob/main/evaluation/logistic_regression.xlsx) for details.
