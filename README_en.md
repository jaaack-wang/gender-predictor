[中文ReadMe](https://github.com/jaaack-wang/gender-predicator/blob/main/README.md)
# Description
Predicting the gender of given Chinese names. Currently, Navie Bayes model and Multi-class Logistic Regression model are provided (with over 93% prediction accuracy). Moreover, both models are trained on the first names and full names. The full-names-trained models can be seen in [Full_Name_Models](https://github.com/jaaack-wang/gender-predicator/tree/main/Full_Name_Models) folder. 

The comparison between the first-names-trained models and the full-names-trained models can be found in [evaluation](https://github.com/jaaack-wang/gender-predicator/tree/main/evaluation) folder. Please note that, when calculating the prediction accuracies, **both types of models do not know whether the inputted names are first names or full names**, unlike the accuracies reported below which were calculated when the first-names-trained models know the types of names inputted (manually set).

## Basic findings

- When the models know the types of names inputted, the first-names-trained models outperform the full-names-trained counterparts by 1%~5% in accuracy. 
- When the types of names inputted are unknown, the full-names-trained models also outperform the first-names-trained models by 1%~5% in accuracy.
- The first-names-trained models determine the types of the inputted names by an inbuilt function that uses dictionary-based max-matching approach, which however is unrealiable and causes a great deal of variance because certain Chinese last names can also occur in first names. In this case, the prediction accuracy for full names dataset is 94%～96%, but only 84%～88% for first names dataset. 
- The full-names-trained models have better performances on first names dataset than full names dataset, which make sense because last names are genderless in the context of Chinese names. 
- In terms of accuracy, although the first-names-trained models are less stable and have greater variance, they have higher uppper limits than the full-names-trained models. Their accuracy is up to 97%, close to or even better than the human-level performance. 
- It follows, if a more accurate model that determines the name type (first or full name) can be trained, first-names-trained models are more promising. 


## Dataset
The dataset comes from the [Comprehensive Chinese Name Corpus (CCNC)](https://github.com/jaaack-wang/ccnc), which contains 
over 3.65 million Chinese names. This is [the script](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/train_dev_test_split.ipynb)
used to divide the dataset into train/dev/test sets. 

For Naive Bayes algorithms, the train and dev sets are combined together 
([script](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/char_gender_pair_converter.ipynb)) 
such that the train set: test set = 8: 2. 

For Multi-class Logistic Regression algorithms, the splitting of train/dev/test sets is a bit complicated. Given the computational cost, the model was actually only trained on 100k Chinese first names in the end. Please refer to [train_dev_test_split.ipynb](https://github.com/jaaack-wang/gender-predicator/blob/main/Multiclass_Logistic_Regression_Gender/train_dev_test_split.ipynb) and [model_training_testing.ipynb](https://github.com/jaaack-wang/gender-predicator/blob/main/Multiclass_Logistic_Regression_Gender/model_training_testing.ipynb) for see more details. 


# ONE. Naive Bayes 
The algorithm is stored in [gender.py](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/gender.py). You can 
check [this notebook](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/gender.ipynb) to see the source code and 
how it can be used. 

This method assumes the independence among different features (i.e., charecters) such that 
[a dictionary](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/data/dict4Gender.json) 
containing charecter-gender pairs was made based on the combined training set. 

## Usage
Download the repository, and then cd to the [Naive_Bayes_Gender](https://github.com/jaaack-wang/gender-predicator/tree/main/Naive_Bayes_Gender).

Examples:
```python
from gender import Gender

# call the Gender class
>>> gender = Gender()
# predicting the gender of a given name
>>> gender.predict('周小窗')
('周小窗',
 {'M': 0.43176377177947284,
  'F': 0.5681203560929395,
  'U': 0.0001158721275876369})
# predicting the gender of given names
# show_all=False returns only optimal predictions. 
>>> names = ['李柔落', '许健康', '黄恺之', '周牧', '梦娜', '爱富']
>>> gender.predict(names, show_all=False)
[('李柔落', 'F', 0.9413547727326725),
 ('许健康', 'M', 0.9945417378532947),
 ('黄恺之', 'M', 0.9298017602220987),
 ('周牧', 'M', 0.7516425755584757),
 ('梦娜', 'F', 0.9995836802664445),
 ('爱富', 'M', 0.9534883720930233)]
```
By default, the `Gender().predict()` function use Laplace smoothing method (`method='lap'`) to adjust the training set such that a certain 
probablity is given to unseen charecters, which defaults to 5000. To use Good-Turing smoothing method, add `"method=gt"` as follows:
```python
from gender import Gender
>>> gender = Gender()
>>> gender.predict('周小窗', method='gt')
('周小窗',
 {'M': 0.4734292530195843,
  'F': 0.5265704585272836,
  'U': 2.8845313211183217e-07})
```
To change the number of unseen characters, add a specified integer to `Gender()` when calling this class, e.g., `gender = Gender(1000)`. 
However, changing the unseen character number appears to make tiny differences for the predictions based on these two smoothing methods. 

## Accuracy
The accuracy is tested against the [test set](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/data/test_ds.txt) 
which contains over 700k unique full names or over 200k unique first names. You can see the deatiled accuracy test 
[here](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/test.ipynb). Basic statistics (the accuracy score for the train set is comparable, see: [train_set_accuracy.ipynb](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/train_set_accuracy.ipynb)):
| Smoothing method | With gender undefined (U) names | Accuracy |
| :---: | :---: | :---: |
| Laplace | Yes | 93.7% |
| Laplace | No | 96.2% |
| Good Turing | Yes | 93.0% |
| Good Turing  | No | 95.5% |

It appears that both methods work better when the gender undefined examples are excluded from the test set. Moreover, Laplace method works slightly better than Good Turing method no matter whether the gender undefined names are excluded or not. 

# TWO. Multi-class Logistic Regression 

The model is trained based on first names, which are converted into word vectors using one-hot encoding. The word vector also presumes an unseen character that will be used to represent all unseen characters. The gender labels, on the other hand, are encoded with numberical values, 0, 1, 2 to represent M, F and U (undefined) respectively. 

The detailed model training and testing process can be seen in [model_training_testing.ipynb](https://github.com/jaaack-wang/gender-predicator/blob/main/Multiclass_Logistic_Regression_Gender/model_training_testing.ipynb).

## Usage 

Download the repository, and then cd to the [Multiclass_Logistic_Regression_Gender](https://github.com/jaaack-wang/gender-predicator/tree/main/Multiclass_Logistic_Regression_Gender). It is essentially similar to the Naive Bayes Model. The differences are that it does not has `method` parameters in `predict` function and in addition it includes an `accuracy` function. 

Examples:
```python
from genderLR import GenderLR
>>> gender = GenderLR()
>>> gender.predict('周小窗')
('周小窗',
 {'M': 0.9951092205495345,
  'F': 0.004890690420910213,
  'U': 8.902955511871448e-08})
# for a list of names
>>> names = ['李柔落', '许健康', '黄恺之', '周牧', '梦娜', '爱富']
>>> gender.predict(names, show_all=False)
[('李柔落', 'F', 0.7406189716495426),
 ('许健康', 'M', 0.9999990047182503),
 ('黄恺之', 'M', 0.9985069564065047),
 ('周牧', 'M', 0.9939343959114006),
 ('梦娜', 'F', 0.9999985293819316),
 ('爱富', 'M', 0.9655679649000578)]
```
The use of `accuracy`:
```python
# first define a list of example = [list1, list2....] where list = [name, gender]
from genderLR import GenderLR
>>> gender = GenderLR()
>>> examples = [['李柔落', 'F'], ['许健康', 'M'], ['黄恺之', 'M'], ['周牧', 'U'], ['梦娜', 'F'], ['爱富', 'M']]
>>> gender.accuracy(examples)
0.8333333333333334
# To see the mismatched case
>>> gender.mismatch
[['name', 'gender', 'pred', 'prob'], 
['周牧', 'U', 'M', 0.9939343959114006]]
```

## Accuracy
Refer to [model_training_testing.ipynb](https://github.com/jaaack-wang/gender-predicator/blob/main/Multiclass_Logistic_Regression_Gender/model_training_testing.ipynb) for details.

| Dataset | train | train | dev1 | dev1 | dev2 | dev2 | test | test |
| :---: | :---: | :---: |:---: | :---: | :---: |:---: | :---: | :---: |
| Include Undefined gender | Y | N | Y | N  | Y | N  | Y | N  |
| Accuracy | 96.2% | 98.0%  | 94.0% | 95.2% | 94.8%  | 96.5% | 94.6% | 97.0% |

