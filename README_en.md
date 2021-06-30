[中文ReadMe](https://github.com/jaaack-wang/gender-predicator/blob/main/README.md)
# Description
Predicting the gender of given Chinese names. Currently, Navie Bayes based method is provided (with over 93% prediction accuracy). 
Planning to try other algorithms (e.g., logistic regression) later on.

## Dataset
The dataset comes from the [Comprehensive Chinese Name Corpus (CCNC)](https://github.com/jaaack-wang/ccnc), which contains 
over 3.65 million Chinese names. This is [the script](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/train_dev_test_split.ipynb)
used to divide the dataset into train/dev/test sets. 

For Naive Bayes algorithms, the train and dev sets are combined together 
([script](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/char_gender_pair_converter.ipynb)) 
such that the train set: test set = 8: 2. 

# Naive Bayes 
The algorithm is stored in [gender.py](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/gender.py). You can 
check [this notebook](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/gender.ipynb) to see the source code and 
how it can be used. 

## Assumption
This method assumes the independence among different features (i.e., charecters) such that 
[a dictionary](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/data/dict4Gender.json) 
containing charecter-gender pair was made based on the combined training set. 

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
By default, the Gender().predict() function use Laplace smoothing method (`method='lap'`) to adjust the training set such that a certain 
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
To change the number of unseen characters, add a specified integer to Gender() when calling this class, e.g., `gender = Gender(1000)`. 
However, changing the unseen character number appears to make tiny differences for the predictions based on these two smoothing methods. 

## Accuracy
The accuracy is tested against the [test set](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/data/test_ds.txt) 
which contains over 700k unique full names or over 200k unique first names. You can see the deatiled accuracy test 
[here](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/test.ipynb). Basic statistics:
| Smoothing method | With gender undefined (U) names | Accuracy |
| :---: | :---: | :---: |
| Laplace | Yes | 93.7% |
| Laplace | No | 96.2% |
| Good Turing | Yes | 93.0% |
| Good Turing  | No | 95.5% |

It appears that both methods work better when the gender undefined examples are excluded from the test set. Moreover, Laplace method works slightly better than Good Turing method no matter whether the gender undefined names are excluded or not. 
