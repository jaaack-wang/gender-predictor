[English ReadMe](https://github.com/jaaack-wang/gender-predicator/blob/main/README_en.md)
# 说明
目前用了朴素贝叶斯来预测中国姓名的性别（总体上93%以上的预测准确率）。之后也会考虑使用逻辑回归之类的算法来尝试做性别预测。

## 数据
实验数据来自内含365万姓名语例的[大型中文姓名语料库](https://github.com/jaaack-wang/ccnc)。该语料库被分为训练集/测试集/预测集
（[代码](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/train_dev_test_split.ipynb)），
但是对于基于朴素贝叶斯的算法，测试集是不需要的，所以训练集和测试集被合二为一
([代码](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/char_gender_pair_converter.ipynb))。
训练集和测试集的比例在此为8比2。

# 朴素贝叶斯
算法主要见于[gender.py](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/gender.py)。后者，可以点击
这个[notebook](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/gender.ipynb)来查看源码和使用方法。

## Assumption
该方法预设不同特征（这里为“字”）间相互独立，所以我用合并的测试集制作了一个字和性别对应的
[匹配字典](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/data/dict4Gender.json)。

## 使用方法
下载该库，并且cd到[Naive_Bayes_Gender](https://github.com/jaaack-wang/gender-predicator/tree/main/Naive_Bayes_Gender)文件夹内.

例子:
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
Gender().predict()方法默认拉普拉斯平滑方法 (`method='lap'`)来平滑测试集的数据，所以一些未见字也会被分与一定的概率。未见字的总字数默认为5000字（这个不重要，见下）。
通过在Gender().predict()方法内添加`"method=gt"`可以使用古德图灵平滑方法，比如:
```python
from gender import Gender
>>> gender = Gender()
>>> gender.predict('周小窗', method='gt')
('周小窗',
 {'M': 0.4734292530195843,
  'F': 0.5265704585272836,
  'U': 2.8845313211183217e-07})
```
为了改变未见字的数量，可以在Gender() 内注明想要的数量，比如： `gender = Gender(1000)`。但是，改变未见字的数量实际上对预测的具体结果没有很大的影响。

## 准确度
准确度是在测试集[test set](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/data/test_ds.txt) 上测得的。
测试集上内含70万以上不同的全名，或者20万以上不同的名字。详细的测试过程见[此](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/test.ipynb)。
基本数据如下：
| 平滑方法 | 是否包括未知性别 | 准确度 |
| :---: | :---: | :---: |
| 拉普拉斯 | 是 | 93.7% |
| 拉普拉斯 | 否 | 96.2% |
| 古德图灵 | 是 | 93.0% |
|古德图灵 | 否 | 93.0% |

结果显示，拉普拉斯平滑方法比古德图灵平滑方法更准确，尤其是当未知性别没有被包括的时候。
