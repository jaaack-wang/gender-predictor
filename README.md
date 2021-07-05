[English ReadMe](https://github.com/jaaack-wang/gender-predicator/blob/main/README_en.md)
# 说明
目前用了朴素贝叶斯和逻辑回归两个模型来预测中国姓名的性别（总体上93%以上的预测准确率），并且两个模型都是基于名字 (不包括姓) 来训练的。
之后也会考虑其他算法 (如：浅层神经网络) 来尝试做性别预测或者用全名来重新训练模型。

## 数据
实验数据来自内含365万姓名语例的[大型中文姓名语料库](https://github.com/jaaack-wang/ccnc)。该语料库被分为训练集/测试集/预测集
（[代码](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/train_dev_test_split.ipynb)），
但是对于基于朴素贝叶斯的算法，测试集是不需要的，所以训练集和测试集被合二为一
([代码](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/char_gender_pair_converter.ipynb))。
训练集和测试集的比例在此为8比2。逻辑回归模型的训练集/测试集/预测集的划分比较复杂，详情请见 [train_dev_test_split.ipynb](https://github.com/jaaack-wang/gender-predicator/blob/main/Multiclass_Logistic_Regression_Gender/train_dev_test_split.ipynb) 和 [model_training_testing.ipynb](https://github.com/jaaack-wang/gender-predicator/blob/main/Multiclass_Logistic_Regression_Gender/model_training_testing.ipynb) 两个文件。

## 问题
由于两个模型都是基于名字来训练的，所以确定输入的name是否是全名的准确度，会影响模型的准确度。目前两个模型都通过自建的基于字典的最大匹配方法来切分名字，但是这个方法有一定的缺陷。由于中国姓氏众多，不少人的名字里面也可能含有姓氏关键字，因此会被误切从而影响模型的预测。大体上来看，如果建测试集的名字当作全名来处理，模型整体准确度会下降7%左右。

这个问题有两种解决方案：
- 用全名来训练两个模型，而不仅仅是名字。
- 提高姓名区分方法的准确度（在字典和最大匹配上添加概率模型？）。

# 一、朴素贝叶斯
算法主要见于[gender.py](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/gender.py)。后者，可以点击
这个[notebook](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/gender.ipynb)来查看源码和使用方法。

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
基本数据如下：（模型在训练集上的准确度与此类似，详见：[train_set_accuracy.ipynb](https://github.com/jaaack-wang/gender-predicator/blob/main/Naive_Bayes_Gender/train_set_accuracy.ipynb)）
| 平滑方法 | 是否包括未知性别 | 准确度 |
| :---: | :---: | :---: |
| 拉普拉斯 | 是 | 93.7% |
| 拉普拉斯 | 否 | 96.2% |
| 古德图灵 | 是 | 93.0% |
|古德图灵 | 否 | 95.5% |

结果显示，当未知性别没被包括时，两个平滑方法的预测准确度都有小幅度提高，但是拉普拉斯平滑方法总的来说比古德图灵平滑方法稍微更准确些。

# 二、逻辑回归
模型基于名字训练而成。名字用独热编码的方法转化为词向量 (word vector)，性别则简单地用0，1，2 来分别代表“男”，“女”和“未知”三个性别类别。词向量也预设了一个未见字的存在，来应对所有的未见字输入。

模型训练过程和解释详见 [model_training_testing.ipynb](https://github.com/jaaack-wang/gender-predicator/blob/main/Multiclass_Logistic_Regression_Gender/model_training_testing.ipynb)。另外，这里的逻辑回归更准确地说是多分类逻辑回归 (Multi-class Logistic Regression).

## 使用方法
下载该库，并且cd到[Multiclass_Logistic_Regression_Gender](https://github.com/jaaack-wang/gender-predicator/tree/main/Multiclass_Logistic_Regression_Gender)文件夹内.基本使用方法和上述的朴素贝叶斯模型相似，只是`predict`里面没有`method`变量，另外又多了一个`accuracy`的函数：

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
Accuracy的使用方法如下：
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

## 准确度

详见 [model_training_testing.ipynb](https://github.com/jaaack-wang/gender-predicator/blob/main/Multiclass_Logistic_Regression_Gender/model_training_testing.ipynb)。

| 数据集 | 训练集 | 训练集 | 测试集1 | 测试集1 | 测试集2 | 测试集2 | 预测集 | 预测集 |
| :---: | :---: | :---: |:---: | :---: | :---: |:---: | :---: | :---: |
| 是否包括未知性别 | 是 | 否 | 是 | 否 | 是 | 否 | 是 | 否 |
| 准确度 | 96.2% | 98.0%  | 94.0% | 95.2% | 94.8%  | 96.5% | 94.6% | 97.0% |
