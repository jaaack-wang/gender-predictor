'''
Author: Zhengxiang (Jack) Wang 
Date: 2021-07-04
GitHub: https://github.com/jaaack-wang 
About: Multi-class logistic regression classifier that predicts gender of Chinese names.
'''


from utils import *
import numpy as np


class GenderLR:
    
    def __init__(self):
        self._theta = np.load('data/params.npy')
        self._dic = load_char_dic()
        self.mismatch = 'You should run accuracy() first.'
        
    def _sigmoid(self, z):
        return 1/(1+np.exp(-z))
    
    def predict(self, name, show_all=True, full_name=False):
        def run(name):            
            if not full_name:
                X = name2vec(name, self._dic)
            else:
                fname = getFirstName(name)
                X = name2vec(fname, self._dic)
            prob = self._sigmoid(np.squeeze(X @ self._theta))
            prob = prob/np.sum(prob)
            if show_all:
                res.append((name, {'M': prob[0], 'F': prob[1], 'U': prob[2]}))
            else:
                M, F, U = prob
                if M==F and F==U: res.append((name, 'M=F=U', M))
                elif M == np.max(prob): res.append((name, 'M', M))
                elif F>U: res.append((name, 'F', F))
                else: res.append((name, 'U', U))
        
        res = []
        if isinstance(name, str):
            run(name)
            return res[0]
        elif isinstance(name, list):
            for n in name:
                run(n)
            return res
        
    def accuracy(self, examples, exclude_U=False, full_name=False):
        right = 0
        mismatch = [['name', 'gender', 'pred', 'prob']]
        smp_sz = len(examples)
        if not exclude_U:
            for example in examples:
                name, gender = example
                _, pred, prob = self.predict(name, show_all=False, full_name=full_name)
                if gender == pred: right += 1
                else: mismatch.append([name, gender, pred, prob])
        else:
            for example in examples:
                name, gender = example
            if gender != 'U':
                _, pred, prob = self.predict(name, show_all=False, full_name=full_name)
                if gender == pred: 
                    right += 1
                else: mismatch.append([name, gender, pred, prob])
            else:
                smp_sz -= 1
        
        self.mismatch = mismatch
        return right/smp_sz
