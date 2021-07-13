'''
Author: Zhengxiang (Jack) Wang 
Date: 2021-07-13
GitHub: https://github.com/jaaack-wang 
About: Naive-bayes-based predictions of the genders
of (a) given name(s) in Chinese.
'''
import json
from collections import defaultdict, Counter


class Gender:
    '''Predict the gender(s) of (a) given name(s) in Chinese.
    
    Basic usage:
    =================================
    name = a name or a list of names
    gender = Gender()
    gender.predict(name)
    =================================
    
    Gender().predict()
    =================================
    Paras:
        name: str or list
        method: "lap" or "gt", defaults to "lap"
            lap --> adjusts the training set by laplace smoothing
            gt --> adjusts the training set by Good Turing smoothing
        show_all: bool, defaults to True
            True --> Returns the probablities for all genders (M, F, U)
            False --> Returns the predicted gender with optimal probablity
    =================================
    
    Notes:
    =================================
    The two smoothing methods assume unseen characters to add. The training set contains 
    about 5000 unique characters for M and F and the default number of unseen characters
    is set to be 5000, although it turns out to be very insignificant. 
    
    To reset this number, when calling the Gender class, make gender=Gender(your_num).
    '''
    def __init__(self, num_unseen_chars=5000):
        # num of unseen chars for Chinese names
        self._unseen = num_unseen_chars
        self.name = 'You have not entered a name yet'
        # loading the unsmoothed training set
        self.genderDict = self._loadDict()
        # laplace-adjusted genderDict
        self._lapDict = self._laplace()
        # frequency-based good-turing dict
        self._gtDict = self._goodTuring()
        
    def _loadDict(self):
        genderDict = json.load(open('data/dict4Gender.json', 'r'))
        genderDict = {k: Counter(v) for k, v in genderDict.items()}
        return defaultdict(Counter, genderDict)
        
    
    def _laplace(self):
        '''Converts the dict into one suitable for laplace smoothing.
        '''
        lapDict = self.genderDict.copy()
        total = lapDict.pop('total')
        # number of unique chars used for each gender
        distinct = Counter([gender for v in lapDict.values() for gender in v.keys()])
        for g in ['M', 'F', 'U']:
            # add the estimated unseen chars (suppose 5000) to each 
            # gender category both in terms of distinct chars and total chars
            distinct[g] += self._unseen
            total[g] += distinct[g]
        lapDict['total'] = total
        return lapDict
        
    def _goodTuring(self):
        '''Rerturns a dict that contains the occurences info for each freq category
        '''
        genDict = self.genderDict.copy()
        total = genDict.pop('total')
        gtDict = defaultdict(Counter, {0: {'M': self._unseen, 'F': self._unseen, 'U': self._unseen}})
        for V in genDict.values():
            for k, v in V.items():
                gtDict[v][k] += 1
        return gtDict
        
    
    def _naiveBayesP(self, name, gender, method='lap'):
        '''Returns the naive bayes probablity of a given gender for a given name.
        '''
        def getNr(r):
            idx = 0
            Nr = self._gtDict[r][gender]
            while not Nr:
                Nr = self._gtDict[r-idx][gender]
                idx += 1
            
            return Nr
        
        if method == 'lap':
            total_char = sum(self._lapDict['total'].values())
            gender_char = self._lapDict['total'][gender]
            p_gender =  gender_char / total_char
            for char in name:
                char_dict = self._lapDict[char]
                p_char_g = (char_dict[gender] + 1) / gender_char
                p_gender *= p_char_g
            return p_gender
        
        elif method == 'gt':
            total_char = sum(self.genderDict['total'].values())
            gender_char = self.genderDict['total'][gender]
            p_gender =  gender_char / total_char
            for char in name:
                char_dict = self.genderDict[char]
                r = char_dict[gender]
                Nr, NrPlus1 = getNr(r), getNr(r+1)
                r_adj = (r + 1) * NrPlus1 / Nr
                p_char_g = r_adj / gender_char
                p_gender *= p_char_g
            return p_gender
        else:
            raise ValueError(f'{method} not available. Please use\n'
            '"lap" --> for laplace-adjust prediction (default).\n'
            '"gt" --> for good-turing-adjusted prediction.')
                
    def predict(self, name, method='lap', show_all=True):
        '''Returns the probablities of genders for (a) given name(s).
        '''
        def run(name, show_all):
            pM = self._naiveBayesP(name, 'M', method=method)
            pF = self._naiveBayesP(name, 'F', method=method)
            pU = self._naiveBayesP(name, 'U', method=method)
            totalP = pM + pF + pU
            pM, pF, pU = pM/totalP, pF/totalP, pU/totalP
            if show_all:
                res.append((self.name, {'M': pM, 'F': pF, 'U': pU}))
            else:
                if pM==pF and pM==pU: res.append((self.name, 'M=F=Undefined', pM))
                elif pM == max(pM, pF, pU): res.append((self.name, 'M', pM))
                elif pF > pU: res.append((self.name, 'F', pF))
                else: res.append((self.name, 'Undefined', pU))
        
        res = []
        if isinstance(name, str):
            self.name = name
            run(name, show_all)
            return res[0]
        elif isinstance(name, list):
            for n in name:
                self.name = n
                run(n, show_all)
            return res
        else:
            raise TypeError('name must be either a str or a list')
