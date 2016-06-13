#!/usr/local/bin/python2.7
# encoding: utf-8

from os.path import join, exists
import sys
import numpy as np
import loader

""" 未対応文字に対応する処理 """
def check_word(_word, char_copus):
    word_modified = ''
    for c in _word:
        if char_copus.has_key(ord(c)):
            word_modified = word_modified + c
        else:
            pass
    return word_modified

""" 動的マッチング コスト計算 """
def dp(wordA, wordB, char_copus, features):
    total = 0.0
    right = np.asarray([0, 1])
    top = np.asarray([1, 0])
    topRight = np.asarray([1, 1])
    direction = []
    direction.append(right)
    direction.append(top)
    direction.append(topRight)

    def computeCost(_p, char_copus=char_copus, features=features):
        id = char_copus[ord(wordA[_p[0]])]
        vecA = features[id]
        id = char_copus[ord(wordB[_p[1]])]
        vecB = features[id]
        val = np.sqrt(np.sum(np.power(vecA - vecB, 2)) / vecA.shape[0])
        return val

    p = np.asarray([0, 0])
    while True:
        val = computeCost(p)
        total = total + val
        next = None
        min_val = 9999
        for d in direction:
            target = p + d
            if target[0] >= len(wordA) or target[1] >= len(wordB):
                continue
            _val = computeCost(target)
            if _val <= min_val:
                min_val = _val
                next = target
        p = next
        if p is None:
            return total
        else:
            pass
    
def dp_matching(text, word_copus):
    char_copus, char_features, chars = loader.setup()
    words = []
    score = []
    print text,
    print '  ->  ',
    text = check_word(text, char_copus)
    print text + ' (modified)'
    if len(text) == 0:
        return None
    
    for _text in word_copus:
        _text = unicode(_text, 'utf-8').rstrip()
        _text = check_word(_text, char_copus)
        if not len(_text) == 0:
            _val = dp(text, _text, char_copus, char_features)
            words.append(_text)
            score.append(_val)
        else:
            pass
    return sorted(zip(words, score), key=lambda x: x[1])
    

if __name__ == '__main__':
    with open('words.dat') as f:
        word_copus = f.readlines()
    text = 'ほぅね\'ん草'
    text = unicode(text, 'utf-8')
    ret = dp_matching(text, word_copus)
    for i, (word, score) in enumerate(ret):
        if i == 3:
            break
        else:
            print word
