# coding: utf-8

# # import packages

# In[24]:


import  nltk


# In[49]:


from sklearn.tree import DecisionTreeClassifier


# In[51]:


from sklearn.feature_extraction import DictVectorizer


# In[52]:


from sklearn.pipeline import Pipeline


# # import Data set from NLTK

# In[25]:


tagged_sentences = nltk.corpus.treebank.tagged_sents()


# In[28]:


print(len(tagged_sentences))


# # preparing Features from this nltk Data annotated Data

# In[29]:


def features(sentence, index):
    return {
        'word': sentence[index],
        'is_first': index == 0,
        'is_last': index == len(sentence) - 1,
        'is_capitalized': sentence[index][0].upper() == sentence[index][0],
        'is_all_caps': sentence[index].upper() == sentence[index],
        'is_all_lower': sentence[index].lower() == sentence[index],
        'prefix-1': sentence[index][0],
        'prefix-2': sentence[index][:2],
        'prefix-3': sentence[index][:3],
        'suffix-1': sentence[index][-1],
        'suffix-2': sentence[index][-2:],
        'suffix-3': sentence[index][-3:],
        'prev_word': '' if index == 0 else sentence[index - 1],
        'next_word': '' if index == len(sentence) - 1 else sentence[index + 1],
        'has_hyphen': '-' in sentence[index],
        'is_numeric': sentence[index].isdigit(),
        'capitals_inside': sentence[index][1:].lower() != sentence[index][1:]
    }


# # here we are bulding untag function to clea that data

# In[30]:


def untag(tagged_sentence):
    return [w for w, t in tagged_sentence]


# # splitting the Data-set into Train and Testing purpose

# In[31]:


cutoff = int(.75 * len(tagged_sentences))
print(len(tagged_sentences))
print(cutoff)


# In[33]:


training_sentences = tagged_sentences[:cutoff]
print(len(training_sentences))


# In[34]:


test_sentences = tagged_sentences[cutoff:]
print(len(test_sentences))


# # Now we are making original Data-set with our  own features

# In[36]:


def transform_to_dataset(tagged_sentences):
    X, y = [], []
 
    for tagged in tagged_sentences:
        for index in range(len(tagged)):
            X.append(features(untag(tagged), index))
            y.append(tagged[index][1])
 
    return X, y


# In[38]:


X, y = transform_to_dataset(training_sentences)


# In[41]:


print(len(X))


# In[43]:


print(len(y))


# # Now we are training our Machine_Learning Model

# In[54]:


clf = Pipeline([
    ('vectorizer', DictVectorizer(sparse=False)),
    ('classifier', DecisionTreeClassifier(criterion='entropy'))
])


# In[64]:


print(len(X[:20000]))
print(len(y[:20000]))


# In[69]:


clf.fit(X[:20000], y[:20000])


# # trainin is completed now prepare Test-Data

# In[59]:


X_test, y_test = transform_to_dataset(test_sentences)


# # Now check the accuracy through Test-Data

# In[81]:


accuracy=clf.score(X_test, y_test)


# In[82]:


print(accuracy)
