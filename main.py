# Section A1a Import libraries

import warnings; warnings.simplefilter('ignore')

from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics, svm
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn import decomposition, ensemble
import nltk, pandas, numpy,  string
import textblob
import matplotlib
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

from keras.preprocessing import text, sequence
from keras import layers, models, optimizers
import pandas as pd
from io import StringIO

from nltk import sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2

#from textProc import tokenizeText
print('Section 1a File retrieval complete')

print("A1a import libraries complete")
# Section A1b - Tokenize and label text
# TC-NLTK p9

def tokenizeText(filename, textlabel):
    file = open(filename, 'rt')
    text = file.read()
    file.close()

    # Split into sentences

    sentences = sent_tokenize(text)

    # Prepare removal of non-character
    table = str.maketrans('\n',' ',string.punctuation)
    sentences = [w.translate(table) for w in sentences]

    labels = [label] * len(sentences)

    # Create dataframe of texts and labels
    trainDF = pd.DataFrame()
    trainDF['text'] = sentences
    trainDF['label'] = labels
    # Drop NAs
    trainDF.dropna(inplace = True)
    print("label ",label, " ", len(trainDF))
    # Return tokenized file
    return trainDF

filename = '../MLSources/Right.txt'
label = 'Right'
trainDF1 = tokenizeText(filename, label)
print("Section 1b(i) ", filename, " complete")

filename = '../MLSources/Left.txt'
label = 'Left'
trainDF2 = tokenizeText(filename, label)
print("Section 1b(i) ", filename, " complete")

trainDF = trainDF1.append(trainDF2)
print('A1b Tokenization complete')

print("Section2 - Factorise individual elements by tag")

# M-CTC 2
col = ['text', 'label']
df = trainDF[col]

df.columns = ['Text', 'Label']
df['category_id'] = df['Label'].factorize()[0]
# In the next line dropna() removes spurious nan dictionary entry
category_id_df = df[['Label', 'category_id']].dropna().drop_duplicates().sort_values('category_id')

category_to_id = dict(category_id_df.values)
id_to_category = dict(category_id_df[['category_id', 'Label']].values)
df.head()

#fig = plt.figure(figsize = (8,6))

CDV_bar = df.groupby('Label').Text.count().plot.bar(ylim = 0, title = "Label count")
#plt.title("Count of corpus elements by tag")
#plt.show

print("element 2, Plot complete")

print("3. Extract features")

# M-CTC 5.1

tfidf = TfidfVectorizer(sublinear_tf = True, min_df = 5, norm = 'l2', encoding = 'latin-1', ngram_range = (1,2), stop_words ='english')
features = tfidf.fit_transform(df.Text).toarray()
labels = df.category_id
features.shape

print("Correlate unigrams and bigrams with tags and display top n")

# M-CTC 5.2
from sklearn.feature_selection import chi2
import numpy as np
N = 100
for Label, category_id in sorted(category_to_id.items()):
    features_chi2 = chi2(features, labels == category_id)
    indices = np.argsort(features_chi2[0])
    feature_names = np.array(tfidf.get_feature_names())[indices]
    unigrams = [v for v in feature_names if len(v.split(' ')) == 1]
    bigrams = [v for v in feature_names if len(v.split(' ')) == 2]
    print("# '{}':".format(Label))
    print("   . Most correlated unigrams:\n.{}".format('\n.'.join(unigrams[-N:])))
    print("   . Most correlated bigrams:\n.{}".format('\n.'.join(bigrams[-N:])))
##########################################################

print("Section B - Create models")

# M-CTC 7
# Predictions and models

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
import seaborn as sns


# Create train / test sets
x_train, x_test, y_train, y_test = train_test_split(df['Text'], df['Label'], random_state = 0)
count_vect = CountVectorizer()
x_train_counts = count_vect.fit_transform(x_train)
tfidf_transformer = TfidfTransformer()
x_train_tfidf = tfidf_transformer.fit_transform(x_train_counts)
print("Train / Test sets created")
print("\nSection B - Create models\n")
print("B1 - Naive Bayes model initiate")

# M-CTC 7 (cont)

clf = MultinomialNB().fit(x_train_tfidf, y_train)

model = MultinomialNB()
x_train1, x_test1, y_train1, y_test1, indices_train1, indices_test1 = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(x_train1, y_train1)
y_pred1 = model.predict(x_test1)
conf_mat = confusion_matrix(y_test1, y_pred1)
fig, ax = plt.subplots(figsize=(4,4))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Label.values, yticklabels=category_id_df.Label.values)
# g.set_axis_labels("Actual", "Predicted")
# g.set_title(model, fontsize = 12)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title(model, fontsize = 18)
fig.savefig('Bayes.png')

print("B1 - Naive Bayes complete\n")

print("B2 - Logistic regression model initiate")

clf = LogisticRegression().fit(x_train_tfidf, y_train)

model = LogisticRegression()
x_train1, x_test1, y_train1, y_test1, indices_train1, indices_test1 = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(x_train1, y_train1)
y_pred1 = model.predict(x_test1)
conf_mat = confusion_matrix(y_test1, y_pred1)
fig, ax = plt.subplots(figsize=(4,4))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Label.values, yticklabels=category_id_df.Label.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title(model, fontsize = 18)
fig.savefig('Logrec.png')

print("B2 - Logistic regression complete\n")

print("B3 - Random forest model initiate ")
clf = RandomForestClassifier().fit(x_train_tfidf, y_train)

model = RandomForestClassifier()
x_train1, x_test1, y_train1, y_test1, indices_train1, indices_test1 = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(x_train1, y_train1)
y_pred1 = model.predict(x_test1)
conf_mat = confusion_matrix(y_test1, y_pred1)
fig, ax = plt.subplots(figsize=(4,4))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Label.values, yticklabels=category_id_df.Label.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title(model, fontsize = 18)
fig.savefig('RandomForest.png')


print("B3 - Random forest classifier complete\n")

print("B4 - Linear SVM model initiate ")
clf = LinearSVC().fit(x_train_tfidf, y_train)

# M-CTC 10
model = LinearSVC()
x_train1, x_test1, y_train1, y_test1, indices_train1, indices_test1 = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(x_train1, y_train1)
y_pred1 = model.predict(x_test1)
conf_mat = confusion_matrix(y_test1, y_pred1)
fig, ax = plt.subplots(figsize=(4,4))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Label.values, yticklabels=category_id_df.Label.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title(model, fontsize = 18)
fig.savefig('LinearSVM.png')


print("B4 - Linear SVM complete\n")

print("B5 - Stochastic gradient descent  model initiate")
clf = SGDClassifier().fit(x_train_tfidf, y_train)

model = SGDClassifier()
x_train1, x_test1, y_train1, y_test1, indices_train1, indices_test1 = train_test_split(features, labels, df.index, test_size=0.33, random_state=0)
model.fit(x_train1, y_train1)
y_pred1 = model.predict(x_test1)
conf_mat = confusion_matrix(y_test1, y_pred1)
fig, ax = plt.subplots(figsize=(4,4))
sns.heatmap(conf_mat, annot=True, fmt='d',
            xticklabels=category_id_df.Label.values, yticklabels=category_id_df.Label.values)
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title(model, fontsize = 18)
fig.savefig('SGD.png')

print("B5 - Stochastic gradient descent classifier complete\n")
#plt = ""
# Section C - Model comparison


# N CTC p 9
print("Section C - Model comparison\n")
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
print("C1 - create cross validation from models")
from sklearn.model_selection import cross_val_score
models = [
    RandomForestClassifier(n_estimators=200, max_depth=3, random_state=0),
    LinearSVC(),
    MultinomialNB(),
    LogisticRegression(random_state=0),
    SGDClassifier()
]
CV = 5
cv_df = pd.DataFrame(index=range(CV * len(models)))
entries = []
for model in models:
    model_name = model.__class__.__name__
    accuracies = cross_val_score(model, features, labels, scoring='accuracy', cv=CV)
    for fold_idx, accuracy in enumerate(accuracies):
        entries.append((model_name, fold_idx, accuracy))
print("C1 - Model Cross Validation Complete")
cv_df = pd.DataFrame(entries, columns=['model_name', 'fold_idx', 'accuracy'])
fig, ax = plt.subplots(figsize=(8,8))
sns.boxplot(x='model_name', y='accuracy', data=cv_df)
ax.set_title('Model comparison')
ax.set_ylabel('Models')
ax.set_xlabel('Accuracy')
plt.xticks(rotation=15)
fig.savefig('accuracy1.png')
ax.set_title('Model comparison (with data points)')
ax.set_ylabel('Models')
ax.set_xlabel('Accuracy')
sns.stripplot(x='model_name', y='accuracy', data=cv_df,
              jitter=True, edgecolor="gray", linewidth=2)
plt.xticks(rotation=15)
fig.savefig('accuracy2.png')
#plt.xticks(rotation=45)
#¢plt.title("Classification method comparison")

#plt.show()
