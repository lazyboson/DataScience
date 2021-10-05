import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn import metrics

sms = pd.read_excel('train.xlsx')
test = pd.read_excel('test.xlsx')

    
uniques =  sms['Final Category'].unique()
d = {uniques[x]:x for x in range(len(uniques))}

     
test['label_num'] = test['Final'].map(d)   
sms['label_num'] = sms['Final Category'].map(d)

X = sms.message.values.astype('unicode')

y = sms.label_num

X_test = test.message.values.astype('unicode')

# instantiate the vectorizer
vect = CountVectorizer(lowercase=False)
# learn training data vocabulary, then use it to create a document-term matrix
X_train_dtm = vect.fit_transform(X)
y_test = test.label_num

nb = MultinomialNB()
SGD_classifier = SGDClassifier()
log_classifier = LogisticRegression(multi_class='multinomial', solver='newton-cg')
# train the model
nb.fit(X_train_dtm, y)
X_test_dtm = vect.transform(X_test)

SGD_classifier.fit(X_train_dtm, y)
log_classifier.fit(X_train_dtm, y)


y_pred_class = nb.predict(X_test_dtm)
SGD_y_pred_class = SGD_classifier.predict(X_test_dtm)
log_y_pred_class = log_classifier.predict(X_test_dtm)


print "MNB Classifier: accurecy ", metrics.accuracy_score(y_test, y_pred_class)
print "SGD Classifier: accurecy ", metrics.accuracy_score(y_test, SGD_y_pred_class)
# print the confusion matrix
print "Logistic Regression: accurecy ", metrics.accuracy_score(y_test, log_y_pred_class)

## print message text for the false positives (ham incorrectly classified as spam)
#print X_test[y_test < y_pred_class]
#
## print message text for the false negatives (spam incorrectly classified as ham)
#print X_test[y_test > y_pred_class]
