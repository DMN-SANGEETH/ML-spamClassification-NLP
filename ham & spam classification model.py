import pandas as pd
messages = pd.read_csv("SMSSpamCollection", sep='\t', names=["lable", "message"])

#Data cleaning and Data Preprocessing

import re
import nltk
#nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

ps = PorterStemmer()
wordnet = WordNetLemmatizer()
corpus = []

for i in range(0,len(messages)):
    review = re.sub('[^a-zA-Z]', ' ',messages['message'][i])
    review = review.lower()
    review = review.split()
    #review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = [wordnet.lemmatize(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
# Creatin BAG model
'''from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000)
X =  cv.fit_transform(corpus).toarray()'''


# Creatin TF-IDE model
from sklearn.feature_extraction.text import TfidfVectorizer
cvy = TfidfVectorizer(max_features=5000)
X  = cvy.fit_transform(corpus).toarray()

y = pd.get_dummies(messages['lable'])

Y=y.iloc[:,1].values

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train,Y_train)
Y_pred=spam_detect_model.predict(X_test)

from sklearn.metrics import confusion_matrix
confusion_metrics=confusion_matrix(Y_test, Y_pred)

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(Y_test, Y_pred)

