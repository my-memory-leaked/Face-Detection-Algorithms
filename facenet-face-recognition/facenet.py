from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import Normalizer
from sklearn.svm import SVC
import numpy as np

# load the embeddings dataset
data = np.load('process/faces-embeddings-yunet.npz')
emdTrainX, trainy, emdTestX, testy = data['arr_0'], data['arr_1'], data['arr_2'], data['arr_3']
print("Dataset: train=%d, test=%d" % (emdTrainX.shape[0], emdTestX.shape[0]))
# normalize input vectors
in_encoder = Normalizer()
emdTrainX_norm = in_encoder.transform(emdTrainX)
emdTestX_norm = in_encoder.transform(emdTestX)
# label encode targets
out_encoder = LabelEncoder()
out_encoder.fit(trainy)
trainy_enc = out_encoder.transform(trainy)
testy_enc = out_encoder.transform(testy)
# fit model
# model = SVC(kernel='sigmoid', class_weight='balanced', probability=True)
model = SVC(kernel='linear', probability=True)
# model = SVC(kernel='rbf', probability=True)
model.fit(emdTrainX_norm, trainy_enc)
# predict
yhat_train = model.predict(emdTrainX_norm)
yhat_test = model.predict(emdTestX_norm)
# score
score_train = accuracy_score(trainy_enc, yhat_train)
score_test = accuracy_score(testy_enc, yhat_test)
# summarize
print('Settings: kernel= %s' % (model.kernel)) #+ ' degree= %d' % (model.degree))
print('Accuracy: train=%.3f, test=%.3f' % (score_train*100, score_test*100))