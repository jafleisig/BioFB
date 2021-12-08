from sklearn.neural_network import MLPClassifier
from scipy.io import loadmat
import numpy as np
from joblib import dump, load
from sklearn import metrics
import matplotlib.pyplot as plt

# Get the partitioned data from the .mat file
mat_data = loadmat('training_data_truncated.mat')
features_train = mat_data['features_train']
features_test = mat_data['features_test']
targets_train = mat_data['targets_train']
targets_test = mat_data['targets_test']

train = False
if train:

    clf = MLPClassifier(hidden_layer_sizes=(100, 100), activation='logistic', solver='sgd', learning_rate_init=0.01,
                        max_iter=50000, shuffle=False, random_state=4, verbose=False, tol=1e-6, momentum=0.9)

    # Train the network
    clf.fit(features_train, targets_train)

    dump(clf, 'sklearn_nn.joblib')

else:
    # Load in the network
    clf = load('sklearn_nn.joblib')


# See the parameters for this network
params = clf.get_params()

# Test the network
testoutput = clf.predict(features_test)
testoutput_proba = clf.predict_proba(features_test)

accuracy = float(np.sum(testoutput == targets_test))/float((2*len(targets_test)))
mse = metrics.mean_squared_error(targets_test, testoutput)

testoutput_proba_incorrect = []
targets_test_incorrect = []
for i in range(len(targets_test)):
    testoutput_proba_incorrect = testoutput_proba_incorrect + [testoutput_proba[i][1]]
    targets_test_incorrect = targets_test_incorrect + [targets_test[i][1]]

fpr, tpr, thresholds = metrics.roc_curve(targets_test_incorrect, testoutput_proba_incorrect, pos_label=1)
auc = metrics.auc(fpr, tpr)

print(accuracy)
print(mse)
print(auc)

fig = plt.figure()
plt.plot(fpr, tpr)
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.title("ROC Curve for Classification Neural Network")
plt.ylabel("True Positive Rate")
plt.xlabel("False Positive Rate")
plt.show()




