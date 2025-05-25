from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, classification_report
import numpy as np

# load data

data_config = {
    'train_feature': 'chunk_{}.fX.npy',      #train feature file name format
    'train_label': 'chunk_{}.fy.npy',        #train label file name format
    'test_feature': 'chunk_{}.fX.npy',       #test feature file name format
    'test_label': 'chunk_{}.fy.npy',         #test label file name format
    'train_chunk_ids': [0,1],
    'test_chunk_ids': [0,1]

}

def load_data(feature_pattern, label_pattern, chunk_ids):
    X_chunks = []
    y_chunks = []
    for chunk_id in chunk_ids:
        #load feature and label data from file
        feature_file = feature_pattern.format(chunk_id)
        X_chunks.append(np.load(feature_file))
        label_file = label_pattern.format(chunk_id)
        y_chunks.append(np.load(label_file))
    
    #concatenate all chunks into single arrays
    X = np.concatenate(X_chunks, axis=0)
    y = np.concatenate(y_chunks, axis=0)
    
    assert X.shape[0] == y.shape[0], "Feature and label arrays must have the same length"
    return X, y

print("Loading data...")
X_train, y_train = load_data(data_config['train_feature'], data_config['train_label'], data_config['train_chunk_ids'])

print("Training model...")
X_test, y_test = load_data(data_config['test_feature'], data_config['test_label'], data_config['test_chunk_ids'])

#data preprocessing
print("Preprocessing data...")
X_train = X_train.reshape(X_train.shape[0], -1) #(N, 257*4)
X_test = X_test.reshape(X_test.shape[0], -1)

#train model
model = XGBClassifier(
    objective='binary:logistic',
    n_estimators=300,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=1,
    random_state=42,
    n_jobs=-1,
    eval_metric='error'
)

#train model
model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=10
    )

#model evaluation
print("Evaluating model...")
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy: {:.2f}%".format(accuracy*100))
print(classification_report(y_test, y_pred))

#model save and load
print("Saving model...")
model.save_model('xgboost.model')
print("Model saved as xgboost.model")