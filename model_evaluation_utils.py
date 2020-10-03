import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.preprocessing import LabelBinarizer
from sklearn.preprocessing import StandardScaler

def train_predict_model(classifier, train_features, train_labels, test_features, test_labels):
    train_lab = LabelBinarizer().fit_transform(np.array(train_labels))
    classifier.fit(train_features, train_lab.ravel())
    return classifier.predict(test_features)

def display_model_performance_metrics(true_labels, predicted_labels, classes=['positive', 'negative']):
    total_classes = len(classes)
    level_labels = [total_classes*[0], list(range(total_classes))]
    matrix = confusion_matrix(y_true=true_labels, y_pred=predicted_labels, 
                                  labels=classes)
    print('\nModel Performance metrics:','--'*15,sep='\n')
    print(f'Accuracy : {(matrix[0][0] + matrix[1][1])/(matrix[0][0] + matrix[1][1]+matrix[0][1] + matrix[1][0])*100:0.2f}%')
    print(f'Precision : {matrix[0][0]/(matrix[0][0] + matrix[1][0])*100:0.2f}%')
    print(f'Recall : {matrix[0][0]/(matrix[0][0] + matrix[0][1])*100:0.2f}%')
    print(f'F1 Score : {2*((matrix[0][0]/(matrix[0][0] + matrix[1][0]))*(matrix[0][0]/(matrix[0][0] + matrix[0][1])))/((matrix[0][0]/(matrix[0][0] + matrix[1][0]))+(matrix[0][0]/(matrix[0][0] + matrix[0][1])))*100:0.2f}%')
    print('\nModel Classification reports:','--'*15,sep='\n')
    print(classification_report(true_labels,predicted_labels))
    cm_frame = pd.DataFrame(data=matrix, 
                        columns=pd.MultiIndex(levels=[['Predicted:'], classes], 
                                              codes=level_labels), 
                        index=pd.MultiIndex(levels=[['Actual:'], classes], 
                                            codes=level_labels))
    print('Prediction Confusion Matrix : ','--'*15,sep='\n')
    print(cm_frame)