from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, roc_auc_score, recall_score
from sklearn.metrics import roc_curve
from sklearn.pipeline import Pipeline
from scikeras.wrappers import KerasClassifier
from keras import models
from keras import layers

class ModelWrapper(Pipeline):
  
  def __init__(self, steps=None, memory=None, verbose=False, base_preprocessor=None, base_model=None, tag=None):
    self.tag = tag
    self.base_preprocessor = base_preprocessor
    self.base_model = base_model
    self.matrix = None
    self.accuracy = None
    self.precision = None
    self.recall = None
    self.roc_auc_score = None
    self.fpr = None
    self.tpr = None
    self.thresholds = None
    if steps is None:
      super().__init__(steps=[('preprocessor', self.base_preprocessor), 
                      ('model', self.base_model)], memory = memory, verbose = verbose)
    else:
      super().__init__(steps=steps, memory = memory, verbose = verbose)

  def evaluate(self, y_true, y_pred):    
    self.matrix = confusion_matrix(y_true = y_true, y_pred = y_pred)
    self.precision = precision_score(y_true = y_true, y_pred = y_pred, average = 'binary')
    self.accuracy = accuracy_score(y_true = y_true, y_pred = y_pred)
    self.recall = recall_score(y_true = y_true, y_pred = y_pred, average = 'binary')
    print("Confusion matrix : " + self.matrix.__repr__() + "\nPrecision : " + self.precision.__repr__() + "\nAccuracy : " + self.accuracy.__repr__() + "\nRecall : " + self.recall.__repr__())
  
  def calculate_roc(self, y_true, y_score):
    self.roc_auc_score = roc_auc_score(y_true = y_true, y_score = y_score)
    self.fpr, self.tpr, self.thresholds = roc_curve(y_true = y_true, y_score = y_score)
    print("Roc_auc_score : " + self.roc_auc_score.__repr__())

  def get_performances(self):
    return {'Tag' : self.tag, 'Description' : self.base_model, 'Confusion matrix' : self.matrix, 'Accuracy' : self.accuracy, 'Precision' : self.precision, 'Recall' : self.recall, 'ROC' : self.roc_auc_score}

  def __repr__(self):
    try:
      l_best_estimator = self.named_steps['model'].best_estimator_.__repr__()
      l_best_score = self.named_steps['model'].best_score_.__repr__()
      return "Tag : " + self.tag.__repr__() + "\nModel name : " + self.base_model.__repr__() + "\nBest estimator : " + l_best_estimator + "\nBest score : " + l_best_score
    except:
      return "Tag : " + self.tag.__repr__() + "\nModel name : " + self.base_model.__repr__() + "\n" + super().__repr__()

def model_nn():
    nn = models.Sequential([
        layers.Input(shape=(10,)),
        layers.Dense(units=60, kernel_initializer='uniform', activation='relu'),
        layers.Dense(units=12, kernel_initializer='uniform', activation='relu'),
        layers.Dense(units=1, kernel_initializer='uniform', activation='sigmoid')
    ])
    nn.compile(optimizer = 'adam', loss ='binary_crossentropy', metrics = ['accuracy', 'MeanSquaredError', 'AUC'])
    return nn

