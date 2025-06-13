from sklearn.ensemble import RandomForestClassifier
import joblib

class FraudDetector:
    def __init__(self):
        self.model = RandomForestClassifier(n_estimators=100)
    
    def train(self, X, y):
        self.model.fit(X, y)
        joblib.dump(self.model, 'model.pkl')
    
    def predict(self, X):
        model = joblib.load('model.pkl')
        return model.predict(X)
