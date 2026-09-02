import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import os

class DiseasePredictor:
    def __init__(self, models_dir=None):
        if models_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.models_dir = os.path.join(current_dir, 'trained_models')
        else:
            self.models_dir = models_dir

        self.models = {}
        self.label_encoder = LabelEncoder()
        self.symptom_columns = []
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.best_model_name = None
        self.best_model = None

    def _init_models(self):
        self.models = {
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                max_depth=15,
                random_state=42,
                n_jobs=-1
            ),
            'naive_bayes': GaussianNB(),
            'svm': SVC(
                kernel='rbf',
                C=10,
                gamma='scale',
                probability=True,
                random_state=42
            )
        }

    def load_data(self, symptom_data):
        if symptom_data is None or symptom_data.empty:
            raise ValueError("Symptom data is empty")

        X = symptom_data.drop('label', axis=1)
        y = symptom_data['label']

        self.symptom_columns = X.columns.tolist()

        y_encoded = self.label_encoder.fit_transform(y)
        self.classes = self.label_encoder.classes_

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
        )

        self.symptom_data = symptom_data
        print(f"Data loaded: {len(X)} samples, {len(self.symptom_columns)} symptoms")
        print(f"Classes: {len(self.classes)} diseases")

    def train_models(self):
        self._init_models()
        results = {}

        for name, model in self.models.items():
            model.fit(self.X_train, self.y_train)
            y_pred = model.predict(self.X_test)
            accuracy = accuracy_score(self.y_test, y_pred)
            results[name] = accuracy
            print(f"{name} accuracy: {accuracy:.4f}")

        self.best_model_name = max(results, key=results.get)
        self.best_model = self.models[self.best_model_name]
        print(f"Best model: {self.best_model_name} with accuracy: {results[self.best_model_name]:.4f}")

        self.save_models()

    def predict(self, symptoms):
        if not symptoms:
            return []

        feature_vector = {col: 0 for col in self.symptom_columns}
        for symptom in symptoms:
            s_clean = symptom.strip().lower().replace(' ', '_')
            if s_clean in feature_vector:
                feature_vector[s_clean] = 1
            elif symptom in feature_vector:
                feature_vector[symptom] = 1

        input_df = pd.DataFrame([feature_vector])

        if hasattr(self.best_model, 'predict_proba'):
            probabilities = self.best_model.predict_proba(input_df)[0]
        else:
            pred = self.best_model.predict(input_df)
            probabilities = np.zeros(len(self.classes))
            probabilities[pred[0]] = 1

        top_indices = np.argsort(probabilities)[::-1][:5]
        results = [(self.classes[i], float(probabilities[i])) for i in top_indices if probabilities[i] > 0.001]

        return results

    def get_all_symptoms(self):
        return self.symptom_columns

    def get_best_model_name(self):
        return self.best_model_name or 'random_forest'

    def save_models(self):
        os.makedirs(self.models_dir, exist_ok=True)

        for name, model in self.models.items():
            with open(os.path.join(self.models_dir, f'{name}.pkl'), 'wb') as f:
                pickle.dump(model, f)

        with open(os.path.join(self.models_dir, 'label_encoder.pkl'), 'wb') as f:
            pickle.dump(self.label_encoder, f)

        with open(os.path.join(self.models_dir, 'symptom_columns.pkl'), 'wb') as f:
            pickle.dump(self.symptom_columns, f)

        print("Models saved successfully")

    def load_models(self):
        model_files = ['random_forest', 'naive_bayes', 'svm']
        for name in model_files:
            filepath = os.path.join(self.models_dir, f'{name}.pkl')
            try:
                with open(filepath, 'rb') as f:
                    self.models[name] = pickle.load(f)
                print(f"Loaded {name} model")
            except FileNotFoundError:
                print(f"Model {filepath} not found, training required")
                return False

        try:
            with open(os.path.join(self.models_dir, 'label_encoder.pkl'), 'rb') as f:
                self.label_encoder = pickle.load(f)
                self.classes = self.label_encoder.classes_

            with open(os.path.join(self.models_dir, 'symptom_columns.pkl'), 'rb') as f:
                self.symptom_columns = pickle.load(f)
        except FileNotFoundError:
            return False

        self.best_model = self.models.get('random_forest')
        self.best_model_name = 'random_forest'

        return True
