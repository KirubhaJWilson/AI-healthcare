import pandas as pd
import os
import ast

class DataLoader:
    def __init__(self, data_dir=None):
        if data_dir is None:
            current_dir = os.path.dirname(os.path.abspath(__file__))
            self.data_dir = os.path.normpath(os.path.join(current_dir, '..', 'data'))
        else:
            self.data_dir = data_dir

        self.symptom_data = None
        self.description_data = {}
        self.diet_data = {}
        self.medication_data = {}
        self.precaution_data = {}
        self.workout_data = {}

    def load_symptom_data(self):
        file_path = os.path.join(self.data_dir, 'disease_prediction.csv')
        if os.path.exists(file_path):
            self.symptom_data = pd.read_csv(file_path)
            print(f"Symptom data loaded: {len(self.symptom_data)} rows")
            return self.symptom_data
        else:
            print(f"Warning: {file_path} not found")
            return pd.DataFrame()

    def load_descriptions(self):
        file_path = os.path.join(self.data_dir, 'description.csv')
        self.description_data = {}
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                disease = str(row['Disease']).strip()
                description = row['Description']
                self.description_data[disease] = description
            print(f"Descriptions loaded: {len(self.description_data)} diseases")
        return self.description_data

    def load_diets(self):
        file_path = os.path.join(self.data_dir, 'diets.csv')
        self.diet_data = {}
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                disease = str(row['Disease']).strip()
                diet_str = str(row['Diet'])
                try:
                    if diet_str.startswith('['):
                        diet_list = ast.literal_eval(diet_str)
                    else:
                        diet_list = [d.strip() for d in diet_str.split(',')]
                except Exception:
                    diet_list = [d.strip() for d in diet_str.split(',')]
                self.diet_data[disease] = diet_list
            print(f"Diets loaded: {len(self.diet_data)} diseases")
        return self.diet_data

    def load_medications(self):
        file_path = os.path.join(self.data_dir, 'medications.csv')
        self.medication_data = {}
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                disease = str(row['Disease']).strip()
                med_str = str(row['Medication'])
                try:
                    if med_str.startswith('['):
                        med_list = ast.literal_eval(med_str)
                    else:
                        med_list = [m.strip() for m in med_str.split(',')]
                except Exception:
                    med_list = [m.strip() for m in med_str.split(',')]
                self.medication_data[disease] = med_list
            print(f"Medications loaded: {len(self.medication_data)} diseases")
        return self.medication_data

    def load_precautions(self):
        file_path = os.path.join(self.data_dir, 'precautions.csv')
        self.precaution_data = {}
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                disease = str(row['Disease']).strip()
                precautions = []
                for col in ['Precaution_1', 'Precaution_2', 'Precaution_3', 'Precaution_4']:
                    if col in row and pd.notna(row[col]) and str(row[col]).strip():
                        precautions.append(str(row[col]).strip())
                self.precaution_data[disease] = precautions
            print(f"Precautions loaded: {len(self.precaution_data)} diseases")
        return self.precaution_data

    def load_workouts(self):
        file_path = os.path.join(self.data_dir, 'workout.csv')
        self.workout_data = {}
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            for _, row in df.iterrows():
                disease = str(row['Disease']).strip()
                workout_str = str(row['Workouts'])
                try:
                    if workout_str.startswith('['):
                        workout_list = ast.literal_eval(workout_str)
                    else:
                        workout_list = [w.strip() for w in workout_str.split(',')]
                except Exception:
                    workout_list = [w.strip() for w in workout_str.split(',')]
                self.workout_data[disease] = workout_list
            print(f"Workouts loaded: {len(self.workout_data)} diseases")
        return self.workout_data

    def load_all(self):
        self.load_symptom_data()
        self.load_descriptions()
        self.load_diets()
        self.load_medications()
        self.load_precautions()
        self.load_workouts()
        return {
            'symptoms': self.symptom_data,
            'descriptions': self.description_data,
            'diets': self.diet_data,
            'medications': self.medication_data,
            'precautions': self.precaution_data,
            'workouts': self.workout_data
        }
