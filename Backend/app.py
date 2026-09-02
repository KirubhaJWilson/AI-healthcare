from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_jwt_extended import JWTManager, create_access_token, jwt_required, get_jwt_identity
import os
import sys
from datetime import timedelta

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.disease_predictor import DiseasePredictor
from models.data_loader import DataLoader

app = Flask(__name__, static_folder='../Frontend/public', static_url_path='')
CORS(app)

app.config['JWT_SECRET_KEY'] = os.environ.get('JWT_SECRET_KEY', 'clinica-ai-secret-key-2026')
app.config['JWT_ACCESS_TOKEN_EXPIRES'] = timedelta(hours=24)
jwt = JWTManager(app)

data_loader = DataLoader()
predictor = DiseasePredictor()

print("Loading datasets...")
data_dict = data_loader.load_all()
symptom_data = data_dict.get('symptoms')
description_data = data_dict.get('descriptions', {})
diet_data = data_dict.get('diets', {})
medication_data = data_dict.get('medications', {})
precaution_data = data_dict.get('precautions', {})
workout_data = data_dict.get('workouts', {})

print("Loading/Training models...")
if not predictor.load_models():
    predictor.load_data(symptom_data)
    predictor.train_models()
print("Models ready!")

ALL_SYMPTOMS = predictor.get_all_symptoms()

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'message': 'CLINICA AI Backend Platform is operational'})

@app.route('/api/symptoms', methods=['GET'])
def get_symptoms():
    return jsonify({
        'symptoms': ALL_SYMPTOMS,
        'count': len(ALL_SYMPTOMS)
    })

@app.route('/api/predict', methods=['POST'])
def predict_disease():
    try:
        data = request.get_json() or {}
        symptoms = data.get('symptoms', [])

        if not symptoms:
            return jsonify({'error': 'No symptoms provided'}), 400

        results = predictor.predict(symptoms)

        response = []
        for disease, probability in results:
            description = description_data.get(disease, "Description not available for this disease.")
            diet = diet_data.get(disease, [])
            medications = medication_data.get(disease, [])
            precautions = precaution_data.get(disease, [])
            workouts = workout_data.get(disease, [])

            response.append({
                'disease': disease,
                'probability': round(probability * 100, 2),
                'description': description,
                'diet': diet,
                'medications': medications,
                'precautions': precautions,
                'workouts': workouts
            })

        return jsonify({
            'predictions': response,
            'model_used': predictor.get_best_model_name()
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/description/<disease>', methods=['GET'])
def get_description(disease):
    description = description_data.get(disease, "Description not available")
    return jsonify({'disease': disease, 'description': description})

@app.route('/api/diet/<disease>', methods=['GET'])
def get_diet(disease):
    diet = diet_data.get(disease, [])
    return jsonify({'disease': disease, 'diet': diet})

@app.route('/api/medications/<disease>', methods=['GET'])
def get_medications(disease):
    medications = medication_data.get(disease, [])
    return jsonify({'disease': disease, 'medications': medications})

@app.route('/api/precautions/<disease>', methods=['GET'])
def get_precautions(disease):
    precautions = precaution_data.get(disease, [])
    return jsonify({'disease': disease, 'precautions': precautions})

@app.route('/api/workout/<disease>', methods=['GET'])
def get_workout(disease):
    workouts = workout_data.get(disease, [])
    return jsonify({'disease': disease, 'workouts': workouts})

@app.route('/api/auth/register', methods=['POST'])
def register():
    try:
        data = request.get_json() or {}
        email = data.get('email')
        password = data.get('password')
        role = data.get('role', 'patient')
        name = data.get('name', email.split('@')[0] if email else 'User')

        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        access_token = create_access_token(identity={'email': email, 'role': role, 'name': name})
        return jsonify({
            'message': 'User registered successfully',
            'token': access_token,
            'user': {'email': email, 'role': role, 'name': name}
        }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/login', methods=['POST'])
def login():
    try:
        data = request.get_json() or {}
        email = data.get('email')
        password = data.get('password')
        role = data.get('role', 'patient')

        if not email or not password:
            return jsonify({'error': 'Email and password are required'}), 400

        access_token = create_access_token(identity={'email': email, 'role': role, 'name': email.split('@')[0]})
        return jsonify({
            'token': access_token,
            'access_token': access_token,
            'user': {'email': email, 'role': role, 'name': email.split('@')[0]}
        }), 200
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/auth/verify', methods=['GET'])
@jwt_required()
def verify():
    current_user = get_jwt_identity()
    return jsonify({'user': current_user}), 200

@app.route('/api/fhir/patient', methods=['POST'])
def create_fhir_patient():
    try:
        data = request.get_json() or {}
        return jsonify({
            'resourceType': 'Patient',
            'id': f"patient-{data.get('id', '1001')}",
            'active': True,
            'name': [{'text': data.get('name', 'John Doe')}],
            'gender': data.get('gender', 'unknown').lower(),
            'status': 'created'
        }), 201
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_index():
    if os.path.exists('/app/index.html'):
        return send_from_directory('/app', 'index.html')
    return jsonify({'message': 'Clinica AI API operational'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 3000))
    app.run(debug=False, host='0.0.0.0', port=port)
