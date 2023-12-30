from flask import Flask, request, jsonify
import pandas as pd
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)
@app.route('/api', methods=['POST'])
def predict_datapoint():
    if request.method == 'POST':
        try:
            data = request.get_json()
            gender = data.get('gender')
            race_ethnicity = data.get('ethnicity')
            parental_level_of_education = data.get('parental_level_of_education')
            lunch = data.get('lunch')
            test_preparation_course = data.get('test_preparation_course')
            reading_score = float(data.get('reading_score'))
            writing_score = float(data.get('writing_score'))

            custom_data = CustomData(
                gender=gender,
                race_ethnicity=race_ethnicity,
                parental_level_of_education=parental_level_of_education,
                lunch=lunch,
                test_preparation_course=test_preparation_course,
                reading_score=reading_score,
                writing_score=writing_score
            )

            pred_df = custom_data.get_data_as_data_frame()

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            return jsonify(results=results[0])

        except Exception as e:
            return jsonify(error=str(e)), 400
    else:
        return jsonify(error="Invalid request method"), 405

if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)
