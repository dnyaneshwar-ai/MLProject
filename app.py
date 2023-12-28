# app.py
import logging
from flask import Flask, request, jsonify
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)
app.config['DEBUG'] = True

logging.basicConfig(level=logging.DEBUG)

@app.route('/predictdata', methods=['POST'])
def predict_datapoint():
    try:
        data = CustomData(
            gender=request.form.get('gender'),
            race_ethnicity=request.form.get('ethnicity'),
            parental_level_of_education=request.form.get('parental_level_of_education'),
            lunch=request.form.get('lunch'),
            test_preparation_course=request.form.get('test_preparation_course'),
            reading_score=float(request.form.get('writing_score')),
            writing_score=float(request.form.get('reading_score'))
        )

        pred_df = data.get_data_as_data_frame()
        logging.debug(f"Input DataFrame: {pred_df}")
        logging.debug("Before Prediction")

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)
        logging.debug("After Prediction")

        return jsonify({"results": results[0]})

    except Exception as e:
        logging.error(f"Error in predict_datapoint: {e}")
        return jsonify({"error": "An error occurred during prediction."})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)

