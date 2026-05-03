from flask import Flask, render_template, request
from logic import predict_disease_web, ensure_models_trained, get_config

app = Flask(__name__)
ensure_models_trained()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction_text = None
    confidence = None
    disease = None
    error_msg = None

    if request.method == 'POST':
        disease = request.form.get('disease')

        if disease in ['diabetes', 'heart', 'liver']:
            try:
                features = get_config()[disease]['features']

                user_input = []
                for feature in features:
                    val = request.form.get(feature)
                    if val is None or val.strip() == '':
                        raise ValueError(f"Missing input for {feature}")
                    user_input.append(float(val))

                prediction, confidence = predict_disease_web(disease, user_input)

                if prediction == 1:
                    prediction_text = "Disease detected!"
                elif prediction == 0:
                    prediction_text = "No disease detected"
                else:
                    prediction_text = "Unable to make prediction"

            except Exception as e:
                error_msg = f"Input error: {e}"
                prediction_text = None
                confidence = None
        else:
            error_msg = "Invalid disease selected."

    return render_template('index.html', 
                           prediction=prediction_text, 
                           confidence=confidence, 
                           disease=disease,
                           error_msg=error_msg)

if __name__ == '__main__':
    app.run(debug=True)
