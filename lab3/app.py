from flask import Flask, jsonify
from model import predict, generate_data, train_model, save_model, load_model

app = Flask(__name__)

X_train, X_test, y_train, y_test = generate_data()
modelTrain = train_model(X_train, y_train)
save_model(modelTrain, "model.joblib")
model = load_model("model.joblib")


@app.route("/predict", methods=["POST"])
def predict_endpoint():
    prediction = predict(model, X_test)
    return jsonify({"prediction": prediction.tolist()})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
