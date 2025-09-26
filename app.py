from flask import Flask, request, jsonify, render_template
from werkzeug.exceptions import BadRequest
import joblib, json, os
from functools import wraps
from base64 import b64decode

app = Flask(__name__, template_folder="templates", static_folder="static")

model = joblib.load("models/iris_clf.joblib")
metadata = json.load(open("models/metadata.json", encoding="utf-8"))
FEATURES = ["sepal_length", "sepal_width", "petal_length", "petal_width"]

AUTH_USER = os.getenv("APP_USER", "student")
AUTH_PASS = os.getenv("APP_PASS", "password")


def check_auth(header):
    if not header or not header.startswith("Basic "):
        return False
    try:
        user, pw = b64decode(header.split()[1]).decode().split(":", 1)
        return user == AUTH_USER and pw == AUTH_PASS
    except Exception:
        return False


def requires_auth(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        if not check_auth(request.headers.get("Authorization")):
            return (
                jsonify({"error": "Unauthorized"}),
                401,
                {"WWW-Authenticate": "Basic"},
            )
        return f(*args, **kwargs)

    return wrapper


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return {"status": "ok", "model": "iris_clf", "accuracy": metadata["accuracy"]}


@app.post("/predict")
@requires_auth
def predict():
    data = request.get_json(silent=True)
    if data is None:
        raise BadRequest("Expected application/json body")

    def infer(item):
        try:
            x = [float(item[k]) for k in FEATURES]
        except KeyError as e:
            raise BadRequest(f"Missing feature {e.args[0]}")
        y = int(model.predict([x])[0])
        proba = model.predict_proba([x])[0]
        return {
            "class_id": y,
            "class_name": metadata["target_names"][y],
            "proba": {
                metadata["target_names"][i]: float(proba[i]) for i in range(len(proba))
            },
        }

    if isinstance(data, dict):
        out = infer(data)
    elif isinstance(data, list):
        out = [infer(obj) for obj in data]
    else:
        raise BadRequest("JSON must be an object or an array")
    return jsonify(out)


if __name__ == "__main__":
    # Для Colab будем запускать иначе; локально так ок
    app.run(host="0.0.0.0", port=5000)
