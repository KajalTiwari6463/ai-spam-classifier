from fastapi import FastAPI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

app = FastAPI()

# Train model once
messages = ["win money", "hello friend", "free offer", "meeting tomorrow"]
labels = [1, 0, 1, 0]

vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(messages)

model = LogisticRegression()
model.fit(X, labels)

@app.get("/")
def home():
    return {"message": "Spam Classifier API Running"}

@app.get("/predict")
def predict(msg: str):
    msg_vec = vectorizer.transform([msg])
    result = model.predict(msg_vec)[0]
    
    return {
        "message": msg,
        "prediction": int(result)  # 1 = Spam, 0 = Not Spam
    }