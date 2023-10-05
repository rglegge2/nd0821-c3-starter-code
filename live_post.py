import requests

body = {
  "age": 39,
  "workclass": "State-gov",
  "fnlgt": 77516,
  "education": "Bachelors",
  "education_num": 13,
  "marital_status": "Never-married",
  "occupation": "Adm-clerical",
  "relationship": "Not-in-family",
  "race": "White",
  "sex": "Male",
  "capital_gain": 2174,
  "capital_loss": 0,
  "hours_per_week": 40,
  "native_country": "United_States"
}

inference_endpoint = 'https://salary-model-api.onrender.com/inference'
res = requests.post(url=inference_endpoint, json=body)

print(f"Sending POST to url: {inference_endpoint} with body: {body}")
print(f"Response status code: {res}")
print(f"Inference result: {res.text}")