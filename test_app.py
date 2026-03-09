from fastapi.testclient import TestClient
from app import app

client = TestClient(app)

def test_prediction():
    response = client.post("/predict", json={
        "features": [6, 2.7, 4.5, 1.5]
    })

    assert response.status_code == 200
    assert "prediction" in response.json()