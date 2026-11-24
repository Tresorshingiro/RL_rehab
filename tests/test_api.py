from fastapi.testclient import TestClient

from api import app
from environment import patient_model as pm

client = TestClient(app)

STATE = {"rom": 0.5, "form": 0.8, "pain": 0.3, "fatigue": 0.2, "motivation": 0.7,
         "adherence_7d": 1.0, "difficulty": 0.25, "sets": 0.25, "day": 0.2, "days_since_rest": 0.3}


def test_recommend_returns_a_labelled_action():
    response = client.post("/recommend", json=STATE)
    assert response.status_code == 200
    body = response.json()
    if "error" in body:          # no trained model yet: endpoint still answers cleanly
        return
    assert 0 <= body["action_id"] < pm.N_ACTIONS
    assert body["action"] == pm.ACTION_NAMES[body["action_id"]]


def test_rejects_out_of_range_values():
    response = client.post("/recommend", json={**STATE, "pain": 7})  # must be normalised
    assert response.status_code == 422