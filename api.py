from fastapi import FastAPI
from pydantic import BaseModel, Field

from environment import patient_model as pm
from play import choose_model
from training.evaluate import load_policy

EXPLANATIONS = [
    "Keep today's plan unchanged.",
    "The patient is coping well: make the exercise harder.",
    "Make the exercise easier to protect form and limit pain.",
    "Add one set to increase training volume.",
    "Remove one set to reduce load.",
    "Prescribe a rest day so pain and fatigue can settle.",
    "Do gentle range-of-motion (mobility) work today instead of strength work.",
    "Send an encouraging reminder to keep the patient engaged.",
    "Ask the physiotherapist to review this patient.",
]


class PatientState(BaseModel):
    """The 10 observations, each already normalised to 0..1 (see the observation table)."""

    rom: float = Field(ge=0, le=1, description="range of motion / target")
    form: float = Field(ge=0, le=1)
    pain: float = Field(ge=0, le=1, description="pain / 10")
    fatigue: float = Field(ge=0, le=1, description="fatigue / 10")
    motivation: float = Field(ge=0, le=1)
    adherence_7d: float = Field(ge=0, le=1)
    difficulty: float = Field(ge=0, le=1, description="(level - 1) / 4")
    sets: float = Field(ge=0, le=1, description="(sets - 1) / 4")
    day: float = Field(ge=0, le=1, description="day / 42")
    days_since_rest: float = Field(ge=0, le=1, description="min(days, 14) / 14")


app = FastAPI(title="Adaptive Home Rehab Coach")
AGENT_NAME, MODEL_PATH = choose_model()
ACT = load_policy(MODEL_PATH) if MODEL_PATH else None


@app.post("/recommend")
def recommend(state: PatientState):
    if ACT is None:
        return {"error": "no trained model found; run training first"}
    observation = [getattr(state, name) for name in pm.OBS_NAMES]
    action = ACT(observation)
    return {
        "action_id": action,
        "action": pm.ACTION_NAMES[action],
        "explanation": EXPLANATIONS[action],
        "agent": AGENT_NAME,
    }