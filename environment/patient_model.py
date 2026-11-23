from dataclasses import dataclass
import numpy as np


# episode
MAX_DAYS = 42
ROM_TARGET = 1.0  # range of motion is stored as a fraction
N_ACTIONS = 9
MAX_FLAGS = 3     # physiotherapist reviews allowed per episode

ACTION_NAMES = [
    "Keep plan",
    "Increase difficulty",
    "Decrease difficulty",
    "Add a set",
    "Remove a set",
    "Rest day",
    "Mobility exercise",
    "Motivation reminder",
    "Flag physiotherapist",
]

OBS_NAMES = [
    "rom","form", "pain", "fatigue", "motivation", "adherence_7d", "difficulty", "sets", "day", "days_since_rest",
]

# transitiona
ROM_GAIN_SCALE = 0.06            # ROM gained per day at full load, perfect form, recovery 1.0
MOBILITY_LOAD = 0.08             # mobility session = light fixed load
MOBILITY_PAIN_FACTOR = 0.3       # mobility hurts much less than strength work
PAIN_PER_LOAD = 2.2              # pain added by a full-load session (x pain sensitivity)
PAIN_PER_BAD_FORM = 1.5          # extra pain form quality is 0
PAIN_PER_OVERREACH = 1.0         # extra pain per difficulty level above capability
PAIN_DAILY_RELIEF = 0.6          # natural pain decrease every day
REST_PAIN_RELIEF = 1.5           # extra pain decrease on a rest day
REVIEW_PAIN_RELIEF = 3.5         # physiotherapist review lowers pain a lot
FATIGUE_PER_LOAD = 4.0           # fatigue added by a full-load session
FATIGUE_DAILY_RECOVERY = 1.0     # fatigue removed per day(x fatigue recovery)
REST_FATIGUE_RELIEF = 2.5        # extra fatigue removed on a rest day
FORM_FATIGUE_DROP = 0.05         # form lost per fatigue point
FORM_OVERREACH_DROP = 0.2        # form lost per difficulty level above capability
SKIP_BASE = 0.03                 # chance of skipping even when everything is fine
SKIP_PER_PAIN = 0.12             # added skip chance per pain point above 5
SKIP_PER_LOW_MOTIVATION = 0.5    # added skip chance at motivation 0
MOTIVATION_DRIFT = 0.1           # pull of motivation back toward the patient's baseline
MOTIVATION_PAIN_DROP = 0.05      # motivation lost per day per pain above 6
MOTIVATION_SKIP_DROP = 0.08      # missing a session is demoralising
MOTIVATION_STREAK_DROP = 0.02    # per day once 7+ days without rest
MOTIVATION_PROGRESS_GAIN = 0.02  # completing a session feels good
REMINDER_BOOST = 0.15            # first reminder effect
REMINDER_DECAY = 0.5             # each recent reminder halves the next one's effect
REMINDER_MEMORY_DAYS = 7         # reminders older than this no longer count
SAFE_DIFFICULTY = 1              # level the physiotherapist resets the plan to
NOISE_STD = 0.05                 # relative noise on gains, pain and motivation


# reward
R_ROM_GAIN = 100.0               # reward per unit of ROM gained (0.02 gain -> +2.0)
R_GOOD_FORM = 0.2                # form quality >= GOOD_FORM
GOOD_FORM = 0.7
R_SESSION_DONE = 0.2
R_PAIN_PER_POINT = 0.5           # per pain point above PAIN_LIMIT
PAIN_LIMIT = 6.0
R_SKIP = -0.5
R_FLAG = -2.0
R_UNNECESSARY = -0.3
R_SUCCESS_BASE = 10.0
R_SUCCESS_EARLY = 20.0           # scaled by the fraction of days left
R_FAILURE = -20.0                # injury or dropout
INJURY_PAIN = 9.0
INJURY_DAYS = 3
HIGH_MOTIVATION = 0.85           # reminders above this are "unnecessary"
LOW_PAIN_FOR_FLAG = 4.0          # flagging below this pain is "unnecessary"


# outcomes
OUTCOME_RUNNING, OUTCOME_SUCCESS, OUTCOME_INJURY = 0, 1, 2
OUTCOME_DROPOUT, OUTCOME_TIME_LIMIT = 3, 4
OUTCOME_NAMES = ["Running", "Success", "Injury", "Dropout", "Time limit"]


# Profile ranges: "train" is what the agent sees during training
# "unseen" is wider and harder, used only for the generalisation test.
PROFILE_RANGES = {
    "train": {
        "recovery_rate": (0.7, 1.3),
        "pain_sensitivity": (0.7, 1.3),
        "baseline_motivation": (0.55, 0.9),
        "fatigue_recovery": (0.7, 1.3),
    },
    "unseen": {
        "recovery_rate": (0.5, 1.5),
        "pain_sensitivity": (0.6, 1.6),
        "baseline_motivation": (0.4, 0.95),
        "fatigue_recovery": (0.5, 1.5)
    },
}

@dataclass
class PatientProfile:
    """Hidden traits of one patient. The agent never observes these"""
    recovery_rate: float
    pain_sensitivity: float
    baseline_motivation: float
    fatigue_recovery: float

def sample_profile(rng: np.random.Generator, mode: str = "train") -> PatientProfile:
    """Draw a new patient from the chosen profile range."""
    ranges = PROFILE_RANGES[mode]
    values = {name: float(rng.uniform(low, high)) for name, (low, high) in ranges.items()}
    return PatientProfile(**values)