from gymnasium.utils.env_checker import check_env

from environment import RehabCoachEnv
from environment import patient_model as pm

def make_env(seed=0):
    env = RehabCoachEnv()
    env.reset(seed=seed)
    return env

def test_gymnasium_api():
    # check_env verifies spaces, reset/step signatures, dtypes and seeding
    check_env(RehabCoachEnv(), skip_render_check=True)

def test_start_state_matches_spec():
    for seed in range(20):
        env = make_env(seed)
        assert 0.3 <= env.rom <= 0.5
        assert 3.0 <= env.pain <= 5.0
        assert (env.difficulty, env.sets) == (1, 2)
        assert env.motivation == env.profile.baseline_motivation

def test_rest_day_lowers_fatigue_and_pain():
    env = make_env()
    env.fatigue, env.pain = 6.0, 5.0
    env.step(5)
    assert env.fatigue < 6.0 and env.pain < 5.0
    assert env.status == "rest"

def test_rest_day_gives_no_rom_gain():
    env = make_env()
    rom_before = env.rom
    env.step(5)
    assert env.rom == rom_before

def test_harder_plan_gains_more_rom_but_more_pain():
    easy, hard = make_env(1), make_env(1)
    hard.difficulty, hard.sets = 3, 4
    easy.motivation = hard.motivation = 1.0     # no skips, so both exercise
    easy.step(0)
    hard.step(0)
    assert hard.rom > easy.rom
    assert hard.pain > easy.pain

def test_flag_limit_and_safe_reset():
    env = make_env()
    env.difficulty = 4
    for _ in range(pm.MAX_FLAGS):
        env.pain = 7.0
        env.step(8)
    assert env.flags_used == pm.MAX_FLAGS
    assert env.difficulty == pm.SAFE_DIFFICULTY
    env.pain = 7.0
    env.step(8)
    assert env.flags_used == pm.MAX_FLAGS

def test_injury_after_three_high_pain_days():
    env = make_env()
    for day in range(pm.INJURY_DAYS):
        env.pain, env.motivation = 10.0, 1.0   # forc a high pain streak
        _, reward, terminated, _, info = env.step(0)
    assert terminated
    assert info["outcome_name"] == "Injury"
    assert reward <= pm.R_FAILURE + 2.0       # big terminal penalty dominates


def test_truncation_at_day_42_and_info_reports_outcome():
    env = make_env()
    for _ in range(pm.MAX_DAYS):
        env.rom = 0.1                 # never succeed
        env.pain, env.motivation = 2.0, 1.0
        _, _, terminated, truncated, info = env.step(5)
        if terminated or truncated:
            break
    assert truncated and not terminated
    assert info["outcome_name"] == "Time limit" 