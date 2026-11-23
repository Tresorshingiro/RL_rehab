import gymnasium as gym
import numpy as np
from gymnasium import spaces

from environment import dynamics
from environment import patient_model as pm


class RehabCoachEnv(gym.Env):
    """Adaptive Home Rehab Coach, The agent adjusts a patient's home exercise plan each day."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}

    def __init__(self, render_mode=None, profile_mode="train"):
        self.render_mode = render_mode
        self.profile_mode = profile_mode
        self.action_space = spaces.Discrete(pm.N_ACTIONS)
        self.observation_space = spaces.Box(0.0, 1.0, shape=(10,), dtype=np.float32)
        self.renderer = None  # created lazily so training never imports OpenGL

    # gym API
    def reset(self, seed = None, options = None):
        super().reset(seed=seed)
        rng = self.np_random
        self.profile = pm.sample_profile(rng, self.profile_mode)
        self.rom = rng.uniform(0.3, 0.5)             # fraction of the target ROM
        self.pain = rng.uniform(3.0, 5.0)
        self.fatigue = 2.0
        self.motivation = self.profile.baseline_motivation
        self.form = 0.8
        self.difficulty, self.sets = 1, 2
        self.day, self.days_since_rest = 0,0
        self.flags_used, self.high_pain_days = 0, 0
        self.reminder_days = []                        # days on which reminders were sent
        self.recent_days = [1] * 7                     # 1 = adherent day, 0 = skipped
        self.history = []                              # per-day records for render/export
        self.last_action, self.last_reward, self.total_reward = None, 0.0, 0.0
        self.status, self.outcome = "start", pm.OUTCOME_RUNNING
        return self._observation(), self._info()

    def step(self, action):
        action = int(action)
        reward = self._apply_plan_action(action)
        reward += self._simulate_day(action)
        self.day += 1
        reward += self._check_termination()
        self.last_action, self.last_reward = action, reward
        self.total_reward += reward
        self.history.append({"day": self.day, "action": action, "status": self.status,
                             "reward": round(float(reward), 4),
                             "rom": round(float(self.rom), 4),
                             "pain": round(float(self.pain),3)})
        terminated = self.outcome in (pm.OUTCOME_SUCCESS, pm.OUTCOME_INJURY, pm.OUTCOME_DROPOUT)
        truncated = self.outcome == pm.OUTCOME_TIME_LIMIT
        if self.render_mode == "human":
            self.render()
        return self._observation(), float(reward), terminated, truncated, self._info()

    def render(self):
        if self.render_mode is None:
            return None
        if self.renderer is None:
            from environment.rendering import RehabRenderer
            #lazy import 
            self.renderer = RehabRenderer()
        return self.renderer.draw(self, return_rgb=self.render_mode == "rgb_array")

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None

    # actions
    def _apply_plan_action(self, action):
        """Change the plan. Returns the penalty for wasteful actions"""
        wasted = False
        if action == 1:
            wasted = self.difficulty == 5
            self.difficulty = min(5, self.difficulty + 1)
        elif action == 2:
            wasted = self.difficulty == 1
            self.difficulty = max(1, self.difficulty -1)
        elif action == 3:
            wasted = self.sets == 5
            self.sets = min(5, self.sets + 1)
        elif action == 4:
            wasted = self.sets == 1
            self.sets = max(1, self.sets -1)
        elif action == 7:
            wasted = self.motivation >= pm.HIGH_MOTIVATION
            self.reminder_days = [d for d in self.reminder_days if self.day - d < pm.REMINDER_MEMORY_DAYS]
            boost = dynamics.reminder_boost(len(self.reminder_days))
            self.motivation = min(1.0, self.motivation + boost)
            self.reminder_days.append(self.day)
        elif action == 8:
            wasted = self.flags_used >= pm.MAX_FLAGS or self.pain < pm.LOW_PAIN_FOR_FLAG
        return pm.R_UNNECESSARY if wasted else 0.0

    # one day
    def _simulate_day(self, action):
        """Run the day's session (or rest/review/skip) and return its reward."""
        reward = 0.0
        self.pain = max(0.0, self.pain - pm.PAIN_DAILY_RELIEF)
        self.fatigue = max(0.0, self.fatigue - pm.FATIGUE_DAILY_RECOVERY * self.profile.fatigue_recovery)

        if action == 8 and self.flags_used < pm.MAX_FLAGS:
            self.status = "review"
            self.flags_used += 1
            self.pain = max(0.0, self.pain - pm.REVIEW_PAIN_RELIEF)
            self.difficulty = pm.SAFE_DIFFICULTY
            reward += pm.R_FLAG
        elif action == 5:
            self.status = "rest"
            self.days_since_rest = -1
            self.pain = max(0.0, self.pain - pm.REST_PAIN_RELIEF)
            self.fatigue = max(0.0, self.fatigue - pm.REST_FATIGUE_RELIEF * self.profile.fatigue_recovery)
        elif self.np_random.random() < dynamics.skip_probability(self.pain, self.motivation):
            self.status = "skipped"
            self.motivation -= pm.MOTIVATION_SKIP_DROP * self._noise()
            reward += pm.R_SKIP
        else:
            self.status = "mobility" if action == 6 else "session"
            reward += self._run_session(mobility=action == 6)

        self.recent_days = self.recent_days[1:] + [0 if self.status == "skipped" else 1]
        self.days_since_rest += 1
        drift, drop = dynamics.motivation_change(self.motivation, self.profile.baseline_motivation,
                                                 self.pain, self.days_since_rest)
        self.motivation = float(np.clip(self.motivation + drift - drop * self._noise(), 0.0, 1.0))
        if self.pain > pm.PAIN_LIMIT:
            reward -= pm.R_PAIN_PER_POINT * (self.pain - pm.PAIN_LIMIT)
        return reward

    def _run_session(self, mobility):
        """Exercise: ROM gain vs pain and fatigue, all shaped by form quality"""
        over = dynamics.overreach(self.difficulty, self.rom)
        load = dynamics.training_load(self.difficulty, self.sets, mobility)
        self.form = float(np.clip(dynamics.form_quality(self.fatigue, over) * self._noise(), 0.0, 1.0))

        gain = max(0.0, dynamics.rom_gain(self.profile.recovery_rate, load, self.form) * self._noise())
        self.rom = min(1.2, self.rom + gain)
        added_pain = dynamics.session_pain(self.profile.pain_sensitivity, load, self.form, over, mobility)
        self.pain = min(10.0, self.pain + added_pain * self._noise())
        self.fatigue = min(10.0, self.fatigue + pm.FATIGUE_PER_LOAD * load * self._noise())
        self.motivation += pm.MOTIVATION_PROGRESS_GAIN

        reward = pm.R_ROM_GAIN * gain + pm.R_SESSION_DONE
        if self.form >= pm.GOOD_FORM:
            reward += pm.R_GOOD_FORM
        return reward

    # endings
    def _check_termination(self):
        """Set self.outcome and return the terminal bonus or penalty."""
        if self.pain >= pm.INJURY_PAIN:
            self.high_pain_days += 1
        else:
            self.high_pain_days = 0
        if self.rom >= pm.ROM_TARGET:
            self.outcome = pm.OUTCOME_SUCCESS
            return dynamics.success_bonus(self.day)
        if self.high_pain_days >= pm.INJURY_DAYS:
            self.outcome = pm.OUTCOME_INJURY
            return pm.R_FAILURE
        if self.motivation <= 0.0:
            self.outcome = pm.OUTCOME_DROPOUT
            return pm.R_FAILURE
        if self.day >= pm.MAX_DAYS:
            self.outcome = pm.OUTCOME_TIME_LIMIT
        return 0.0

    # helpers
    def _noise(self):
        """Multiplicative noise 1.0 so no two patients days are identical"""
        return 1.0 + self.np_random.normal(0.0, pm.NOISE_STD)

    def _observation(self):
        obs = [
            self.rom / pm.ROM_TARGET,
            self.form,
            self.pain / 10.0,
            self.fatigue / 10.0,
            self.motivation,
            sum(self.recent_days) / 7.0,
            (self.difficulty - 1) / 4.0,
            (self.sets - 1) / 4.0,
            self.day / pm.MAX_DAYS,
            min(self.days_since_rest, 14) / 14.0,
        ]
        return np.clip(np.array(obs, dtype=np.float32), 0.0, 1.0)

    def _info(self):
        return {
            "day": self.day,
            "outcome": np.int64(self.outcome),
            "outcome_name": pm.OUTCOME_NAMES[self.outcome],
            "status": self.status,
            "flags_used": self.flags_used,
        }
