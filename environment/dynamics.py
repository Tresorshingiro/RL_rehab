from environment import patient_model as pm


def capability(rom):
    """Difficulty level the joint can handle today: 1 at ROM 0, 5 at the target."""
    return 1.0 + 4.0 * rom


def overreach(difficulty, rom):
    """How many levels the plan is above what the patient can handle"""
    return max(0.0, difficulty - capability(rom))


def training_load(difficulty, sets, mobility):
    """0.04 (level 1, 1 set) .. 1.0 (level 5, 5 sets). Mobility is a light fixed load."""
    return pm.MOBILITY_LOAD if mobility else difficulty * sets / 25.0


def form_quality(fatigue, overreach_levels):
    """Form drops when tired or when the exercise is too hard"""
    return 1.0 - pm.FORM_FATIGUE_DROP * fatigue - pm.FORM_OVERREACH_DROP * overreach_levels


def rom_gain(recovery_rate, load, form):
    """Harder training gains more ROM; poor form wastes up to half of it."""
    return pm.ROM_GAIN_SCALE * recovery_rate * load * (0.5 + 0.5 * form)


def session_pain(pain_sensitivity, load, form, overreach_levels, mobility):
    """pain added by a session: load, bad form and overreach all hurt"""
    pain = (pm.PAIN_PER_LOAD * pain_sensitivity * load 
            + pm.PAIN_PER_BAD_FORM * (1.0 - form)
            + pm.PAIN_PER_OVERREACH * overreach_levels)
    return pain * (pm.MOBILITY_PAIN_FACTOR if mobility else 1.0)


def skip_probability(pain, motivation):
    """patients skip more in pain (above 5) or unmotivated."""
    p = pm.SKIP_BASE
    p += pm.SKIP_PER_PAIN * max(0.0, pain - 5.0)
    p += pm.SKIP_PER_LOW_MOTIVATION * (1.0 - motivation)
    return min(0.9, p)


def motivation_change(motivation, baseline, pain, days_since_rest):
    drift = pm.MOTIVATION_DRIFT * (baseline - motivation)
    drop = pm.MOTIVATION_PAIN_DROP * max(0.0, pain - pm.PAIN_LIMIT)
    if days_since_rest >= 7:
        drop += pm.MOTIVATION_STREAK_DROP
    return drift, drop

def reminder_boost(recent_reminders):
    """Each reminder in the last week halves the next one's effect."""
    return pm.REMINDER_BOOST * pm.REMINDER_DECAY ** recent_reminders


def success_bonus(day):
    """Reaching the target earlier earns more."""
    days_left = (pm.MAX_DAYS - day) / pm.MAX_DAYS
    return pm.R_SUCCESS_BASE + pm.R_SUCCESS_EARLY * days_left