import pygame

from environment import patient_model as pm

PANEL_SIZE = (340, 720)
PHONE_PIXELS = (160, 280)
BG = (24, 28, 36, 235)
TEXT = (235, 235, 235)
MUTED = (150, 155, 165)
BAR_BG = (60, 64, 75)
BAR_COLORS = {
    "Pain": (230, 120, 60), "Fatigue": (200, 170, 60), "Range of motion": (80, 170, 240),
    "Motivation": (120, 210, 120), "Form quality": (180, 130, 230), "Adherence (7d)": (90, 200, 200),
}
PAIN_ALERT = (230, 40, 40)
BANNER_COLORS = {"Success": (40, 160, 70), "Injury": (190, 40, 40),
                 "Dropout": (190, 110, 30), "Time limit": (90, 90, 110)}

_FONTS = {}


def font(size, bold=False):
    if (size, bold) not in _FONTS:
        _FONTS[(size, bold)] = pygame.font.SysFont("arial", size, bold=bold)
    return _FONTS[(size, bold)]


def _text(surface, text, pos, size=16, color=TEXT, bold=False):
    surface.blit(font(size, bold).render(text, True, color), pos)


def _bar(surface, label, value, y, color):
    """Label, value and a horizontal bar for a 0..1 quantity."""
    _text(surface, label, (16, y), 15, MUTED)
    _text(surface, f"{value * 100:.0f}%", (280, y), 15)
    pygame.draw.rect(surface, BAR_BG, (16, y + 20, 308, 10), border_radius=4)
    pygame.draw.rect(surface, color, (16, y + 20, int(308 * value), 10), border_radius=4)


def _sparkline(surface, label, values, y, color, low, high, target=None):
    """Small trend line of `values` scaled between low and high."""
    _text(surface, label, (16, y), 14, MUTED)
    box = pygame.Rect(16, y + 18, 308, 44)
    pygame.draw.rect(surface, BAR_BG, box, border_radius=4)
    if target is not None:  # dashed-looking target line
        ty = box.bottom - (target - low) / (high - low) * box.height
        for x in range(box.left, box.right, 8):
            pygame.draw.line(surface, (240, 240, 240), (x, ty), (x + 4, ty))
    if len(values) >= 2:
        step = box.width / (pm.MAX_DAYS - 1)
        points = [(box.left + i * step, box.bottom - (min(max(v, low), high) - low) / (high - low) * box.height)
                  for i, v in enumerate(values)]
        pygame.draw.lines(surface, color, False, points, 2)


def hud_surface(env):
    """The right-hand panel with live bars, trends, rewards and the action log."""
    panel = pygame.Surface(PANEL_SIZE, pygame.SRCALPHA)
    panel.fill(BG)
    obs = env._observation()
    _text(panel, f"Day {env.day} / {pm.MAX_DAYS}", (16, 12), 26, bold=True)
    action = pm.ACTION_NAMES[env.last_action] if env.last_action is not None else "-"
    _text(panel, f"Action: {action}", (16, 48), 17)
    _text(panel, f"Reward: {env.last_reward:+.2f}    Total: {env.total_reward:+.1f}", (16, 72), 16)
    _text(panel, f"Plan: difficulty {env.difficulty}/5, {env.sets} sets   Flags {env.flags_used}/3", (16, 96), 14, MUTED)

    pain_color = PAIN_ALERT if env.pain > pm.PAIN_LIMIT else BAR_COLORS["Pain"]
    bars = [("Pain", obs[2], pain_color), ("Fatigue", obs[3], BAR_COLORS["Fatigue"]),
            ("Range of motion", obs[0], BAR_COLORS["Range of motion"]),
            ("Motivation", obs[4], BAR_COLORS["Motivation"]),
            ("Form quality", obs[1], BAR_COLORS["Form quality"]),
            ("Adherence (7d)", obs[5], BAR_COLORS["Adherence (7d)"])]
    for i, (label, value, color) in enumerate(bars):
        _bar(panel, label, float(value), 128 + i * 40, color)

    roms = [h["rom"] for h in env.history]
    pains = [h["pain"] for h in env.history]
    _sparkline(panel, "Range of motion (dashed = target)", roms, 374, BAR_COLORS["Range of motion"], 0.2, 1.1, 1.0)
    _sparkline(panel, "Pain (dashed = 6)", pains, 446, BAR_COLORS["Pain"], 0.0, 10.0, pm.PAIN_LIMIT)

    _text(panel, "Last 5 actions", (16, 522), 15, MUTED)
    for i, record in enumerate(reversed(env.history[-5:])):
        line = f"Day {record['day']:>2}: {pm.ACTION_NAMES[record['action']]:<21} {record['reward']:+.2f}"
        _text(panel, line, (16, 544 + i * 22), 14)
    return panel


def phone_surface(env, progress=1.0):
    """What the capstone app shows: pose overlay, reps, notifications."""
    screen = pygame.Surface(PHONE_PIXELS, pygame.SRCALPHA)
    w, h = PHONE_PIXELS
    if env.outcome == pm.OUTCOME_DROPOUT:
        screen.fill((5, 5, 5, 255))                       # phone turned off
        return screen
    screen.fill((235, 242, 250, 255))
    pygame.draw.rect(screen, (40, 90, 160), (0, 0, w, 26))
    _text(screen, "RehabCoach", (8, 4), 15, (255, 255, 255), bold=True)
    if env.status in ("session", "mobility"):
        _mini_skeleton(screen, env.form)
        reps = env.sets * (6 + 2 * env.difficulty)
        _text(screen, f"Reps {int(progress * reps)}/{reps}" if env.status == "session" else "Stretch", (10, h - 30), 18, (20, 20, 20), True)
    else:
        messages = {"rest": "Rest day", "skipped": "Session missed", "review": "Video call: Physio",
                    "start": "Ready"}
        color = (200, 30, 30) if env.status == "skipped" else (30, 30, 30)
        _text(screen, messages.get(env.status, ""), (10, h // 2), 18, color, True)
    if env.last_action == 7:                              # motivation reminder bubble
        pygame.draw.rect(screen, (255, 210, 60), (8, 34, w - 16, 34), border_radius=10)
        _text(screen, "Reminder: you can do it!", (16, 42), 15, (30, 30, 30))
    return screen


def _mini_skeleton(screen, form):
    if form >= 0.7:
        color = (40, 170, 60)       # good form: green
    elif form >= 0.45:
        color = (230, 150, 20)      # so-so form: amber
    else:
        color = (210, 30, 30)       # poor form: red
    points = [(80, 90), (80, 150), (60, 200), (60, 250), (100, 200), (100, 250)]
    for a, b in [(0, 1), (1, 2), (2, 3), (1, 4), (4, 5)]:
        pygame.draw.line(screen, color, points[a], points[b], 4)
    pygame.draw.circle(screen, color, (80, 78), 12, 3)


def banner_surface(env):
    """Shown when the episode ends: result and the day it ended."""
    name = pm.OUTCOME_NAMES[env.outcome]
    banner = pygame.Surface((520, 90), pygame.SRCALPHA)
    banner.fill((*BANNER_COLORS[name], 230))
    _text(banner, f"{name}", (20, 10), 36, bold=True)
    _text(banner, f"Episode ended on day {env.day}   Total reward {env.total_reward:+.1f}", (20, 56), 18)
    return banner