import math

from OpenGL.GL import glMultMatrixf, glPopMatrix, glPushMatrix, glRotatef, glScalef, glTranslatef

from environment.gl_shapes import box, cylinder_between, ring, sphere

# ---------------------------------------------------------------- body layout
SHIN, THIGH, TORSO = 0.45, 0.45, 0.55
UPPER_ARM, FOREARM = 0.28, 0.26
HIP_WIDTH, SHOULDER_WIDTH = 0.12, 0.18
HEAD_RADIUS = 0.11
SKIN = (0.55, 0.38, 0.28)
HAIR = (0.12, 0.08, 0.06)
SHIRT = (0.2, 0.45, 0.75)
PHYSIO_SHIRT = (0.15, 0.6, 0.55)
TROUSERS = (0.22, 0.22, 0.26)
SHOES = (0.1, 0.1, 0.12)
GREEN, AMBER, RED = (0.2, 0.8, 0.3), (0.95, 0.65, 0.1), (0.9, 0.15, 0.15)

# (start radius, end radius) of each tapered limb
THIGH_R, SHIN_R = (0.075, 0.056), (0.056, 0.04)
UPPER_ARM_R, FOREARM_R = (0.052, 0.042), (0.04, 0.032)


# ---------------------------------------------------------------- vector helpers
def _add(p, dx=0.0, dy=0.0, dz=0.0):
    return (p[0] + dx, p[1] + dy, p[2] + dz)


def _plus(a, b):
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def _sub(a, b):
    return (a[0] - b[0], a[1] - b[1], a[2] - b[2])


def _scale(v, k):
    return (v[0] * k, v[1] * k, v[2] * k)


def _dot(a, b):
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def _unit(v):
    n = math.sqrt(_dot(v, v))
    return _scale(v, 1.0 / n) if n > 1e-9 else (0.0, 1.0, 0.0)


# ---------------------------------------------------------------- poses
def _upper_body(pelvis, lean, hands):
    """Torso, head and shoulders above `pelvis`; `hands` gives both hand positions."""
    neck = _add(pelvis, TORSO * math.sin(lean), TORSO * math.cos(lean))
    pose = {
        "pelvis": pelvis, "neck": neck, "head": _add(neck, 0.05, 0.16),
        "l_shoulder": _add(neck, dz=-SHOULDER_WIDTH), "r_shoulder": _add(neck, dz=SHOULDER_WIDTH),
        "l_hand": hands[0], "r_hand": hands[1],
    }
    return pose


def squat_pose(base, depth, knee_cave):
    """Strength exercise. depth 0..1; knee_cave > 0 shows poor form (knees fall in)."""
    knee_angle, hip_angle, lean = 0.7 * depth, 1.4 * depth, 0.6 * depth
    pose = {}
    for side, sign in (("l", -1), ("r", 1)):
        ankle = _add(base, dy=0.08, dz=sign * HIP_WIDTH)
        knee = _add(ankle, SHIN * math.sin(knee_angle), SHIN * math.cos(knee_angle), -sign * knee_cave)
        hip = _add(knee, -THIGH * math.sin(hip_angle), THIGH * math.cos(hip_angle), sign * knee_cave)
        pose.update({f"{side}_ankle": ankle, f"{side}_knee": knee, f"{side}_hip": hip})
    pelvis = _add(pose["l_hip"], dz=HIP_WIDTH)
    reach = 0.15 + 0.35 * depth                      # arms forward for balance
    hands = [_add(pelvis, reach + 0.1, 0.35, -0.2), _add(pelvis, reach + 0.1, 0.35, 0.2)]
    pose.update(_upper_body(pelvis, lean, hands))
    return pose


def knee_lift_pose(base, lift):
    """Mobility exercise: slow standing knee lift on the right leg. lift 0..1."""
    pose = {}
    for side, sign in (("l", -1), ("r", 1)):
        hip = _add(base, dy=SHIN + THIGH + 0.08, dz=sign * HIP_WIDTH)
        angle = 1.3 * lift if side == "r" else 0.0  # thigh rotates forward
        knee = _add(hip, THIGH * math.sin(angle), -THIGH * math.cos(angle))
        ankle = _add(knee, dy=-SHIN)
        pose.update({f"{side}_hip": hip, f"{side}_knee": knee, f"{side}_ankle": ankle})
    pelvis = _add(pose["l_hip"], dz=HIP_WIDTH)
    hands = [_add(pelvis, 0.05, 0.1, -0.3), _add(pelvis, 0.05, 0.1, 0.3)]  # arms out for balance
    pose.update(_upper_body(pelvis, 0.0, hands))
    return pose


def sitting_pose(seat, holding_knee=False):
    """Sitting on a sofa (rest/skip) or on the floor holding the knee (injury)."""
    pose = {}
    for side, sign in (("l", -1), ("r", 1)):
        hip = _add(seat, dz=sign * HIP_WIDTH)
        knee = _add(hip, THIGH * 0.95, THIGH * 0.3 if holding_knee else 0.0)
        drop = min(SHIN, knee[1] - 0.05)            # feet stop at the floor
        ankle = _add(knee, 0.1, -drop)
        pose.update({f"{side}_hip": hip, f"{side}_knee": knee, f"{side}_ankle": ankle})
    pelvis = seat
    if holding_knee:
        hands = [_add(pose["r_knee"], dz=-0.06), _add(pose["r_knee"], dy=0.05, dz=0.06)]
        lean = 0.5
    else:
        hands = [_add(pelvis, 0.3, 0.05, -0.25), _add(pelvis, 0.3, 0.05, 0.25)]
        lean = -0.2  # leaning back, relaxed
    pose.update(_upper_body(pelvis, lean, hands))
    return pose


def standing_pose(base):
    """Neutral pose (used for the physiotherapist)."""
    return knee_lift_pose(base, 0.0)


def form_color(form):
    if form >= 0.7:
        return GREEN
    return AMBER if form >= 0.45 else RED


# ---------------------------------------------------------------- derived joints
def _elbow(shoulder, hand, bend):
    """Two-bone IK: place the elbow so both arm segments keep their length.

    `bend` is the direction the elbow should point. If the hand is out of
    reach, the arm straightens and the hand stops at full reach.
    """
    reach = _sub(hand, shoulder)
    along = _unit(reach)
    dist = max(1e-3, min(math.sqrt(_dot(reach, reach)), 0.999 * (UPPER_ARM + FOREARM)))
    out = _unit(_sub(bend, _scale(along, _dot(bend, along))))   # part of `bend` across the arm
    x = (UPPER_ARM ** 2 - FOREARM ** 2 + dist ** 2) / (2 * dist)
    y = math.sqrt(max(0.0, UPPER_ARM ** 2 - x * x))
    elbow = _plus(shoulder, _plus(_scale(along, x), _scale(out, y)))
    return elbow, _plus(shoulder, _scale(along, dist))


def _with_elbows(pose):
    pose = dict(pose)
    for side, sign in (("l", -1), ("r", 1)):
        bend = (-0.5, -0.6, sign * 0.8)             # elbows point back, down and out
        pose[f"{side}_elbow"], pose[f"{side}_hand"] = _elbow(
            pose[f"{side}_shoulder"], pose[f"{side}_hand"], bend)
    return pose


# ---------------------------------------------------------------- body parts
def _frame(origin, up, side):
    """4x4 matrix for glMultMatrixf: local x = forward, y = `up`, z = `side`."""
    up = _unit(up)
    side = _unit(_sub(side, _scale(up, _dot(side, up))))
    forward = _cross(up, side)
    return [*forward, 0.0, *up, 0.0, *side, 0.0, *origin, 1.0]


def _ellipsoid(center, up, side, radii, color):
    """Sphere stretched to `radii` = (forward, up, side), turned to face the body's front."""
    glPushMatrix()
    glMultMatrixf(_frame(center, up, side))
    glScalef(*radii)
    sphere((0.0, 0.0, 0.0), 1.0, color)
    glPopMatrix()


def _shoe(ankle):
    """Flat box under the ankle, toes pointing forward (+x)."""
    box(_add(ankle, 0.05, -0.045), (0.22, 0.07, 0.1), SHOES)


def _limb(a, b, radii, color):
    cylinder_between(a, b, radii[0], color, end_radius=radii[1])
    sphere(b, radii[1], color)                      # rounds off the joint


def draw_figure(pose, shirt=SHIRT, joint_color=None, bad_joint=None, pulse=0.0, origin=(0, 0, 0), yaw=0.0):
    """Draw the body part by part.

    `joint_color` colours the rings around the knees (form quality); None
    draws no rings (the physiotherapist). `bad_joint` gets a pulsing red
    marker. `origin` and `yaw` (degrees) place and turn the whole figure,
    e.g. to sit facing out of the sofa.
    """
    p = _with_elbows(pose)
    up = _unit(_sub(p["neck"], p["pelvis"]))
    side = _sub(p["r_shoulder"], p["l_shoulder"])
    glPushMatrix()
    glTranslatef(*origin)
    glRotatef(yaw, 0, 1, 0)

    for s in ("l", "r"):                            # legs
        sphere(p[f"{s}_hip"], THIGH_R[0], TROUSERS)
        _limb(p[f"{s}_hip"], p[f"{s}_knee"], THIGH_R, TROUSERS)
        _limb(p[f"{s}_knee"], p[f"{s}_ankle"], SHIN_R, TROUSERS)
        _shoe(p[f"{s}_ankle"])

    _ellipsoid(_plus(p["pelvis"], _scale(up, 0.04)), up, side, (0.11, 0.1, 0.19), TROUSERS)   # hips
    _ellipsoid(_plus(p["pelvis"], _scale(up, 0.31)), up, side, (0.13, 0.26, 0.19), shirt)     # chest
    cylinder_between(p["l_shoulder"], p["r_shoulder"], 0.065, shirt)                           # shoulders

    for s in ("l", "r"):                            # arms: short sleeve, bare forearm
        sphere(p[f"{s}_shoulder"], 0.065, shirt)
        _limb(p[f"{s}_shoulder"], p[f"{s}_elbow"], UPPER_ARM_R, shirt)
        sphere(p[f"{s}_elbow"], FOREARM_R[0], SKIN)
        _limb(p[f"{s}_elbow"], p[f"{s}_hand"], FOREARM_R, SKIN)
        sphere(p[f"{s}_hand"], 0.045, SKIN)

    cylinder_between(p["neck"], p["head"], 0.045, SKIN)                                        # neck
    sphere(p["head"], HEAD_RADIUS, SKIN)
    sphere(_add(p["head"], -0.02, 0.04), HEAD_RADIUS * 0.93, HAIR)                             # hair

    if joint_color is not None:                     # form rings around both knees
        for s in ("l", "r"):
            leg = _sub(p[f"{s}_ankle"], p[f"{s}_hip"])
            ring(p[f"{s}_knee"], leg, 0.085, 0.014, joint_color)
    if bad_joint is not None:
        sphere(p[bad_joint], 0.085 + 0.03 * pulse, RED)
    glPopMatrix()