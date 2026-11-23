from OpenGL.GL import GL_LIGHTING, glDisable, glEnable, glPopMatrix, glPushMatrix, glRotatef, glTranslatef

from environment.gl_shapes import box, cylinder_between, floor_grid, sphere, textured_quad_3d

# ---------------------------------------------------------------- layout (metres)
ROOM_HALF = 3.0
FLOOR_COLOR, GRID_COLOR = (0.62, 0.55, 0.45), (0.5, 0.44, 0.36)
WALL_COLOR = (0.78, 0.72, 0.62)
MAT_CENTER, MAT_SIZE, MAT_COLOR = (0.0, 0.01, 0.0), (1.6, 0.02, 0.9), (0.3, 0.55, 0.6)
SOFA_CENTER = (-1.9, 0.0, -1.9)
SOFA_SEAT = (-1.9, 0.45, -1.75)         # where the patient sits on rest/skip days
SOFA_COLOR = (0.55, 0.25, 0.25)
WEIGHTS_START = (-0.4, 0.0, 0.75)       # first dumbbell; more appear in a row along +x
WINDOW_CENTER, WINDOW_SIZE = (-0.6, 1.7, -2.98), (1.4, 1.0)
CALENDAR_ORIGIN = (0.9, 2.4, -2.97)     # top-left square, on the back wall
CALENDAR_CELL = 0.17
PHONE_BASE = (1.4, 0.0, 0.1)            # tripod position in front of the mat
PHONE_YAW = 45.0                        # degrees: screen turned toward patient and camera
PHONE_SIZE = (0.32, 0.56)

CALENDAR_COLORS = {
    None: (0.8, 0.8, 0.84), "session": (0.2, 0.75, 0.3), "mobility": (0.2, 0.75, 0.3),
    "rest": (0.6, 0.6, 0.6), "skipped": (0.85, 0.2, 0.2), "review": (0.25, 0.45, 0.9),
}


def window_light(phase):
    """Sky colour across the day: morning (pale) -> noon (bright) -> evening (orange)."""
    morning, noon, evening = (0.55, 0.7, 0.95), (0.45, 0.75, 1.0), (1.0, 0.5, 0.25)
    if phase < 0.5:
        a, b, t = morning, noon, phase / 0.5
    else:
        a, b, t = noon, evening, (phase - 0.5) / 0.5
    return tuple(a[i] + (b[i] - a[i]) * t for i in range(3))


def draw_room(phase):
    box((0, -0.01, 0), (2 * ROOM_HALF, 0.02, 2 * ROOM_HALF), FLOOR_COLOR)
    floor_grid(ROOM_HALF, 0.5, GRID_COLOR)
    box((0, 1.5, -ROOM_HALF), (2 * ROOM_HALF, 3.0, 0.02), WALL_COLOR)   # back wall
    box((-ROOM_HALF, 1.5, 0), (0.02, 3.0, 2 * ROOM_HALF), WALL_COLOR)   # left wall
    x, y, z = WINDOW_CENTER
    w, h = WINDOW_SIZE
    box((x, y, z + 0.01), (w + 0.1, h + 0.1, 0.02), (0.4, 0.3, 0.2))    # frame
    glDisable(GL_LIGHTING)                                              # the sky glows
    box((x, y, z + 0.02), (w, h, 0.02), window_light(phase))            # sky = time of day
    glEnable(GL_LIGHTING)


def draw_mat():
    box(MAT_CENTER, MAT_SIZE, MAT_COLOR)


def draw_sofa():
    x, _, z = SOFA_CENTER
    box((x, 0.22, z), (1.4, 0.44, 0.7), SOFA_COLOR)                     # seat
    box((x, 0.65, z - 0.3), (1.4, 0.5, 0.15), SOFA_COLOR)               # backrest
    for dx in (-0.7, 0.7):
        box((x + dx, 0.35, z), (0.15, 0.7, 0.7), SOFA_COLOR)            # arms


def draw_weights(difficulty, sets):
    """One dumbbell per set; dumbbells get bigger with difficulty."""
    size = 0.03 + 0.01 * difficulty
    for i in range(sets):
        x = WEIGHTS_START[0] + i * 0.25
        z = WEIGHTS_START[2]
        y = size
        cylinder_between((x, y, z - 0.12), (x, y, z + 0.12), 0.015, (0.3, 0.3, 0.3))
        sphere((x, y, z - 0.12), size, (0.15, 0.15, 0.15))
        sphere((x, y, z + 0.12), size, (0.15, 0.15, 0.15))


def draw_calendar(history, max_days):
    """42 squares on the wall, 7 per row, coloured by what happened each day."""
    left, top, z = CALENDAR_ORIGIN
    box((left + 0.55, top - 0.45, z), (1.3, 1.15, 0.01), (0.97, 0.96, 0.9))   # paper
    for day in range(max_days):
        status = history[day]["status"] if day < len(history) else None
        row, col = divmod(day, 7)
        x = left + 0.03 + col * CALENDAR_CELL
        y = top - 0.08 - row * CALENDAR_CELL
        box((x, y, z + 0.01), (CALENDAR_CELL * 0.8, CALENDAR_CELL * 0.8, 0.01), CALENDAR_COLORS[status])
        if status == "review":                                         # "Review" marker
            sphere((x, y, z + 0.03), 0.03, (1.0, 1.0, 0.2))


def draw_phone(screen_surface):
    """Tripod with the capstone app's phone; its screen is a live texture.

    Drawn in a local frame where the screen faces -x, then turned by PHONE_YAW.
    """
    glPushMatrix()
    glTranslatef(*PHONE_BASE)
    glRotatef(PHONE_YAW, 0, 1, 0)
    for foot in ((0.12, -0.15), (0.12, 0.15), (-0.12, 0.0)):
        cylinder_between((foot[0], 0.0, foot[1]), (0, 1.0, 0), 0.012, (0.2, 0.2, 0.2))
    w, h = PHONE_SIZE
    box((0.02, 1.0 + h / 2, 0), (0.03, h + 0.04, w + 0.04), (0.05, 0.05, 0.05))
    sx = -0.001
    corners = [(sx, 1.0, -w / 2), (sx, 1.0, w / 2), (sx, 1.0 + h, w / 2), (sx, 1.0 + h, -w / 2)]
    textured_quad_3d("phone", screen_surface, corners)
    glPopMatrix()