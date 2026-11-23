import math
import os

import numpy as np
import pygame
from OpenGL.GL import (
    GL_AMBIENT, GL_AMBIENT_AND_DIFFUSE, GL_COLOR_BUFFER_BIT, GL_COLOR_MATERIAL,
    GL_DEPTH_BUFFER_BIT, GL_DEPTH_TEST, GL_DIFFUSE, GL_FRONT_AND_BACK, GL_LIGHT0,
    GL_LIGHTING, GL_MODELVIEW, GL_NORMALIZE, GL_POSITION, GL_PROJECTION, GL_RGB,
    GL_UNSIGNED_BYTE, glClear, glClearColor, glColorMaterial, glEnable, glLightfv,
    glLoadIdentity, glMatrixMode, glReadPixels, glViewport,
)
from OpenGL.GLU import gluLookAt, gluPerspective

from environment import figure, hud, scene
from environment.gl_shapes import overlay_2d
from environment import patient_model as pm

# ---------------------------------------------------------------- settings
WINDOW_SIZE = (1280, 720)
PANEL_WIDTH = hud.PANEL_SIZE[0]
CAMERA_EYE = (-0.6, 2.4, 4.8)
CAMERA_TARGET = (0.1, 0.9, -0.4)
FPS = 30
FRAMES_PER_DAY = 45        # 1.5 s per simulated day in human mode
END_HOLD_FRAMES = 90       # keep the end banner on screen for 3 s
PATIENT_BASE = (0.0, 0.02, 0.0)


class RehabRenderer:
    def __init__(self, visible=True):
        pygame.init()
        pygame.font.init()
        flags = pygame.OPENGL | pygame.DOUBLEBUF | (0 if visible else pygame.HIDDEN)
        pygame.display.set_mode(WINDOW_SIZE, flags)
        pygame.display.set_caption("Adaptive Home Rehab Coach")
        self.clock = pygame.time.Clock()
        self.frames_per_day = FRAMES_PER_DAY
        self.window_closed = False
        self.last_frame = None
        glEnable(GL_DEPTH_TEST)
        glEnable(GL_LIGHTING)
        glEnable(GL_LIGHT0)
        glEnable(GL_NORMALIZE)
        glEnable(GL_COLOR_MATERIAL)
        glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

    # ------------------------------------------------------------ public
    def draw(self, env, return_rgb=False):
        """Animate one simulated day. Returns an RGB array if asked."""
        if return_rgb:
            ended = env.outcome != pm.OUTCOME_RUNNING
            self._draw_frame(env, 0.5, show_banner=ended)
            return self.last_frame
        for frame in range(self.frames_per_day):
            phase = (frame + 1) / self.frames_per_day   # 0 = morning, 1 = evening
            self._draw_frame(env, phase, show_banner=False)
        if env.outcome != pm.OUTCOME_RUNNING:
            for _ in range(END_HOLD_FRAMES):
                self._draw_frame(env, 1.0, show_banner=True)
        return None

    def _read_pixels(self):
        """Copy the frame we just drew. Must run *before* flip()."""
        w, h = WINDOW_SIZE
        data = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
        image = np.frombuffer(data, dtype=np.uint8).reshape(h, w, 3)
        return np.flipud(image).copy()              # OpenGL rows start at the bottom

    def save_screenshot(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        surface = pygame.surfarray.make_surface(self.last_frame.swapaxes(0, 1))
        pygame.image.save(surface, path)
        return path

    def close(self):
        pygame.quit()

    # ------------------------------------------------------------ one frame
    def _draw_frame(self, env, phase, show_banner):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.window_closed = True
        self._setup_3d(phase)
        tick = pygame.time.get_ticks() / 1000.0
        scene.draw_room(phase)
        scene.draw_mat()
        scene.draw_sofa()
        scene.draw_weights(env.difficulty, env.sets)
        scene.draw_calendar(env.history, pm.MAX_DAYS)
        scene.draw_phone(hud.phone_surface(env, progress=phase))
        self._draw_people(env, tick)
        self._draw_2d(env, show_banner)
        self.last_frame = self._read_pixels()       # kept for screenshots
        pygame.display.flip()
        self.clock.tick(FPS)

    def _setup_3d(self, phase):
        w, h = WINDOW_SIZE
        glClearColor(0.12, 0.13, 0.16, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glViewport(0, 0, w - PANEL_WIDTH, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45.0, (w - PANEL_WIDTH) / h, 0.1, 50.0)
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        gluLookAt(*CAMERA_EYE, *CAMERA_TARGET, 0, 1, 0)
        # The light comes through the window and takes the sky's colour.
        sky = scene.window_light(phase)
        glLightfv(GL_LIGHT0, GL_POSITION, (scene.WINDOW_CENTER[0], 2.5, 1.0, 1.0))
        light = [0.45 + 0.4 * channel for channel in sky]      # half white, half sky colour
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (*light, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (0.35, 0.35, 0.38, 1.0))

    def _draw_people(self, env, tick):
        """Choose the patient's pose from the day's status and the outcome."""
        joint_color = figure.form_color(env.form)
        bad_joint = None
        origin, yaw = (0, 0, 0), 0.0
        pulse = 0.5 + 0.5 * math.sin(tick * 6)
        if env.outcome == pm.OUTCOME_INJURY:
            pose = figure.sitting_pose((0.0, 0.12, 0.0), holding_knee=True)
            joint_color, bad_joint = figure.RED, "r_knee"
        elif env.outcome == pm.OUTCOME_DROPOUT or env.status in ("rest", "skipped"):
            pose = figure.sitting_pose((0.0, scene.SOFA_SEAT[1], 0.0))
            origin, yaw = (scene.SOFA_SEAT[0], 0.0, scene.SOFA_SEAT[2]), -90.0  # face out of the sofa
        elif env.status == "session":
            pose, bad_joint = self._squat(env, tick)
        elif env.status == "mobility":
            lift = (0.3 + 0.7 * env.rom) * (0.5 - 0.5 * math.cos(tick * 1.2))  # slow
            pose = figure.knee_lift_pose(PATIENT_BASE, min(1.0, lift))
        else:
            pose = figure.standing_pose(PATIENT_BASE)
        figure.draw_figure(pose, joint_color=joint_color, bad_joint=bad_joint, pulse=pulse, origin=origin, yaw=yaw)
        if env.status == "review":
            physio = figure.standing_pose((0.6, 0.02, -0.8))
            figure.draw_figure(physio, shirt=figure.PHYSIO_SHIRT)

    def _squat(self, env, tick):
        """Depth grows with difficulty and current ROM; speed grows with sets."""
        reps_per_second = 0.4 + 0.12 * env.sets
        max_depth = (0.25 + 0.12 * env.difficulty) * (0.5 + 0.5 * env.rom)
        depth = max_depth * (0.5 - 0.5 * math.cos(2 * math.pi * reps_per_second * tick))
        pose = figure.squat_pose(PATIENT_BASE, min(1.0, depth), knee_cave=0.12 * (1 - env.form))
        bad_joint = None
        if env.form < pm.GOOD_FORM:
            overreach = env.difficulty > 1 + 4 * env.rom
            bad_joint = "r_knee" if overreach else "r_hip"   # knee strain vs fatigue
        return pose, bad_joint

    def _draw_2d(self, env, show_banner):
        w, h = WINDOW_SIZE
        glViewport(0, 0, w, h)
        overlay_2d("hud", hud.hud_surface(env), w - PANEL_WIDTH, 0, WINDOW_SIZE)
        if show_banner:
            overlay_2d("banner", hud.banner_surface(env), (w - PANEL_WIDTH - 520) // 2, 40, WINDOW_SIZE)


def save_env_screenshot(env, path="assets/environment.png"):
    """Draw the env's current state once and save it (used for the report)."""
    if env.renderer is None:
        env.renderer = RehabRenderer()
    env.renderer.draw(env, return_rgb=True)
    return env.renderer.save_screenshot(path)


if __name__ == "__main__":
    # Random-agent demo: `uv run python -m environment.rendering`
    import random

    from environment import RehabCoachEnv

    demo_env = RehabCoachEnv(render_mode="human")
    demo_env.reset(seed=1)
    demo_env.render()
    demo_env.renderer.frames_per_day = 10
    finished = False
    while not finished:
        _, _, terminated, truncated, _ = demo_env.step(random.randrange(pm.N_ACTIONS))
        finished = terminated or truncated or demo_env.renderer.window_closed
        if demo_env.day == 12:
            print("screenshot:", demo_env.renderer.save_screenshot("assets/environment.png"))
    demo_env.close()