import math

import pygame
from OpenGL.GL import (
    GL_BLEND, GL_DEPTH_TEST, GL_LIGHTING, GL_LINEAR, GL_LINES, GL_MODELVIEW,
    GL_ONE_MINUS_SRC_ALPHA, GL_PROJECTION, GL_QUAD_STRIP, GL_QUADS, GL_RGBA, GL_SRC_ALPHA,
    GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_TEXTURE_MIN_FILTER, GL_UNSIGNED_BYTE,
    glBegin, glBindTexture, glBlendFunc, glColor3f, glColor4f, glDisable, glEnable,
    glEnd, glGenTextures, glLoadIdentity, glMatrixMode, glNormal3f, glOrtho,
    glPopMatrix, glPushMatrix, glRotatef, glScalef, glTexCoord2f, glTexImage2D,
    glTexParameteri, glTranslatef, glVertex2f, glVertex3f,
)
from OpenGL.GLU import gluCylinder, gluNewQuadric, gluSphere

_QUADRIC = None
_TEXTURES = {}

# (normal, four corners) of a unit cube centred on the origin
_CUBE_FACES = [
    ((0, 0, 1), [(-1, -1, 1), (1, -1, 1), (1, 1, 1), (-1, 1, 1)]),
    ((0, 0, -1), [(-1, -1, -1), (-1, 1, -1), (1, 1, -1), (1, -1, -1)]),
    ((0, 1, 0), [(-1, 1, -1), (-1, 1, 1), (1, 1, 1), (1, 1, -1)]),
    ((0, -1, 0), [(-1, -1, -1), (1, -1, -1), (1, -1, 1), (-1, -1, 1)]),
    ((1, 0, 0), [(1, -1, -1), (1, 1, -1), (1, 1, 1), (1, -1, 1)]),
    ((-1, 0, 0), [(-1, -1, -1), (-1, -1, 1), (-1, 1, 1), (-1, 1, -1)]),
]


def _quadric():
    global _QUADRIC
    if _QUADRIC is None:
        _QUADRIC = gluNewQuadric()
    return _QUADRIC


def box(center, size, color):
    """Axis-aligned box. `size` is the full width, height and depth."""
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(*center)
    glScalef(size[0] / 2, size[1] / 2, size[2] / 2)
    glBegin(GL_QUADS)
    for normal, corners in _CUBE_FACES:
        glNormal3f(*normal)
        for corner in corners:
            glVertex3f(*corner)
    glEnd()
    glPopMatrix()


def sphere(center, radius, color):
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(*center)
    gluSphere(_quadric(), radius, 16, 12)
    glPopMatrix()


def _point_z_along(direction):
    """Rotate the current matrix so its +z axis points along `direction`. Returns the length."""
    dx, dy, dz = direction
    length = math.sqrt(dx * dx + dy * dy + dz * dz)
    if length < 1e-6:
        return 0.0
    angle = math.degrees(math.acos(max(-1.0, min(1.0, dz / length))))
    axis = (-dy, dx, 0.0) if abs(dx) + abs(dy) > 1e-6 else (1.0, 0.0, 0.0)
    glRotatef(angle, *axis)
    return length


def cylinder_between(start, end, radius, color, end_radius=None):
    """Cylinder from `start` to `end`. Give `end_radius` for a tapered limb."""
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(*start)
    # gluCylinder points along +z; rotate +z onto the limb direction.
    length = _point_z_along([end[i] - start[i] for i in range(3)])
    if length > 0.0:
        top = radius if end_radius is None else end_radius
        gluCylinder(_quadric(), radius, top, length, 16, 1)
    glPopMatrix()


def ring(center, axis, radius, tube, color, segments=24, sides=8):
    """Torus at `center` wrapped around `axis` (form markers around the knees)."""
    glColor3f(*color)
    glPushMatrix()
    glTranslatef(*center)
    _point_z_along(axis)
    for i in range(segments):
        glBegin(GL_QUAD_STRIP)
        for j in range(sides + 1):
            v = 2 * math.pi * j / sides
            for k in (i, i + 1):
                u = 2 * math.pi * k / segments
                glNormal3f(math.cos(v) * math.cos(u), math.cos(v) * math.sin(u), math.sin(v))
                r = radius + tube * math.cos(v)
                glVertex3f(r * math.cos(u), r * math.sin(u), tube * math.sin(v))
        glEnd()
    glPopMatrix()


def floor_grid(half_size, step, color):
    glDisable(GL_LIGHTING)
    glColor3f(*color)
    glBegin(GL_LINES)
    n = int(half_size / step)
    for i in range(-n, n + 1):
        glVertex3f(i * step, 0.002, -half_size)
        glVertex3f(i * step, 0.002, half_size)
        glVertex3f(-half_size, 0.002, i * step)
        glVertex3f(half_size, 0.002, i * step)
    glEnd()
    glEnable(GL_LIGHTING)


def _upload_texture(name, surface):
    """Copy a pygame surface into an OpenGL texture (re-used every frame)."""
    if name not in _TEXTURES:
        _TEXTURES[name] = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, _TEXTURES[name])
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    data = pygame.image.tobytes(surface, "RGBA", True)
    w, h = surface.get_size()
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, data)


def textured_quad_3d(name, surface, corners):
    """Draw `surface` on a 3D rectangle. corners: bottom-left, bottom-right, top-right, top-left."""
    _upload_texture(name, surface)
    glDisable(GL_LIGHTING)
    glEnable(GL_TEXTURE_2D)
    glColor4f(1, 1, 1, 1)
    glBegin(GL_QUADS)
    for (u, v), corner in zip([(0, 0), (1, 0), (1, 1), (0, 1)], corners):
        glTexCoord2f(u, v)
        glVertex3f(*corner)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    glEnable(GL_LIGHTING)


def overlay_2d(name, surface, x, y, window_size):
    """Draw `surface` flat on screen with its top-left at pixel (x, y)."""
    win_w, win_h = window_size
    w, h = surface.get_size()
    _upload_texture(name, surface)
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, win_w, win_h, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_LIGHTING)
    glDisable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glEnable(GL_TEXTURE_2D)
    glColor4f(1, 1, 1, 1)
    glBegin(GL_QUADS)
    for (u, v), (px, py) in zip([(0, 1), (1, 1), (1, 0), (0, 0)], [(x, y), (x + w, y), (x + w, y + h), (x, y + h)]):
        glTexCoord2f(u, v)
        glVertex2f(px, py)
    glEnd()
    glDisable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)
    glPopMatrix()