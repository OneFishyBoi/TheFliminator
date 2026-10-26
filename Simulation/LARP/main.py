import threading
import time
from collections import deque

import numpy as np
import pygame

from setup import Grid

# ---- Simulation settings: change these to suit your run ----
NX = 20
NY = 20
X_LENGTH = 1.0
Y_LENGTH = 1.0
DENSITY = 1.0e3
VISCOSITY = 1.0e-2
INITIAL_DT = 0.01
TOTAL_TIME = 5.0
PRESSURE_TOLERANCE = 1.0e-2
INITIAL_VELOCITY = np.array([0.0, 0.0], dtype=float)
INITIAL_PRESSURE = 0.0
INITIAL_VOLUME_FRACTION = 1.0

# ---- Window settings ----
WIDTH, HEIGHT = 1120, 760
PANEL_W = 250
FPS = 60
MAX_SAVED_FRAMES = 500
PARAMETER_FIELDS = (
    ("nx", "X grid partitions", int),
    ("ny", "Y grid partitions", int),
    ("x_length", "X length", float),
    ("y_length", "Y length", float),
    ("density", "Density", float),
    ("viscosity", "Viscosity", float),
    ("dt", "Initial time step", float),
    ("total_time", "Total simulation time", float),
    ("pressure_tolerance", "Pressure tolerance", float),
    ("velocity_x", "Initial velocity X", float),
    ("velocity_y", "Initial velocity Y", float),
    ("pressure", "Initial pressure", float),
    ("volume_fraction", "Initial volume fraction", float),
    ("body_force_x", "Body force X", float),
    ("body_force_y", "Body force Y", float),
)
BACKGROUND = (24, 27, 33)
PANEL = (35, 39, 47)
TEXT = (230, 233, 238)
MUTED = (160, 168, 180)
ACCENT = (80, 170, 245)


def pressure_to_surface(pressure, size):
    """Blue-white-red pressure map, scaled robustly around its mean."""
    p = np.asarray(pressure, dtype=float)
    finite = np.isfinite(p)
    if not finite.any():
        p = np.zeros_like(p)
    else:
        fill = float(np.mean(p[finite]))
        p = np.where(finite, p, fill)
    p = p - np.mean(p)
    scale = float(np.max(np.abs(p)))
    if scale > 0:
        z = np.clip(p / scale, -1.0, 1.0)
    else:
        z = np.zeros_like(p)

    # Convert grid's (x,y) indexing to screen image coordinates.
    z = np.flipud(z.T)
    rgb = np.empty((*z.shape, 3), dtype=np.uint8)
    neg = z < 0
    pos = ~neg
    a = np.abs(z)
    # Blue -> white for negative values; white -> red for positive values.
    rgb[neg, 0] = (255 * (1 - a[neg])).astype(np.uint8)
    rgb[neg, 1] = (255 * (1 - a[neg])).astype(np.uint8)
    rgb[neg, 2] = 255
    rgb[pos, 0] = 255
    rgb[pos, 1] = (255 * (1 - a[pos])).astype(np.uint8)
    rgb[pos, 2] = (255 * (1 - a[pos])).astype(np.uint8)
    surf = pygame.surfarray.make_surface(rgb)
    return pygame.transform.smoothscale(surf, size)


class CFDViewer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("TheFliminator — CFD viewer")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("consolas", 18)
        self.small = pygame.font.SysFont("consolas", 15)
        self.title_font = pygame.font.SysFont("consolas", 23, bold=True)

        self.parameters = {
            "nx": NX,
            "ny": NY,
            "x_length": X_LENGTH,
            "y_length": Y_LENGTH,
            "density": DENSITY,
            "viscosity": VISCOSITY,
            "dt": INITIAL_DT,
            "total_time": TOTAL_TIME,
            "pressure_tolerance": PRESSURE_TOLERANCE,
            "velocity_x": float(INITIAL_VELOCITY[0]),
            "velocity_y": float(INITIAL_VELOCITY[1]),
            "pressure": INITIAL_PRESSURE,
            "volume_fraction": INITIAL_VOLUME_FRACTION,
            "body_force_x": 0.0,
            "body_force_y": 0.0,
        }
        self.grid = self.make_grid()
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.worker = None
        self.frames = deque(maxlen=MAX_SAVED_FRAMES)
        self.running = False
        self.finished = False
        self.status = "Ready"
        self.sim_time = 0.0
        self.pressure = self.grid.get_pressures().copy()
        self.velocity = self.grid.get_velocities().copy()
        self.normalise_arrows = True
        self.show_pressure = True
        self.show_arrows = True
        self.playing = False
        self.frame_index = 0
        self.last_frame_time = time.time()
        self.replay_fps = 12
        self.show_settings = False
        self.parameter_draft = {}
        self.active_parameter = None
        self.parameter_text = ""
        self.parameter_select_all = False
        self.parameter_scroll = 0
        self.parameter_error = ""

    def make_grid(self):
        params = self.parameters
        body_force = np.zeros((params["nx"], params["ny"], 2), dtype=float)
        body_force[:, :, 0] = params["body_force_x"]
        body_force[:, :, 1] = params["body_force_y"]
        return Grid(
            density=params["density"],
            viscosity=params["viscosity"],
            x_length=params["x_length"],
            y_length=params["y_length"],
            x_partitions=params["nx"],
            y_partitions=params["ny"],
            dt=params["dt"],
            total_time=params["total_time"],
            initial_velocity=np.array(
                [params["velocity_x"], params["velocity_y"]], dtype=float
            ),
            initial_pressure=params["pressure"],
            initial_volume_fraction=params["volume_fraction"],
            body_force=body_force,
        )

    def report(self, t, pressures, velocities):
        with self.lock:
            self.sim_time = t
            self.pressure = pressures
            self.velocity = velocities
            self.frames.append((t, pressures.copy(), velocities.copy()))
            self.frame_index = len(self.frames) - 1

    def start(self):
        if self.running:
            return
        # Start from the current grid state; create a fresh grid after a completed run.
        if self.finished or self.stop_event.is_set():
            self.grid = self.make_grid()
            self.pressure = self.grid.get_pressures().copy()
            self.velocity = self.grid.get_velocities().copy()
            self.sim_time = 0.0
            self.frames.clear()
            self.frame_index = 0
        self.stop_event.clear()
        self.finished = False
        self.running = True
        self.playing = False
        self.status = "Running"
        self.worker = threading.Thread(target=self.run_simulation, daemon=True)
        self.worker.start()

    def run_simulation(self):
        try:
            self.grid.simulate(
                pressure_error_tolerance=self.parameters["pressure_tolerance"],
                progress_callback=self.report,
                stop_requested=self.stop_event.is_set,
            )
            self.status = "Stopped" if self.stop_event.is_set() else "Finished"
        except Exception as exc:
            self.status = f"Error: {exc}"
        finally:
            self.running = False
            self.finished = True

    def stop(self):
        if self.running:
            self.stop_event.set()
            self.status = "Stopping…"

    def change_replay_fps(self, delta):
        self.replay_fps = max(1, min(60, self.replay_fps + delta))
        self.last_frame_time = time.time()

    def text(self, value, x, y, colour=TEXT, font=None):
        img = (font or self.font).render(str(value), True, colour)
        self.screen.blit(img, (x, y))

    def button(self, rect, label, enabled=True):
        colour = (58, 65, 77) if enabled else (47, 50, 57)
        pygame.draw.rect(self.screen, colour, rect, border_radius=6)
        pygame.draw.rect(self.screen, (82, 91, 106), rect, width=1, border_radius=6)
        surf = self.font.render(label, True, TEXT if enabled else MUTED)
        self.screen.blit(surf, surf.get_rect(center=rect.center))

    def open_settings(self):
        if self.running:
            return
        self.show_settings = True
        self.parameter_draft = {
            key: str(value) for key, value in self.parameters.items()
        }
        self.active_parameter = None
        self.parameter_error = ""
        self.parameter_scroll = 0

    def settings_layout(self):
        w, h = self.screen.get_size()
        panel = pygame.Rect(0, 0, min(620, w - 32), min(690, h - 32))
        panel.center = (w // 2, h // 2)
        row_top = panel.top + 76
        row_bottom = panel.bottom - 78
        row_height = 38
        visible_rows = max(1, (row_bottom - row_top) // row_height)
        max_scroll = max(0, len(PARAMETER_FIELDS) - visible_rows)
        self.parameter_scroll = min(self.parameter_scroll, max_scroll)
        return panel, row_top, row_bottom, row_height, visible_rows, max_scroll

    def save_settings(self):
        try:
            updated = {}
            for key, label, value_type in PARAMETER_FIELDS:
                value = value_type(self.parameter_draft[key].strip())
                if not np.isfinite(value):
                    raise ValueError(f"{label} must be a finite number.")
                updated[key] = value

            for key in ("nx", "ny"):
                if updated[key] < 3:
                    raise ValueError("Grid partitions must be at least 3.")
            for key, label in (
                ("x_length", "X length"),
                ("y_length", "Y length"),
                ("density", "Density"),
                ("dt", "Initial time step"),
                ("pressure_tolerance", "Pressure tolerance"),
            ):
                if updated[key] <= 0:
                    raise ValueError(f"{label} must be greater than zero.")
            if updated["viscosity"] < 0:
                raise ValueError("Viscosity cannot be negative.")
            if updated["total_time"] < 0:
                raise ValueError("Total simulation time cannot be negative.")
            if not 0 <= updated["volume_fraction"] <= 1:
                raise ValueError("Initial volume fraction must be between 0 and 1.")
        except (KeyError, ValueError, OverflowError) as exc:
            self.parameter_error = str(exc) or "Enter valid parameter values."
            return

        self.parameters = updated
        self.grid = self.make_grid()
        self.pressure = self.grid.get_pressures().copy()
        self.velocity = self.grid.get_velocities().copy()
        self.sim_time = 0.0
        self.frames.clear()
        self.frame_index = 0
        self.playing = False
        self.finished = False
        self.stop_event.clear()
        self.status = "Ready"
        self.show_settings = False
        self.parameter_error = ""

    def draw_settings(self):
        panel, row_top, row_bottom, row_height, visible_rows, max_scroll = self.settings_layout()
        overlay = pygame.Surface(self.screen.get_size(), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 175))
        self.screen.blit(overlay, (0, 0))
        pygame.draw.rect(self.screen, PANEL, panel, border_radius=10)
        pygame.draw.rect(self.screen, (82, 91, 106), panel, width=1, border_radius=10)
        self.text("Initial parameters", panel.left + 22, panel.top + 16, font=self.title_font)
        self.text("Body force is uniform across the grid. Click a value to edit.",
                  panel.left + 22, panel.top + 47, MUTED, self.small)

        old_clip = self.screen.get_clip()
        self.screen.set_clip(pygame.Rect(
            panel.left + 12, row_top, panel.width - 24, row_bottom - row_top
        ))
        for visible_index in range(visible_rows):
            field_index = self.parameter_scroll + visible_index
            if field_index >= len(PARAMETER_FIELDS):
                break
            key, label, _ = PARAMETER_FIELDS[field_index]
            y = row_top + visible_index * row_height
            row = pygame.Rect(panel.left + 16, y, panel.width - 32, row_height - 2)
            pygame.draw.rect(self.screen, (43, 48, 58), row, border_radius=5)
            self.text(label, row.left + 10, row.top + 8, font=self.small)
            value_rect = pygame.Rect(row.right - 164, row.top + 3, 154, row.height - 6)
            active = key == self.active_parameter
            pygame.draw.rect(
                self.screen, (60, 74, 91) if active else (32, 36, 43),
                value_rect, border_radius=4
            )
            pygame.draw.rect(
                self.screen, ACCENT if active else (70, 76, 87),
                value_rect, width=1, border_radius=4
            )
            value = self.parameter_text if active else self.parameter_draft[key]
            self.text(value, value_rect.left + 7, value_rect.top + 5,
                      TEXT, self.small)
        self.screen.set_clip(old_clip)

        if max_scroll:
            self.text(f"{self.parameter_scroll + 1}-{min(len(PARAMETER_FIELDS), self.parameter_scroll + visible_rows)} of {len(PARAMETER_FIELDS)}",
                      panel.right - 145, row_bottom + 5, MUTED, self.small)
        if self.parameter_error:
            self.text(self.parameter_error, panel.left + 20, panel.bottom - 68,
                      (255, 130, 120), self.small)
        self.button(pygame.Rect(panel.right - 226, panel.bottom - 48, 96, 34), "Cancel")
        self.button(pygame.Rect(panel.right - 116, panel.bottom - 48, 96, 34), "Save")
        self.settings_panel = panel
        self.settings_rows = (row_top, row_bottom, row_height, visible_rows)

    def plot_rect(self):
        w, h = self.screen.get_size()
        return pygame.Rect(PANEL_W + 24, 64, max(100, w - PANEL_W - 48), max(100, h - 125))

    def draw_arrows(self, rect, velocity):
        nx, ny, _ = velocity.shape
        stride = max(1, max(nx, ny) // 22)
        # Keep arrows legible while sampling the field.
        for i in range(1 + stride // 2, nx - 1, stride):
            for j in range(1 + stride // 2, ny - 1, stride):
                ux, vy = velocity[i, j]
                speed = float(np.hypot(ux, vy))
                if speed < 1e-12:
                    continue
                x = rect.left + (i + 0.5) / nx * rect.width
                y = rect.bottom - (j + 0.5) / ny * rect.height
                if self.normalise_arrows:
                    length = 17.0
                else:
                    length = min(32.0, speed * 20.0)
                dx = ux / speed * length
                dy = -vy / speed * length
                start = (int(x - dx * 0.45), int(y - dy * 0.45))
                end = (int(x + dx * 0.55), int(y + dy * 0.55))
                pygame.draw.line(self.screen, (22, 25, 30), start, end, 2)
                # Arrow head
                angle = np.arctan2(end[1] - start[1], end[0] - start[0])
                head = 5
                p1 = (end[0] - head * np.cos(angle - 0.55),
                      end[1] - head * np.sin(angle - 0.55))
                p2 = (end[0] - head * np.cos(angle + 0.55),
                      end[1] - head * np.sin(angle + 0.55))
                pygame.draw.polygon(self.screen, (22, 25, 30), [end, p1, p2])

    def draw(self):
        self.screen.fill(BACKGROUND)
        w, h = self.screen.get_size()
        pygame.draw.rect(self.screen, PANEL, (0, 0, PANEL_W, h))
        self.text("TheFliminator", 18, 17, font=self.title_font)
        self.text("2D incompressible flow", 18, 47, MUTED, self.small)

        self.button(pygame.Rect(18, 88, 100, 36), "Start", not self.running)
        self.button(pygame.Rect(130, 88, 100, 36), "Stop", self.running)
        self.text(f"Status: {self.status}", 18, 139, ACCENT, self.small)
        self.text(f"t = {self.sim_time:.4f} s", 18, 166)
        self.text(f"Saved frames: {len(self.frames)}", 18, 192, MUTED, self.small)

        self.button(pygame.Rect(18, 210, PANEL_W - 36, 34), "Initial parameters", not self.running)
        pygame.draw.line(self.screen, (70, 76, 87), (18, 258), (PANEL_W - 18, 258), 1)
        self.text("Display", 18, 273, font=self.title_font)
        self.text("[P] Pressure map", 18, 310, TEXT if self.show_pressure else MUTED)
        self.text("[V] Velocity arrows", 18, 339, TEXT if self.show_arrows else MUTED)
        self.text("[N] Normalise arrow lengths", 18, 368, TEXT if self.normalise_arrows else MUTED)
        self.text("Click text to toggle", 18, 391, MUTED, self.small)

        pygame.draw.line(self.screen, (70, 76, 87), (18, 425), (PANEL_W - 18, 425), 1)
        self.text("Replay", 18, 441, font=self.title_font)
        self.button(pygame.Rect(18, 481, 100, 34), "Play" if not self.playing else "Pause",
                    bool(self.frames))
        self.button(pygame.Rect(130, 481, 100, 34), "Live")
        self.text("← / → step frame", 18, 526, MUTED, self.small)
        self.text("Drag slider below plot", 18, 548, MUTED, self.small)
        self.button(pygame.Rect(18, 570, 38, 30), "-")
        fps_label = self.small.render(f"{self.replay_fps} FPS", True, TEXT)
        self.screen.blit(fps_label, fps_label.get_rect(center=(PANEL_W // 2, 585)))
        self.button(pygame.Rect(PANEL_W - 56, 570, 38, 30), "+")

        rect = self.plot_rect()
        with self.lock:
            p = self.pressure.copy()
            vel = self.velocity.copy()
            current_t = self.sim_time
            frames = list(self.frames)
        if frames and not self.running and 0 <= self.frame_index < len(frames):
            current_t, p, vel = frames[self.frame_index]
        if self.show_pressure:
            surf = pressure_to_surface(p, rect.size)
            self.screen.blit(surf, rect.topleft)
        else:
            self.screen.fill((245, 245, 245), rect)
        pygame.draw.rect(self.screen, (110, 118, 130), rect, width=1)
        if self.show_arrows:
            self.draw_arrows(rect, vel)
        self.text(f"Pressure + velocity  |  t={current_t:.4f}s", rect.left, 27, MUTED, self.small)

        # Timeline is draggable for replay.
        timeline = pygame.Rect(rect.left, h - 42, rect.width, 12)
        pygame.draw.rect(self.screen, (70, 76, 87), timeline, border_radius=5)
        if len(frames) > 1:
            frac = self.frame_index / (len(frames) - 1)
            marker_x = timeline.left + int(frac * timeline.width)
            pygame.draw.circle(self.screen, ACCENT, (marker_x, timeline.centery), 7)
        else:
            pygame.draw.circle(self.screen, ACCENT, (timeline.left, timeline.centery), 7)
        self.timeline_rect = timeline
        if self.show_settings:
            self.draw_settings()
        pygame.display.flip()

    def handle_click(self, pos):
        x, y = pos
        if self.show_settings:
            panel, row_top, row_bottom, row_height, visible_rows, _ = self.settings_layout()
            if pygame.Rect(panel.right - 226, panel.bottom - 48, 96, 34).collidepoint(pos):
                self.show_settings = False
                return
            if pygame.Rect(panel.right - 116, panel.bottom - 48, 96, 34).collidepoint(pos):
                if self.active_parameter is not None:
                    self.parameter_draft[self.active_parameter] = self.parameter_text
                self.save_settings()
                return
            if row_top <= y < row_bottom and panel.left <= x < panel.right:
                visible_index = (y - row_top) // row_height
                field_index = self.parameter_scroll + visible_index
                if visible_index < visible_rows and field_index < len(PARAMETER_FIELDS):
                    if self.active_parameter is not None:
                        self.parameter_draft[self.active_parameter] = self.parameter_text
                    key = PARAMETER_FIELDS[field_index][0]
                    self.active_parameter = key
                    self.parameter_text = self.parameter_draft[key]
                    self.parameter_select_all = True
                    self.parameter_error = ""
            elif not panel.collidepoint(pos):
                self.show_settings = False
            return
        if pygame.Rect(18, 88, 100, 36).collidepoint(pos):
            self.start()
        elif pygame.Rect(130, 88, 100, 36).collidepoint(pos):
            self.stop()
        elif pygame.Rect(18, 210, PANEL_W - 36, 34).collidepoint(pos):
            self.open_settings()
        elif 306 <= y <= 331 and x < PANEL_W:
            self.show_pressure = not self.show_pressure
        elif 333 <= y <= 359 and x < PANEL_W:
            self.show_arrows = not self.show_arrows
        elif 362 <= y <= 388 and x < PANEL_W:
            self.normalise_arrows = not self.normalise_arrows
        elif pygame.Rect(18, 481, 100, 34).collidepoint(pos):
            if self.frames:
                self.playing = not self.playing
                self.last_frame_time = time.time()
        elif pygame.Rect(130, 481, 100, 34).collidepoint(pos):
            self.playing = False
            self.frame_index = max(0, len(self.frames) - 1)
        elif pygame.Rect(18, 570, 38, 30).collidepoint(pos):
            self.change_replay_fps(-1)
        elif pygame.Rect(PANEL_W - 56, 570, 38, 30).collidepoint(pos):
            self.change_replay_fps(1)
        elif hasattr(self, "timeline_rect") and self.timeline_rect.inflate(0, 20).collidepoint(pos):
            self.playing = False
            self.set_frame_from_x(x)

    def set_frame_from_x(self, x):
        if len(self.frames) < 2:
            return
        timeline = self.timeline_rect
        frac = np.clip((x - timeline.left) / timeline.width, 0.0, 1.0)
        self.frame_index = int(round(frac * (len(self.frames) - 1)))
        self.sim_time = self.frames[self.frame_index][0]

    def run(self):
        running = True
        while running:
            self.clock.tick(FPS)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.stop()
                    running = False
                elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                    self.handle_click(event.pos)
                elif event.type == pygame.MOUSEWHEEL and self.show_settings:
                    panel, _, _, _, _, max_scroll = self.settings_layout()
                    if panel.collidepoint(pygame.mouse.get_pos()):
                        self.parameter_scroll = max(
                            0, min(max_scroll, self.parameter_scroll - event.y)
                        )
                elif event.type == pygame.KEYDOWN and self.show_settings:
                    if event.key == pygame.K_ESCAPE:
                        self.show_settings = False
                    elif event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        if self.active_parameter is not None:
                            self.parameter_draft[self.active_parameter] = self.parameter_text
                            self.active_parameter = None
                    elif self.active_parameter is not None:
                        if event.key == pygame.K_BACKSPACE:
                            if self.parameter_select_all:
                                self.parameter_text = ""
                            else:
                                self.parameter_text = self.parameter_text[:-1]
                            self.parameter_select_all = False
                        elif event.key == pygame.K_a and event.mod & pygame.KMOD_CTRL:
                            self.parameter_text = ""
                            self.parameter_select_all = False
                        elif event.unicode and event.unicode in "0123456789.eE+-":
                            if self.parameter_select_all:
                                self.parameter_text = event.unicode
                            else:
                                self.parameter_text += event.unicode
                            self.parameter_select_all = False
                elif event.type == pygame.MOUSEMOTION and event.buttons[0]:
                    if (not self.show_settings and hasattr(self, "timeline_rect")
                            and self.timeline_rect.inflate(0, 20).collidepoint(event.pos)):
                        self.playing = False
                        self.set_frame_from_x(event.pos[0])
                elif event.type == pygame.KEYDOWN:
                    if self.show_settings:
                        continue
                    if event.key == pygame.K_ESCAPE:
                        self.stop()
                        running = False
                    elif event.key == pygame.K_SPACE:
                        self.playing = not self.playing if self.frames else False
                    elif event.key == pygame.K_p:
                        self.show_pressure = not self.show_pressure
                    elif event.key == pygame.K_v:
                        self.show_arrows = not self.show_arrows
                    elif event.key == pygame.K_n:
                        self.normalise_arrows = not self.normalise_arrows
                    elif event.key == pygame.K_LEFT and self.frames:
                        self.playing = False
                        self.frame_index = max(0, self.frame_index - 1)
                    elif event.key == pygame.K_RIGHT and self.frames:
                        self.playing = False
                        self.frame_index = min(len(self.frames) - 1, self.frame_index + 1)

            if self.playing and len(self.frames) > 1 and time.time() - self.last_frame_time >= 1 / self.replay_fps:
                self.frame_index += 1
                if self.frame_index >= len(self.frames):
                    self.frame_index = 0
                self.sim_time = self.frames[self.frame_index][0]
                self.last_frame_time = time.time()
            self.draw()

        pygame.quit()


if __name__ == "__main__":
    CFDViewer().run()
