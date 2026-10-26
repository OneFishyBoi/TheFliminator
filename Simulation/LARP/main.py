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
BACKGROUND = (10, 16, 28)
PANEL = (17, 26, 42)
PANEL_RAISED = (23, 35, 54)
TEXT = (231, 239, 249)
MUTED = (139, 157, 180)
ACCENT = (64, 210, 193)
ACCENT_BLUE = (74, 169, 255)


def rgb_field_to_surface(velocity, pressure, size):
    """Encode speed as cyan and signed, mean-relative pressure as orange or violet."""
    speed = np.hypot(velocity[..., 0], velocity[..., 1])
    pressure = np.asarray(pressure, dtype=float)
    finite_pressure = np.isfinite(pressure)
    pressure_mean = float(np.mean(pressure[finite_pressure])) if finite_pressure.any() else 0.0
    fields = (
        (speed, np.array((20, 235, 255), dtype=float), None),
        (
            pressure - pressure_mean,
            np.array((255, 91, 28), dtype=float),
            np.array((177, 55, 255), dtype=float),
        ),
    )
    rgb = np.zeros((*fields[0][0].shape, 3), dtype=float)
    for field, positive, negative in fields:
        finite = np.isfinite(field)
        if finite.any():
            fill = float(np.mean(field[finite]))
            values = np.where(finite, field, fill)
        else:
            values = np.zeros_like(field)
        scale = float(np.percentile(np.abs(values), 99))
        if scale > 0:
            intensity = np.clip(np.abs(values) / scale, 0.0, 1.0) ** 0.65
            if negative is None:
                colour = positive
            else:
                colour = np.where((values >= 0)[..., None], positive, negative)
            rgb += np.flip((intensity[..., None] * colour), axis=1)

    # Preserve vivid hues when several fields overlap instead of clipping them to white.
    peak = np.max(rgb, axis=2, keepdims=True)
    rgb = np.divide(rgb, np.maximum(peak, 1.0), out=np.zeros_like(rgb))
    rgb = (rgb * 235 + (5, 8, 18)).clip(0, 255).astype(np.uint8)
    surf = pygame.surfarray.make_surface(rgb)
    return pygame.transform.smoothscale(surf, size)


class CFDViewer:
    def __init__(self):
        pygame.init()
        pygame.display.set_caption("TheFliminator — CFD viewer")
        self.screen = pygame.display.set_mode((WIDTH, HEIGHT), pygame.RESIZABLE)
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("segoeui", 17)
        self.small = pygame.font.SysFont("segoeui", 14)
        self.title_font = pygame.font.SysFont("segoeui", 23, bold=True)

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
        self.show_rgb = True
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

    def button(self, rect, label, enabled=True, primary=False):
        if primary and enabled:
            colour, border, text_colour = (35, 157, 146), (73, 218, 199), (8, 25, 35)
        else:
            colour = PANEL_RAISED if enabled else (27, 34, 47)
            border = (48, 66, 88) if enabled else (39, 47, 60)
            text_colour = TEXT if enabled else MUTED
        pygame.draw.rect(self.screen, colour, rect, border_radius=8)
        pygame.draw.rect(self.screen, border, rect, width=1, border_radius=8)
        surf = self.font.render(label, True, text_colour)
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
        self.button(pygame.Rect(panel.right - 116, panel.bottom - 48, 96, 34), "Save", primary=True)
        self.settings_panel = panel
        self.settings_rows = (row_top, row_bottom, row_height, visible_rows)

    def plot_rect(self):
        w, h = self.screen.get_size()
        return pygame.Rect(PANEL_W + 24, 64, max(100, w - PANEL_W - 48), max(100, h - 125))

    def draw_arrows(self, rect, velocity):
        nx, ny, _ = velocity.shape
        stride = max(1, max(nx, ny) // 22)
        for i in range(stride // 2, nx, stride):
            for j in range(stride // 2, ny, stride):
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
                pygame.draw.line(self.screen, (8, 19, 31), start, end, 4)
                pygame.draw.line(self.screen, (184, 246, 239), start, end, 2)
                # Arrow head
                angle = np.arctan2(end[1] - start[1], end[0] - start[0])
                head = 5
                p1 = (end[0] - head * np.cos(angle - 0.55),
                      end[1] - head * np.sin(angle - 0.55))
                p2 = (end[0] - head * np.cos(angle + 0.55),
                      end[1] - head * np.sin(angle + 0.55))
                pygame.draw.polygon(self.screen, (8, 19, 31), [end, p1, p2])
                p1 = (end[0] - (head - 1) * np.cos(angle - 0.55),
                      end[1] - (head - 1) * np.sin(angle - 0.55))
                p2 = (end[0] - (head - 1) * np.cos(angle + 0.55),
                      end[1] - (head - 1) * np.sin(angle + 0.55))
                pygame.draw.polygon(self.screen, (184, 246, 239), [end, p1, p2])

    def draw_walls(self, rect, nx, ny):
        x_edges = [round(rect.left + i * rect.width / nx) for i in range(nx + 1)]
        y_edges = [round(rect.top + j * rect.height / ny) for j in range(ny + 1)]
        left = pygame.Rect(x_edges[0], rect.top, x_edges[1] - x_edges[0], rect.height)
        right = pygame.Rect(x_edges[-2], rect.top, x_edges[-1] - x_edges[-2], rect.height)
        bottom = pygame.Rect(x_edges[1], y_edges[-2],
                             x_edges[-2] - x_edges[1], y_edges[-1] - y_edges[-2])
        pygame.draw.rect(self.screen, (43, 61, 85), left)
        pygame.draw.rect(self.screen, (43, 61, 85), right)
        pygame.draw.rect(self.screen, (43, 61, 85), bottom)

        def shade_wall(wall_rect, horizontal, reverse=False):
            span = wall_rect.width if horizontal else wall_rect.height
            for offset in range(span):
                fraction = offset / max(1, span - 1)
                if reverse:
                    fraction = 1 - fraction
                shade = int(29 + 27 * fraction)
                colour = (shade, shade + 14, shade + 33)
                if horizontal:
                    pygame.draw.line(self.screen, colour,
                                     (wall_rect.left + offset, wall_rect.top),
                                     (wall_rect.left + offset, wall_rect.bottom - 1))
                else:
                    pygame.draw.line(self.screen, colour,
                                     (wall_rect.left, wall_rect.top + offset),
                                     (wall_rect.right - 1, wall_rect.top + offset))

        shade_wall(left, True)
        shade_wall(right, True, reverse=True)
        shade_wall(bottom, False)
        pygame.draw.line(self.screen, (164, 197, 224),
                         (left.right - 1, left.top), (left.right - 1, left.bottom - 1), 2)
        pygame.draw.line(self.screen, (164, 197, 224),
                         (right.left, right.top), (right.left, right.bottom - 1), 2)
        pygame.draw.line(self.screen, (164, 197, 224),
                         (bottom.left, bottom.top), (bottom.right - 1, bottom.top), 2)
        pygame.draw.line(self.screen, (20, 32, 49),
                         (left.left, left.top), (left.left, left.bottom - 1), 2)
        pygame.draw.line(self.screen, (20, 32, 49),
                         (right.right - 1, right.top), (right.right - 1, right.bottom - 1), 2)
        pygame.draw.line(self.screen, (20, 32, 49),
                         (bottom.left, bottom.bottom - 1), (bottom.right - 1, bottom.bottom - 1), 2)

        for edge in y_edges[1:-1]:
            pygame.draw.line(self.screen, (35, 49, 69),
                             (left.left, edge), (left.right - 1, edge), 1)
            pygame.draw.line(self.screen, (35, 49, 69),
                             (right.left, edge), (right.right - 1, edge), 1)
        for edge in x_edges[1:-1]:
            pygame.draw.line(self.screen, (35, 49, 69),
                             (edge, bottom.top), (edge, bottom.bottom - 1), 1)
        for y0, y1 in zip(y_edges[:-1], y_edges[1:]):
            center_y = (y0 + y1) // 2
            for wall_rect in (left, right):
                center_x = wall_rect.centerx
                pygame.draw.circle(self.screen, (20, 32, 49),
                                   (center_x, center_y), 3)
                pygame.draw.circle(self.screen, (153, 184, 211),
                                   (center_x - 1, center_y - 1), 1)
        for x0, x1 in zip(x_edges[1:-2], x_edges[2:-1]):
            center_x = (x0 + x1) // 2
            pygame.draw.circle(self.screen, (20, 32, 49),
                               (center_x, bottom.centery), 3)
            pygame.draw.circle(self.screen, (153, 184, 211),
                               (center_x - 1, bottom.centery - 1), 1)

        lid_rect = pygame.Rect(x_edges[1], rect.top,
                               x_edges[-2] - x_edges[1], y_edges[1] - y_edges[0])
        pygame.draw.rect(self.screen, (17, 103, 103), lid_rect)
        for offset in range(lid_rect.height):
            fraction = offset / max(1, lid_rect.height - 1)
            colour = (int(38 - 14 * fraction), int(207 - 65 * fraction),
                      int(184 - 47 * fraction))
            pygame.draw.line(self.screen, colour,
                             (lid_rect.left, lid_rect.top + offset),
                             (lid_rect.right - 1, lid_rect.top + offset))
        pygame.draw.line(self.screen, (184, 255, 233),
                         (lid_rect.left, lid_rect.top),
                         (lid_rect.right - 1, lid_rect.top), 2)
        pygame.draw.line(self.screen, (12, 88, 89),
                         (lid_rect.left, lid_rect.bottom - 1),
                         (lid_rect.right - 1, lid_rect.bottom - 1), 2)
        for edge in x_edges[2:-2]:
            pygame.draw.line(self.screen, (32, 151, 139),
                             (edge, lid_rect.top + 3), (edge, lid_rect.bottom - 3), 1)

        arrow_y = lid_rect.centery
        arrow_spacing = max(18, min(48, lid_rect.width // 12))
        for arrow_x in range(lid_rect.left + 9, lid_rect.right - 12, arrow_spacing):
            tip = (arrow_x + 10, arrow_y)
            pygame.draw.line(self.screen, (12, 79, 77),
                             (arrow_x, arrow_y), tip, 4)
            pygame.draw.line(self.screen, (235, 255, 246),
                             (arrow_x, arrow_y), tip, 2)
            pygame.draw.line(self.screen, (235, 255, 246),
                             (arrow_x + 5, arrow_y - 4), tip, 2)
            pygame.draw.line(self.screen, (235, 255, 246),
                             (arrow_x + 5, arrow_y + 4), tip, 2)

    def draw(self):
        self.screen.fill(BACKGROUND)
        w, h = self.screen.get_size()
        pygame.draw.rect(self.screen, PANEL, (0, 0, PANEL_W, h))
        pygame.draw.line(self.screen, (35, 51, 73), (PANEL_W - 1, 0), (PANEL_W - 1, h), 1)
        pygame.draw.rect(self.screen, ACCENT, (18, 20, 34, 34), border_radius=10)
        pygame.draw.circle(self.screen, (8, 48, 58), (35, 37), 8, 2)
        pygame.draw.circle(self.screen, (8, 48, 58), (35, 37), 3)
        self.text("TheFliminator", 62, 17, font=self.title_font)
        self.text("FLOW LAB  /  2D SOLVER", 63, 47, MUTED, self.small)

        self.button(pygame.Rect(18, 88, 100, 36), "Run", not self.running, primary=True)
        self.button(pygame.Rect(130, 88, 100, 36), "Stop", self.running)
        status_colour = ACCENT if self.running else ACCENT_BLUE if self.finished else MUTED
        pygame.draw.rect(self.screen, PANEL_RAISED, (18, 137, PANEL_W - 36, 27), border_radius=7)
        pygame.draw.circle(self.screen, status_colour, (31, 150), 4)
        self.text(self.status, 42, 140, status_colour, self.small)
        self.text(f"{self.sim_time:.4f}", 18, 172, font=self.title_font)
        self.text("SIMULATION TIME  /  SECONDS", 18, 202, MUTED, self.small)
        self.text(f"{len(self.frames):,}  frames captured", 18, 227, MUTED, self.small)

        self.button(pygame.Rect(18, 258, PANEL_W - 36, 36), "Initial parameters", not self.running)
        pygame.draw.line(self.screen, (35, 51, 73), (18, 311), (PANEL_W - 18, 311), 1)
        self.text("VISUALISATION", 18, 326, MUTED, self.small)
        for row_y, label, key, enabled in (
            (352, "RGB speed + pressure", "P", self.show_rgb),
            (384, "Velocity vectors", "V", self.show_arrows),
            (416, "Normalise vector length", "N", self.normalise_arrows),
        ):
            row = pygame.Rect(14, row_y - 4, PANEL_W - 28, 29)
            if enabled:
                pygame.draw.rect(self.screen, PANEL_RAISED, row, border_radius=6)
            pygame.draw.circle(self.screen, ACCENT if enabled else (59, 73, 91),
                               (27, row.centery), 4)
            self.text(label, 39, row_y, TEXT if enabled else MUTED, self.small)
            key_surface = self.small.render(key, True, MUTED)
            self.screen.blit(key_surface, (PANEL_W - 31, row_y))

        pygame.draw.line(self.screen, (35, 51, 73), (18, 458), (PANEL_W - 18, 458), 1)
        self.text("FRAME REPLAY", 18, 474, MUTED, self.small)
        self.button(pygame.Rect(18, 506, 100, 34), "Pause" if self.playing else "Play",
                    bool(self.frames))
        self.button(pygame.Rect(130, 506, 100, 34), "Live")
        self.text("← / →  step frame", 18, 550, MUTED, self.small)
        self.text("PLAYBACK SPEED", 18, 577, MUTED, self.small)
        self.button(pygame.Rect(18, 602, 38, 30), "−")
        fps_label = self.small.render(f"{self.replay_fps} FPS", True, TEXT)
        self.screen.blit(fps_label, fps_label.get_rect(center=(PANEL_W // 2, 617)))
        self.button(pygame.Rect(PANEL_W - 56, 602, 38, 30), "+")

        rect = self.plot_rect()
        with self.lock:
            p = self.pressure.copy()
            vel = self.velocity.copy()
            current_t = self.sim_time
            frames = list(self.frames)
        if frames and not self.running and 0 <= self.frame_index < len(frames):
            current_t, p, vel = frames[self.frame_index]
        nx, ny = p.shape
        x_edges = [round(rect.left + i * rect.width / nx) for i in range(nx + 1)]
        y_edges = [round(rect.top + j * rect.height / ny) for j in range(ny + 1)]
        field_rect = pygame.Rect(
            x_edges[1], y_edges[1],
            x_edges[-2] - x_edges[1], y_edges[-2] - y_edges[1],
        )
        interior_pressure = p[1:-1, 1:-1]
        interior_velocity = vel[1:-1, 1:-1]
        if self.show_rgb:
            surf = rgb_field_to_surface(interior_velocity, interior_pressure, field_rect.size)
            self.screen.blit(surf, field_rect.topleft)
        else:
            self.screen.fill(PANEL_RAISED, field_rect)
        if max(nx, ny) <= 48 and self.show_rgb:
            mesh = pygame.Surface(field_rect.size, pygame.SRCALPHA)
            for i in range(1, nx - 2):
                x = round(i * field_rect.width / (nx - 2))
                pygame.draw.line(mesh, (224, 242, 255, 17), (x, 0), (x, field_rect.height), 1)
            for j in range(1, ny - 2):
                y = round(j * field_rect.height / (ny - 2))
                pygame.draw.line(mesh, (224, 242, 255, 17), (0, y), (field_rect.width, y), 1)
            self.screen.blit(mesh, field_rect.topleft)
        self.draw_walls(rect, nx, ny)
        if self.show_arrows:
            self.draw_arrows(field_rect, interior_velocity)
        self.text("SPEED + PRESSURE", rect.left, 27, TEXT, self.small)
        legend_x = rect.left + 185
        for label, positive, negative in (
            ("speed", (20, 235, 255), None),
            ("p", (255, 91, 28), (177, 55, 255)),
        ):
            pygame.draw.rect(self.screen, positive, (legend_x, 29, 5, 10), border_radius=2)
            if negative is not None:
                pygame.draw.rect(self.screen, negative,
                                 (legend_x + 6, 29, 5, 10), border_radius=2)
            self.text(label, legend_x + (15 if negative is not None else 9),
                      27, MUTED, self.small)
            legend_x += 62 if negative is not None else 55
        self.text(f"t = {current_t:.4f} s", rect.right - 110, 27, MUTED, self.small)
        self.text("Speed: cyan  ·  pressure: orange (+) / violet (−)  ·  99th percentile contrast",
                  rect.left, 45, MUTED, self.small)

        # Timeline is draggable for replay.
        timeline = pygame.Rect(rect.left, h - 41, rect.width, 6)
        pygame.draw.rect(self.screen, PANEL_RAISED, timeline, border_radius=4)
        if len(frames) > 1:
            progress = self.frame_index / (len(frames) - 1)
            progress_rect = pygame.Rect(timeline.left, timeline.top,
                                        int(timeline.width * progress), timeline.height)
            pygame.draw.rect(self.screen, ACCENT, progress_rect, border_radius=4)
        if len(frames) > 1:
            frac = self.frame_index / (len(frames) - 1)
            marker_x = timeline.left + int(frac * timeline.width)
            pygame.draw.circle(self.screen, TEXT, (marker_x, timeline.centery), 6)
            pygame.draw.circle(self.screen, ACCENT, (marker_x, timeline.centery), 3)
        else:
            pygame.draw.circle(self.screen, TEXT, (timeline.left, timeline.centery), 6)
            pygame.draw.circle(self.screen, ACCENT, (timeline.left, timeline.centery), 3)
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
        elif pygame.Rect(18, 258, PANEL_W - 36, 36).collidepoint(pos):
            self.open_settings()
        elif 348 <= y <= 377 and x < PANEL_W:
            self.show_rgb = not self.show_rgb
        elif 380 <= y <= 409 and x < PANEL_W:
            self.show_arrows = not self.show_arrows
        elif 412 <= y <= 441 and x < PANEL_W:
            self.normalise_arrows = not self.normalise_arrows
        elif pygame.Rect(18, 506, 100, 34).collidepoint(pos):
            if self.frames:
                self.playing = not self.playing
                self.last_frame_time = time.time()
        elif pygame.Rect(130, 506, 100, 34).collidepoint(pos):
            self.playing = False
            self.frame_index = max(0, len(self.frames) - 1)
        elif pygame.Rect(18, 602, 38, 30).collidepoint(pos):
            self.change_replay_fps(-1)
        elif pygame.Rect(PANEL_W - 56, 602, 38, 30).collidepoint(pos):
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
                        self.show_rgb = not self.show_rgb
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
