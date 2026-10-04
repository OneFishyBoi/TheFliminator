import sys
from dataclasses import dataclass

import pygame


WINDOW_WIDTH = 1100
WINDOW_HEIGHT = 760
FPS = 60

BACKGROUND_COLOR = (23, 24, 30)
PANEL_COLOR = (35, 37, 46)
TEXT_COLOR = (230, 232, 238)
SLIDER_TRACK_COLOR = (65, 70, 84)
SLIDER_HANDLE_COLOR = (190, 196, 214)
SLIDER_ACTIVE_COLOR = (230, 236, 255)
GRID_BACKGROUND = (255, 255, 255)
GRID_LINE_COLOR = (0, 0, 0)

MIN_GRID_N = 8


@dataclass
class Slider:
    x: int
    y: int
    width: int
    min_value: int
    max_value: int
    value: int
    handle_radius: int = 10
    active: bool = False

    @property
    def track_rect(self) -> pygame.Rect:
        return pygame.Rect(self.x, self.y, self.width, 6)

    def _value_to_pos(self, value: int) -> int:
        if self.max_value == self.min_value:
            return self.x
        ratio = (value - self.min_value) / (self.max_value - self.min_value)
        return self.x + int(round(ratio * self.width))

    def _pos_to_value(self, px: int) -> int:
        if self.max_value == self.min_value:
            return self.min_value
        clamped_x = max(self.x, min(self.x + self.width, px))
        ratio = (clamped_x - self.x) / self.width
        return int(round(self.min_value + ratio * (self.max_value - self.min_value)))

    def handle_event(self, event: pygame.event.Event) -> bool:
        changed = False
        handle_x = self._value_to_pos(self.value)
        handle_center = (handle_x, self.y + 3)
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if pygame.Rect(
                handle_center[0] - self.handle_radius,
                handle_center[1] - self.handle_radius,
                self.handle_radius * 2,
                self.handle_radius * 2,
            ).collidepoint(event.pos) or self.track_rect.inflate(0, 14).collidepoint(event.pos):
                self.active = True
                new_value = self._pos_to_value(event.pos[0])
                if new_value != self.value:
                    self.value = new_value
                    changed = True
        elif event.type == pygame.MOUSEMOTION and self.active:
            new_value = self._pos_to_value(event.pos[0])
            if new_value != self.value:
                self.value = new_value
                changed = True
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.active = False
        return changed

    def draw(self, surface: pygame.Surface) -> None:
        pygame.draw.rect(surface, SLIDER_TRACK_COLOR, self.track_rect, border_radius=3)
        handle_x = self._value_to_pos(self.value)
        handle_color = SLIDER_ACTIVE_COLOR if self.active else SLIDER_HANDLE_COLOR
        pygame.draw.circle(surface, handle_color, (handle_x, self.y + 3), self.handle_radius)


def build_palette() -> list[tuple[int, int, int]]:
    return [
        (20, 20, 20),
        (255, 255, 255),
        (255, 59, 48),
        (255, 149, 0),
        (255, 214, 10),
        (52, 199, 89),
        (48, 176, 255),
        (10, 132, 255),
        (94, 92, 230),
        (191, 90, 242),
        (255, 55, 95),
        (142, 142, 147),
    ]


def create_grid_surface(size: int, fill_color: tuple[int, int, int]) -> pygame.Surface:
    grid = pygame.Surface((size, size))
    grid.fill(fill_color)
    return grid


def resize_grid_surface(current: pygame.Surface, new_size: int) -> pygame.Surface:
    if current.get_width() == new_size:
        return current
    return pygame.transform.scale(current, (new_size, new_size))


def get_grid_layout(window_w: int, window_h: int, panel_w: int, margin: int, grid_n: int) -> tuple[int, int, int, int]:
    grid_area = min(window_h - margin * 2, window_w - panel_w - margin * 3)
    grid_area = max(1, grid_area)
    cell_size = max(1, grid_area // grid_n)
    used_area = cell_size * grid_n
    origin_x = margin + (grid_area - used_area) // 2
    origin_y = margin + (grid_area - used_area) // 2
    return origin_x, origin_y, used_area, cell_size


def cell_from_mouse(
    mx: int,
    my: int,
    origin_x: int,
    origin_y: int,
    used_area: int,
    cell_size: int,
    grid_n: int,
) -> tuple[int, int] | None:
    if mx < origin_x or my < origin_y or mx >= origin_x + used_area or my >= origin_y + used_area:
        return None
    gx = (mx - origin_x) // cell_size
    gy = (my - origin_y) // cell_size
    if gx < 0 or gy < 0 or gx >= grid_n or gy >= grid_n:
        return None
    return gx, gy


def draw_grid(
    target: pygame.Surface,
    grid_surface: pygame.Surface,
    origin_x: int,
    origin_y: int,
    used_area: int,
    cell_size: int,
) -> None:
    target.fill(GRID_BACKGROUND, pygame.Rect(origin_x, origin_y, used_area, used_area))
    if cell_size == 1:
        target.blit(grid_surface, (origin_x, origin_y))
    else:
        scaled = pygame.transform.scale(grid_surface, (used_area, used_area))
        target.blit(scaled, (origin_x, origin_y))

    if cell_size >= 6:
        for offset in range(0, used_area + 1, cell_size):
            x = origin_x + offset
            y = origin_y + offset
            pygame.draw.line(target, GRID_LINE_COLOR, (x, origin_y), (x, origin_y + used_area), 1)
            pygame.draw.line(target, GRID_LINE_COLOR, (origin_x, y), (origin_x + used_area, y), 1)

    pygame.draw.rect(target, GRID_LINE_COLOR, pygame.Rect(origin_x, origin_y, used_area, used_area), width=1)


def draw_sidebar(
    target: pygame.Surface,
    panel_x: int,
    panel_w: int,
    font: pygame.font.Font,
    slider: Slider,
    palette: list[tuple[int, int, int]],
    selected_index: int,
) -> list[pygame.Rect]:
    sidebar = pygame.Rect(panel_x, 0, panel_w, WINDOW_HEIGHT)
    pygame.draw.rect(target, PANEL_COLOR, sidebar)

    title = font.render("FluidSim Grid Painter", True, TEXT_COLOR)
    target.blit(title, (panel_x + 16, 16))

    grid_label = font.render(f"Grid size: {slider.value} x {slider.value}", True, TEXT_COLOR)
    target.blit(grid_label, (panel_x + 16, 70))
    slider.draw(target)

    help_text = font.render("Drag on grid to paint cells", True, TEXT_COLOR)
    target.blit(help_text, (panel_x + 16, 122))

    palette_title = font.render("Color palette", True, TEXT_COLOR)
    target.blit(palette_title, (panel_x + 16, 168))

    swatch_rects = get_palette_rects(panel_x, len(palette))
    for idx, color in enumerate(palette):
        rect = swatch_rects[idx]
        pygame.draw.rect(target, color, rect)
        border_w = 3 if idx == selected_index else 1
        border_color = SLIDER_ACTIVE_COLOR if idx == selected_index else (15, 15, 18)
        pygame.draw.rect(target, border_color, rect, width=border_w)
    return swatch_rects


def get_palette_rects(panel_x: int, palette_len: int) -> list[pygame.Rect]:
    swatch_size = 28
    gap = 10
    cols = 4
    start_x = panel_x + 16
    start_y = 200
    rects: list[pygame.Rect] = []
    for idx in range(palette_len):
        row = idx // cols
        col = idx % cols
        x = start_x + col * (swatch_size + gap)
        y = start_y + row * (swatch_size + gap)
        rects.append(pygame.Rect(x, y, swatch_size, swatch_size))
    return rects


def main() -> None:
    pygame.init()
    pygame.display.set_caption("Fluid Simulation Grid Prototype")
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    font = pygame.font.SysFont("Segoe UI", 22)

    panel_width = 300
    margin = 20
    grid_n = 64

    grid_max_from_layout = max(
        MIN_GRID_N,
        min(WINDOW_HEIGHT - margin * 2, WINDOW_WIDTH - panel_width - margin * 3),
    )
    slider = Slider(
        x=WINDOW_WIDTH - panel_width + 20,
        y=100,
        width=panel_width - 40,
        min_value=MIN_GRID_N,
        max_value=grid_max_from_layout,
        value=grid_n,
    )

    palette = build_palette()
    selected_color_index = 0
    selected_color = palette[selected_color_index]
    grid_surface = create_grid_surface(grid_n, (255, 255, 255))
    painting = False

    running = True
    while running:
        origin_x, origin_y, used_area, cell_size = get_grid_layout(
            WINDOW_WIDTH, WINDOW_HEIGHT, panel_width, margin, grid_n
        )

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
                continue

            if slider.handle_event(event):
                grid_n = slider.value
                grid_surface = resize_grid_surface(grid_surface, grid_n)

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                on_grid = cell_from_mouse(
                    event.pos[0],
                    event.pos[1],
                    origin_x,
                    origin_y,
                    used_area,
                    cell_size,
                    grid_n,
                )
                painting = on_grid is not None
            elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
                painting = False

            if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                mouse_x, mouse_y = event.pos
                panel_x = WINDOW_WIDTH - panel_width
                if mouse_x >= panel_x:
                    swatches = get_palette_rects(panel_x, len(palette))
                    for idx, swatch in enumerate(swatches):
                        if swatch.collidepoint(event.pos):
                            selected_color_index = idx
                            selected_color = palette[idx]
                            painting = False
                            break

            if event.type == pygame.MOUSEMOTION and not (event.buttons[0] or painting):
                continue

            if painting and (event.type == pygame.MOUSEMOTION or event.type == pygame.MOUSEBUTTONDOWN):
                cell = cell_from_mouse(
                    event.pos[0],
                    event.pos[1],
                    origin_x,
                    origin_y,
                    used_area,
                    cell_size,
                    grid_n,
                )
                if cell is not None:
                    grid_surface.set_at(cell, selected_color)

        screen.fill(BACKGROUND_COLOR)
        draw_grid(screen, grid_surface, origin_x, origin_y, used_area, cell_size)
        draw_sidebar(
            screen,
            WINDOW_WIDTH - panel_width,
            panel_width,
            font,
            slider,
            palette,
            selected_color_index,
        )

        pygame.display.flip()
        clock.tick(FPS)

    pygame.quit()
    sys.exit(0)


if __name__ == "__main__":
    main()
