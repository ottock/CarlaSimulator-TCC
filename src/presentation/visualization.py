# imports
import math
import pygame
import numpy as np
import logging
from typing import Optional, Tuple
import queue
import threading
import time

# constants
logger = logging.getLogger(__name__)

# ── Color palette (dark tech theme) ───────────────────────────────
BG         = (8,   12,  22)
PANEL_BG   = (12,  18,  32)
PANEL_BDR  = (40,  70,  120)
GRID_MIN   = (20,  32,  52)
GRID_MAJ   = (35,  55,  85)
TXT_PRI    = (200, 220, 255)
TXT_DIM    = (90,  120, 160)
ACCENT     = (0,   200, 255)
EGO_CLR    = (255, 80,  80)
OBS_CLR    = (255, 200, 0)
OBS_MKR    = (255, 160, 0)
CLOSE_CLR  = (255, 60,  60)
MED_CLR    = (255, 140, 0)
FAR_CLR    = (0,   200, 100)
WHITE      = (255, 255, 255)


# classes
class LidarVisualization:
    """Real-time LIDAR obstacle visualization with dark tech UI.

    Renders LIDAR point clouds, obstacle metadata, and camera frames in a
    single monitoring window.
    """

    def __init__(
        self,
        width: int = 1200,
        height: int = 700,
        title: str = "CARLA – LIDAR Obstacle Monitor",
    ):
        """Initialize LIDAR visualization window.

        Args:
            width: Window width in pixels.
            height: Window height in pixels.
            title: Window title string.
        """
        self.width  = width
        self.height = height
        self.title  = title
        self.min_width = 900
        self.min_height = 560

        self.display = None
        self.clock   = None

        self.lidar_data_queue  = queue.Queue(maxsize=2)
        self.camera_data_queue = queue.Queue(maxsize=2)
        self.running    = True
        self.paused     = False
        self.stop_event = threading.Event()

        self.range_min  = -50
        self.range_max  =  50
        self.height_min = -3
        self.height_max =  8

        # Fonts – populated in run()
        self._f_lg = None
        self._f_md = None
        self._f_sm = None
        self._f_xs = None

        # Layout rects – populated in run()
        self.top_view_rect   = None
        self.camera_rect     = None
        self.side_view_rect  = None
        self.front_view_rect = None
        self.info_rect       = None

        # Rolling buffer – accumulate last N LiDAR scans to fill gaps between
        # sensor rotations and prevent point flickering.
        self._pts_buffer: list = []
        self._pts_buffer_size: int = 3

        logger.info(f"LidarVisualization initialised: {width}x{height}")

    def _get_merged_points(self) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return merged points and mask across the rolling scan buffer.

        Returns:
            Tuple of (points array of shape (N, 3), vehicle_mask array or None).
        """
        if not self._pts_buffer:
            return np.zeros((0, 4), dtype=np.float32), None
        pts_list  = [b["points"]       for b in self._pts_buffer]
        mask_list = [b["vehicle_mask"] for b in self._pts_buffer]
        pts = np.concatenate(pts_list, axis=0)
        if any(m is not None for m in mask_list):
            merged_masks = [
                m if m is not None else np.zeros(len(b["points"]), dtype=bool)
                for m, b in zip(mask_list, self._pts_buffer)
            ]
            mask = np.concatenate(merged_masks, axis=0)
        else:
            mask = None
        return pts, mask

    def _recalculate_layout(self) -> None:
        """Recompute adaptive panel layout for the current window size.

        Updates panel rectangles based on the active display dimensions.
        """
        HEADER_H = 40
        MARGIN   = 10

        info_h = max(120, min(220, self.height // 4))
        view_h = max(180, self.height - HEADER_H - (4 * MARGIN) - info_h)

        views_top  = HEADER_H + MARGIN
        available_w = self.width - (3 * MARGIN)

        # Left column: LIDAR top view (square, up to 42% of available width)
        top_w   = max(260, min(view_h, int(available_w * 0.42)))
        right_w = max(320, available_w - top_w)
        right_left = MARGIN + top_w + MARGIN

        # Right column: camera takes upper 62%, side+front share the rest
        cam_h  = max(120, int(view_h * 0.62))
        sub_h  = max(60, view_h - cam_h - MARGIN)
        sub_w  = max(120, (right_w - MARGIN) // 2)

        info_top     = views_top + view_h + MARGIN
        info_h_final = max(80, self.height - info_top - MARGIN)

        self.top_view_rect   = pygame.Rect(MARGIN, views_top, top_w, view_h)
        self.camera_rect     = pygame.Rect(right_left, views_top, right_w, cam_h)
        self.side_view_rect  = pygame.Rect(
            right_left,
            views_top + cam_h + MARGIN,
            sub_w,
            sub_h,
        )
        self.front_view_rect = pygame.Rect(
            right_left + sub_w + MARGIN,
            views_top + cam_h + MARGIN,
            sub_w,
            sub_h,
        )
        self.info_rect = pygame.Rect(
            MARGIN,
            info_top,
            self.width - (2 * MARGIN),
            info_h_final,
        )

    # ── public API ────────────────────────────────────────────────

    def update_lidar_data(self, data: dict) -> None:
        """Enqueue incoming LIDAR data for rendering.

        Args:
            data: LIDAR data dictionary from sensor callback.
        """
        try:
            self.lidar_data_queue.put_nowait(data)
        except queue.Full:
            pass

    def update_camera_data(self, data: dict) -> None:
        """Enqueue incoming camera frame for rendering.

        Args:
            data: Camera data dictionary from sensor callback.
        """
        try:
            self.camera_data_queue.put_nowait(data)
        except queue.Full:
            pass

    def start(self) -> threading.Thread:
        """Start the visualization loop in a background thread.

        Returns:
            The started visualization thread.
        """
        thread = threading.Thread(target=self.run, daemon=False)
        thread.start()
        logger.info("Visualization thread started")
        time.sleep(0.5)
        return thread

    def stop(self) -> None:
        """Signal the visualization loop to stop.

        Stops the background render loop and wakes any waiting state.
        """
        self.running = False
        self.stop_event.set()
        logger.info("Visualization stop requested")

    # ── coordinate helpers ────────────────────────────────────────

    def _w2sx(self, wx: float, rect: pygame.Rect) -> int:
        """Convert world X coordinate to screen X within a rect.

        Args:
            wx: World X value within range_min..range_max.
            rect: Target screen rectangle.

        Returns:
            Screen X pixel coordinate.
        """
        n = (wx - self.range_min) / (self.range_max - self.range_min)
        return int(rect.left + n * rect.width)

    def _w2sy_top(self, wy: float, rect: pygame.Rect) -> int:
        """Convert world Y coordinate to screen Y for top view.

        Args:
            wy: World Y value (inverted so +Y is up on screen).
            rect: Target screen rectangle.

        Returns:
            Screen Y pixel coordinate.
        """
        n = (wy - self.range_min) / (self.range_max - self.range_min)
        return int(rect.top + (1 - n) * rect.height)

    def _w2sy_elev(self, wz: float, rect: pygame.Rect) -> int:
        """Convert world Z coordinate to screen Y for elevation views.

        Args:
            wz: World Z value within height_min..height_max.
            rect: Target screen rectangle.

        Returns:
            Screen Y pixel coordinate.
        """
        n = (wz - self.height_min) / (self.height_max - self.height_min)
        return int(rect.top + (1 - n) * rect.height)

    def _z_color(self, z: float) -> Tuple[int, int, int]:
        """Map height value to RGB color using blue→cyan→green→yellow gradient.

        Args:
            z: Height value in world units.

        Returns:
            RGB tuple.
        """
        n = max(0.0, min(1.0, (z - self.height_min) / (self.height_max - self.height_min)))
        if n < 0.33:
            t = n / 0.33
            return (0, int(t * 200), int(255 * (1 - t)))
        if n < 0.66:
            t = (n - 0.33) / 0.33
            return (0, int(200 + t * 55), int(t * 80))
        t = (n - 0.66) / 0.34
        return (int(t * 255), 255 - int(t * 55), 0)

    def _compute_point_colors(
        self,
        points: np.ndarray,
        obs_mask: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """Compute RGB colors for each point in a point cloud.

        Uses a blue→cyan→green→yellow height gradient; obstacle points
        are overridden with amber (OBS_CLR).

        Args:
            points: Point cloud array of shape (N, 3).
            obs_mask: Optional boolean mask of obstacle points.

        Returns:
            Array of shape (N, 3) with uint8 RGB colors.
        """
        z = points[:, 2]
        n = np.clip(
            (z - self.height_min) / (self.height_max - self.height_min), 0.0, 1.0
        )
        r = np.zeros(len(z), dtype=np.float32)
        g = np.zeros(len(z), dtype=np.float32)
        b = np.zeros(len(z), dtype=np.float32)

        m1 = n < 0.33
        m2 = (n >= 0.33) & (n < 0.66)
        m3 = n >= 0.66

        if m1.any():
            t = n[m1] / 0.33
            g[m1] = t * 200
            b[m1] = (1.0 - t) * 255

        if m2.any():
            t = (n[m2] - 0.33) / 0.33
            g[m2] = 200 + t * 55
            b[m2] = t * 80

        if m3.any():
            t = (n[m3] - 0.66) / 0.34
            r[m3] = t * 255
            g[m3] = 255 - t * 55

        colors = np.stack([r, g, b], axis=1).clip(0, 255).astype(np.uint8)

        if obs_mask is not None and len(obs_mask) == len(points):
            obs_idx = np.where(obs_mask)[0]
            if len(obs_idx):
                colors[obs_idx] = (255, 200, 0)

        return colors

    def _render_points_surfarray(
        self,
        rect: pygame.Rect,
        sx: np.ndarray,
        sy: np.ndarray,
        colors: np.ndarray,
    ) -> None:
        """Write all points into the display via pixel-array access.

        Draws 2×2 pixel blocks per point for denser, continuous coverage.

        Args:
            rect: Clipping rectangle on the display surface.
            sx: Absolute display X coordinates as int32 array.
            sy: Absolute display Y coordinates as int32 array.
            colors: Point colors as (N, 3) uint8 RGB array.
        """
        in_rect = (
            (sx >= rect.left) & (sx < rect.right  - 1) &
            (sy >= rect.top)  & (sy < rect.bottom - 1)
        )
        xs  = sx[in_rect]
        ys  = sy[in_rect]
        clr = colors[in_rect]

        if not len(xs):
            return

        arr = pygame.surfarray.pixels3d(self.display)   # shape: (W, H, 3)
        try:
            for dx in (0, 1):
                for dy in (0, 1):
                    arr[xs + dx, ys + dy] = clr
        finally:
            del arr  # release pixel lock

    @staticmethod
    def _dist_color(dist: float) -> Tuple[int, int, int]:
        """Return RGB color based on obstacle distance.

        Args:
            dist: Distance in metres.

        Returns:
            RGB tuple (close=red, medium=orange, far=green).
        """
        if dist < 15:
            return CLOSE_CLR
        if dist < 30:
            return MED_CLR
        return FAR_CLR

    @staticmethod
    def _bearing_str(lx: float, ly: float) -> str:
        """Convert local XY position to cardinal bearing label.

        Args:
            lx: Local X coordinate (forward axis).
            ly: Local Y coordinate (lateral axis).

        Returns:
            Portuguese bearing label such as 'FRENTE' or 'DIREITA'.
        """
        deg = math.degrees(math.atan2(ly, lx)) % 360
        if deg < 22.5 or deg >= 337.5: return "FRENTE"
        if deg < 67.5:                  return "FRT-DIR"
        if deg < 112.5:                 return "DIREITA"
        if deg < 157.5:                 return "TRS-DIR"
        if deg < 202.5:                 return "TRASEIRA"
        if deg < 247.5:                 return "TRS-ESQ"
        if deg < 292.5:                 return "ESQUERDA"
        return "FRT-ESQ"

    # ── panel primitives ──────────────────────────────────────────

    def _draw_panel(self, rect: pygame.Rect) -> None:
        """Draw a filled panel background with border.

        Args:
            rect: Panel rectangle.
        """
        pygame.draw.rect(self.display, PANEL_BG,  rect)
        pygame.draw.rect(self.display, PANEL_BDR, rect, 1)

    def _panel_title(self, rect: pygame.Rect, text: str) -> None:
        """Draw a small label at the top-left corner of a panel.

        Args:
            rect: Panel rectangle.
            text: Label text.
        """
        lbl = self._f_sm.render(text, True, ACCENT)
        bg  = pygame.Rect(rect.left + 5, rect.top + 2,
                          lbl.get_width() + 6, lbl.get_height() + 4)
        pygame.draw.rect(self.display, PANEL_BG, bg)
        self.display.blit(lbl, (rect.left + 8, rect.top + 4))

    def _draw_grid_xy(self, rect: pygame.Rect) -> None:
        """Draw XY grid lines inside a rectangle.

        Args:
            rect: Drawing area rectangle.
        """
        for d in range(10, 55, 10):
            col = GRID_MAJ if d % 20 == 0 else GRID_MIN
            for val in (d, -d):
                pygame.draw.line(self.display, col,
                                 (self._w2sx(val, rect), rect.top),
                                 (self._w2sx(val, rect), rect.bottom))
                sy = self._w2sy_top(val, rect)
                pygame.draw.line(self.display, col,
                                 (rect.left, sy), (rect.right, sy))

    def _draw_grid_xz(self, rect: pygame.Rect) -> None:
        """Draw XZ elevation grid lines inside a rectangle.

        Args:
            rect: Drawing area rectangle.
        """
        for d in range(10, 55, 10):
            col = GRID_MAJ if d % 20 == 0 else GRID_MIN
            for val in (d, -d):
                sx = self._w2sx(val, rect)
                pygame.draw.line(self.display, col,
                                 (sx, rect.top), (sx, rect.bottom))
        for h in range(int(self.height_min), int(self.height_max) + 1, 2):
            col = GRID_MAJ if h == 0 else GRID_MIN
            sy = self._w2sy_elev(h, rect)
            pygame.draw.line(self.display, col,
                             (rect.left, sy), (rect.right, sy))

    # ── header ────────────────────────────────────────────────────

    def _draw_header(self, detected: list) -> None:
        """Draw the top header bar with title and object count.

        Args:
            detected: List of detected obstacle dictionaries.
        """
        bar = pygame.Rect(0, 0, self.width, 38)
        pygame.draw.rect(self.display, PANEL_BG, bar)
        pygame.draw.line(self.display, PANEL_BDR, (0, 37), (self.width, 37))

        title_s = self._f_lg.render("CARLA  ·  LIDAR OBSTACLE MONITOR", True, ACCENT)
        self.display.blit(title_s, (14, 8))

        n   = len(detected)
        s   = self._f_md.render(f"OBJETOS: {n}", True, TXT_DIM)
        self.display.blit(s, (self.width // 2 - s.get_width() // 2, 10))

        state_str = "  PAUSADO" if self.paused else "  EM EXECUÇÃO"
        state_col = (255, 180, 0)  if self.paused else FAR_CLR
        ss = self._f_sm.render(state_str, True, state_col)
        self.display.blit(ss, (self.width - ss.get_width() - 14, 11))

    # ── top view ──────────────────────────────────────────────────

    def _draw_top_view(
        self,
        points: np.ndarray,
        obs_mask: Optional[np.ndarray],
    ) -> None:
        """Draw the top-down XY LIDAR view.

        Args:
            points: Point cloud array of shape (N, 3).
            obs_mask: Optional boolean obstacle mask.
        """
        rect = self.top_view_rect
        self._draw_panel(rect)
        self._draw_grid_xy(rect)

        cx = rect.left + rect.width  // 2
        cy = rect.top  + rect.height // 2

        # Distance ring circles (drawn before points as background context)
        sx_m = rect.width  / (self.range_max - self.range_min)
        sy_m = rect.height / (self.range_max - self.range_min)
        for ring_m in (10, 20, 30, 40):
            rx  = int(ring_m * sx_m)
            ry  = int(ring_m * sy_m)
            col = GRID_MAJ if ring_m % 20 == 0 else GRID_MIN
            pygame.draw.ellipse(self.display, col,
                                pygame.Rect(cx - rx, cy - ry, rx * 2, ry * 2), 1)

        # LIDAR points — all points, numpy-vectorised, 2×2 pixel blocks
        if len(points) > 0:
            z_ok = (points[:, 2] >= self.height_min) & (points[:, 2] <= self.height_max)
            pts  = points[z_ok]
            mask = obs_mask[z_ok] if obs_mask is not None and len(obs_mask) == len(points) else None
            if len(pts):
                s_x = (
                    (pts[:, 1] - self.range_min) / (self.range_max - self.range_min)
                    * rect.width + rect.left
                ).astype(np.int32)
                s_y = (
                    (1.0 - (pts[:, 0] - self.range_min) / (self.range_max - self.range_min))
                    * rect.height + rect.top
                ).astype(np.int32)
                self._render_points_surfarray(rect, s_x, s_y, self._compute_point_colors(pts, mask))

        pad = 6
        for text, sx_off, sy_off in (
            ("F", cx - 4,          rect.top    + pad),
            ("B", cx - 4,          rect.bottom - 16),
            ("R", rect.right - 16, cy - 8),
            ("L", rect.left + pad, cy - 8),
        ):
            self.display.blit(self._f_sm.render(text, True, TXT_DIM), (sx_off, sy_off))

        # Ego vehicle
        vl, vw = 16, 9
        pygame.draw.rect(self.display, EGO_CLR,
                         (cx - vw // 2, cy - vl // 2, vw, vl))
        pygame.draw.polygon(self.display, WHITE, [
            (cx, cy - vl // 2 - 1),
            (cx - 4, cy - vl // 2 - 7),
            (cx + 4, cy - vl // 2 - 7),
        ])

        self._panel_title(rect, "TOP VIEW (XY)")

    # ── side view ─────────────────────────────────────────────────

    def _draw_side_view(
        self,
        points: np.ndarray,
        obs_mask: Optional[np.ndarray],
    ) -> None:
        """Draw the side elevation XZ LIDAR view.

        Args:
            points: Point cloud array of shape (N, 3).
            obs_mask: Optional boolean obstacle mask.
        """
        rect = self.side_view_rect
        self._draw_panel(rect)
        self._draw_grid_xz(rect)

        # Ground reference
        gy = self._w2sy_elev(-2.0, rect)
        if rect.top <= gy <= rect.bottom:
            pygame.draw.line(self.display, GRID_MAJ,
                             (rect.left, gy), (rect.right, gy), 1)

        if len(points) > 0:
            mask = obs_mask if obs_mask is not None and len(obs_mask) == len(points) else None
            s_x = (
                (points[:, 0] - self.range_min) / (self.range_max - self.range_min)
                * rect.width + rect.left
            ).astype(np.int32)
            s_y = (
                (1.0 - (points[:, 2] - self.height_min) / (self.height_max - self.height_min))
                * rect.height + rect.top
            ).astype(np.int32)
            self._render_points_surfarray(rect, s_x, s_y, self._compute_point_colors(points, mask))

        self._panel_title(rect, "VISTA LATERAL (XZ)")

    # ── front view ────────────────────────────────────────────────

    def _draw_front_view(
        self,
        points: np.ndarray,
        obs_mask: Optional[np.ndarray],
    ) -> None:
        """Draw the front elevation YZ LIDAR view.

        Args:
            points: Point cloud array of shape (N, 3).
            obs_mask: Optional boolean obstacle mask.
        """
        rect = self.front_view_rect
        self._draw_panel(rect)
        self._draw_grid_xz(rect)

        gy = self._w2sy_elev(-2.0, rect)
        if rect.top <= gy <= rect.bottom:
            pygame.draw.line(self.display, GRID_MAJ,
                             (rect.left, gy), (rect.right, gy), 1)

        if len(points) > 0:
            mask = obs_mask if obs_mask is not None and len(obs_mask) == len(points) else None
            s_x = (
                (points[:, 1] - self.range_min) / (self.range_max - self.range_min)
                * rect.width + rect.left
            ).astype(np.int32)
            s_y = (
                (1.0 - (points[:, 2] - self.height_min) / (self.height_max - self.height_min))
                * rect.height + rect.top
            ).astype(np.int32)
            self._render_points_surfarray(rect, s_x, s_y, self._compute_point_colors(points, mask))

        self._panel_title(rect, "VISTA FRONTAL (YZ)")

    # ── camera view ───────────────────────────────────────────────

    def _draw_camera_view(self, camera_data: Optional[dict]) -> None:
        """Draw the RGB camera image panel.

        Args:
            camera_data: Camera data dictionary, or None if no frame is available.
        """
        rect = self.camera_rect
        self._draw_panel(rect)

        if camera_data is not None:
            try:
                img = camera_data["image"]          # (H, W, 3) BGR from CARLA
                rgb = img[:, :, ::-1]               # BGR → RGB
                # surfarray expects (W, H, 3); numpy array is (H, W, 3)
                surf = pygame.surfarray.make_surface(rgb.transpose(1, 0, 2))
                scaled = pygame.transform.scale(surf, (rect.width, rect.height))
                self.display.blit(scaled, (rect.left, rect.top))
                pygame.draw.rect(self.display, PANEL_BDR, rect, 1)
            except Exception as exc:
                logger.debug("Camera render error: %s", exc)
                msg = self._f_sm.render("Erro ao renderizar câmera", True, CLOSE_CLR)
                self.display.blit(msg, (
                    rect.left + rect.width  // 2 - msg.get_width()  // 2,
                    rect.top  + rect.height // 2 - msg.get_height() // 2,
                ))
        else:
            msg = self._f_md.render("Aguardando câmera RGB...", True, TXT_DIM)
            self.display.blit(msg, (
                rect.left + rect.width  // 2 - msg.get_width()  // 2,
                rect.top  + rect.height // 2 - msg.get_height() // 2,
            ))

        self._panel_title(rect, "CÂMERA RGB (FRONTAL)")

    # ── info panel ────────────────────────────────────────────────

    def _draw_info_panel(
        self,
        detected: list,
        frame: int,
        timestamp: float,
    ) -> None:
        """Draw the obstacle information table panel.

        Args:
            detected: List of detected obstacle dictionaries.
            frame: Current simulation frame number.
            timestamp: Current simulation timestamp in seconds.
        """
        rect = self.info_rect
        self._draw_panel(rect)

        # Section title
        title_s = self._f_sm.render(
            f"OBSTÁCULOS DETECTADOS  ({len(detected)} total)", True, ACCENT
        )
        self.display.blit(title_s, (rect.left + 8, rect.top + 6))

        if not detected:
            msg = self._f_md.render("Nenhum obstáculo detectado", True, TXT_DIM)
            self.display.blit(msg, (rect.left + 8, rect.top + 30))
        else:
            # Column header positions
            col_x = [rect.left + 8, rect.left + 55, rect.left + 160,
                     rect.left + 270, rect.left + 380]
            hdr_y = rect.top + 24
            for hx, h in zip(col_x, ["#", "DISTÂNCIA", "DIREÇÃO", "PONTOS", "STATUS"]):
                self.display.blit(self._f_xs.render(h, True, TXT_DIM), (hx, hdr_y))

            sep_y = hdr_y + 14
            pygame.draw.line(self.display, PANEL_BDR,
                             (rect.left + 4, sep_y), (rect.right - 4, sep_y))

            row_h   = 22
            max_r   = min(len(detected), (rect.height - 60) // row_h)
            for idx, obj in enumerate(detected[:max_r]):
                ry = sep_y + 4 + idx * row_h
                if idx % 2 == 0:
                    pygame.draw.rect(self.display, (15, 22, 38),
                                     pygame.Rect(rect.left + 2, ry - 2,
                                                 rect.width - 4, row_h))
                dc  = self._dist_color(obj["distance"])
                brg = self._bearing_str(obj["local_x"], obj["local_y"])
                ok  = obj["distance"] < 10
                cells = [
                    (f"{idx + 1}",                TXT_DIM),
                    (f"{obj['distance']:.1f} m",  dc),
                    (brg,                          TXT_PRI),
                    (str(obj.get("point_count", "?")), TXT_DIM),
                    ("PRÓXIMO" if ok else "OK", CLOSE_CLR if ok else FAR_CLR),
                ]
                for cx_pos, (text, col) in zip(col_x, cells):
                    self.display.blit(self._f_sm.render(text, True, col), (cx_pos, ry))


        # Frame / timestamp footer
        ft = self._f_xs.render(f"Frame #{frame}  |  T: {timestamp:.2f}s", True, TXT_DIM)
        self.display.blit(ft, (rect.right - ft.get_width() - 8, rect.bottom - 14))

    # ── HUD overlay ───────────────────────────────────────────────

    def _draw_hud(self) -> None:
        """Draw the HUD overlay with keyboard hints.

        Renders the shortcut legend at the bottom of the window.
        """
        hints = self._f_xs.render("SPACE pausar  ·  ESC fechar", True, TXT_DIM)
        self.display.blit(hints, (8, self.height - 14))

    def _draw_waiting(self) -> None:
        """Draw the waiting message for missing LIDAR data.

        Shows a centered placeholder while no sensor frame is available.
        """
        msg = self._f_lg.render("Aguardando dados do LIDAR...", True, TXT_DIM)
        self.display.blit(msg, (
            self.width  // 2 - msg.get_width()  // 2,
            self.height // 2 - msg.get_height() // 2,
        ))

    # ── main loop ─────────────────────────────────────────────────

    def run(self) -> None:
        """Run the pygame visualization loop until stopped.

        Handles event processing, queue draining, and frame rendering.
        """
        try:
            pygame.init()
            self.width = max(self.min_width, self.width)
            self.height = max(self.min_height, self.height)
            self.display = pygame.display.set_mode(
                (self.width, self.height), pygame.RESIZABLE
            )
            pygame.display.set_caption(self.title)
            self.clock = pygame.time.Clock()

            self._f_lg = pygame.font.Font(None, 28)
            self._f_md = pygame.font.Font(None, 23)
            self._f_sm = pygame.font.Font(None, 19)
            self._f_xs = pygame.font.Font(None, 16)

            self._recalculate_layout()

            logger.info("Pygame window ready")

            frame_count = 0
            last_lidar  = None
            last_camera = None

            while self.running and not self.stop_event.is_set():
                try:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            self.running = False
                        elif event.type == pygame.VIDEORESIZE:
                            self.width = max(self.min_width, event.w)
                            self.height = max(self.min_height, event.h)
                            self.display = pygame.display.set_mode(
                                (self.width, self.height), pygame.RESIZABLE
                            )
                            self._recalculate_layout()
                        elif event.type == pygame.KEYDOWN:
                            if event.key == pygame.K_SPACE:
                                self.paused = not self.paused
                                if not self.paused:
                                    self._pts_buffer.clear()
                                logger.info("Visualization %s",
                                            "paused" if self.paused else "resumed")
                            elif event.key == pygame.K_ESCAPE:
                                self.running = False
                except Exception:
                    pass

                try:
                    lidar_data, camera_data = self._get_latest_data()
                    if not self.paused:
                        if lidar_data is not None:
                            last_lidar = lidar_data
                            self._pts_buffer.append({
                                "points":       lidar_data["points"],
                                "vehicle_mask": lidar_data.get("vehicle_mask"),
                            })
                            if len(self._pts_buffer) > self._pts_buffer_size:
                                self._pts_buffer.pop(0)
                        if camera_data is not None:
                            last_camera = camera_data

                    self.display.fill(BG)

                    detected = last_lidar.get("detected_objects", []) if last_lidar else []
                    self._draw_header(detected)

                    if last_lidar is not None:
                        pts, obs_mask = self._get_merged_points()
                        ts    = last_lidar.get("timestamp", 0.0)
                        frame = last_lidar.get("frame", 0)

                        self._draw_top_view(pts, obs_mask)
                        self._draw_camera_view(last_camera)
                        self._draw_side_view(pts, obs_mask)
                        self._draw_front_view(pts, obs_mask)
                        self._draw_info_panel(detected, frame, ts)
                    else:
                        for r in (self.top_view_rect, self.camera_rect,
                                  self.side_view_rect, self.front_view_rect,
                                  self.info_rect):
                            self._draw_panel(r)
                        self._draw_camera_view(last_camera)
                        self._draw_waiting()

                    self._draw_hud()
                    frame_count += 1

                except Exception as e:
                    logger.error("Render error: %s", e)

                pygame.display.flip()
                self.clock.tick(30)

        except Exception as e:
            logger.error("Fatal visualization error: %s", e)
        finally:
            try:
                pygame.quit()
            except Exception:
                pass
            logger.info("Visualization stopped")

    def _get_latest_data(self) -> Tuple[Optional[dict], Optional[dict]]:
        """Drain queues and return the latest LIDAR and camera data.

        Returns:
            Tuple of (lidar_data, camera_data); either may be None if no new data arrived.
        """
        lidar_data = camera_data = None
        try:
            while True:
                lidar_data = self.lidar_data_queue.get_nowait()
        except queue.Empty:
            pass
        try:
            while True:
                camera_data = self.camera_data_queue.get_nowait()
        except queue.Empty:
            pass
        return lidar_data, camera_data