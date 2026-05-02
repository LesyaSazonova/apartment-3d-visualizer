import cv2
import numpy as np
import math
import os
from dataclasses import dataclass
from typing import List, Tuple, Optional
from enum import Enum

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from config import (DEFAULT_PIXELS_PER_METER, AUTO_SCALE_PIXELS_PER_METER,
                        EXPECTED_PLAN_WIDTH_METERS, EXPECTED_PLAN_HEIGHT_METERS)
except ImportError:
    DEFAULT_PIXELS_PER_METER = 70.0
    AUTO_SCALE_PIXELS_PER_METER = None
    EXPECTED_PLAN_WIDTH_METERS  = (5.0, 30.0)
    EXPECTED_PLAN_HEIGHT_METERS = (3.0, 25.0)


class OpeningType(Enum):
    DOOR = "door"
    WINDOW = "window"


@dataclass
class WallSegment:
    start: Tuple[float, float]
    end: Tuple[float, float]
    thickness: float = 0.2

    @property
    def length(self):
        dx = self.end[0] - self.start[0]
        dy = self.end[1] - self.start[1]
        return math.sqrt(dx * dx + dy * dy)

    @property
    def angle(self):
        return math.atan2(self.end[1] - self.start[1],
                          self.end[0] - self.start[0])

    @property
    def is_horizontal(self):
        a = abs(self.angle)
        return a < math.radians(10) or a > math.radians(170)

    @property
    def is_vertical(self):
        return abs(abs(self.angle) - math.pi / 2) < math.radians(10)


@dataclass
class Opening:
    opening_type: OpeningType
    position: Tuple[float, float]
    width: float
    height: float
    wall_segment: Optional[WallSegment] = None
    sill_height: float = 0.0
    angle: float = 0.0


@dataclass
class FloorPlanData:
    contour_points: List[Tuple[float, float]]
    wall_segments: List[WallSegment]
    openings: List[Opening]
    rooms: List[List[Tuple[float, float]]]
    image_width: int = 0
    image_height: int = 0
    scale: float = 1.0
    bounding_box: Tuple[float, float, float, float] = (0, 0, 0, 0)


class ImageProcessor:
    def __init__(self, pixels_per_meter=DEFAULT_PIXELS_PER_METER):
        if AUTO_SCALE_PIXELS_PER_METER is not None:
            self.pixels_per_meter = float(AUTO_SCALE_PIXELS_PER_METER)
        else:
            self.pixels_per_meter = pixels_per_meter
        self._img_w = 0
        self._img_h = 0
        self._deskew_angle = 0.0
        self._auto_scaled = False

    def _auto_scale(self, plan_width_px, plan_height_px):
        if AUTO_SCALE_PIXELS_PER_METER is not None:
            return

        w_min, w_max = EXPECTED_PLAN_WIDTH_METERS
        h_min, h_max = EXPECTED_PLAN_HEIGHT_METERS
        w_mid = (w_min + w_max) / 2
        h_mid = (h_min + h_max) / 2

        scale_w = plan_width_px  / w_mid
        scale_h = plan_height_px / h_mid
        auto_scale = (scale_w + scale_h) / 2
        auto_scale = max(30.0, min(300.0, auto_scale))

        old_scale = self.pixels_per_meter
        self.pixels_per_meter = auto_scale
        self._auto_scaled = True
        print(f"[APT] Автомасштаб: {old_scale:.1f} → {auto_scale:.1f} px/m "
              f"(план {plan_width_px}x{plan_height_px}px "
              f"≈ {plan_width_px/auto_scale:.1f}x{plan_height_px/auto_scale:.1f}м)")

    def process(self, filepath):
        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Файл не найден: {filepath}")
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Не удалось загрузить: {filepath}")

        self._img_h, self._img_w = image.shape[:2]
        print(f"[APT] ===== НАЧАЛО =====")
        print(f"[APT] Изображение: {self._img_w}x{self._img_h}")

        self._gray_raw = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        gray        = self._preprocess_scan(image)
        gray        = self._deskew(gray)
        print(f"[APT] Поворот скана исправлен: {self._deskew_angle:.2f}°")

        binary      = self._binarize(gray)
        clean       = self._remove_text_and_dimensions(binary)
        wall_mask, detail_mask = self._split_walls_details(clean)
        wall_mask, crop_rect   = self._crop_to_plan(wall_mask)
        detail_mask  = self._apply_crop(detail_mask,  crop_rect)
        clean        = self._apply_crop(clean,         crop_rect)
        binary_crop  = self._apply_crop(binary,        crop_rect)

        plan_w_px = crop_rect[2] - crop_rect[0]
        plan_h_px = crop_rect[3] - crop_rect[1]
        self._auto_scale(plan_w_px, plan_h_px)

        self._gray_for_arcs = self._apply_crop(self._gray_raw, crop_rect)

        door_arcs   = self._find_door_arcs(detail_mask)
        print(f"[APT] Дуг дверей: {len(door_arcs)}")

        walls_clean = self._remove_arcs(clean, door_arcs)
        contour_px  = self._floodfill_contour(wall_mask)
        print(f"[APT] Контур: {len(contour_px)} точек")

        # Центрлинии стен берём только из маски толстых стен: в «clean» могут выживать тонкие линии полотен дверей.
        all_lines   = self._find_all_lines(wall_mask)
        print(f"[APT] Линий Hough: {len(all_lines)}")

        walls_px    = self._group_and_merge(all_lines)
        print(f"[APT] Стен (до фильтра): {len(walls_px)}")

        walls_px    = self._filter_walls_by_contour(walls_px, contour_px,
                                                     binary_crop.shape)
        print(f"[APT] Стен (после фильтра): {len(walls_px)}")

        walls_px    = self._deduplicate_walls(walls_px)
        print(f"[APT] Стен (после дедупликации): {len(walls_px)}")

        # Проёмы берём только из разрывов в стенах. Дуга рядом с разрывом — дверь, иначе окно.
        gap_openings = self._gaps_to_openings(binary_crop, walls_px, door_arcs,
                                              self.pixels_per_meter)
        walls_px = self._filter_non_structural_segments_near_openings(
            walls_px, gap_openings, self.pixels_per_meter
        )

        doors   = [o for o in gap_openings if o.opening_type == OpeningType.DOOR]
        windows = [o for o in gap_openings if o.opening_type == OpeningType.WINDOW]
        print(f"[APT][DBG] openings_from_gaps={len(gap_openings)} doors={len(doors)} windows={len(windows)}")

        ox, oy = crop_rect[0], crop_rect[1]
        scale  = self.pixels_per_meter

        contour_m = [((x + ox) / scale, (y + oy) / scale) for x, y in contour_px]
        walls_m   = []
        for w in walls_px:
            wm = WallSegment(
                start=((w.start[0]+ox)/scale, (w.start[1]+oy)/scale),
                end  =((w.end[0]  +ox)/scale, (w.end[1]  +oy)/scale),
                thickness=0.2)
            if wm.length > 0.05:
                walls_m.append(wm)

        # Сохраняем привязку проёма к конкретному сегменту стены, чтобы потом не вырезать на соседней параллельной стене.
        wall_px_to_m = {id(walls_px[i]): walls_m[i] for i in range(min(len(walls_px), len(walls_m)))}

        openings_m = []
        for o in doors + windows:
            bound_wall_m = None
            if o.wall_segment is not None:
                bound_wall_m = wall_px_to_m.get(id(o.wall_segment))
            openings_m.append(Opening(
                opening_type=o.opening_type,
                position    =((o.position[0]*scale+ox)/scale,
                              (o.position[1]*scale+oy)/scale),
                width=o.width, height=o.height,
                sill_height=o.sill_height, angle=o.angle,
                wall_segment=bound_wall_m,
            ))

        all_x = [p[0] for p in contour_m]
        all_y = [p[1] for p in contour_m]
        if not all_x:
            raise ValueError("Пустой контур")

        bbox  = (min(all_x), min(all_y), max(all_x), max(all_y))
        cx_m  = (bbox[0]+bbox[2])/2
        cy_m  = (bbox[1]+bbox[3])/2

        contour_c  = [(x-cx_m, y-cy_m) for x, y in contour_m]
        walls_c    = [WallSegment(start=(w.start[0]-cx_m, w.start[1]-cy_m),
                                  end  =(w.end[0]  -cx_m, w.end[1]  -cy_m),
                                  thickness=w.thickness) for w in walls_m]
        # Привязку сохраняем и после центрирования координат.
        wall_m_to_c = {}
        for i in range(len(walls_m)):
            wall_m_to_c[id(walls_m[i])] = walls_c[i]

        openings_c = []
        for o in openings_m:
            bound_wall_c = None
            if o.wall_segment is not None:
                bound_wall_c = wall_m_to_c.get(id(o.wall_segment))
            openings_c.append(Opening(
                opening_type=o.opening_type,
                position=(o.position[0]-cx_m, o.position[1]-cy_m),
                width=o.width, height=o.height,
                sill_height=o.sill_height, angle=o.angle,
                wall_segment=bound_wall_c,
            ))

        openings_c = self._dedup_openings(openings_c, tol=0.5)
        bbox_c = (bbox[0]-cx_m, bbox[1]-cy_m, bbox[2]-cx_m, bbox[3]-cy_m)

        print(f"[APT] Размер: {bbox[2]-bbox[0]:.1f}x{bbox[3]-bbox[1]:.1f} м")
        print(f"[APT] Итого: {len(walls_c)} стен, {len(openings_c)} проёмов")
        print(f"[APT] ===== ГОТОВО =====")

        return FloorPlanData(
            contour_points=contour_c, wall_segments=walls_c,
            openings=openings_c, rooms=[],
            image_width=self._img_w, image_height=self._img_h,
            scale=self.pixels_per_meter, bounding_box=bbox_c)

    def _opening_bbox_px(self, opening, ppm):
        cx = opening.position[0] * ppm
        cy = opening.position[1] * ppm
        ow = max(opening.width * ppm, 1.0)
        # Окно/дверь в плане: вдоль стены — шире, поперёк — узкая зона символа.
        along = max(ow * 0.62, ppm * 0.22, 8.0)
        across = max(ppm * 0.30, 10.0)
        if opening.wall_segment is not None and opening.wall_segment.is_horizontal:
            return (cx - along, cy - across, cx + along, cy + across)
        if opening.wall_segment is not None and opening.wall_segment.is_vertical:
            return (cx - across, cy - along, cx + across, cy + along)
        # Fallback без привязки к стене.
        return (cx - along, cy - along, cx + along, cy + along)

    @staticmethod
    def _pt_in_bbox(px, py, bbox, pad=0.0):
        x1, y1, x2, y2 = bbox
        return (x1 - pad) <= px <= (x2 + pad) and (y1 - pad) <= py <= (y2 + pad)

    def _filter_non_structural_segments_near_openings(self, walls_px, openings_px, ppm):
        """
        Удаляет короткие неструктурные сегменты (линии символов окна/двери),
        которые попали в walls_px рядом с detected opening.
        """
        if not walls_px or not openings_px:
            return walls_px

        kept = []
        removed = []
        min_symbol_len = max(ppm * 0.22, 8.0)
        max_symbol_len = max(ppm * 1.10, 30.0)

        # Отладка для проблемной зоны: левое окно на внешней стене.
        dbg_left_window = None
        windows = [o for o in openings_px if o.opening_type == OpeningType.WINDOW]
        if windows:
            dbg_left_window = min(windows, key=lambda o: o.position[0])
            dbg_bbox = self._opening_bbox_px(dbg_left_window, ppm)
            print(
                f"[APT][DBG][SYM] left_window center=({dbg_left_window.position[0]*ppm:.1f},{dbg_left_window.position[1]*ppm:.1f}) "
                f"bbox=({dbg_bbox[0]:.1f},{dbg_bbox[1]:.1f},{dbg_bbox[2]:.1f},{dbg_bbox[3]:.1f})"
            )

        for w in walls_px:
            midx = (w.start[0] + w.end[0]) / 2.0
            midy = (w.start[1] + w.end[1]) / 2.0
            wlen = w.length
            drop_reason = None

            for op in openings_px:
                if op.wall_segment is w:
                    continue
                bbox = self._opening_bbox_px(op, ppm)
                if not self._pt_in_bbox(midx, midy, bbox, pad=max(ppm * 0.08, 3.0)):
                    continue

                # Сегменты символов обычно короткие, находятся внутри bbox проёма и близко к оси "родной" стены.
                if not (min_symbol_len <= wlen <= max_symbol_len):
                    continue

                if op.wall_segment is not None:
                    dist_to_opening_wall = self._pt_seg(midx, midy, op.wall_segment)
                    if dist_to_opening_wall > max(ppm * 0.65, op.width * ppm * 0.95):
                        continue

                # Оставляем только реальные конструктивные стены:
                # короткий сегмент внутри символа проёма считаем неструктурным.
                drop_reason = f"inside_opening_symbol_{op.opening_type.value}"
                break

            if drop_reason is None:
                kept.append(w)
            else:
                removed.append((w, drop_reason))

        if removed:
            print(f"[APT][DBG][SYM] removed_symbol_segments={len(removed)}")
            for w, reason in removed[:20]:
                print(
                    f"[APT][DBG][SYM] remove seg=({w.start[0]:.1f},{w.start[1]:.1f})->({w.end[0]:.1f},{w.end[1]:.1f}) "
                    f"len={w.length:.1f}px reason={reason}"
                )

        if dbg_left_window is not None:
            dbg_bbox = self._opening_bbox_px(dbg_left_window, ppm)
            nearby = []
            for w in walls_px:
                midx = (w.start[0] + w.end[0]) / 2.0
                midy = (w.start[1] + w.end[1]) / 2.0
                if self._pt_in_bbox(midx, midy, dbg_bbox, pad=max(ppm * 0.15, 5.0)):
                    nearby.append(w)
            print(f"[APT][DBG][SYM] left_window nearby_segments={len(nearby)}")
            for w in nearby[:20]:
                was_removed = any(w is rw for rw, _ in removed)
                print(
                    f"[APT][DBG][SYM] near_left_window seg=({w.start[0]:.1f},{w.start[1]:.1f})->({w.end[0]:.1f},{w.end[1]:.1f}) "
                    f"len={w.length:.1f}px removed={was_removed}"
                )

        return kept

    def _preprocess_scan(self, image):
        gray  = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
        gray  = clahe.apply(gray)
        sigma = max(self._img_w, self._img_h) // 20
        bg    = cv2.GaussianBlur(gray, (0, 0), sigmaX=sigma)
        gray  = cv2.addWeighted(gray, 1.5, bg, -0.5, 128)
        return np.clip(gray, 0, 255).astype(np.uint8)

    def _deskew(self, gray):
        _, bw   = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        edges   = cv2.Canny(bw, 50, 150)
        lines   = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

        if lines is None:
            self._deskew_angle = 0.0
            return gray

        angles = []
        for line in lines[:200]:
            theta     = line[0][1]
            angle_deg = math.degrees(theta)
            if angle_deg < 15:
                angles.append(angle_deg)
            elif angle_deg > 165:
                angles.append(angle_deg - 180)
            elif 75 < angle_deg < 105:
                angles.append(angle_deg - 90)

        if not angles:
            self._deskew_angle = 0.0
            return gray

        angle = float(np.median(angles))
        if abs(angle) > 15:
            self._deskew_angle = 0.0
            return gray

        self._deskew_angle = angle
        h, w = gray.shape[:2]
        M    = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        return cv2.warpAffine(gray, M, (w, h),
                              flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_REPLICATE)

    def _binarize(self, gray):
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        _, b_otsu = cv2.threshold(blurred, 0, 255,
                                   cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        b_small   = cv2.adaptiveThreshold(blurred, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 15, 4)
        b_large   = cv2.adaptiveThreshold(blurred, 255,
                                           cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY_INV, 51, 8)
        binary = cv2.bitwise_or(cv2.bitwise_or(b_otsu, b_small), b_large)
        k      = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        return cv2.morphologyEx(binary, cv2.MORPH_CLOSE, k, iterations=1)

    def _remove_text_and_dimensions(self, binary):
        img_size = max(self._img_w, self._img_h)
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        clean = np.zeros_like(binary)

        min_area_noise = (img_size * 0.001) ** 2
        min_area_wall  = (img_size * 0.004) ** 2

        for i in range(1, num_labels):
            area = stats[i, cv2.CC_STAT_AREA]
            w    = stats[i, cv2.CC_STAT_WIDTH]
            h    = stats[i, cv2.CC_STAT_HEIGHT]
            if area < min_area_noise:
                continue
            aspect    = max(w, h) / max(min(w, h), 1)
            thin_side = min(w, h)

            if aspect > 10 and thin_side < 5:
                continue
            if area < min_area_wall and aspect < 3:
                continue
            if area < min_area_wall * 2 and aspect > 6:
                continue

            clean[labels == i] = 255
        return clean

    def _split_walls_details(self, clean):
        img_size = max(self._img_w, self._img_h)

        k_big = max(int(img_size * 0.018), 10)
        horiz_big = cv2.morphologyEx(clean, cv2.MORPH_OPEN,
                     cv2.getStructuringElement(cv2.MORPH_RECT, (k_big, 1)))
        vert_big  = cv2.morphologyEx(clean, cv2.MORPH_OPEN,
                     cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_big)))

        k_small = max(int(img_size * 0.005), 3)
        horiz_small = cv2.morphologyEx(clean, cv2.MORPH_OPEN,
                       cv2.getStructuringElement(cv2.MORPH_RECT, (k_small, 1)))
        vert_small  = cv2.morphologyEx(clean, cv2.MORPH_OPEN,
                       cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_small)))

        k_mid = max(int(img_size * 0.010), 6)
        horiz_mid = cv2.morphologyEx(clean, cv2.MORPH_OPEN,
                     cv2.getStructuringElement(cv2.MORPH_RECT, (k_mid, 1)))
        vert_mid  = cv2.morphologyEx(clean, cv2.MORPH_OPEN,
                     cv2.getStructuringElement(cv2.MORPH_RECT, (1, k_mid)))

        wall_mask = cv2.bitwise_or(
            cv2.bitwise_or(horiz_big, vert_big),
            cv2.bitwise_or(horiz_small, vert_small))
        wall_mask = cv2.bitwise_or(wall_mask,
            cv2.bitwise_or(horiz_mid, vert_mid))

        k_close   = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        wall_mask = cv2.morphologyEx(wall_mask, cv2.MORPH_CLOSE, k_close, iterations=2)
        wall_mask = cv2.dilate(wall_mask,
                               cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)),
                               iterations=1)

        detail_mask = cv2.bitwise_and(clean, cv2.bitwise_not(wall_mask))
        detail_mask = cv2.morphologyEx(detail_mask, cv2.MORPH_OPEN,
                       cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=1)
        return wall_mask, detail_mask

    def _crop_to_plan(self, wall_mask):
        h, w     = wall_mask.shape[:2]
        h_proj   = np.sum(wall_mask > 0, axis=0)
        v_proj   = np.sum(wall_mask > 0, axis=1)
        h_active = np.where(h_proj > h * 0.02)[0]
        v_active = np.where(v_proj > w * 0.02)[0]
        if len(h_active) < 2 or len(v_active) < 2:
            return wall_mask, (0, 0, w, h)
        pad = max(int(max(w, h) * 0.015), 8)
        x1  = max(0, int(h_active[0])  - pad)
        x2  = min(w, int(h_active[-1]) + pad)
        y1  = max(0, int(v_active[0])  - pad)
        y2  = min(h, int(v_active[-1]) + pad)
        if x2 - x1 < w * 0.4 or y2 - y1 < h * 0.4:
            return wall_mask, (0, 0, w, h)
        return wall_mask[y1:y2, x1:x2], (x1, y1, x2, y2)

    def _apply_crop(self, img, rect):
        return img[rect[1]:rect[3], rect[0]:rect[2]].copy()

    def _find_door_arcs(self, detail_mask):
        if not hasattr(self, '_gray_for_arcs') or self._gray_for_arcs is None:
            return []

        gray = self._gray_for_arcs
        img_size = max(gray.shape[1], gray.shape[0])

        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        min_r = max(int(img_size * 0.030), 18)
        max_r = max(int(img_size * 0.14),  60)

        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1.0,
            minDist=int(img_size * 0.045),
            param1=40,
            param2=18,
            minRadius=min_r,
            maxRadius=max_r
        )

        arcs = []
        if circles is None:
            return arcs

        h, w = gray.shape[:2]
        for c in circles[0]:
            cx, cy, r = float(c[0]), float(c[1]), float(c[2])

            margin = r * 0.3
            if not (margin < cx < w - margin and margin < cy < h - margin):
                continue

            roi_x1 = max(0, int(cx - r))
            roi_y1 = max(0, int(cy - r))
            roi_x2 = min(w, int(cx + r))
            roi_y2 = min(h, int(cy + r))

            _, local_bin = cv2.threshold(
                gray[roi_y1:roi_y2, roi_x1:roi_x2], 0, 255,
                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
            )
            fill = np.count_nonzero(local_bin) / max(local_bin.size, 1)
            
            if fill > 0.80:
                continue
            if fill < 0.002:
                continue

            bw = int(r * 2)
            bh = int(r * 2)
            x  = int(cx - r)
            y  = int(cy - r)

            arc_img = np.zeros((bh + 2, bw + 2), dtype=np.uint8)
            cv2.circle(arc_img, (int(r) + 1, int(r) + 1), int(r), 255, 2)
            cnts, _ = cv2.findContours(arc_img, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
            contour = cnts[0] if cnts else np.array([[[x, y]]], dtype=np.int32)
            contour = contour + np.array([[[x, y]]], dtype=np.int32)

            arcs.append({
                'bbox':    (x, y, bw, bh),
                'center':  (cx, cy),
                'radius':  r,
                'contour': contour
            })

        return arcs

    def _remove_arcs(self, clean, arcs):
        result = clean.copy()
        for arc in arcs:
            cv2.drawContours(result, [arc['contour']], -1, 0, thickness=4)
            x, y, bw, bh = arc['bbox']
            pad = int(max(bw, bh) * 0.12)
            x1, y1 = max(0, x - pad), max(0, y - pad)
            x2, y2 = min(result.shape[1], x + bw + pad), min(result.shape[0], y + bh + pad)
            roi = result[y1:y2, x1:x2]
            if roi.size > 0:
                ks = max(int(max(bw, bh) * 0.20), 5)
                hl = cv2.morphologyEx(roi, cv2.MORPH_OPEN,
                      cv2.getStructuringElement(cv2.MORPH_RECT, (ks, 1)))
                vl = cv2.morphologyEx(roi, cv2.MORPH_OPEN,
                      cv2.getStructuringElement(cv2.MORPH_RECT, (1, ks)))
                result[y1:y2, x1:x2] = cv2.bitwise_or(hl, vl)
        return cv2.morphologyEx(result, cv2.MORPH_CLOSE,
                cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)), iterations=1)

    def _floodfill_contour(self, wall_mask):
        h, w   = wall_mask.shape[:2]
        # Перед floodfill «запечатываем» проёмы: иначе контур может протечь через вход и дать выступы пола.
        # Размер ядра завязан на масштаб (px/м), чтобы одинаково работать на разных изображениях.
        k = max(15, int(self.pixels_per_meter * 1.10))
        k = min(k, int(min(w, h) * 0.20))
        closed = cv2.morphologyEx(
            wall_mask,
            cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (k, k)),
            iterations=3
        )
        flood  = closed.copy()
        mask   = np.zeros((h + 2, w + 2), np.uint8)
        fv     = 128
        for pt in [(0, 0), (w - 1, 0), (0, h - 1), (w - 1, h - 1)]:
            if flood[pt[1], pt[0]] == 0:
                cv2.floodFill(flood, mask, pt, fv)
        for x in range(0, w, 10):
            if flood[0, x] == 0:     cv2.floodFill(flood, mask, (x, 0),     fv)
            if flood[h - 1, x] == 0: cv2.floodFill(flood, mask, (x, h - 1), fv)
        for y in range(0, h, 10):
            if flood[y, 0] == 0:     cv2.floodFill(flood, mask, (0, y),     fv)
            if flood[y, w - 1] == 0: cv2.floodFill(flood, mask, (w - 1, y), fv)

        inside = np.zeros((h, w), dtype=np.uint8)
        inside[flood != fv] = 255
        inside = cv2.morphologyEx(inside, cv2.MORPH_CLOSE,
                  cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=3)

        contours, _ = cv2.findContours(inside, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            raise ValueError("Контур не найден")
        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        main = None
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if w * h * 0.01 < area < w * h * 0.98:
                main = cnt
                break
        if main is None:
            main = contours[0]

        peri = cv2.arcLength(main, True)
        best = None
        for eps in [0.002, 0.003, 0.005, 0.007, 0.01, 0.015, 0.02]:
            approx = cv2.approxPolyDP(main, eps * peri, True)
            if 4 <= len(approx) <= 30:
                best = approx
                if len(approx) <= 16:
                    break
        if best is None:
            best = cv2.approxPolyDP(main, 0.025 * peri, True)

        pts = [(float(p[0][0]), float(p[0][1])) for p in best]
        return self._remove_close_points(self._orthogonalize(pts))

    def _find_all_lines(self, walls_clean):
        h, w     = walls_clean.shape[:2]
        img_size = max(w, h)

        thick = cv2.dilate(walls_clean,
                 cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)), iterations=2)
        edges = cv2.Canny(thick, 40, 120, apertureSize=3)

        l1 = cv2.HoughLinesP(edges, 1, np.pi / 180, 30,
              minLineLength=int(img_size * 0.05),  maxLineGap=int(img_size * 0.012))
        l2 = cv2.HoughLinesP(edges, 1, np.pi / 180, 20,
              minLineLength=int(img_size * 0.015), maxLineGap=int(img_size * 0.015))
        l3 = cv2.HoughLinesP(edges, 1, np.pi / 180, 15,
              minLineLength=int(img_size * 0.008), maxLineGap=int(img_size * 0.008))

        all_l = []
        for ls in [l1, l2, l3]:
            if ls is not None:
                all_l.extend(ls.tolist())

        min_len = int(img_size * 0.008)
        raw = []
        for line in all_l:
            x1, y1, x2, y2 = line[0]
            seg = WallSegment(start=(float(x1), float(y1)), end=(float(x2), float(y2)))
            if not (seg.is_horizontal or seg.is_vertical):
                continue
            if seg.length < min_len:
                continue
            if self._line_score(walls_clean, x1, y1, x2, y2) < 0.30:
                continue
            raw.append(self._snap(seg))
        return raw

    def _group_and_merge(self, lines):
        if not lines:
            return []
        coords   = [c for l in lines for c in [l.start[0], l.end[0], l.start[1], l.end[1]]]
        img_size = max(coords) if coords else 100

        max_gap = max(img_size * 0.025, 12)

        h_l = sorted([l for l in lines if l.is_horizontal], key=lambda l: l.start[1])
        v_l = sorted([l for l in lines if l.is_vertical],   key=lambda l: l.start[0])
        walls = self._group_dir(h_l, 'h', max_gap) + self._group_dir(v_l, 'v', max_gap)
        return self._final_merge(walls, max_gap)

    def _group_dir(self, sorted_lines, d, max_gap):
        if not sorted_lines:
            return []
        used = [False] * len(sorted_lines)
        walls = []
        for i in range(len(sorted_lines)):
            if used[i]:
                continue
            group = [sorted_lines[i]]
            used[i] = True
            for j in range(len(sorted_lines)):
                if j == i or used[j]:
                    continue
                li, lj = sorted_lines[i], sorted_lines[j]
                if d == 'h':
                    dist = abs(li.start[1] - lj.start[1])
                    ov   = self._ov(min(li.start[0], li.end[0]), max(li.start[0], li.end[0]),
                                    min(lj.start[0], lj.end[0]), max(lj.start[0], lj.end[0]))
                else:
                    dist = abs(li.start[0] - lj.start[0])
                    ov   = self._ov(min(li.start[1], li.end[1]), max(li.start[1], li.end[1]),
                                    min(lj.start[1], lj.end[1]), max(lj.start[1], lj.end[1]))
                if dist < max_gap and ov > min(li.length, lj.length) * 0.10:
                    group.append(lj)
                    used[j] = True
            if d == 'h':
                ay = sum(l.start[1] for l in group) / len(group)
                mn = min(min(l.start[0], l.end[0]) for l in group)
                mx = max(max(l.start[0], l.end[0]) for l in group)
                walls.append(WallSegment(start=(mn, ay), end=(mx, ay)))
            else:
                ax = sum(l.start[0] for l in group) / len(group)
                mn = min(min(l.start[1], l.end[1]) for l in group)
                mx = max(max(l.start[1], l.end[1]) for l in group)
                walls.append(WallSegment(start=(ax, mn), end=(ax, mx)))
        return walls

    def _final_merge(self, walls, max_gap):
        for _ in range(5):
            merged = []
            used = [False] * len(walls)
            changed = False
            for i in range(len(walls)):
                if used[i]:
                    continue
                best = walls[i]
                used[i] = True
                for j in range(i + 1, len(walls)):
                    if used[j]:
                        continue
                    wi, wj = best, walls[j]
                    if wi.is_horizontal and wj.is_horizontal and abs(wi.start[1] - wj.start[1]) < max_gap:
                        o = self._ov(min(wi.start[0], wi.end[0]), max(wi.start[0], wi.end[0]),
                                     min(wj.start[0], wj.end[0]), max(wj.start[0], wj.end[0]))
                        if o > min(wi.length, wj.length) * 0.10:
                            ay   = (wi.start[1] + wj.start[1]) / 2
                            best = WallSegment(
                                start=(min(wi.start[0], wi.end[0], wj.start[0], wj.end[0]), ay),
                                end  =(max(wi.start[0], wi.end[0], wj.start[0], wj.end[0]), ay))
                            used[j] = True
                            changed = True
                    elif wi.is_vertical and wj.is_vertical and abs(wi.start[0] - wj.start[0]) < max_gap:
                        o = self._ov(min(wi.start[1], wi.end[1]), max(wi.start[1], wi.end[1]),
                                     min(wj.start[1], wj.end[1]), max(wj.start[1], wj.end[1]))
                        if o > min(wi.length, wj.length) * 0.10:
                            ax   = (wi.start[0] + wj.start[0]) / 2
                            best = WallSegment(
                                start=(ax, min(wi.start[1], wi.end[1], wj.start[1], wj.end[1])),
                                end  =(ax, max(wi.start[1], wi.end[1], wj.start[1], wj.end[1])))
                            used[j] = True
                            changed = True
                merged.append(best)
            walls = merged
            if not changed:
                break
        return walls

    def _ov(self, a1, a2, b1, b2):
        return max(0, min(a2, b2) - max(a1, b1))

    def _deduplicate_walls(self, walls):
        if not walls:
            return walls

        coords = [c for w in walls for c in [w.start[0], w.end[0], w.start[1], w.end[1]]]
        img_size = max(coords) if coords else 100
        wall_thickness_px = img_size * 0.02

        used = [False] * len(walls)
        result = []

        for i in range(len(walls)):
            if used[i]:
                continue
            wi = walls[i]
            best_partner = None
            best_dist = float('inf')

            for j in range(i + 1, len(walls)):
                if used[j]:
                    continue
                wj = walls[j]

                if wi.is_horizontal != wj.is_horizontal:
                    continue
                if wi.is_vertical != wj.is_vertical:
                    continue

                if wi.is_horizontal:
                    dist = abs(wi.start[1] - wj.start[1])
                    ov = self._ov(min(wi.start[0], wi.end[0]), max(wi.start[0], wi.end[0]),
                                  min(wj.start[0], wj.end[0]), max(wj.start[0], wj.end[0]))
                    min_len = min(wi.length, wj.length)
                else:
                    dist = abs(wi.start[0] - wj.start[0])
                    ov = self._ov(min(wi.start[1], wi.end[1]), max(wi.start[1], wi.end[1]),
                                  min(wj.start[1], wj.end[1]), max(wj.start[1], wj.end[1]))
                    min_len = min(wi.length, wj.length)

                if dist < wall_thickness_px and ov > min_len * 0.5:
                    if dist < best_dist:
                        best_dist = dist
                        best_partner = j

            if best_partner is not None:
                wj = walls[best_partner]
                if wi.is_horizontal:
                    ay = (wi.start[1] + wj.start[1]) / 2
                    mn = min(wi.start[0], wi.end[0], wj.start[0], wj.end[0])
                    mx = max(wi.start[0], wi.end[0], wj.start[0], wj.end[0])
                    merged = WallSegment(start=(mn, ay), end=(mx, ay))
                else:
                    ax = (wi.start[0] + wj.start[0]) / 2
                    mn = min(wi.start[1], wi.end[1], wj.start[1], wj.end[1])
                    mx = max(wi.start[1], wi.end[1], wj.start[1], wj.end[1])
                    merged = WallSegment(start=(ax, mn), end=(ax, mx))
                result.append(merged)
                used[best_partner] = True
            else:
                result.append(wi)
            used[i] = True

        return result

    def _filter_walls_by_contour(self, walls, contour_pts, shape):
        if len(contour_pts) < 3:
            return walls
        h, w    = shape[:2]
        cnt_arr = np.array(contour_pts, dtype=np.int32)
        mask    = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(mask, [cnt_arr], 255)
        margin  = max(h, w) // 20
        mask    = cv2.dilate(mask, cv2.getStructuringElement(
                              cv2.MORPH_RECT, (margin, margin)))
        filtered = []
        for wall in walls:
            pts_to_check = [wall.start, wall.end,
                            ((wall.start[0] + wall.end[0]) / 2,
                             (wall.start[1] + wall.end[1]) / 2)]
            inside = sum(1 for px, py in pts_to_check
                         if mask[int(np.clip(py, 0, h - 1)),
                                 int(np.clip(px, 0, w - 1))] > 0)
            if inside >= 2:
                filtered.append(wall)
        return filtered

    def _arcs_to_doors(self, arcs, walls_px):
        return []

    def _arc_near_gap(self, cx, cy, width_px, wall, arcs):
        if not arcs:
            return None

        best_arc = None
        best_score = None
        x1, y1 = wall.start
        x2, y2 = wall.end
        dx, dy = x2 - x1, y2 - y1
        wall_len = math.sqrt(dx * dx + dy * dy)
        if wall_len < 1e-6:
            return None

        for arc in arcs:
            ax, ay = arc['center']
            r = arc['radius']

            dist_to_wall = self._pt_seg(ax, ay, wall)
            # Дуга должна матчиться локально к разрыву, иначе окна будут ошибочно становиться дверями.
            ppm = float(self.pixels_per_meter) if self.pixels_per_meter else 70.0
            r_m = r / max(ppm, 1.0)
            if not (0.30 <= r_m <= 1.80):
                continue

            # Требуем близость центра дуги к центру разрыва (евклидово расстояние), чтобы отсеять случайные окружности.
            d_gap = math.hypot(ax - cx, ay - cy)
            max_d_gap = max(width_px * 1.35, r * 1.60, ppm * 0.95)
            if d_gap > max_d_gap:
                continue

            # Центр дуги должен быть рядом со стеной, но не «где-то далеко».
            if dist_to_wall > max(r * 1.35, ppm * 0.55):
                continue

            t_gap = ((cx - x1) * dx + (cy - y1) * dy) / (wall_len * wall_len)
            t_arc = ((ax - x1) * dx + (ay - y1) * dy) / (wall_len * wall_len)
            along_dist = abs(t_gap - t_arc) * wall_len

            # Дополнительная локализация вдоль оси стены.
            max_along = max(width_px * 1.20, r * 1.35, ppm * 1.00)
            if along_dist > max_along:
                continue

            score = along_dist + dist_to_wall * 0.55 + d_gap * 0.35
            if best_score is None or score < best_score:
                best_score = score
                best_arc = arc

        return best_arc

    def _detect_inner_line_pattern(self, binary_img, wall, cx, cy, width_px):
        """
        Проверяет, что внутри разрыва стены есть тонкие параллельные линии символа проёма.
        Нужен именно для валидации/отладки детекции разрыва; на выбор door/window не влияет.
        """
        h, w = binary_img.shape[:2]
        ppm = float(self.pixels_per_meter) if self.pixels_per_meter else 70.0
        along = int(max(width_px * 0.65, ppm * 0.18, 8))
        probe = int(max(ppm * 0.22, 8))

        if wall.is_horizontal:
            x = int(np.clip(cx, 0, w - 1))
            y1 = int(np.clip(cy - probe, 0, h - 1))
            y2 = int(np.clip(cy + probe + 1, 0, h))
            profile = binary_img[y1:y2, x] if y2 > y1 else np.array([], dtype=np.uint8)
        else:
            y = int(np.clip(cy, 0, h - 1))
            x1 = int(np.clip(cx - probe, 0, w - 1))
            x2 = int(np.clip(cx + probe + 1, 0, w))
            profile = binary_img[y, x1:x2] if x2 > x1 else np.array([], dtype=np.uint8)

        if profile.size == 0:
            return False, {"groups": 0, "thin_groups": 0, "profile_px": 0}

        bp = (profile > 128).astype(np.uint8)
        groups = 0
        thin_groups = 0
        in_g = False
        run = 0
        max_thin = max(2, int(len(bp) * 0.20))
        min_thin = 1
        for v in bp:
            if v:
                if not in_g:
                    groups += 1
                    in_g = True
                    run = 1
                else:
                    run += 1
            elif in_g:
                if min_thin <= run <= max_thin:
                    thin_groups += 1
                in_g = False
                run = 0
        if in_g and min_thin <= run <= max_thin:
            thin_groups += 1

        # Для устойчивости принимаем 2 тонких линии как «идеал», но допускаем 1 при частичной потере штриха.
        has_pattern = thin_groups >= 2 or (thin_groups == 1 and groups >= 2 and along >= 8)
        return has_pattern, {
            "groups": int(groups),
            "thin_groups": int(thin_groups),
            "profile_px": int(profile.size),
        }

    def _pt_seg(self, px, py, w):
        x1, y1 = w.start
        x2, y2 = w.end
        dx, dy = x2 - x1, y2 - y1
        if dx == 0 and dy == 0:
            return math.sqrt((px - x1) ** 2 + (py - y1) ** 2)
        t = max(0, min(1, ((px - x1) * dx + (py - y1) * dy) / (dx * dx + dy * dy)))
        return math.sqrt((px - x1 - t * dx) ** 2 + (py - y1 - t * dy) ** 2)

    def _find_windows(self, original_binary, detail_mask, walls_px):
        scale = self.pixels_per_meter
        bh, bw = original_binary.shape[:2]
        img_size = max(bw, bh)
        windows = []

        min_wall_px = max(int(scale * 0.8), int(img_size * 0.03))

        for wall in walls_px:
            if wall.length < min_wall_px:
                continue
            num = max(int(wall.length * 0.6), 30)
            scores = []

            ps = max(40, int(img_size * 0.04))

            for i in range(num + 1):
                t  = i / num
                px = int(wall.start[0] + t * (wall.end[0] - wall.start[0]))
                py = int(wall.start[1] + t * (wall.end[1] - wall.start[1]))
                if not (5 <= px < bw - 5 and 5 <= py < bh - 5):
                    scores.append((t, 'wall'))
                    continue

                if wall.is_horizontal:
                    profile = original_binary[max(0, py - ps):min(bh, py + ps + 1), px]
                else:
                    profile = original_binary[py, max(0, px - ps):min(bw, px + ps + 1)]

                if len(profile) < 8:
                    scores.append((t, 'wall'))
                    continue
                scores.append((t, self._analyze_profile(profile)))

            segs = self._extract_segments(scores)
            for s, e, tp in segs:
                if tp != 'window':
                    continue
                sl = (e - s) * wall.length
                min_win = max(15, scale * 0.5)
                max_win = min(wall.length * 0.82, scale * 3.0)
                if sl < min_win or sl > max_win:
                    continue
                gc = (s + e) / 2
                px = wall.start[0] + gc * (wall.end[0] - wall.start[0])
                py = wall.start[1] + gc * (wall.end[1] - wall.start[1])
                windows.append(Opening(
                    opening_type=OpeningType.WINDOW,
                    position=(px / scale, py / scale),
                    width=sl / scale,
                    height=1.2,
                    sill_height=0.9,
                    angle=wall.angle
                ))
        return windows

    def _scan_wall_for_gaps(self, binary_img, wall):
        h, w = binary_img.shape[:2]
        num = max(int(wall.length * 1.0), 60)
        probe = max(6, int(max(h, w) * 0.012))

        filled = []
        for i in range(num + 1):
            t  = i / num
            px = int(wall.start[0] + t * (wall.end[0] - wall.start[0]))
            py = int(wall.start[1] + t * (wall.end[1] - wall.start[1]))
            if not (0 <= px < w and 0 <= py < h):
                filled.append(True)
                continue
            if wall.is_horizontal:
                strip = binary_img[max(0, py - probe):min(h, py + probe + 1), px]
            else:
                strip = binary_img[py, max(0, px - probe):min(w, px + probe + 1)]
            if strip.size == 0:
                filled.append(True)
                continue
            density = float(np.count_nonzero(strip > 128)) / strip.size
            # Порог плотности завышен намеренно: тонкие линии символов (полотно двери) не должны разрывать проём.
            filled.append(density > 0.25)

        gaps = []
        in_gap = False
        gap_start_idx = 0
        for idx, f in enumerate(filled):
            if not f and not in_gap:
                gap_start_idx = idx
                in_gap = True
            elif f and in_gap:
                gap_end_idx = idx
                s = gap_start_idx / num
                e = gap_end_idx / num
                width_px = (e - s) * wall.length
                gaps.append(((s + e) / 2, width_px))
                in_gap = False
        if in_gap:
            s = gap_start_idx / num
            e = 1.0
            width_px = (e - s) * wall.length
            gaps.append(((s + e) / 2, width_px))

        return gaps

    def _gaps_to_openings(self, binary_img, walls_px, door_arcs, scale):
        extra = []
        for wall in walls_px:
            if wall.length < scale * 0.3:
                continue

            raw_gaps = self._scan_wall_for_gaps(binary_img, wall)
            for center_t, width_px in raw_gaps:
                w_m = width_px / scale
                if not (0.35 <= w_m <= 2.6):
                    continue

                cx = wall.start[0] + center_t * (wall.end[0] - wall.start[0])
                cy = wall.start[1] + center_t * (wall.end[1] - wall.start[1])
                inner_ok, inner_info = self._detect_inner_line_pattern(binary_img, wall, cx, cy, width_px)
                arc = self._arc_near_gap(cx, cy, width_px, wall, door_arcs)

                wall_dbg = f"(({wall.start[0]:.0f},{wall.start[1]:.0f})->({wall.end[0]:.0f},{wall.end[1]:.0f}))"
                if arc is not None:
                    print(
                        f"[APT][DBG][OPENING] center=({cx:.0f},{cy:.0f}) wall={wall_dbg} "
                        f"inner_pattern={inner_ok} groups={inner_info['groups']}/{inner_info['thin_groups']} "
                        f"arc_found=True arc_r={arc['radius']:.0f}px final=door"
                    )
                    width_m = min(max(w_m, 0.65), 1.35)
                    extra.append(Opening(
                        opening_type=OpeningType.DOOR,
                        position=(cx / scale, cy / scale),
                        width=width_m,
                        height=2.1,
                        sill_height=0.0,
                        angle=wall.angle,
                        wall_segment=wall,
                    ))
                    continue

                # Разрыв без дуги считаем окном.
                print(
                    f"[APT][DBG][OPENING] center=({cx:.0f},{cy:.0f}) wall={wall_dbg} "
                    f"inner_pattern={inner_ok} groups={inner_info['groups']}/{inner_info['thin_groups']} "
                    f"arc_found=False final=window"
                )
                extra.append(Opening(
                    opening_type=OpeningType.WINDOW,
                    position=(cx / scale, cy / scale),
                    width=w_m,
                    height=1.2,
                    sill_height=0.9,
                    angle=wall.angle,
                    wall_segment=wall,
                ))
        return extra

    def _dedup_openings(self, openings, tol=0.5):
        if not openings:
            return openings

        ordered = sorted(openings, key=lambda o: 0 if o.opening_type == OpeningType.DOOR else 1)
        kept = []
        for o in ordered:
            duplicate_index = None
            for idx, k in enumerate(kept):
                dx = o.position[0] - k.position[0]
                dy = o.position[1] - k.position[1]
                same_place = math.sqrt(dx * dx + dy * dy) < tol
                same_wall = abs(math.sin(o.angle - k.angle)) < 0.25
                if same_place and same_wall:
                    duplicate_index = idx
                    break

            if duplicate_index is None:
                kept.append(o)
                continue

            existing = kept[duplicate_index]
            if existing.opening_type == OpeningType.WINDOW and o.opening_type == OpeningType.DOOR:
                kept[duplicate_index] = o

        return kept

    def _analyze_profile(self, profile):
        if len(profile) == 0:
            return 'wall'

        bp = (profile > 128).astype(int)
        density = np.sum(bp) / len(bp)

        if density < 0.015:
            return 'gap'

        n = len(bp)
        groups  = 0
        in_g    = False
        widths  = []
        centers = []
        cw = 0
        cs = 0
        for i, v in enumerate(bp):
            if v == 1:
                if not in_g:
                    groups += 1; in_g = True; cw = 1; cs = i
                else:
                    cw += 1
            else:
                if in_g:
                    widths.append(cw)
                    centers.append(cs + cw / 2.0)
                    in_g = False; cw = 0
        if in_g:
            widths.append(cw)
            centers.append(cs + cw / 2.0)

        if not widths:
            return 'gap'

        max_w = max(widths)

        if groups == 1:
            if max_w > n * 0.25:
                return 'wall'
            pos = centers[0]
            if n * 0.2 < pos < n * 0.8:
                return 'window'
            return 'wall'

        if max_w > n * 0.28:
            return 'wall'

        edge_margin = n * 0.22
        min_center  = min(centers)
        max_center  = max(centers)
        all_in_mid  = (min_center > edge_margin) and (max_center < n - edge_margin)
        spans_edges = (min_center <= edge_margin) and (max_center >= n - edge_margin)

        if groups == 2:
            if spans_edges:
                if max_w > 10:
                    return 'wall'
                if density < 0.12:
                    return 'window'
                return 'wall'
            if all_in_mid:
                return 'window'
            if max_w <= 5 and density < 0.15:
                return 'window'
            return 'wall'

        if 3 <= groups <= 4:
            if max_w <= 14:
                return 'window'
            if max_w > 22:
                return 'wall'
            return 'window'

        return 'wall'

    def _extract_segments(self, scored):
        if not scored:
            return []
        segs = []
        ct = scored[0][1]
        cs = scored[0][0]
        for i in range(1, len(scored)):
            t, p = scored[i]
            if p != ct:
                segs.append((cs, scored[i - 1][0], ct))
                ct = p
                cs = t
        segs.append((cs, scored[-1][0], ct))
        return segs

    def _line_score(self, binary, x1, y1, x2, y2):
        h, w = binary.shape[:2]
        hits = total = 0
        for i in range(26):
            t = i / 25
            px, py = int(x1 + t * (x2 - x1)), int(y1 + t * (y2 - y1))
            if 0 <= px < w and 0 <= py < h:
                total += 1
                roi = binary[max(0, py - 2):min(h, py + 3), max(0, px - 2):min(w, px + 3)]
                if float(np.mean(roi)) > 50:
                    hits += 1
        return hits / max(total, 1)

    def _snap(self, s):
        if s.is_horizontal:
            ay = (s.start[1] + s.end[1]) / 2
            return WallSegment(start=(min(s.start[0], s.end[0]), ay),
                               end  =(max(s.start[0], s.end[0]), ay))
        elif s.is_vertical:
            ax = (s.start[0] + s.end[0]) / 2
            return WallSegment(start=(ax, min(s.start[1], s.end[1])),
                               end  =(ax, max(s.start[1], s.end[1])))
        return s

    def _orthogonalize(self, pts, td=12.0):
        if len(pts) < 3:
            return pts
        r = list(pts)
        th = math.radians(td)
        for _ in range(5):
            ch = False
            for i in range(len(r)):
                p, c = r[(i - 1) % len(r)], r[i]
                dx, dy = c[0] - p[0], c[1] - p[1]
                if abs(dx) < 1 and abs(dy) < 1:
                    continue
                a = math.atan2(abs(dy), abs(dx))
                if a < th:
                    r[i] = (c[0], p[1]); ch = True
                elif a > math.pi / 2 - th:
                    r[i] = (p[0], c[1]); ch = True
            if not ch:
                break
        return r

    def _remove_close_points(self, pts, md=5):
        if len(pts) < 3:
            return pts
        cl = [pts[0]]
        for i in range(1, len(pts)):
            if math.sqrt((pts[i][0] - cl[-1][0]) ** 2 + (pts[i][1] - cl[-1][1]) ** 2) > md:
                cl.append(pts[i])
        return cl