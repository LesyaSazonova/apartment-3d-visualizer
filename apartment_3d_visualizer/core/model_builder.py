"""
Построение 3D-модели по данным плана.

Примечания по эвристикам:
- линии дверных полотен на 2D могут детектиться как «стены» — фильтруем их по привязке к проёмам;
- для булевых вырезов используется solver=EXACT: на тонких стенах он стабильнее FAST.
"""
import bpy
import bmesh
import math
import os
import cv2
import numpy as np
from mathutils import Matrix, Euler, Vector

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import DEFAULT_WALL_HEIGHT, DEFAULT_WALL_THICKNESS, DEFAULT_MATERIALS

_ADDON_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_TEX_DIR       = os.path.join(_ADDON_DIR, "textures")
_MAP_KEYS      = {
    'diff':  ['diff','color','albedo','basecolor','col','base_col'],
    'rough': ['rough','roughness'],
    'nor':   ['nor_gl','nor_dx','nor','normal','nrm'],
}
_IMG_EXTS      = {'.jpg','.jpeg','.png','.exr','.tga','.hdr'}
_PRESET_FOLDER = {
    'PARQUET':'parquet','TILE':'tile','CONCRETE':'concrete',
    'CARPET':'carpet','MARBLE':'marble',
}

def _find_tex_maps(preset):
    folder = os.path.join(_TEX_DIR, _PRESET_FOLDER.get(preset,''))
    if not os.path.isdir(folder): return {}
    files  = [f for f in os.listdir(folder)
              if os.path.splitext(f)[1].lower() in _IMG_EXTS]
    found  = {}
    for mt, kws in _MAP_KEYS.items():
        for fn in files:
            if any(kw in fn.lower() for kw in kws):
                found[mt] = os.path.join(folder, fn); break
    return found


class ModelBuilder:

    COLUMN_OBJ_PROP = "apartment_type"
    COLUMN_OBJ_KIND = "column"

    CORNER_SNAP_TOL = 0.15
    
    # Толщина стен должна быть одинаковой по всей модели.
    # Константы оставлены для совместимости, но в построении используется self.wall_thickness.
    EXTERIOR_WALL_THICKNESS  = 0.30
    INTERIOR_WALL_THICKNESS  = 0.20
    PARTITION_THICKNESS      = 0.10
    
    EXTERIOR_MIN_LENGTH = 4.0
    PARTITION_MAX_LENGTH = 2.0

    def __init__(self, wall_height=DEFAULT_WALL_HEIGHT,
                 wall_thickness=DEFAULT_WALL_THICKNESS):
        self.wall_height    = wall_height
        self.wall_thickness = wall_thickness
        self._apartment_collection = None
        self._walls_collection     = None
        self._openings_collection  = None
        self._floors_collection    = None
        self._floor_obj   = None
        self._wall_objects = []
        self._wall_meta    = []

    @staticmethod
    def _segment_length(seg):
        try:
            sx, sy = seg.start
            ex, ey = seg.end
            return math.hypot(ex - sx, ey - sy)
        except Exception:
            return 0.0

    def _remove_fake_doors(self, walls, openings):
        """
        Исключает из генерации фейковые перегородки,
        образованные прямыми линиями открытых дверей на 2D плане.
        """
        valid = []
        for w in walls:
            is_fake = False
            for op in openings:
                if op.opening_type.value == "door":
                    mx = (w.start[0] + w.end[0]) / 2
                    my = (w.start[1] + w.end[1]) / 2
                    d_mid = math.hypot(mx - op.position[0], my - op.position[1])
                    # Если стена короткая и её центр находится прямо в зоне дверного проема
                    if d_mid < op.width * 0.9 and w.length < op.width * 1.6:
                        is_fake = True
                        break
            if not is_fake:
                valid.append(w)
        return valid

    def build_from_plan_data(self, plan_data):
        self.clear_scene()
        self._create_collections()

        # Линии дверных полотен на плане часто попадают в детект стен — убираем их до снапа/мерджа.
        clean_walls = self._remove_fake_doors(plan_data.wall_segments, plan_data.openings)

        print(f"[APT][DBG] raw_walls={len(plan_data.wall_segments)} cleaned(fake-door-filter)={len(clean_walls)}")

        aligned = self._snap_walls_to_grid(clean_walls)
        print(f"[APT][DBG] snap_to_grid={len(aligned)}")
        aligned = self._merge_collinear_walls(aligned, plan_data.bounding_box)
        print(f"[APT][DBG] merge_collinear={len(aligned)}")
        # После мерджа повторно стыкуем пересечения, чтобы углы сошлись в общих вершинах.
        aligned = self._extend_to_intersections(aligned, self.CORNER_SNAP_TOL)
        aligned = self._remove_isolated_walls(aligned, self.CORNER_SNAP_TOL)
        aligned = self._remove_overlapping_parallel_walls(aligned)
        aligned = self._normalize_collinear_wall_groups(aligned)
        aligned = self._snap_wall_junction_endpoints(aligned)
        aligned = self._normalize_orthogonal_junctions(aligned)
        aligned = self._normalize_outer_corner_junctions(aligned, plan_data.bounding_box)
        aligned = self._stitch_adjacent_collinear_segments(aligned)
        aligned = self._snap_wall_junction_endpoints(aligned)
        aligned = self._remove_high_overlap_parallel_duplicates(aligned)
        aligned = self._filter_suspicious_walls_near_openings(aligned, plan_data.openings)
        aligned = self._remove_high_overlap_parallel_duplicates(aligned)
        self._debug_print_bottom_right_segments(aligned, plan_data.bounding_box)
        print(f"[APT] Стен snap-to-grid: {len(aligned)}")

        # Keep pre-trim geometry for robust floor outline.
        floor_outline_walls = list(aligned)

        # Junction/column system: trim walls so they do not overlap in joints.
        junctions = self._find_wall_junctions(aligned, openings=plan_data.openings)
        aligned = self._trim_walls_for_columns(aligned, junctions)

        # Пол строим по финальной очищенной геометрии стен, чтобы он не выступал за периметр.
        self._build_floor(plan_data.contour_points, floor_outline_walls, plan_data.bounding_box)

        classified = self._classify_walls(aligned, plan_data.bounding_box)
        print(f"[APT] Классификация: внешних={sum(1 for w in classified if w['type']=='ext')}, "
              f"внутренних={sum(1 for w in classified if w['type']=='int')}, "
              f"перегородок={sum(1 for w in classified if w['type']=='part')}")

        built_wall_keys = set()
        for wall_data in classified:
            seg = wall_data['seg']
            sx, sy = seg.start
            ex, ey = seg.end
            if abs(sx - ex) < abs(sy - ey):
                if sy > ey:
                    sy, ey = ey, sy
                sx = ex = (sx + ex) / 2.0
            else:
                if sx > ex:
                    sx, ex = ex, sx
                sy = ey = (sy + ey) / 2.0
            q = (
                round(sx / 0.02), round(sy / 0.02),
                round(ex / 0.02), round(ey / 0.02),
                round(wall_data['thickness'] / 0.01)
            )
            if q in built_wall_keys:
                continue
            built_wall_keys.add(q)
            self._build_wall_segment(
                seg.start,
                seg.end,
                wall_data['thickness'],
                source_seg=seg,
                source_type=wall_data['type'],
            )

        self._build_columns(junctions, openings=plan_data.openings)

        print(f"[APT][DBG] openings_total={len(plan_data.openings)} "
              f"doors={sum(1 for o in plan_data.openings if o.opening_type.value=='door')} "
              f"windows={sum(1 for o in plan_data.openings if o.opening_type.value=='window')}")

        print(f"[APT] Построено {len(self._wall_objects)} стен, вырезаем проёмы...")

        ok = fail = 0
        skipped = {
            "no_matching_wall": 0,
            "bound_mismatch": 0,
            "ambiguous_wall_match": 0,
            "projection_outside_segment": 0,
            "boolean_failed": 0,
        }
        for opening in plan_data.openings:
            result, reason = self._cut_opening(opening)
            if result:
                ok += 1
            else:
                fail += 1
                if reason in skipped:
                    skipped[reason] += 1

        self._purge_cutters()

        print(f"[APT] Проёмов: {ok} вырезано, {fail} пропущено")
        if fail:
            print(f"[APT][DBG] skipped_breakdown={skipped}")
        self._apply_default_materials()
        self._setup_lighting(plan_data.bounding_box)
        print("[APT] Модель готова!")
        return self._apartment_collection

    def _classify_walls(self, walls, bbox):
        if not walls:
            return []
        
        minx, miny, maxx, maxy = bbox
        margin = 0.5
        
        result = []
        for seg in walls:
            length = math.hypot(seg.end[0]-seg.start[0], seg.end[1]-seg.start[1])
            
            mid_x = (seg.start[0] + seg.end[0]) / 2
            mid_y = (seg.start[1] + seg.end[1]) / 2
            
            near_border = (
                abs(mid_x - minx) < margin or abs(mid_x - maxx) < margin or
                abs(mid_y - miny) < margin or abs(mid_y - maxy) < margin
            )
            
            # Толщина стен намеренно единая; тип оставлен для возможных материалов/логики.
            if near_border and length >= self.EXTERIOR_MIN_LENGTH:
                wtype = 'ext'
            elif length < self.PARTITION_MAX_LENGTH:
                wtype = 'part'
            elif near_border:
                wtype = 'ext'
            else:
                wtype = 'int'
            thick = self.wall_thickness
            
            result.append({
                'seg': seg,
                'type': wtype,
                'thickness': thick,
                'length': length,
            })
        
        return result

    def _cut_opening(self, opening):
        if opening.opening_type.value == "door":
            cut_h      = opening.height
            cut_bottom = 0.0
        else:
            cut_h      = opening.height
            cut_bottom = opening.sill_height
        cut_top = cut_bottom + cut_h

        ox, oy = opening.position[0], opening.position[1]
        print(
            f"[APT][DBG][CUT] opening_center=({ox:.2f},{oy:.2f}) "
            f"type={opening.opening_type.value} z_min={cut_bottom:.2f} z_max={cut_top:.2f}"
        )

        best_meta = None
        best_score = float('inf')
        opening_angle = getattr(opening, 'angle', None)

        # Жёсткая привязка к исходной стене нужна, чтобы проём не переносился на соседний параллельный сегмент.
        bound_wall = getattr(opening, 'wall_segment', None)
        if bound_wall is not None:
            best_meta, best_score, amb = self._match_bound_opening_to_wall(opening, bound_wall)
            if amb:
                print(f"[APT] ✗ Ambiguous wall match for bound opening ({ox:.2f}, {oy:.2f})")
                return False, "ambiguous_wall_match"

        # Подбор по ближайшей стене — только если явной привязки нет.
        if best_meta is None and bound_wall is None:
            for meta in self._wall_meta:
                dist = self._point_to_segment_dist(
                    ox, oy,
                    meta['start'][0], meta['start'][1],
                    meta['end'][0],   meta['end'][1]
                )
                max_allowed = meta['thickness'] * 3.0 + 0.5
                if dist > max_allowed:
                    continue
                score = dist
                if opening_angle is not None:
                    angle_diff = abs(self._angle_diff(meta['angle'], opening_angle))
                    if angle_diff < math.radians(15) or abs(angle_diff - math.pi) < math.radians(15):
                        score *= 0.5
                x1, y1 = meta['start']
                x2, y2 = meta['end']
                _, _, t = self._project_point_to_segment(ox, oy, x1, y1, x2, y2)
                if t <= 0.02 or t >= 0.98:
                    score *= 2.0
                if score < best_score:
                    best_score = score
                    best_meta = meta

        # Если проём был явно привязан, но стену после очистки найти не удалось — лучше пропустить, чем вырезать не там.
        if best_meta is None and bound_wall is not None:
            print(f"[APT] ✗ Bound opening couldn't match a wall; skipped ({ox:.2f}, {oy:.2f})")
            return False, "bound_mismatch"

        if best_meta is None:
            print(f"[APT] ✗ Нет стен для проёма ({ox:.2f}, {oy:.2f})")
            return False, "no_matching_wall"

        wall_obj = best_meta['obj']
        angle    = best_meta['angle']
        thick    = best_meta['thickness']
        print(
            f"[APT][DBG][CUT] matched_wall={wall_obj.name} "
            f"start=({best_meta['start'][0]:.2f},{best_meta['start'][1]:.2f}) "
            f"end=({best_meta['end'][0]:.2f},{best_meta['end'][1]:.2f})"
        )

        x1, y1 = best_meta['start']
        x2, y2 = best_meta['end']
        proj_x, proj_y, t = self._project_point_to_segment(ox, oy, x1, y1, x2, y2)
        if t <= 0.01 or t >= 0.99:
            # Не вырезаем на самом конце сегмента: это часто даёт артефакты на стыках и ложные «дыры» рядом.
            print(f"[APT] ✗ Opening projection outside wall span; skipped ({ox:.2f}, {oy:.2f}) t={t:.3f}")
            return False, "projection_outside_segment"

        hw = opening.width / 2 + 0.03
        hh = cut_h / 2 + 0.03
        z_center = cut_bottom + cut_h / 2
        depth = thick * 3.0 + 0.2

        cutter_mesh = bpy.data.meshes.new(f"Cutter_{opening.opening_type.value}")
        cutter_obj = bpy.data.objects.new(f"Cutter_{opening.opening_type.value}", cutter_mesh)
        helper_created = True
        helper_deleted = False
        
        bm = bmesh.new()
        coords = [
            (-hw, -depth, -hh), ( hw, -depth, -hh), ( hw,  depth, -hh), (-hw,  depth, -hh),
            (-hw, -depth,  hh), ( hw, -depth,  hh), ( hw,  depth,  hh), (-hw,  depth,  hh),
        ]
        verts = [bm.verts.new(c) for c in coords]
        bm.verts.ensure_lookup_table()
        for f in [(0,1,2,3),(4,7,6,5),(0,4,5,1),(2,6,7,3),(0,3,7,4),(1,5,6,2)]:
            bm.faces.new([verts[i] for i in f])
        bm.normal_update()
        bmesh.ops.recalc_face_normals(bm, faces=bm.faces)
        bm.to_mesh(cutter_mesh)
        bm.free()
        
        cutter_obj.location = (proj_x, proj_y, z_center)
        cutter_obj.rotation_euler = (0, 0, angle)
        self._openings_collection.objects.link(cutter_obj)

        cutter_obj.hide_viewport = True
        cutter_obj.hide_render = True
        cutter_obj.display_type = 'WIRE'

        mod = wall_obj.modifiers.new(name="Cut_Opening", type='BOOLEAN')
        mod.operation = 'DIFFERENCE'
        mod.object = cutter_obj
        mod.solver = 'EXACT'

        success = self._apply_modifier_safe(wall_obj, mod)
        
        if success:
            try:
                bm_fix = bmesh.new()
                bm_fix.from_mesh(wall_obj.data)
                bmesh.ops.recalc_face_normals(bm_fix, faces=bm_fix.faces)
                bm_fix.normal_update()
                bm_fix.to_mesh(wall_obj.data)
                bm_fix.free()
                wall_obj.data.update()
            except Exception as e:
                print(f"[APT] Пересчёт нормалей не удался: {e}")
        
        if success:
            print(f"[APT] ✓ {opening.opening_type.value} вырезан в {wall_obj.name} "
                  f"(dist={best_score:.3f}m)")
        else:
            print(f"[APT] ✗ Не удалось применить Boolean для {opening.opening_type.value}")
            try:
                wall_obj.modifiers.remove(mod)
            except:
                pass

        # Резак удаляем сразу: оставленные cutter-объекты легко начинают влиять на последующие операции.
        try:
            for col in list(cutter_obj.users_collection):
                try:
                    col.objects.unlink(cutter_obj)
                except:
                    pass
            cutter_data = cutter_obj.data
            bpy.data.objects.remove(cutter_obj, do_unlink=True)
            if cutter_data and hasattr(cutter_data, 'users') and cutter_data.users == 0:
                bpy.data.meshes.remove(cutter_data)
            helper_deleted = True
        except Exception as e:
            print(f"[APT] Ошибка удаления резака {cutter_obj.name}: {e}")
        print(
            f"[APT][DBG][CUT] cut_success={success} helper_created={helper_created} "
            f"helper_deleted={helper_deleted}"
        )

        return (success, None) if success else (False, "boolean_failed")

    def _match_bound_opening_to_wall(self, opening, bound_wall):
        """Подбор «привязанного» проёма к финальному списку стен по геометрии (устойчиво к мерджу сегментов)."""
        ox, oy = opening.position[0], opening.position[1]
        bs = getattr(bound_wall, 'start', None)
        be = getattr(bound_wall, 'end', None)
        if bs is None or be is None:
            return None, float('inf'), False

        dx = be[0] - bs[0]
        dy = be[1] - bs[1]
        bound_is_h = abs(dx) >= abs(dy)

        candidates = []
        for meta in self._wall_meta:
            sx, sy = meta['start']
            ex, ey = meta['end']
            mdx = ex - sx
            mdy = ey - sy
            meta_is_h = abs(mdx) >= abs(mdy)
            if meta_is_h != bound_is_h:
                continue

            # Порог по расстоянию до оси стены: не даём проёму перепрыгнуть на параллельную стену.
            dist = self._point_to_segment_dist(ox, oy, sx, sy, ex, ey)
            max_centerline = meta['thickness'] * 1.75 + 0.20
            if dist > max_centerline:
                continue

            # Проекция должна попадать внутрь сегмента (иначе часто режет угол/торец).
            _, _, t = self._project_point_to_segment(ox, oy, sx, sy, ex, ey)
            margin_t = 0.02
            if t < margin_t or t > 1.0 - margin_t:
                continue

            # Требуем перекрытие по «длинной» оси, чтобы не матчить стену через коридор.
            if meta_is_h:
                b1, b2 = sorted([bs[0], be[0]])
                m1, m2 = sorted([sx, ex])
            else:
                b1, b2 = sorted([bs[1], be[1]])
                m1, m2 = sorted([sy, ey])
            overlap = max(0.0, min(b2, m2) - max(b1, m1))
            if overlap < 0.20:
                continue

            score = dist
            candidates.append((score, meta))

        if not candidates:
            return None, float('inf'), False

        candidates.sort(key=lambda x: x[0])
        best_score, best_meta = candidates[0]

        # Если кандидатов несколько и они слишком близки по score — не угадываем.
        if len(candidates) > 1 and (candidates[1][0] - best_score) < 0.05:
            return None, best_score, True

        return best_meta, best_score, False

    def _apply_modifier_safe(self, obj, mod):
        mod_name = mod.name
        
        try:
            win = bpy.context.window
            scr = win.screen if win else None
            area3d = None
            if scr:
                for area in scr.areas:
                    if area.type == 'VIEW_3D':
                        area3d = area
                        break
            
            if area3d:
                region = next((r for r in area3d.regions if r.type == 'WINDOW'), None)
                
                with bpy.context.temp_override(
                    window=win,
                    area=area3d,
                    region=region,
                    active_object=obj,
                    selected_objects=[obj],
                    selected_editable_objects=[obj],
                    object=obj,
                ):
                    bpy.ops.object.modifier_apply(modifier=mod_name)
                return True
        except Exception as e:
            print(f"[APT] Метод 1 не сработал: {e}")
        
        try:
            bpy.ops.object.select_all(action='DESELECT')
            obj.select_set(True)
            bpy.context.view_layer.objects.active = obj
            bpy.ops.object.modifier_apply(modifier=mod_name)
            return True
        except Exception as e:
            print(f"[APT] Метод 2 не сработал: {e}")
        
        try:
            bpy.context.view_layer.update()
            depsgraph = bpy.context.evaluated_depsgraph_get()
            eval_obj = obj.evaluated_get(depsgraph)
            new_mesh = bpy.data.meshes.new_from_object(eval_obj)
            
            old_mesh = obj.data
            obj.data = new_mesh
            obj.modifiers.remove(mod)
            
            if old_mesh and old_mesh.users == 0:
                bpy.data.meshes.remove(old_mesh)
            return True
        except Exception as e:
            print(f"[APT] Метод 3 не сработал: {e}")
        
        return False

    def _purge_cutters(self):
        to_remove = []
        for obj in list(bpy.data.objects):
            if obj.name.startswith("Cutter") or obj.name.startswith("DebugCutter"):
                to_remove.append(obj)
        
        if self._openings_collection:
            for obj in list(self._openings_collection.objects):
                if obj not in to_remove:
                    to_remove.append(obj)
        
        for obj in to_remove:
            try:
                for col in list(obj.users_collection):
                    try:
                        col.objects.unlink(obj)
                    except:
                        pass
                mesh = obj.data
                bpy.data.objects.remove(obj, do_unlink=True)
                if mesh and hasattr(mesh, 'users') and mesh.users == 0:
                    try:
                        bpy.data.meshes.remove(mesh)
                    except:
                        pass
            except Exception as e:
                print(f"[APT] Ошибка удаления резака {obj.name}: {e}")
        
        if to_remove:
            print(f"[APT] Удалено {len(to_remove)} резаков/лишних объектов")
        
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D':
                    area.tag_redraw()

    @staticmethod
    def _point_to_segment_dist(px, py, x1, y1, x2, y2):
        dx, dy = x2-x1, y2-y1
        if dx == 0 and dy == 0:
            return math.hypot(px-x1, py-y1)
        t  = max(0.0, min(1.0, ((px-x1)*dx+(py-y1)*dy)/(dx*dx+dy*dy)))
        nx = x1 + t * dx
        ny = y1 + t * dy
        return math.hypot(px-nx, py-ny)

    @staticmethod
    def _project_point_to_segment(px, py, x1, y1, x2, y2):
        dx, dy = x2-x1, y2-y1
        L2 = dx*dx + dy*dy
        if L2 == 0:
            return x1, y1, 0.0
        t = max(0.0, min(1.0, ((px-x1)*dx+(py-y1)*dy)/L2))
        return x1+t*dx, y1+t*dy, t

    @staticmethod
    def _angle_diff(a1, a2):
        d = a1 - a2
        while d > math.pi: d -= 2 * math.pi
        while d < -math.pi: d += 2 * math.pi
        return d

    def _snap_walls_to_grid(self, wall_segments):
        if not wall_segments:
            return []
        tol = self.CORNER_SNAP_TOL
        walls = []
        for w in wall_segments:
            dx = w.end[0]-w.start[0]; dy = w.end[1]-w.start[1]
            if abs(dx) >= abs(dy):
                ay = (w.start[1]+w.end[1])/2
                sx, ex = sorted([w.start[0], w.end[0]])
                walls.append(_Seg((sx, ay), (ex, ay), self.wall_thickness, True))
            else:
                ax = (w.start[0]+w.end[0])/2
                sy, ey = sorted([w.start[1], w.end[1]])
                walls.append(_Seg((ax, sy), (ax, ey), self.wall_thickness, False))

        x_axes = sorted({w.start[0] for w in walls if not w.is_horiz})
        y_axes = sorted({w.start[1] for w in walls if w.is_horiz})
        x_map  = self._cluster_1d(x_axes, tol)
        y_map  = self._cluster_1d(y_axes, tol)

        for w in walls:
            if w.is_horiz:
                ny = y_map.get(w.start[1], w.start[1])
                w.start = (w.start[0], ny); w.end = (w.end[0], ny)
            else:
                nx = x_map.get(w.start[0], w.start[0])
                w.start = (nx, w.start[1]); w.end = (nx, w.end[1])

        all_x = [c for w in walls for c in [w.start[0], w.end[0]]]
        all_y = [c for w in walls for c in [w.start[1], w.end[1]]]
        x_end = self._cluster_1d(sorted(set(all_x)), tol)
        y_end = self._cluster_1d(sorted(set(all_y)), tol)

        result = []
        for w in walls:
            sx = x_end.get(w.start[0], w.start[0])
            ex = x_end.get(w.end[0],   w.end[0])
            sy = y_end.get(w.start[1], w.start[1])
            ey = y_end.get(w.end[1],   w.end[1])
            if w.is_horiz: ey = sy
            else:          ex = sx
            if math.hypot(ex-sx, ey-sy) >= 0.05:
                result.append(_Seg((sx, sy), (ex, ey), w.thickness, w.is_horiz))

        result = self._extend_to_intersections(result, tol)
        result = self._remove_isolated_walls(result, tol)
        return result

    def _merge_collinear_walls(self, walls, bbox=None):
        """
        Merge/deduplicate collinear overlapping segments to prevent z-fighting,
        apparent thickness changes, and random boolean artifacts caused by
        overlapping wall meshes.
        """
        if not walls:
            return walls

        tol = max(self.CORNER_SNAP_TOL * 0.75, 0.05)
        join_tol = max(self.wall_thickness * 0.75, tol)

        def key_axis(w):
            # Квантуем ось, чтобы «почти одинаковые» линии схлопывались в один бакет.
            return round((w.start[1] if w.is_horiz else w.start[0]) / tol) * tol

        merged = []
        merged_parallel_pairs = 0
        for is_h in (True, False):
            group = [w for w in walls if w.is_horiz == is_h]
            buckets = {}
            for w in group:
                k = key_axis(w)
                buckets.setdefault(k, []).append(w)

            for k, segs in buckets.items():
                if is_h:
                    segs = sorted(segs, key=lambda s: (min(s.start[0], s.end[0]), max(s.start[0], s.end[0])))
                    y = sum(s.start[1] for s in segs) / max(len(segs), 1)
                    cur_s, cur_e = None, None
                    for s in segs:
                        a = min(s.start[0], s.end[0])
                        b = max(s.start[0], s.end[0])
                        if cur_s is None:
                            cur_s, cur_e = a, b
                            continue
                        if a <= cur_e + join_tol:
                            cur_e = max(cur_e, b)
                        else:
                            merged.append(_Seg((cur_s, y), (cur_e, y), self.wall_thickness, True))
                            cur_s, cur_e = a, b
                    if cur_s is not None:
                        merged.append(_Seg((cur_s, y), (cur_e, y), self.wall_thickness, True))
                else:
                    segs = sorted(segs, key=lambda s: (min(s.start[1], s.end[1]), max(s.start[1], s.end[1])))
                    x = sum(s.start[0] for s in segs) / max(len(segs), 1)
                    cur_s, cur_e = None, None
                    for s in segs:
                        a = min(s.start[1], s.end[1])
                        b = max(s.start[1], s.end[1])
                        if cur_s is None:
                            cur_s, cur_e = a, b
                            continue
                        if a <= cur_e + join_tol:
                            cur_e = max(cur_e, b)
                        else:
                            merged.append(_Seg((x, cur_s), (x, cur_e), self.wall_thickness, False))
                            cur_s, cur_e = a, b
                    if cur_s is not None:
                        merged.append(_Seg((x, cur_s), (x, cur_e), self.wall_thickness, False))

        # Финальная дедупликация: одинаковые сегменты в пределах допуска.
        uniq = []
        seen = set()
        for w in merged:
            sx, sy = w.start
            ex, ey = w.end
            if w.is_horiz and sx > ex:
                sx, ex = ex, sx
            if (not w.is_horiz) and sy > ey:
                sy, ey = ey, sy
            q = (
                w.is_horiz,
                round(sx / tol), round(sy / tol),
                round(ex / tol), round(ey / tol),
            )
            if q in seen:
                continue
            seen.add(q)
            uniq.append(_Seg((sx, sy), (ex, ey), self.wall_thickness, w.is_horiz))

        # Мердж параллельных «двойников»: на внешних стенах толстая растровая линия часто даёт два близких центра.
        parallel_tol_inner = max(self.wall_thickness * 0.55, tol * 0.9)
        # На периметре допускаем больший оффсет, чтобы схлопнуть дубликаты внешней стены.
        parallel_tol_outer = max(self.wall_thickness * 2.00, tol * 1.6)

        def is_near_border(seg):
            if not bbox:
                return False
            minx, miny, maxx, maxy = bbox
            mx = (seg.start[0] + seg.end[0]) / 2.0
            my = (seg.start[1] + seg.end[1]) / 2.0
            margin = 0.7
            return (
                abs(mx - minx) < margin or abs(mx - maxx) < margin or
                abs(my - miny) < margin or abs(my - maxy) < margin
            )

        def overlap_1d(a1, a2, b1, b2):
            return max(0.0, min(a2, b2) - max(a1, b1))

        out = []
        for is_h in (True, False):
            segs = [w for w in uniq if w.is_horiz == is_h]
            segs.sort(key=lambda s: (s.start[1] if is_h else s.start[0],
                                     min(s.start[0], s.end[0]) if is_h else min(s.start[1], s.end[1])))
            used = [False] * len(segs)
            for i, a in enumerate(segs):
                if used[i]:
                    continue
                base = a
                used[i] = True

                # Пытаемся поглотить близкие параллельные дубликаты.
                for j in range(i + 1, len(segs)):
                    if used[j]:
                        continue
                    b = segs[j]
                    a_axis = base.start[1] if is_h else base.start[0]
                    b_axis = b.start[1] if is_h else b.start[0]
                    # На внешних стенах оффсет допускаем больше.
                    ptol = parallel_tol_outer if (is_near_border(base) and is_near_border(b)) else parallel_tol_inner
                    if abs(b_axis - a_axis) > ptol:
                        if b_axis - a_axis > ptol:
                            break
                        continue

                    if is_h:
                        a1, a2 = sorted([base.start[0], base.end[0]])
                        b1, b2 = sorted([b.start[0], b.end[0]])
                    else:
                        a1, a2 = sorted([base.start[1], base.end[1]])
                        b1, b2 = sorted([b.start[1], b.end[1]])

                    ov = overlap_1d(a1, a2, b1, b2)
                    if ov < min(a2 - a1, b2 - b1) * 0.75:
                        continue

                    merged_parallel_pairs += 1
                    new_axis = (a_axis + b_axis) / 2.0
                    new_s = min(a1, b1)
                    new_e = max(a2, b2)
                    used[j] = True
                    if is_h:
                        base = _Seg((new_s, new_axis), (new_e, new_axis), self.wall_thickness, True)
                    else:
                        base = _Seg((new_axis, new_s), (new_axis, new_e), self.wall_thickness, False)

                out.append(base)

        if merged_parallel_pairs:
            print(f"[APT][DBG] merged_parallel_pairs={merged_parallel_pairs}")

        out = self._merge_axis_offset_collinear_groups(out)
        return out

    def _merge_axis_offset_collinear_groups(self, walls):
        """
        Merge collinear segments even when their axis differs slightly.
        Helps when one logical wall is split with a small perpendicular offset.
        """
        if not walls:
            return walls

        # Roughly corresponds to ~15-25 px at common plan scales.
        axis_tol = max(self.wall_thickness * 1.6, self.CORNER_SNAP_TOL * 2.0, 0.22)
        axis_tol = min(axis_tol, max(self.wall_thickness * 2.1, 0.38))
        gap_tol = max(self.wall_thickness * 1.35, 0.18)
        overlap_gate = 0.18

        merged_any = 0
        out = []

        def interval(seg):
            if seg.is_horiz:
                return sorted([seg.start[0], seg.end[0]])
            return sorted([seg.start[1], seg.end[1]])

        def axis(seg):
            return seg.start[1] if seg.is_horiz else seg.start[0]

        def ov_1d(a1, a2, b1, b2):
            return max(0.0, min(a2, b2) - max(a1, b1))

        for is_h in (True, False):
            segs = [s for s in walls if s.is_horiz == is_h]
            if not segs:
                continue
            segs = sorted(segs, key=lambda s: axis(s))
            used = [False] * len(segs)

            for i, base in enumerate(segs):
                if used[i]:
                    continue
                used[i] = True
                chain = [base]
                a1, a2 = interval(base)
                axis_sum = axis(base)

                changed = True
                while changed:
                    changed = False
                    chain_axis = axis_sum / max(len(chain), 1)
                    for j, cand in enumerate(segs):
                        if used[j]:
                            continue
                        cand_axis = axis(cand)
                        if abs(cand_axis - chain_axis) > axis_tol:
                            continue
                        c1, c2 = interval(cand)
                        ov = ov_1d(a1, a2, c1, c2)
                        min_len = max(min(a2 - a1, c2 - c1), 1e-6)
                        near_join = c1 <= a2 + gap_tol and c2 >= a1 - gap_tol
                        if (ov / min_len) < overlap_gate and not near_join:
                            continue
                        chain.append(cand)
                        used[j] = True
                        merged_any += 1
                        axis_sum += cand_axis
                        a1 = min(a1, c1)
                        a2 = max(a2, c2)
                        changed = True

                merged_axis = axis_sum / max(len(chain), 1)
                if is_h:
                    out.append(_Seg((a1, merged_axis), (a2, merged_axis), self.wall_thickness, True))
                else:
                    out.append(_Seg((merged_axis, a1), (merged_axis, a2), self.wall_thickness, False))

        if merged_any:
            print(f"[APT][FIX] collinear_axis_offset_merged={merged_any} axis_tol={axis_tol:.3f}")
        return out

    @staticmethod
    def _cluster_1d(values, tol):
        if not values: return {}
        clusters = [[values[0]]]
        for v in sorted(values)[1:]:
            if v - clusters[-1][-1] <= tol: clusters[-1].append(v)
            else: clusters.append([v])
        m = {}
        for c in clusters:
            mean = sum(c)/len(c)
            for v in c: m[v] = mean
        return m

    def _extend_to_intersections(self, walls, tol):
        horiz = [w for w in walls if w.is_horiz]
        vert  = [w for w in walls if not w.is_horiz]
        # Увеличенный допуск помогает «дотянуть» стыки до единой вершины без ручной правки.
        ext   = max(tol*5, self.wall_thickness*2.2)
        for h in horiz:
            hy = h.start[1]
            for v in vert:
                vx = v.start[0]
                vy1, vy2 = sorted([v.start[1], v.end[1]])
                if not (vy1-ext <= hy <= vy2+ext): continue
                if abs(h.start[0]-vx) <= ext: h.start = (vx, hy)
                if abs(h.end[0]  -vx) <= ext: h.end   = (vx, hy)
                if abs(v.start[1]-hy) <= ext: v.start = (vx, hy)
                if abs(v.end[1]  -hy) <= ext: v.end   = (vx, hy)
        out = []
        for w in walls:
            sx, sy = w.start; ex, ey = w.end
            if w.is_horiz and sx > ex: sx, ex = ex, sx
            elif not w.is_horiz and sy > ey: sy, ey = ey, sy
            if math.hypot(ex-sx, ey-sy) >= 0.05:
                out.append(_Seg((sx,sy),(ex,ey),w.thickness,w.is_horiz))
        return out

    def _remove_isolated_walls(self, walls, tol):
        if len(walls) < 2: return walls
        snap = max(tol*4, self.wall_thickness*2.0)
        mink = self.wall_thickness * 2.5
        def connected(wx, wy, oi):
            for j, w2 in enumerate(walls):
                if j == oi: continue
                x1,y1=w2.start; x2,y2=w2.end; dx,dy=x2-x1,y2-y1; L2=dx*dx+dy*dy
                if L2 < 1e-12: d = math.hypot(wx-x1, wy-y1)
                else:
                    tc = max(0., min(1., ((wx-x1)*dx+(wy-y1)*dy)/L2))
                    d  = math.hypot(wx-x1-tc*dx, wy-y1-tc*dy)
                if d <= snap: return True
            return False
        kept = []
        for i, w in enumerate(walls):
            cs = connected(w.start[0], w.start[1], i)
            ce = connected(w.end[0],   w.end[1],   i)
            ln = math.hypot(w.end[0]-w.start[0], w.end[1]-w.start[1])
            if not cs and not ce: continue
            if ln < mink and (not cs or not ce): continue
            kept.append(w)
        print(f"[APT] Стен после фильтра: {len(kept)} (было {len(walls)})")
        return kept

    def _remove_overlapping_parallel_walls(self, walls):
        """
        Удаляет почти совпадающие параллельные сегменты, которые дают визуальный «двойной» контур.
        Оставляет более длинный/базовый сегмент.
        """
        if len(walls) < 2:
            return walls

        axis_tol = max(self.wall_thickness * 0.95, self.CORNER_SNAP_TOL * 1.2, 0.10)
        end_tol = max(self.wall_thickness * 1.20, 0.18)

        used = [False] * len(walls)
        out = []

        def overlap_1d(a1, a2, b1, b2):
            return max(0.0, min(a2, b2) - max(a1, b1))

        for i, a in enumerate(walls):
            if used[i]:
                continue
            keep = a
            for j in range(i + 1, len(walls)):
                if used[j]:
                    continue
                b = walls[j]
                if a.is_horiz != b.is_horiz:
                    continue

                if a.is_horiz:
                    a_axis = a.start[1]
                    b_axis = b.start[1]
                    if abs(a_axis - b_axis) > axis_tol:
                        continue
                    a1, a2 = sorted([a.start[0], a.end[0]])
                    b1, b2 = sorted([b.start[0], b.end[0]])
                else:
                    a_axis = a.start[0]
                    b_axis = b.start[0]
                    if abs(a_axis - b_axis) > axis_tol:
                        continue
                    a1, a2 = sorted([a.start[1], a.end[1]])
                    b1, b2 = sorted([b.start[1], b.end[1]])

                ov = overlap_1d(a1, a2, b1, b2)
                min_len = max(min(a.length, b.length), 1e-6)
                if ov < min_len * 0.80:
                    continue

                # Почти одинаковые концы или один сегмент вложен в другой.
                close_ends = (
                    (abs(a1 - b1) <= end_tol and abs(a2 - b2) <= end_tol) or
                    (a1 >= b1 - end_tol and a2 <= b2 + end_tol) or
                    (b1 >= a1 - end_tol and b2 <= a2 + end_tol)
                )
                if not close_ends:
                    continue

                # Помечаем более короткий как дубликат.
                if a.length >= b.length:
                    used[j] = True
                    keep = a
                else:
                    used[i] = True
                    keep = b
                    break

            if not used[i]:
                out.append(keep)

        if len(out) != len(walls):
            print(f"[APT][DBG] remove_overlapping_parallel: {len(walls)} -> {len(out)}")
        return out

    def _snap_wall_junction_endpoints(self, walls):
        """
        Финальная точечная склейка узлов: если горизонтальный и вертикальный сегменты
        должны сходиться в углу, принудительно задаём им общий endpoint.
        Убирает микро-смещения/ступеньки в углах после merge/dedup.
        """
        if len(walls) < 2:
            return walls

        join_tol = max(self.wall_thickness * 0.90, self.CORNER_SNAP_TOL * 0.90, 0.08)

        # 1) Кластеризуем близкие endpoints в одну точку.
        points = []
        for i, w in enumerate(walls):
            points.append((i, 0, w.start[0], w.start[1]))
            points.append((i, 1, w.end[0], w.end[1]))

        used = [False] * len(points)
        snapped = {}
        for i in range(len(points)):
            if used[i]:
                continue
            cluster = [i]
            used[i] = True
            for j in range(i + 1, len(points)):
                if used[j]:
                    continue
                if math.hypot(points[i][2] - points[j][2], points[i][3] - points[j][3]) <= join_tol:
                    used[j] = True
                    cluster.append(j)
            cx = sum(points[k][2] for k in cluster) / len(cluster)
            cy = sum(points[k][3] for k in cluster) / len(cluster)
            for k in cluster:
                seg_idx, end_idx, _, _ = points[k]
                snapped[(seg_idx, end_idx)] = (cx, cy)

        for i, w in enumerate(walls):
            s = snapped.get((i, 0), w.start)
            e = snapped.get((i, 1), w.end)
            walls[i] = _Seg(s, e, w.thickness, w.is_horiz)

        # 2) Явно дотягиваем ортогональные углы до общей точки (x вертикали, y горизонтали).
        for i, a in enumerate(walls):
            for j, b in enumerate(walls):
                if j <= i:
                    continue
                if a.is_horiz == b.is_horiz:
                    continue

                h = a if a.is_horiz else b
                v = b if a.is_horiz else a
                vx = (v.start[0] + v.end[0]) / 2.0
                hy = (h.start[1] + h.end[1]) / 2.0
                corner = (vx, hy)

                h_ends = [h.start, h.end]
                v_ends = [v.start, v.end]
                h_idx = 0 if math.hypot(h_ends[0][0] - corner[0], h_ends[0][1] - corner[1]) <= math.hypot(h_ends[1][0] - corner[0], h_ends[1][1] - corner[1]) else 1
                v_idx = 0 if math.hypot(v_ends[0][0] - corner[0], v_ends[0][1] - corner[1]) <= math.hypot(v_ends[1][0] - corner[0], v_ends[1][1] - corner[1]) else 1

                if math.hypot(h_ends[h_idx][0] - corner[0], h_ends[h_idx][1] - corner[1]) > join_tol:
                    continue
                if math.hypot(v_ends[v_idx][0] - corner[0], v_ends[v_idx][1] - corner[1]) > join_tol:
                    continue

                if h_idx == 0:
                    h = _Seg(corner, h.end, h.thickness, True)
                else:
                    h = _Seg(h.start, corner, h.thickness, True)
                if v_idx == 0:
                    v = _Seg(corner, v.end, v.thickness, False)
                else:
                    v = _Seg(v.start, corner, v.thickness, False)

                if a.is_horiz:
                    walls[i], walls[j] = h, v
                else:
                    walls[i], walls[j] = v, h

        # 3) Нормализуем направление сегментов.
        out = []
        for w in walls:
            sx, sy = w.start
            ex, ey = w.end
            if w.is_horiz and sx > ex:
                sx, ex = ex, sx
            elif (not w.is_horiz) and sy > ey:
                sy, ey = ey, sy
            if math.hypot(ex - sx, ey - sy) >= 0.05:
                out.append(_Seg((sx, sy), (ex, ey), w.thickness, w.is_horiz))
        return out

    def _normalize_outer_corner_junctions(self, walls, bbox):
        """
        Явная нормализация внешнего угла (в т.ч. правого-нижнего), чтобы
        горизонтальный и вертикальный сегменты разделяли один и тот же endpoint.
        """
        if not walls or not bbox:
            return walls

        minx, miny, maxx, maxy = bbox
        tol_axis = max(self.wall_thickness * 1.2, 0.12)
        tol_end = max(self.wall_thickness * 1.4, 0.16)

        horizontals = [w for w in walls if w.is_horiz]
        verticals = [w for w in walls if not w.is_horiz]
        if not horizontals or not verticals:
            return walls

        # Целимся в правый-нижний внешний угол.
        right_vs = [v for v in verticals if abs(v.start[0] - maxx) <= tol_axis]
        bottom_hs = [h for h in horizontals if abs(h.start[1] - miny) <= tol_axis]
        if not right_vs or not bottom_hs:
            return walls

        v = min(right_vs, key=lambda s: min(abs(s.start[1] - miny), abs(s.end[1] - miny)))
        h = min(bottom_hs, key=lambda s: min(abs(s.start[0] - maxx), abs(s.end[0] - maxx)))

        corner = (float(v.start[0]), float(h.start[1]))

        def snap_seg_endpoint(seg, target):
            ds = math.hypot(seg.start[0] - target[0], seg.start[1] - target[1])
            de = math.hypot(seg.end[0] - target[0], seg.end[1] - target[1])
            if min(ds, de) > tol_end:
                return seg, False
            if ds <= de:
                return _Seg(target, seg.end, seg.thickness, seg.is_horiz), True
            return _Seg(seg.start, target, seg.thickness, seg.is_horiz), True

        v2, vs = snap_seg_endpoint(v, corner)
        h2, hs = snap_seg_endpoint(h, corner)
        if not (vs and hs):
            return walls

        out = []
        replaced_v = replaced_h = False
        for s in walls:
            if (not replaced_v) and s is v:
                out.append(v2)
                replaced_v = True
            elif (not replaced_h) and s is h:
                out.append(h2)
                replaced_h = True
            else:
                out.append(s)

        print(
            f"[APT][DBG][CORNER] bottom_right corner=({corner[0]:.3f},{corner[1]:.3f}) "
            f"h=({h.start[0]:.3f},{h.start[1]:.3f})->({h.end[0]:.3f},{h.end[1]:.3f}) "
            f"v=({v.start[0]:.3f},{v.start[1]:.3f})->({v.end[0]:.3f},{v.end[1]:.3f}) "
            f"snapped_h={hs} snapped_v={vs}"
        )
        return out

    def _normalize_orthogonal_junctions(self, walls):
        """
        Явная сварка ортогональных стыков:
        если горизонтальный и вертикальный сегменты должны встретиться в углу,
        их ближайшие endpoints принудительно ставятся в одну intersection-точку.
        """
        if len(walls) < 2:
            return walls

        tol = max(self.wall_thickness * 1.1, self.CORNER_SNAP_TOL * 0.9, 0.10)
        axis_proximity_tol = max(self.wall_thickness * 1.35, self.CORNER_SNAP_TOL * 1.1, 0.14)
        span_extend_tol = max(self.wall_thickness * 1.7, self.CORNER_SNAP_TOL * 1.5, 0.20)
        changed = 0
        out = list(walls)

        def point_to_point(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        for i, a in enumerate(out):
            for j in range(i + 1, len(out)):
                b = out[j]
                if a.is_horiz == b.is_horiz:
                    continue

                h = a if a.is_horiz else b
                v = b if a.is_horiz else a
                ix = (v.start[0] + v.end[0]) / 2.0
                iy = (h.start[1] + h.end[1]) / 2.0
                inter = (ix, iy)

                hsx, hsy = h.start
                hex_, hey = h.end
                vsx, vsy = v.start
                vex, vey = v.end

                h_min_x, h_max_x = sorted([hsx, hex_])
                v_min_y, v_max_y = sorted([vsy, vey])

                # Allow slightly extended span to catch near-miss corners.
                near_h_span = (h_min_x - span_extend_tol) <= ix <= (h_max_x + span_extend_tol)
                near_v_span = (v_min_y - span_extend_tol) <= iy <= (v_max_y + span_extend_tol)
                if not (near_h_span and near_v_span):
                    continue

                h_ends = [h.start, h.end]
                v_ends = [v.start, v.end]
                h_idx = 0 if point_to_point(h_ends[0], inter) <= point_to_point(h_ends[1], inter) else 1
                v_idx = 0 if point_to_point(v_ends[0], inter) <= point_to_point(v_ends[1], inter) else 1
                if abs(h_ends[h_idx][1] - iy) > axis_proximity_tol:
                    continue
                if abs(v_ends[v_idx][0] - ix) > axis_proximity_tol:
                    continue
                if point_to_point(h_ends[h_idx], inter) > span_extend_tol or point_to_point(v_ends[v_idx], inter) > span_extend_tol:
                    continue

                if h_idx == 0:
                    h2 = _Seg(inter, h.end, h.thickness, True)
                else:
                    h2 = _Seg(h.start, inter, h.thickness, True)
                if v_idx == 0:
                    v2 = _Seg(inter, v.end, v.thickness, False)
                else:
                    v2 = _Seg(v.start, inter, v.thickness, False)

                if a.is_horiz:
                    out[i], out[j] = h2, v2
                    a, b = h2, v2
                else:
                    out[i], out[j] = v2, h2
                    a, b = v2, h2
                changed += 1
                print(
                    f"[APT][FIX] corner_snap inter=({inter[0]:.3f},{inter[1]:.3f}) "
                    f"h=({h.start[0]:.3f},{h.start[1]:.3f})->({h.end[0]:.3f},{h.end[1]:.3f}) "
                    f"v=({v.start[0]:.3f},{v.start[1]:.3f})->({v.end[0]:.3f},{v.end[1]:.3f})"
                )

        if changed:
            print(f"[APT][DBG][CORNER] orthogonal_junctions_welded={changed}")
        return out

    def _remove_high_overlap_parallel_duplicates(self, walls):
        """
        Final pass: remove almost-identical parallel segments with >80% overlap.
        """
        if len(walls) < 2:
            return walls

        axis_tol = max(self.wall_thickness * 1.0, self.CORNER_SNAP_TOL * 1.1, 0.12)
        used = [False] * len(walls)
        out = []
        removed = 0

        def overlap_1d(a1, a2, b1, b2):
            return max(0.0, min(a2, b2) - max(a1, b1))

        for i, a in enumerate(walls):
            if used[i]:
                continue
            keep = a
            for j in range(i + 1, len(walls)):
                if used[j]:
                    continue
                b = walls[j]
                if a.is_horiz != b.is_horiz:
                    continue

                if a.is_horiz:
                    axis_a, axis_b = a.start[1], b.start[1]
                    if abs(axis_a - axis_b) > axis_tol:
                        continue
                    a1, a2 = sorted([a.start[0], a.end[0]])
                    b1, b2 = sorted([b.start[0], b.end[0]])
                else:
                    axis_a, axis_b = a.start[0], b.start[0]
                    if abs(axis_a - axis_b) > axis_tol:
                        continue
                    a1, a2 = sorted([a.start[1], a.end[1]])
                    b1, b2 = sorted([b.start[1], b.end[1]])

                ov = overlap_1d(a1, a2, b1, b2)
                min_len = max(min(a2 - a1, b2 - b1), 1e-6)
                if ov / min_len < 0.80:
                    continue

                if self._segment_length(b) <= self._segment_length(keep):
                    used[j] = True
                    removed += 1
                    print(
                        f"[APT][FIX] drop_parallel_duplicate "
                        f"A=({a.start[0]:.3f},{a.start[1]:.3f})->({a.end[0]:.3f},{a.end[1]:.3f}) "
                        f"B=({b.start[0]:.3f},{b.start[1]:.3f})->({b.end[0]:.3f},{b.end[1]:.3f})"
                    )
                else:
                    used[i] = True
                    keep = b
                    removed += 1
                    break

            if not used[i]:
                out.append(keep)

        if removed:
            print(f"[APT][FIX] high_overlap_parallel_removed={removed}")
        return out

    def _find_wall_junctions(self, walls, openings=None):
        """
        Detect corner / T-junction / endpoint nodes for column-based joins.
        """
        if not walls:
            print("[APT][JUNCTION] found 0 corners, 0 T-junctions, 0 endpoints")
            return []

        tol = max(self.wall_thickness * 1.5, self.CORNER_SNAP_TOL * 1.3, 0.12)
        endpoints = []
        for wi, w in enumerate(walls):
            endpoints.append({
                "wall_idx": wi,
                "end_key": "start",
                "pt": (float(w.start[0]), float(w.start[1])),
                "is_horiz": bool(w.is_horiz),
            })
            endpoints.append({
                "wall_idx": wi,
                "end_key": "end",
                "pt": (float(w.end[0]), float(w.end[1])),
                "is_horiz": bool(w.is_horiz),
            })

        def p_dist(a, b):
            return math.hypot(a[0] - b[0], a[1] - b[1])

        def point_to_segment_with_t(px, py, x1, y1, x2, y2):
            dx, dy = x2 - x1, y2 - y1
            L2 = dx * dx + dy * dy
            if L2 <= 1e-12:
                return math.hypot(px - x1, py - y1), 0.0
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / L2))
            nx = x1 + t * dx
            ny = y1 + t * dy
            return math.hypot(px - nx, py - ny), t

        def endpoint_ref_id(ref):
            return (ref["wall_idx"], ref["end_key"])

        connected = set()
        junctions = []
        corner_count = 0
        t_count = 0
        endpoint_count = 0

        # 1) Corner junctions by endpoint clusters containing horizontal + vertical walls.
        used = [False] * len(endpoints)
        for i in range(len(endpoints)):
            if used[i]:
                continue
            cluster = [i]
            used[i] = True
            for j in range(i + 1, len(endpoints)):
                if used[j]:
                    continue
                if p_dist(endpoints[i]["pt"], endpoints[j]["pt"]) <= tol:
                    used[j] = True
                    cluster.append(j)

            refs = [endpoints[k] for k in cluster]
            has_h = any(r["is_horiz"] for r in refs)
            has_v = any((not r["is_horiz"]) for r in refs)
            if not (has_h and has_v):
                continue

            px = sum(r["pt"][0] for r in refs) / len(refs)
            py = sum(r["pt"][1] for r in refs) / len(refs)
            trim_ends = set()
            wall_ids = set()
            for r in refs:
                connected.add(endpoint_ref_id(r))
                trim_ends.add(endpoint_ref_id(r))
                wall_ids.add(r["wall_idx"])
            junctions.append({
                "junction_type": "corner",
                "position": (px, py),
                "connected_walls": sorted(wall_ids),
                "trim_ends": trim_ends,
            })
            corner_count += 1

        # 2) T-junctions: endpoint of one wall lies on body of another wall.
        seen_t = set()
        for ref in endpoints:
            rid = endpoint_ref_id(ref)
            if rid in connected:
                continue
            px, py = ref["pt"]
            best = None
            best_dist = float("inf")
            for wi, w in enumerate(walls):
                if wi == ref["wall_idx"]:
                    continue
                d, t = point_to_segment_with_t(px, py, w.start[0], w.start[1], w.end[0], w.end[1])
                if d > tol:
                    continue
                # Must hit segment body (not its own endpoints).
                if t <= 0.10 or t >= 0.90:
                    continue
                if d < best_dist:
                    best = wi
                    best_dist = d
            if best is None:
                continue

            key = (round(px / tol), round(py / tol), ref["wall_idx"], best)
            if key in seen_t:
                continue
            seen_t.add(key)
            connected.add(rid)
            junctions.append({
                "junction_type": "t_junction",
                "position": (px, py),
                "connected_walls": sorted({ref["wall_idx"], best}),
                "trim_ends": {rid},  # trim only ending wall, not through wall
            })
            t_count += 1

        # 3) Unconnected endpoints.
        for ref in endpoints:
            rid = endpoint_ref_id(ref)
            if rid in connected:
                continue
            px, py = ref["pt"]
            junctions.append({
                "junction_type": "endpoint",
                "position": (px, py),
                "connected_walls": [ref["wall_idx"]],
                "trim_ends": {rid},
            })
            endpoint_count += 1

        print(f"[APT][JUNCTION] found {corner_count} corners, {t_count} T-junctions, {endpoint_count} endpoints")
        return junctions

    def _trim_walls_for_columns(self, walls, junctions):
        """
        Shorten wall ends participating in junctions to make room for columns.
        """
        if not walls or not junctions:
            return walls

        # EXACT flush trim for column joins:
        # column size in plan = wall_thickness x wall_thickness,
        # therefore each trimmed endpoint shifts by wall_thickness/2.
        trim_dist = self.wall_thickness * 0.5
        min_len_to_trim = self.wall_thickness * 2.0
        trim_map = {}
        for j in junctions:
            for wi, end_key in j.get("trim_ends", set()):
                trim_map.setdefault(wi, {"start": False, "end": False})[end_key] = True

        out = []
        trimmed_ends = 0
        for wi, w in enumerate(walls):
            flags = trim_map.get(wi)
            if not flags:
                out.append(w)
                continue

            sx, sy = w.start
            ex, ey = w.end
            dx, dy = ex - sx, ey - sy
            ln = math.hypot(dx, dy)
            if ln < max(min_len_to_trim, 0.05):
                out.append(w)
                continue

            ux, uy = dx / ln, dy / ln
            nsx, nsy = sx, sy
            nex, ney = ex, ey

            if flags.get("start"):
                nsx += ux * trim_dist
                nsy += uy * trim_dist
                trimmed_ends += 1
            if flags.get("end"):
                nex -= ux * trim_dist
                ney -= uy * trim_dist
                trimmed_ends += 1

            # Keep strict orthogonality.
            if w.is_horiz:
                y_axis = (nsy + ney) * 0.5
                nsy = y_axis
                ney = y_axis
            else:
                x_axis = (nsx + nex) * 0.5
                nsx = x_axis
                nex = x_axis

            if math.hypot(nex - nsx, ney - nsy) >= 0.05:
                out.append(_Seg((nsx, nsy), (nex, ney), w.thickness, w.is_horiz))

        if trimmed_ends:
            print(f"[APT][JUNCTION] trimmed_wall_ends={trimmed_ends} trim_dist={trim_dist:.3f}")
        return out

    def _build_columns(self, junctions, openings=None):
        if not junctions:
            print("[APT][COLUMN] built 0 columns")
            return

        opening_pts = []
        if openings:
            for op in openings:
                try:
                    opening_pts.append((float(op.position[0]), float(op.position[1])))
                except Exception:
                    continue
        opening_tol = max(self.wall_thickness * 1.2, 0.25)

        def near_opening(pos):
            for ox, oy in opening_pts:
                if math.hypot(pos[0] - ox, pos[1] - oy) <= opening_tol:
                    return True
            return False

        wall_mat = bpy.data.materials.get("Wall_Material")
        built = 0
        for idx, j in enumerate(junctions):
            # No column for collinear end-to-end joins.
            if j.get("junction_type") == "collinear":
                continue
            pos = j.get("position", (0.0, 0.0))
            if near_opening(pos):
                continue

            # FIX: column plan size must be EXACTLY wall thickness (no multipliers).
            jx, jy = float(pos[0]), float(pos[1])
            col_size = float(self.wall_thickness)
            hw = col_size * 0.5
            hd = col_size * 0.5
            hh = self.wall_height * 0.5

            mesh = bpy.data.meshes.new(f"ColumnMesh_{idx}")
            obj = bpy.data.objects.new(f"Column_{idx}", mesh)
            obj[self.COLUMN_OBJ_PROP] = self.COLUMN_OBJ_KIND
            bm = bmesh.new()
            coords = [
                (-hw, -hd, -hh), ( hw, -hd, -hh), ( hw,  hd, -hh), (-hw,  hd, -hh),
                (-hw, -hd,  hh), ( hw, -hd,  hh), ( hw,  hd,  hh), (-hw,  hd,  hh),
            ]
            verts = [bm.verts.new(c) for c in coords]
            bm.verts.ensure_lookup_table()
            for f in [(0,1,2,3),(4,7,6,5),(0,4,5,1),(2,6,7,3),(0,3,7,4),(1,5,6,2)]:
                bm.faces.new([verts[i] for i in f])
            bm.normal_update()
            bm.to_mesh(mesh)
            bm.free()

            # FIX: column center is EXACTLY at junction intersection.
            obj.location = (jx, jy, hh)
            if self._walls_collection:
                self._walls_collection.objects.link(obj)
            elif self._apartment_collection:
                self._apartment_collection.objects.link(obj)
            else:
                bpy.context.scene.collection.objects.link(obj)

            if wall_mat:
                obj.data.materials.append(wall_mat)

            # Keep columns in wall object registry for materials/updates.
            self._wall_objects.append(obj)
            built += 1
            print(
                f"[APT][COLUMN] Column idx={idx} pos=({jx:.3f},{jy:.3f}) "
                f"size={col_size:.3f}x{col_size:.3f} (wall_thickness={self.wall_thickness:.3f})"
            )

        print(f"[APT][COLUMN] built {built} columns")

    def _debug_print_bottom_right_segments(self, walls, bbox):
        if not walls or not bbox:
            return
        minx, miny, maxx, maxy = bbox
        tol = max(self.wall_thickness * 1.6, 0.20)
        near = []
        for i, s in enumerate(walls):
            sx, sy = s.start
            ex, ey = s.end
            if (
                max(sx, ex) >= (maxx - tol) or
                min(sx, ex) >= (maxx - tol) or
                min(sy, ey) <= (miny + tol)
            ):
                near.append((i, s))
        if not near:
            return
        print("[APT][DBG][CORNER] segments_near_bottom_right:")
        for i, s in near:
            print(
                f"[APT][DBG][CORNER]  #{i} "
                f"({s.start[0]:.4f},{s.start[1]:.4f}) -> ({s.end[0]:.4f},{s.end[1]:.4f}) "
                f"{'H' if s.is_horiz else 'V'}"
            )

    def _stitch_adjacent_collinear_segments(self, walls):
        """
        Финальная сшивка соседних коллинеарных сегментов:
        убирает маленькие швы/ступеньки, когда одна стена случайно разбилась на 2 куска.
        """
        if not walls:
            return walls

        axis_tol = max(self.CORNER_SNAP_TOL * 0.65, self.wall_thickness * 0.45, 0.05)
        gap_tol = max(self.wall_thickness * 0.85, 0.12)
        out = []

        for is_h in (True, False):
            segs = [s for s in walls if s.is_horiz == is_h]
            if not segs:
                continue

            # Бакеты по оси (y для горизонталей, x для вертикалей).
            buckets = {}
            for s in segs:
                axis = s.start[1] if is_h else s.start[0]
                key = round(axis / axis_tol)
                buckets.setdefault(key, []).append(s)

            for _, group in buckets.items():
                if is_h:
                    group = sorted(group, key=lambda s: min(s.start[0], s.end[0]))
                    axis = sum(s.start[1] for s in group) / len(group)
                    cur_s = min(group[0].start[0], group[0].end[0])
                    cur_e = max(group[0].start[0], group[0].end[0])
                    for s in group[1:]:
                        s1 = min(s.start[0], s.end[0])
                        s2 = max(s.start[0], s.end[0])
                        if s1 <= cur_e + gap_tol:
                            cur_e = max(cur_e, s2)
                        else:
                            out.append(_Seg((cur_s, axis), (cur_e, axis), self.wall_thickness, True))
                            cur_s, cur_e = s1, s2
                    out.append(_Seg((cur_s, axis), (cur_e, axis), self.wall_thickness, True))
                else:
                    group = sorted(group, key=lambda s: min(s.start[1], s.end[1]))
                    axis = sum(s.start[0] for s in group) / len(group)
                    cur_s = min(group[0].start[1], group[0].end[1])
                    cur_e = max(group[0].start[1], group[0].end[1])
                    for s in group[1:]:
                        s1 = min(s.start[1], s.end[1])
                        s2 = max(s.start[1], s.end[1])
                        if s1 <= cur_e + gap_tol:
                            cur_e = max(cur_e, s2)
                        else:
                            out.append(_Seg((axis, cur_s), (axis, cur_e), self.wall_thickness, False))
                            cur_s, cur_e = s1, s2
                    out.append(_Seg((axis, cur_s), (axis, cur_e), self.wall_thickness, False))

        # Нормализуем направление и выкидываем вырожденные.
        norm = []
        for s in out:
            sx, sy = s.start
            ex, ey = s.end
            if s.is_horiz and sx > ex:
                sx, ex = ex, sx
            elif (not s.is_horiz) and sy > ey:
                sy, ey = ey, sy
            if math.hypot(ex - sx, ey - sy) >= 0.05:
                norm.append(_Seg((sx, sy), (ex, ey), s.thickness, s.is_horiz))

        if len(norm) != len(walls):
            print(f"[APT][DBG][CORNER] stitch_adjacent_collinear: {len(walls)} -> {len(norm)}")
        return norm
    def _normalize_collinear_wall_groups(self, walls):
        """
        Нормализация фрагментов одной логической стены:
        все коллинеарные куски в группе получают одну каноническую ось
        (Y для горизонталей, X для вертикалей), чтобы не было смещения блоков.
        """
        if not walls:
            return walls

        axis_tol = max(self.CORNER_SNAP_TOL * 0.8, self.wall_thickness * 0.4, 0.05)
        out = []

        for is_h in (True, False):
            segs = [s for s in walls if s.is_horiz == is_h]
            if not segs:
                continue

            # Грубая группировка по оси.
            buckets = {}
            for s in segs:
                axis = s.start[1] if is_h else s.start[0]
                key = round(axis / axis_tol)
                buckets.setdefault(key, []).append(s)

            # Внутри бакета разбиваем на связные по диапазону группы (чтобы разные стены не склеить).
            for group in buckets.values():
                ordered = sorted(
                    group,
                    key=lambda s: min(s.start[0], s.end[0]) if is_h else min(s.start[1], s.end[1])
                )
                chains = []
                cur = [ordered[0]]
                for s in ordered[1:]:
                    prev = cur[-1]
                    if is_h:
                        p1, p2 = sorted([prev.start[0], prev.end[0]])
                        s1, s2 = sorted([s.start[0], s.end[0]])
                    else:
                        p1, p2 = sorted([prev.start[1], prev.end[1]])
                        s1, s2 = sorted([s.start[1], s.end[1]])
                    if s1 <= p2 + max(self.wall_thickness * 2.2, 0.45):
                        cur.append(s)
                    else:
                        chains.append(cur)
                        cur = [s]
                chains.append(cur)

                for chain in chains:
                    canonical_axis = sum((c.start[1] if is_h else c.start[0]) for c in chain) / len(chain)
                    for c in chain:
                        if is_h:
                            x1, x2 = c.start[0], c.end[0]
                            out.append(_Seg((x1, canonical_axis), (x2, canonical_axis), self.wall_thickness, True))
                        else:
                            y1, y2 = c.start[1], c.end[1]
                            out.append(_Seg((canonical_axis, y1), (canonical_axis, y2), self.wall_thickness, False))

        if len(out) == len(walls):
            print(f"[APT][DBG] normalize_collinear_groups kept={len(out)}")
        else:
            print(f"[APT][DBG] normalize_collinear_groups {len(walls)} -> {len(out)}")
        return out

    def _filter_suspicious_walls_near_openings(self, walls, openings):
        """
        Safety-фильтр перед построением мешей: убирает короткие сегменты-символы
        рядом с проёмами, чтобы они не превращались в отдельные Wall_* объекты.
        """
        if not walls or not openings:
            return walls

        print(f"[APT][DBG][WALL_FILTER] before={len(walls)}")
        removed = []
        kept = []

        def opening_bbox(op):
            ox, oy = op.position
            ow = max(op.width, 0.2)
            along = max(ow * 0.75, 0.22)
            across = max(self.wall_thickness * 2.0, 0.18)
            op_h = abs(math.cos(getattr(op, 'angle', 0.0))) >= abs(math.sin(getattr(op, 'angle', 0.0)))
            if op_h:
                return (ox - along, oy - across, ox + along, oy + across)
            return (ox - across, oy - along, ox + across, oy + along)

        for seg in walls:
            sx, sy = seg.start
            ex, ey = seg.end
            mx, my = (sx + ex) / 2.0, (sy + ey) / 2.0
            s_len = self._segment_length(seg)
            drop_reason = None

            for oi, op in enumerate(openings):
                b = opening_bbox(op)
                x1, y1, x2, y2 = b
                in_mid = (x1 <= mx <= x2) and (y1 <= my <= y2)
                in_endpoints = ((x1 <= sx <= x2) and (y1 <= sy <= y2)) and ((x1 <= ex <= x2) and (y1 <= ey <= y2))
                if not (in_mid or in_endpoints):
                    continue

                # Короткая линия внутри/рядом с символом проёма — почти наверняка не несущая стена.
                if s_len <= max(op.width * 1.35, 0.95):
                    drop_reason = f"short_in_opening_bbox[{oi}] bbox={tuple(round(v,3) for v in b)}"
                    break

                # Если сегмент рядом с осью стены, к которой привязан проём, и заметно короче обычных стен — тоже отбрасываем.
                if op.wall_segment is not None:
                    d = self._point_to_segment_dist(mx, my,
                                                    op.wall_segment.start[0], op.wall_segment.start[1],
                                                    op.wall_segment.end[0], op.wall_segment.end[1])
                    if d <= max(self.wall_thickness * 1.3, 0.24) and s_len <= max(op.width * 1.8, 1.2):
                        drop_reason = f"near_opening_axis[{oi}] d={d:.3f}"
                        break

            if drop_reason:
                removed.append((seg, drop_reason))
            else:
                kept.append(seg)

        print(f"[APT][DBG][WALL_FILTER] after={len(kept)} removed={len(removed)}")
        for seg, reason in removed[:40]:
            print(
                f"[APT][DBG][WALL_FILTER] remove seg=({seg.start[0]:.3f},{seg.start[1]:.3f})->"
                f"({seg.end[0]:.3f},{seg.end[1]:.3f}) len={self._segment_length(seg):.3f} reason={reason}"
            )
        return kept

    def clear_scene(self):
        for obj in bpy.data.objects: obj.select_set(True)
        bpy.ops.object.delete(use_global=False)
        for b in bpy.data.meshes:
            if b.users == 0: bpy.data.meshes.remove(b)
        for b in bpy.data.materials:
            if b.users == 0: bpy.data.materials.remove(b)
        self._wall_objects = []; self._wall_meta = []; self._floor_obj = None

    def _create_collections(self):
        sc = bpy.context.scene.collection
        self._apartment_collection = bpy.data.collections.new("Apartment")
        sc.children.link(self._apartment_collection)
        self._walls_collection = bpy.data.collections.new("Walls")
        self._apartment_collection.children.link(self._walls_collection)
        self._openings_collection = bpy.data.collections.new("Openings")
        self._apartment_collection.children.link(self._openings_collection)
        self._floors_collection = bpy.data.collections.new("Floors")
        self._apartment_collection.children.link(self._floors_collection)

    def _build_floor(self, pts, wall_segments=None, bbox=None):
        """
        Build a clean, flat floor mesh.
        Uses only the provided boundary polygon, cleaned/snapped to grid.
        Avoids using any opening/arc/symbol geometry. All vertices are z=0.
        """
        if len(pts) < 3:
            return

        cleaned = []
        if wall_segments:
            cleaned = self._derive_floor_outline_from_walls(wall_segments, bbox)
        if not cleaned:
            cleaned = self._clean_floor_polygon(list(pts))
        print(f"[APT][DBG] floor_pts_raw={len(pts)} floor_pts_clean={len(cleaned)}")
        if len(cleaned) < 3:
            return

        mesh = bpy.data.meshes.new("Floor")
        obj  = bpy.data.objects.new("Floor", mesh)
        bm   = bmesh.new()

        verts = [bm.verts.new((x, y, 0.0)) for x, y in cleaned]
        bm.verts.ensure_lookup_table()
        edges = []
        for i in range(len(verts)):
            e = bm.edges.new((verts[i], verts[(i + 1) % len(verts)]))
            edges.append(e)

        # Заполняем контур триангуляцией: так устойчивее для вогнутых ортогональных полигонов.
        try:
            bmesh.ops.triangle_fill(bm, edges=edges, use_beauty=True)
        except Exception as e:
            print(f"[APT] Floor triangle_fill failed: {e}")
            try:
                bm.faces.new(verts)
            except Exception as e2:
                print(f"[APT] Floor face creation failed: {e2}")

        # Пол должен быть строго в плоскости z=0.
        for v in bm.verts:
            v.co.z = 0.0

        bm.normal_update()
        bm.to_mesh(mesh)
        bm.free()
        self._floors_collection.objects.link(obj)
        self._floor_obj = obj

    def _derive_floor_outline_from_walls(self, wall_segments, bbox=None):
        if not wall_segments:
            return []

        xs = [p for w in wall_segments for p in (w.start[0], w.end[0])]
        ys = [p for w in wall_segments for p in (w.start[1], w.end[1])]
        if not xs or not ys:
            return []

        minx = min(xs)
        maxx = max(xs)
        miny = min(ys)
        maxy = max(ys)
        if bbox:
            minx = min(minx, bbox[0])
            miny = min(miny, bbox[1])
            maxx = max(maxx, bbox[2])
            maxy = max(maxy, bbox[3])

        pad = max(self.wall_thickness * 2.5, 0.4)
        minx -= pad
        miny -= pad
        maxx += pad
        maxy += pad

        res = max(min(self.wall_thickness * 0.22, 0.06), 0.02)
        wpx = max(64, int((maxx - minx) / res) + 4)
        hpx = max(64, int((maxy - miny) / res) + 4)
        mask = np.zeros((hpx, wpx), dtype=np.uint8)

        def to_px(x, y):
            px = int(round((x - minx) / res))
            py = int(round((y - miny) / res))
            return int(np.clip(px, 0, wpx - 1)), int(np.clip(py, 0, hpx - 1))

        tpx = max(3, int(round(self.wall_thickness / res)))
        for s in wall_segments:
            x1, y1 = to_px(s.start[0], s.start[1])
            x2, y2 = to_px(s.end[0], s.end[1])
            cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=tpx, lineType=cv2.LINE_8)

        k = max(3, int(tpx * 0.9))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))
        walls = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        flood = walls.copy()
        ff_mask = np.zeros((hpx + 2, wpx + 2), np.uint8)
        for pt in [(0, 0), (wpx - 1, 0), (0, hpx - 1), (wpx - 1, hpx - 1)]:
            if flood[pt[1], pt[0]] == 0:
                cv2.floodFill(flood, ff_mask, pt, 128)
        outside = (flood == 128)
        inside_or_walls = np.where(outside, 0, 255).astype(np.uint8)
        inside_or_walls = cv2.morphologyEx(
            inside_or_walls, cv2.MORPH_CLOSE,
            cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), iterations=1
        )

        cnts, _ = cv2.findContours(inside_or_walls, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return []
        cnt = max(cnts, key=cv2.contourArea)
        peri = cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, max(1.0, 0.006 * peri), True)

        pts = []
        for p in approx:
            px, py = p[0]
            x = minx + px * res
            y = miny + py * res
            pts.append((float(x), float(y)))

        return self._clean_floor_polygon(pts)

    def _clean_floor_polygon(self, pts):
        """
        Strict contour cleanup:
        - snap to grid
        - remove duplicate/near points
        - enforce orthogonal steps
        - remove very small edges
        """
        if not pts:
            return pts

        tol = max(self.CORNER_SNAP_TOL * 0.5, 0.03)

        def snap(v):
            return (round(v[0] / tol) * tol, round(v[1] / tol) * tol)

        snapped = [snap(p) for p in pts]

        # Убираем подряд идущие дубликаты/почти-дубликаты после снапа.
        out = []
        for p in snapped:
            if not out:
                out.append(p)
                continue
            if math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) > tol * 0.5:
                out.append(p)

        if len(out) > 2 and math.hypot(out[0][0] - out[-1][0], out[0][1] - out[-1][1]) <= tol * 0.5:
            out.pop()

        # Локальная ортогонализация: шаг оставляем по доминирующей оси (под прямые стены плана).
        ortho = [out[0]]
        for i in range(1, len(out)):
            px, py = ortho[-1]
            x, y = out[i]
            dx = x - px
            dy = y - py
            if abs(dx) >= abs(dy):
                ortho.append((x, py))
            else:
                ortho.append((px, y))

        # Отрезаем короткие рёбра, которые обычно приходят от шума/символов возле проёмов.
        cleaned = []
        # Порог завязан на толщину стены: так одинаково работает на разных масштабах плана.
        min_edge = max(self.wall_thickness * 0.45, 0.10)
        for p in ortho:
            if not cleaned:
                cleaned.append(p)
                continue
            if math.hypot(p[0] - cleaned[-1][0], p[1] - cleaned[-1][1]) >= min_edge:
                cleaned.append(p)

        # Финальная чистка замыкания контура.
        if len(cleaned) >= 3:
            if math.hypot(cleaned[0][0] - cleaned[-1][0], cleaned[0][1] - cleaned[-1][1]) < min_edge:
                cleaned.pop()

        # Убираем «шипы/ступеньки» на контуре (частая причина выступов пола у входа).
        def simplify_spikes(poly, max_iter=8):
            poly = list(poly)
            for _ in range(max_iter):
                if len(poly) < 4:
                    break
                changed = False
                keep = []
                n = len(poly)
                for i in range(n):
                    a = poly[(i - 1) % n]
                    b = poly[i]
                    c = poly[(i + 1) % n]

                    ab = math.hypot(b[0] - a[0], b[1] - a[1])
                    bc = math.hypot(c[0] - b[0], c[1] - b[1])

                    # Коллинеарные точки не несут информации и часто появляются после снапа.
                    if (a[0] == b[0] == c[0]) or (a[1] == b[1] == c[1]):
                        changed = True
                        continue

                    is_turn = (a[0] == b[0] and b[1] == c[1]) or (a[1] == b[1] and b[0] == c[0])
                    if is_turn and (ab < min_edge * 1.35 or bc < min_edge * 1.35):
                        changed = True
                        continue

                    keep.append(b)
                poly = keep
                if not changed:
                    break
            return poly

        simplified = simplify_spikes(cleaned)
        return simplified

    def _build_wall_segment(self, start, end, thickness=None, source_seg=None, source_type=None):
        if thickness is None: thickness = self.wall_thickness
        dx = end[0]-start[0]; dy = end[1]-start[1]
        length = math.sqrt(dx*dx+dy*dy)
        if length < 0.05: return
        angle = math.atan2(dy, dx)
        cx = (start[0]+end[0])/2; cy = (start[1]+end[1])/2
        # FIX for column-based joints: do not extend wall caps past segment endpoints.
        # Segments are already trimmed by wall_thickness/2 to meet column faces flush.
        extended_length = length

        idx  = len(self._wall_objects)
        mesh = bpy.data.meshes.new(f"Wall_{idx}")
        obj  = bpy.data.objects.new(f"Wall_{idx}", mesh)
        bm   = bmesh.new()
        hl = extended_length / 2; ht = thickness / 2; hh = self.wall_height / 2
        coords = [
            (-hl,-ht,-hh),(hl,-ht,-hh),(hl,ht,-hh),(-hl,ht,-hh),
            (-hl,-ht, hh),(hl,-ht, hh),(hl,ht, hh),(-hl,ht, hh),
        ]
        vs = [bm.verts.new(c) for c in coords]
        bm.verts.ensure_lookup_table()
        for f in [(0,1,2,3),(4,7,6,5),(0,4,5,1),
                  (2,6,7,3),(0,3,7,4),(1,5,6,2)]:
            bm.faces.new([vs[i] for i in f])
        bm.normal_update()
        bm.to_mesh(mesh); bm.free()

        obj.location       = (cx, cy, self.wall_height/2)
        obj.rotation_euler = (0, 0, angle)
        self._walls_collection.objects.link(obj)
        self._wall_objects.append(obj)

        is_h = abs(angle) < math.radians(10) or abs(angle) > math.radians(170)
        orientation = "horizontal" if is_h else "vertical"
        seg_ref = source_seg if source_seg is not None else type("Tmp", (), {"start": start, "end": end})()
        seg_len_dbg = self._segment_length(seg_ref)
        obj["wall_index"] = int(idx)
        obj["start"] = f"({getattr(seg_ref, 'start', start)[0]:.4f},{getattr(seg_ref, 'start', start)[1]:.4f})"
        obj["end"] = f"({getattr(seg_ref, 'end', end)[0]:.4f},{getattr(seg_ref, 'end', end)[1]:.4f})"
        obj["length"] = float(seg_len_dbg if seg_len_dbg > 0 else length)
        obj["orientation"] = orientation
        obj["source_type"] = str(source_type) if source_type is not None else "unknown"

        print(
            f"[APT][DBG][WALL_OBJ] {obj.name} idx={idx} "
            f"start={obj['start']} end={obj['end']} len={obj['length']:.3f} "
            f"orient={orientation} source_type={obj['source_type']} "
            f"loc=({obj.location.x:.3f},{obj.location.y:.3f},{obj.location.z:.3f}) "
            f"dims=({obj.dimensions.x:.3f},{obj.dimensions.y:.3f},{obj.dimensions.z:.3f})"
        )

        self._wall_meta.append({
            'obj':obj,'start':start,'end':end,'cx':cx,'cy':cy,
            'angle':angle,'length':length,'thickness':thickness,'is_horiz':is_h,
            'source_type': source_type if source_type is not None else 'unknown',
        })

    def _apply_default_materials(self):
        wm = self._make_mat("Wall_Material",
                             DEFAULT_MATERIALS["wall"]["color"],
                             DEFAULT_MATERIALS["wall"]["roughness"])
        for w in self._wall_objects:
            if not w.data.materials: w.data.materials.append(wm)
        fm = self._make_mat("Floor_Material",
                             DEFAULT_MATERIALS["floor"]["color"],
                             DEFAULT_MATERIALS["floor"]["roughness"])
        if self._floor_obj and not self._floor_obj.data.materials:
            self._floor_obj.data.materials.append(fm)

    def _make_mat(self, name, color, roughness):
        mat = bpy.data.materials.new(name=name)
        mat.use_nodes = True
        nodes = mat.node_tree.nodes; nodes.clear()
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = color
        bsdf.inputs['Roughness'].default_value  = roughness
        out  = nodes.new('ShaderNodeOutputMaterial'); out.location = (300, 0)
        mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
        mat.diffuse_color = color
        return mat

    @staticmethod
    def _refresh():
        for win in bpy.context.window_manager.windows:
            for area in win.screen.areas:
                if area.type == 'VIEW_3D': area.tag_redraw()

    def set_wall_color(self, color):
        mat = bpy.data.materials.get("Wall_Material")
        if not mat: return
        mat.diffuse_color = color
        if mat.use_nodes:
            for n in mat.node_tree.nodes:
                if n.type == 'BSDF_PRINCIPLED':
                    n.inputs['Base Color'].default_value = color; break
        # Материал обновляем на всех текущих объектах стен (после перестроения список объектов меняется).
        wall_objs = []
        apt = bpy.data.collections.get("Apartment")
        if apt:
            walls_col = bpy.data.collections.get("Walls")
            if walls_col and walls_col in apt.children_recursive:
                wall_objs.extend([o for o in walls_col.objects if o.type == 'MESH'])
        if not wall_objs:
            wall_objs = [o for o in bpy.data.objects if o.type == 'MESH' and o.name.startswith("Wall_")]

        for obj in wall_objs:
            if not obj.data:
                continue
            if not obj.data.materials:
                obj.data.materials.append(mat)
            else:
                obj.data.materials[0] = mat
        self._refresh()

    def set_floor_color(self, color):
        mat = bpy.data.materials.get("Floor_Material")
        if not mat: return
        mat.diffuse_color = color
        if mat.use_nodes:
            for n in mat.node_tree.nodes:
                if n.type == 'BSDF_PRINCIPLED':
                    n.inputs['Base Color'].default_value = color; break
        self._refresh()

    def apply_floor_preset(self, preset):
        maps = _find_tex_maps(preset)
        if maps:
            self._apply_image_floor(preset, maps)
        else:
            self._apply_procedural_floor(preset)
        self._refresh()

    def _apply_image_floor(self, preset, maps):
        mat = bpy.data.materials.get("Floor_Material")
        if not mat: return
        mat.use_nodes = True
        n = mat.node_tree.nodes; l = mat.node_tree.links; n.clear()
        out  = n.new('ShaderNodeOutputMaterial'); out.location  = (700,0)
        bsdf = n.new('ShaderNodeBsdfPrincipled'); bsdf.location = (400,0)
        l.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
        uv = n.new('ShaderNodeTexCoord'); uv.location = (-800,0)
        mp = n.new('ShaderNodeMapping');  mp.location = (-600,0)
        mp.inputs['Scale'].default_value = (4,4,4)
        l.new(uv.outputs['Generated'], mp.inputs['Vector'])
        y = 0
        if 'diff' in maps:
            tx = self._img_node(n, maps['diff'], 'sRGB', (-300,y)); y-=280
            l.new(mp.outputs['Vector'], tx.inputs['Vector'])
            l.new(tx.outputs['Color'],  bsdf.inputs['Base Color'])
        if 'rough' in maps:
            tx = self._img_node(n, maps['rough'], 'Non-Color', (-300,y)); y-=280
            l.new(mp.outputs['Vector'], tx.inputs['Vector'])
            l.new(tx.outputs['Color'],  bsdf.inputs['Roughness'])
        if 'nor' in maps:
            tx = self._img_node(n, maps['nor'], 'Non-Color', (-300,y))
            nm = n.new('ShaderNodeNormalMap'); nm.location = (0,y)
            l.new(mp.outputs['Vector'], tx.inputs['Vector'])
            l.new(tx.outputs['Color'],  nm.inputs['Color'])
            l.new(nm.outputs['Normal'], bsdf.inputs['Normal'])
        sc = {'PARQUET':(0.30,0.17,0.07,1),'TILE':(0.85,0.84,0.81,1),
              'CONCRETE':(0.49,0.49,0.47,1),'CARPET':(0.40,0.30,0.52,1),
              'MARBLE':(0.91,0.89,0.87,1)}
        mat.diffuse_color = sc.get(preset,(0.6,0.5,0.4,1))

    @staticmethod
    def _img_node(nodes, path, cs, loc):
        tx  = nodes.new('ShaderNodeTexImage'); tx.location = loc
        img = bpy.data.images.load(path, check_existing=True)
        img.colorspace_settings.name = cs; tx.image = img
        return tx

    def _apply_procedural_floor(self, preset):
        mat = bpy.data.materials.get("Floor_Material")
        if not mat: return
        mat.use_nodes = True
        n = mat.node_tree.nodes; l = mat.node_tree.links; n.clear()
        bsdf = n.new('ShaderNodeBsdfPrincipled'); bsdf.location = (300,0)
        out  = n.new('ShaderNodeOutputMaterial'); out.location  = (600,0)
        l.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
        uv = n.new('ShaderNodeTexCoord'); uv.location = (-800,0)
        mp = n.new('ShaderNodeMapping');  mp.location = (-600,0)
        l.new(uv.outputs['Generated'], mp.inputs['Vector'])
        d = {'PARQUET':self._p_parquet,'TILE':self._p_tile,'CONCRETE':self._p_concrete,
             'CARPET':self._p_carpet,  'MARBLE':self._p_marble}
        d.get(preset, self._p_parquet)(n, l, mp, bsdf)
        sc = {'PARQUET':(0.30,0.17,0.07,1),'TILE':(0.85,0.84,0.81,1),
              'CONCRETE':(0.49,0.49,0.47,1),'CARPET':(0.40,0.30,0.52,1),
              'MARBLE':(0.91,0.89,0.87,1)}
        mat.diffuse_color = sc.get(preset,(0.6,0.5,0.4,1))

    def _p_parquet(self,n,l,mp,b):
        mp.inputs['Scale'].default_value=(8,8,8)
        w=n.new('ShaderNodeTexWave'); w.location=(-300,100); w.wave_type='BANDS'; w.bands_direction='X'
        w.inputs['Scale'].default_value=4.; w.inputs['Distortion'].default_value=2.; w.inputs['Detail'].default_value=8.
        ns=n.new('ShaderNodeTexNoise'); ns.location=(-300,-100)
        ns.inputs['Scale'].default_value=20.; ns.inputs['Detail'].default_value=6.
        mx=n.new('ShaderNodeMixRGB'); mx.location=(-100,0); mx.blend_type='MULTIPLY'; mx.inputs['Fac'].default_value=0.4
        cr=n.new('ShaderNodeValToRGB'); cr.location=(100,0)
        cr.color_ramp.elements[0].position=0.3; cr.color_ramp.elements[0].color=(0.20,0.09,0.03,1)
        cr.color_ramp.elements[1].position=0.8; cr.color_ramp.elements[1].color=(0.52,0.28,0.12,1)
        l.new(mp.outputs['Vector'],w.inputs['Vector']); l.new(mp.outputs['Vector'],ns.inputs['Vector'])
        l.new(w.outputs['Color'],mx.inputs['Color1']); l.new(ns.outputs['Color'],mx.inputs['Color2'])
        l.new(mx.outputs['Color'],cr.inputs['Fac']); l.new(cr.outputs['Color'],b.inputs['Base Color'])
        b.inputs['Roughness'].default_value=0.22

    def _p_tile(self,n,l,mp,b):
        mp.inputs['Scale'].default_value=(5,5,5)
        br=n.new('ShaderNodeTexBrick'); br.location=(-200,0)
        br.inputs['Color1'].default_value=(0.91,0.89,0.85,1); br.inputs['Color2'].default_value=(0.83,0.81,0.77,1)
        br.inputs['Mortar'].default_value=(0.68,0.68,0.68,1); br.inputs['Scale'].default_value=5.
        br.inputs['Mortar Size'].default_value=0.025
        l.new(mp.outputs['Vector'],br.inputs['Vector']); l.new(br.outputs['Color'],b.inputs['Base Color'])
        b.inputs['Roughness'].default_value=0.12

    def _p_concrete(self,n,l,mp,b):
        mp.inputs['Scale'].default_value=(3,3,3)
        ns=n.new('ShaderNodeTexNoise'); ns.location=(-300,0)
        ns.inputs['Scale'].default_value=5.; ns.inputs['Detail'].default_value=8.; ns.inputs['Roughness'].default_value=0.7
        cr=n.new('ShaderNodeValToRGB'); cr.location=(-100,0)
        cr.color_ramp.elements[0].position=0.3; cr.color_ramp.elements[0].color=(0.36,0.35,0.33,1)
        cr.color_ramp.elements[1].position=0.7; cr.color_ramp.elements[1].color=(0.61,0.60,0.57,1)
        l.new(mp.outputs['Vector'],ns.inputs['Vector']); l.new(ns.outputs['Fac'],cr.inputs['Fac'])
        l.new(cr.outputs['Color'],b.inputs['Base Color']); b.inputs['Roughness'].default_value=0.85

    def _p_carpet(self,n,l,mp,b):
        mp.inputs['Scale'].default_value=(12,12,12)
        ns=n.new('ShaderNodeTexNoise'); ns.location=(-300,0)
        ns.inputs['Scale'].default_value=30.; ns.inputs['Detail'].default_value=14.; ns.inputs['Roughness'].default_value=0.8
        cr=n.new('ShaderNodeValToRGB'); cr.location=(-100,0)
        cr.color_ramp.elements[0].position=0.2; cr.color_ramp.elements[0].color=(0.22,0.15,0.35,1)
        cr.color_ramp.elements[1].position=0.8; cr.color_ramp.elements[1].color=(0.40,0.30,0.56,1)
        l.new(mp.outputs['Vector'],ns.inputs['Vector']); l.new(ns.outputs['Fac'],cr.inputs['Fac'])
        l.new(cr.outputs['Color'],b.inputs['Base Color']); b.inputs['Roughness'].default_value=0.95

    def _p_marble(self,n,l,mp,b):
        mp.inputs['Scale'].default_value=(2,2,2)
        ns=n.new('ShaderNodeTexNoise'); ns.location=(-400,-100)
        ns.inputs['Scale'].default_value=3.; ns.inputs['Detail'].default_value=10.; ns.inputs['Distortion'].default_value=0.8
        wv=n.new('ShaderNodeTexWave'); wv.location=(-200,0); wv.wave_type='BANDS'
        wv.inputs['Scale'].default_value=2.; wv.inputs['Distortion'].default_value=8.
        cr=n.new('ShaderNodeValToRGB'); cr.location=(0,0)
        cr.color_ramp.elements[0].position=0.0; cr.color_ramp.elements[0].color=(0.93,0.91,0.89,1)
        e1=cr.color_ramp.elements.new(0.4); e1.color=(0.68,0.65,0.62,1)
        cr.color_ramp.elements[1].position=1.0; cr.color_ramp.elements[1].color=(0.96,0.94,0.92,1)
        l.new(mp.outputs['Vector'],ns.inputs['Vector']); l.new(mp.outputs['Vector'],wv.inputs['Vector'])
        l.new(ns.outputs['Fac'],wv.inputs['Detail Roughness']); l.new(wv.outputs['Fac'],cr.inputs['Fac'])
        l.new(cr.outputs['Color'],b.inputs['Base Color'])
        b.inputs['Roughness'].default_value=0.04
        if 'IOR' in b.inputs: b.inputs['IOR'].default_value=1.55

    def set_floor_texture(self, path):
        mat = bpy.data.materials.get("Floor_Material")
        if not mat: return
        mat.use_nodes = True
        n = mat.node_tree.nodes; l = mat.node_tree.links
        bsdf = next((x for x in n if x.type=='BSDF_PRINCIPLED'), None)
        out  = next((x for x in n if x.type=='OUTPUT_MATERIAL'), None)
        if not bsdf or not out:
            n.clear()
            bsdf = n.new('ShaderNodeBsdfPrincipled'); bsdf.location = (300,0)
            out  = n.new('ShaderNodeOutputMaterial'); out.location  = (600,0)
            l.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
        tx = n.new('ShaderNodeTexImage'); tx.location = (-300,0)
        try: tx.image = bpy.data.images.load(path, check_existing=True)
        except Exception as e:
            print(f"[APT] Ошибка текстуры: {e}"); n.remove(tx); return
        uv = n.new('ShaderNodeTexCoord'); uv.location = (-600,0)
        mp = n.new('ShaderNodeMapping');  mp.location = (-450,0)
        mp.inputs['Scale'].default_value = (4,4,4)
        l.new(uv.outputs['Generated'], mp.inputs['Vector'])
        l.new(mp.outputs['Vector'],    tx.inputs['Vector'])
        l.new(tx.outputs['Color'],     bsdf.inputs['Base Color'])
        bsdf.inputs['Roughness'].default_value = 0.3
        self._refresh()

    def _setup_lighting(self, bbox):
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2
        width = max(bbox[2] - bbox[0], 0.01)
        height = max(bbox[3] - bbox[1], 0.01)
        diagonal = max(math.hypot(width, height), 0.5)

        sun = bpy.data.lights.new("Sun",'SUN'); sun.energy = 3.
        s   = bpy.data.objects.new("Sun", sun)
        sun_z = max(diagonal * 1.5, self.wall_height * 3.0)
        s.location       = (cx, cy, sun_z)
        s.rotation_euler = (math.radians(45), 0, math.radians(30))
        self._apartment_collection.objects.link(s)
        area = bpy.data.lights.new("Interior",'AREA')
        area.energy = 200; area.size = 2
        a = bpy.data.objects.new("Interior", area)
        a.location = (cx, cy, self.wall_height * 0.85)
        self._apartment_collection.objects.link(a)

    def _is_column_object(self, obj):
        if obj is None or obj.type != 'MESH':
            return False
        try:
            if obj.get(self.COLUMN_OBJ_PROP) == self.COLUMN_OBJ_KIND:
                return True
        except Exception:
            pass
        return obj.name.startswith("Column_")

    def update_wall_height(self, new_h):
        if self.wall_height == 0:
            return
        factor = new_h / self.wall_height
        for w in self._wall_objects:
            w.scale.z *= factor
            w.location.z *= factor
        self.wall_height = new_h

    def update_wall_thickness(self, new_t):
        if self.wall_thickness == 0:
            return
        factor = new_t / self.wall_thickness
        for w in self._wall_objects:
            # Колонки без поворота: толщина в плане — по X и Y; стены — локальная Y после поворота.
            if self._is_column_object(w):
                w.scale.x *= factor
                w.scale.y *= factor
            else:
                w.scale.y *= factor
        self.wall_thickness = new_t


class _Seg:
    __slots__ = ('start','end','thickness','is_horiz')
    def __init__(self, start, end, thickness, is_horiz):
        self.start = start; self.end = end
        self.thickness = thickness; self.is_horiz = is_horiz