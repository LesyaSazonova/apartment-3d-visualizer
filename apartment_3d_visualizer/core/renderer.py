import bpy
import os
import datetime
import math
from mathutils import Vector, Euler

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RENDER_SETTINGS, CAMERA_PRESETS, RENDERS_DIR


class SceneRenderer:
    def __init__(self):
        self._camera = None
        scene = bpy.context.scene
        scene.render.engine = RENDER_SETTINGS.get("engine", "CYCLES")
        scene.render.resolution_x = RENDER_SETTINGS.get("resolution_x", 1920)
        scene.render.resolution_y = RENDER_SETTINGS.get("resolution_y", 1080)
        if scene.render.engine == "CYCLES":
            scene.cycles.samples = RENDER_SETTINGS.get("samples", 128)
            scene.cycles.use_denoising = True
        scene.render.image_settings.file_format = 'PNG'

    def _ensure_camera(self):
        if self._camera and self._camera.name in bpy.data.objects:
            return self._camera
        cam = bpy.data.cameras.new("RenderCamera")
        obj = bpy.data.objects.new("RenderCamera", cam)
        apt = bpy.data.collections.get("Apartment")
        if apt:
            apt.objects.link(obj)
        else:
            bpy.context.scene.collection.objects.link(obj)
        bpy.context.scene.camera = obj
        self._camera = obj
        return obj

    def _calculate_scene_bounds(self):
        """
        Calculate true bounds from all relevant apartment geometry.
        """
        walls = bpy.data.collections.get("Walls")
        floors = bpy.data.collections.get("Floors")
        furniture = bpy.data.collections.get("Furniture")
        apt = bpy.data.collections.get("Apartment")

        candidates = []
        for col in (walls, floors, furniture):
            if not col:
                continue
            candidates.extend([o for o in col.objects if o.type == 'MESH'])

        # Fallback: generic wall objects may exist outside named collections.
        if not candidates:
            candidates.extend([
                o for o in bpy.data.objects
                if o.type == 'MESH' and "Wall_" in o.name
            ])

        if not candidates and apt:
            candidates.extend([o for o in apt.all_objects if o.type == 'MESH'])
        if not candidates:
            candidates.extend([o for o in bpy.context.scene.objects if o.type == 'MESH'])

        min_x = min_y = float("inf")
        max_x = max_y = float("-inf")
        points_count = 0

        for obj in candidates:
            try:
                for corner in obj.bound_box:
                    world = obj.matrix_world @ Vector(corner)
                    min_x = min(min_x, world.x)
                    min_y = min(min_y, world.y)
                    max_x = max(max_x, world.x)
                    max_y = max(max_y, world.y)
                    points_count += 1
            except Exception:
                continue

        if points_count == 0:
            # Safe fallback for empty scene.
            min_x, min_y, max_x, max_y = -5.0, -5.0, 5.0, 5.0

        width = max(max_x - min_x, 0.01)
        height = max(max_y - min_y, 0.01)
        center_x = (min_x + max_x) * 0.5
        center_y = (min_y + max_y) * 0.5
        diagonal = max(math.hypot(width, height), 0.5)

        out = {
            "bbox": (min_x, min_y, max_x, max_y),
            "center_x": center_x,
            "center_y": center_y,
            "width": width,
            "height": height,
            "diagonal": diagonal,
        }
        print(
            "[APT][CAM] Bounds: X[{:.2f}, {:.2f}] Y[{:.2f}, {:.2f}] W={:.2f} H={:.2f} Diag={:.2f}".format(
                min_x, max_x, min_y, max_y, width, height, diagonal
            )
        )
        return out

    def set_camera_preset(self, name):
        if name not in CAMERA_PRESETS:
            return
        p = CAMERA_PRESETS[name]
        cam = self._ensure_camera()
        bounds = self._calculate_scene_bounds()
        cx = bounds["center_x"]
        cy = bounds["center_y"]
        width = bounds["width"]
        height = bounds["height"]
        diag = bounds["diagonal"]
        center = Vector((cx, cy, 0.0))

        cam.data.type = 'ORTHO' if p.get("type") == "ORTHO" else 'PERSP'
        if cam.data.type == 'ORTHO':
            cam.data.ortho_scale = max(width, height) * 1.4
        else:
            cam.data.lens = p.get("focal_length", 35)

        if name == "top":
            cam.location = (cx, cy, diag * 1.5)
            cam.rotation_euler = Euler((0.0, 0.0, 0.0), 'XYZ')
            dist = (Vector(cam.location) - center).length
            print(
                "[APT][CAM] Preset: {}, Camera pos: ({:.2f}, {:.2f}, {:.2f}), Distance to center: {:.2f}m".format(
                    name, cam.location.x, cam.location.y, cam.location.z, dist
                )
            )
            print(f"[APT][CAM] ortho_scale={cam.data.ortho_scale:.2f} (for top view)")
            return

        wall_h = 2.7
        walls_col = bpy.data.collections.get("Walls")
        if walls_col:
            zmax = []
            for obj in walls_col.objects:
                if obj.type != 'MESH':
                    continue
                try:
                    zmax.extend([(obj.matrix_world @ Vector(c)).z for c in obj.bound_box])
                except Exception:
                    continue
            if zmax:
                wall_h = max(max(zmax), 2.2)

        if name == "perspective":
            cam.location = (cx + diag * 1.2, cy - diag * 1.2, diag * 0.9)
            target = Vector((cx, cy, wall_h * 0.5))
            look = target - Vector(cam.location)
            if look.length > 1e-6:
                cam.rotation_euler = look.to_track_quat('-Z', 'Y').to_euler()
            else:
                cam.rotation_euler = Euler((math.radians(60), 0.0, math.radians(45)), 'XYZ')
            dist = (Vector(cam.location) - center).length
            print(
                "[APT][CAM] Preset: {}, Camera pos: ({:.2f}, {:.2f}, {:.2f}), Distance to center: {:.2f}m".format(
                    name, cam.location.x, cam.location.y, cam.location.z, dist
                )
            )
            return

        if name == "first_person":
            px = cx + 1.5
            py = cy - 1.5
            pad = 0.5
            minx, miny, maxx, maxy = bounds["bbox"]
            # Keep first-person point inside apartment XY bounds with safe padding.
            px = min(max(px, minx + pad), maxx - pad)
            py = min(max(py, miny + pad), maxy - pad)

            # Keep away from walls by iteratively moving to center if too close.
            wall_objs = []
            if walls_col:
                wall_objs = [o for o in walls_col.objects if o.type == 'MESH']
            for _ in range(8):
                too_close = False
                probe = Vector((px, py, 1.7))
                for w in wall_objs:
                    try:
                        local = w.matrix_world.inverted() @ probe
                        hx = max(abs(c[0]) for c in w.bound_box)
                        hy = max(abs(c[1]) for c in w.bound_box)
                        if abs(local.x) <= hx + 0.10 and abs(local.y) <= hy + 0.10:
                            too_close = True
                            break
                    except Exception:
                        continue
                if not too_close:
                    break
                px = (px + cx) * 0.5
                py = (py + cy) * 0.5

            cam.location = (px, py, 1.7)
            cam.rotation_euler = Euler((math.radians(90), 0.0, 0.0), 'XYZ')
            dist = (Vector(cam.location) - center).length
            print(
                "[APT][CAM] Preset: {}, Camera pos: ({:.2f}, {:.2f}, {:.2f}), Distance to center: {:.2f}m".format(
                    name, cam.location.x, cam.location.y, cam.location.z, dist
                )
            )
            return

        # Fallback for custom presets: keep preset rotation, compute sane distance from scene size.
        cam.location = (cx + diag * 1.0, cy - diag * 1.0, diag * 0.8)
        cam.rotation_euler = Euler(p.get("rotation", (1.0, 0.0, 0.8)), 'XYZ')
        print(
            "[APT][CAM] Preset: {}, Camera pos: ({:.2f}, {:.2f}, {:.2f}), Distance to center: {:.2f}m".format(
                name, cam.location.x, cam.location.y, cam.location.z, (Vector(cam.location) - center).length
            )
        )

    def render(self, output_path=None, preset=None):
        # Force camera placement refresh before every render.
        self._calculate_scene_bounds()
        if preset:
            self.set_camera_preset(preset)
        self._ensure_camera()
        if not output_path:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(RENDERS_DIR, f"render_{preset or 'custom'}_{ts}.png")
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        bpy.context.scene.render.filepath = output_path
        bpy.ops.render.render(write_still=True)
        return output_path

    def render_all_presets(self, output_dir=None):
        if not output_dir:
            ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = os.path.join(RENDERS_DIR, f"batch_{ts}")
        os.makedirs(output_dir, exist_ok=True)
        results = {}
        for name in CAMERA_PRESETS:
            self._calculate_scene_bounds()
            path = os.path.join(output_dir, f"{name}.png")
            self.render(output_path=path, preset=name)
            results[name] = path
        return results