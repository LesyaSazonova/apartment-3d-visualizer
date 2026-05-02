import bpy
import bmesh
import json
import os
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field, asdict
from enum import Enum

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import FURNITURE_DIR, CATALOG_FILE


@dataclass
class AssetInfo:
    id: str
    name: str
    category: str
    filename: str
    dimensions: Tuple[float, float, float]
    description: str = ""
    thumbnail: str = ""
    tags: List[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, data):
        data['dimensions'] = tuple(data['dimensions'])
        return cls(**data)


@dataclass
class PlacedAsset:
    asset_info: AssetInfo
    blender_object: Optional[bpy.types.Object] = None
    position: Tuple[float, float, float] = (0, 0, 0)
    rotation: Tuple[float, float, float] = (0, 0, 0)
    scale: Tuple[float, float, float] = (1, 1, 1)


class AssetManager:
    """
    Управление библиотекой мебели и размещением ассетов в сцене.
    
    Возможности:
      • Каталог из 30+ предметов мебели по 7 категориям
      • Процедурная генерация геометрии (без внешних файлов)
      • Интеграция с Blender Asset Browser (drag-and-drop)
      • Проверка коллизий при размещении
      • Импорт пользовательских моделей (.blend, .obj, .fbx)
    """
    
    def __init__(self):
        self.catalog = {}
        self.placed_assets = []
        self._assets_collection = None
        self._load_catalog()
        self._ensure_collection()

    def _ensure_collection(self):
        """Создаёт коллекцию для мебели, если её нет."""
        if "Furniture" not in bpy.data.collections:
            self._assets_collection = bpy.data.collections.new("Furniture")
            apt = bpy.data.collections.get("Apartment")
            if apt:
                apt.children.link(self._assets_collection)
            else:
                bpy.context.scene.collection.children.link(self._assets_collection)
        else:
            self._assets_collection = bpy.data.collections["Furniture"]

    def _load_catalog(self):
        """Загружает каталог из JSON или создаёт стандартный."""
        if os.path.exists(CATALOG_FILE):
            with open(CATALOG_FILE, 'r', encoding='utf-8') as f:
                data = json.load(f)
                for item in data.get("assets", []):
                    info = AssetInfo.from_dict(item)
                    self.catalog[info.id] = info
        else:
            self._create_defaults()
        
        # Создаём каталог для Asset Browser
        self.create_asset_catalog()

    def _save_catalog(self):
        """Сохраняет каталог в JSON."""
        data = {"assets": [i.to_dict() for i in self.catalog.values()]}
        with open(CATALOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)

    # ═══════════════════════════════════════════════════════════════
    # СТАНДАРТНЫЙ КАТАЛОГ (30+ предметов)
    # ═══════════════════════════════════════════════════════════════
    def _create_defaults(self):
        """Создаёт стандартный каталог мебели."""
        items = [
            # Гостиная
            AssetInfo("sofa", "Диван", "Гостиная", "__primitive__", (2.0, 0.9, 0.85),
                      description="Двухместный диван", tags=["диван", "мягкая"]),
            AssetInfo("sofa_corner", "Угловой диван", "Гостиная", "__primitive__", (2.5, 2.0, 0.85),
                      description="Угловой диван", tags=["диван", "угловой"]),
            AssetInfo("armchair", "Кресло", "Гостиная", "__primitive__", (0.9, 0.85, 0.85),
                      description="Мягкое кресло", tags=["кресло", "мягкая"]),
            AssetInfo("tv_stand", "Тумба ТВ", "Гостиная", "__primitive__", (1.5, 0.4, 0.5),
                      description="Тумба под телевизор", tags=["тумба", "тв"]),
            AssetInfo("table_coffee", "Журнальный столик", "Гостиная", "__primitive__", (1.0, 0.6, 0.45),
                      description="Низкий журнальный столик", tags=["стол", "журнальный"]),
            AssetInfo("bookshelf", "Стеллаж", "Гостиная", "__primitive__", (0.8, 0.3, 1.8),
                      description="Открытый стеллаж", tags=["стеллаж", "полка"]),

            # Спальня
            AssetInfo("bed_double", "Кровать двуспальная", "Спальня", "__primitive__", (1.8, 2.1, 0.5),
                      description="Двуспальная кровать 180×200", tags=["кровать", "двуспальная"]),
            AssetInfo("bed_single", "Кровать односпальная", "Спальня", "__primitive__", (0.9, 2.0, 0.5),
                      description="Односпальная кровать 90×200", tags=["кровать", "односпальная"]),
            AssetInfo("wardrobe", "Шкаф-купе", "Спальня", "__primitive__", (2.0, 0.6, 2.2),
                      description="Двухдверный шкаф-купе", tags=["шкаф", "купе"]),
            AssetInfo("wardrobe_small", "Шкаф", "Спальня", "__primitive__", (1.0, 0.6, 2.2),
                      description="Однодверный шкаф", tags=["шкаф"]),
            AssetInfo("commode", "Комод", "Спальня", "__primitive__", (1.0, 0.5, 0.8),
                      description="Комод с ящиками", tags=["комод", "ящики"]),
            AssetInfo("nightstand", "Тумба прикроватная", "Спальня", "__primitive__", (0.45, 0.4, 0.55),
                      description="Прикроватная тумба", tags=["тумба", "прикроватная"]),

            # Кухня
            AssetInfo("table_dining", "Обеденный стол", "Кухня", "__primitive__", (1.4, 0.8, 0.75),
                      description="Прямоугольный обеденный стол", tags=["стол", "обеденный"]),
            AssetInfo("table_round", "Круглый стол", "Кухня", "__primitive__", (1.0, 1.0, 0.75),
                      description="Круглый обеденный стол", tags=["стол", "круглый"]),
            AssetInfo("chair", "Стул", "Кухня", "__primitive__", (0.45, 0.45, 0.9),
                      description="Обеденный стул", tags=["стул"]),
            AssetInfo("kitchen_counter", "Кухонный гарнитур", "Кухня", "__primitive__", (2.4, 0.6, 0.85),
                      description="Нижние шкафы со столешницей", tags=["кухня", "столешница"]),
            AssetInfo("kitchen_upper", "Навесные шкафы", "Кухня", "__primitive__", (2.4, 0.35, 0.7),
                      description="Верхние навесные шкафы", tags=["кухня", "навесные"]),
            AssetInfo("fridge", "Холодильник", "Кухня", "__primitive__", (0.6, 0.65, 1.8),
                      description="Двухкамерный холодильник", tags=["холодильник", "техника"]),
            AssetInfo("stove", "Плита", "Кухня", "__primitive__", (0.6, 0.6, 0.85),
                      description="Кухонная плита", tags=["плита", "техника"]),

            # Ванная
            AssetInfo("bath", "Ванна", "Ванная", "__primitive__", (1.7, 0.7, 0.6),
                      description="Стандартная ванна 170см", tags=["ванна"]),
            AssetInfo("shower", "Душевая кабина", "Ванная", "__primitive__", (0.9, 0.9, 2.0),
                      description="Душевая кабина 90×90", tags=["душ", "кабина"]),
            AssetInfo("toilet", "Унитаз", "Ванная", "__primitive__", (0.4, 0.65, 0.4),
                      description="Напольный унитаз", tags=["унитаз"]),
            AssetInfo("sink_bath", "Раковина", "Ванная", "__primitive__", (0.6, 0.5, 0.85),
                      description="Раковина с тумбой", tags=["раковина", "тумба"]),
            AssetInfo("washing", "Стиральная машина", "Ванная", "__primitive__", (0.6, 0.6, 0.85),
                      description="Стиральная машина", tags=["стиральная", "техника"]),

            # Кабинет
            AssetInfo("desk", "Письменный стол", "Кабинет", "__primitive__", (1.2, 0.6, 0.75),
                      description="Рабочий стол", tags=["стол", "рабочий", "письменный"]),
            AssetInfo("desk_large", "Рабочий стол большой", "Кабинет", "__primitive__", (1.6, 0.8, 0.75),
                      description="Большой рабочий стол", tags=["стол", "рабочий"]),
            AssetInfo("office_chair", "Офисное кресло", "Кабинет", "__primitive__", (0.6, 0.6, 1.1),
                      description="Кресло на колёсиках", tags=["кресло", "офисное"]),
            AssetInfo("bookcase", "Книжный шкаф", "Кабинет", "__primitive__", (0.8, 0.35, 2.0),
                      description="Закрытый книжный шкаф", tags=["шкаф", "книжный"]),

            # Прихожая
            AssetInfo("hallway_cabinet", "Шкаф прихожая", "Прихожая", "__primitive__", (1.2, 0.4, 2.0),
                      description="Шкаф для прихожей", tags=["шкаф", "прихожая"]),
            AssetInfo("shoe_rack", "Обувница", "Прихожая", "__primitive__", (0.8, 0.3, 0.5),
                      description="Полка для обуви", tags=["обувница", "полка"]),
            AssetInfo("mirror_stand", "Зеркало напольное", "Прихожая", "__primitive__", (0.5, 0.05, 1.7),
                      description="Напольное зеркало", tags=["зеркало"]),

            # Детская
            AssetInfo("bed_child", "Детская кровать", "Детская", "__primitive__", (0.8, 1.6, 0.5),
                      description="Кровать для ребёнка", tags=["кровать", "детская"]),
            AssetInfo("desk_child", "Детский стол", "Детская", "__primitive__", (1.0, 0.5, 0.6),
                      description="Стол для ребёнка", tags=["стол", "детский"]),
            AssetInfo("toy_cabinet", "Шкаф для игрушек", "Детская", "__primitive__", (0.8, 0.4, 1.2),
                      description="Низкий шкаф для игрушек", tags=["шкаф", "игрушки"]),
        ]
        for item in items:
            self.catalog[item.id] = item
        self._save_catalog()

    # ═══════════════════════════════════════════════════════════════
    # КАТЕГОРИИ И ПОИСК
    # ═══════════════════════════════════════════════════════════════
    def get_categories(self):
        """Возвращает список всех категорий."""
        cats = set()
        for info in self.catalog.values():
            cats.add(info.category)
        return sorted(list(cats))

    def get_assets_by_category(self, category):
        """Возвращает ассеты указанной категории."""
        return [i for i in self.catalog.values() if i.category == category]

    def search_assets(self, query):
        """Поиск ассетов по названию, описанию и тегам."""
        q = query.lower()
        return [
            i for i in self.catalog.values()
            if q in i.name.lower() or q in i.description.lower()
            or any(q in t for t in i.tags)
        ]

    # ═══════════════════════════════════════════════════════════════
    # ИНТЕГРАЦИЯ С ASSET BROWSER
    # ═══════════════════════════════════════════════════════════════
    def register_asset_library(self):
        """
        Регистрирует библиотеку мебели в Blender Asset Browser.
        Вызывается при активации аддона.
        """
        lib_path = FURNITURE_DIR
        os.makedirs(lib_path, exist_ok=True)
        
        lib_name = "Apartment Furniture"
        prefs = bpy.context.preferences
        found = False
        
        for lib in prefs.filepaths.asset_libraries:
            if lib.name == lib_name:
                found = True
                break
        
        if not found:
            lib = prefs.filepaths.asset_libraries.new(name=lib_name)
            lib.path = lib_path
            print(f"[APT] Библиотека '{lib_name}' добавлена: {lib_path}")
        
        # Создаём .blend файлы с ассетами
        self._generate_asset_blend_files()

    def _generate_asset_blend_files(self):
        """
        Создаёт .blend файлы с ассетами для Asset Browser.
        По одному файлу на каждую категорию.
        """
        categories = self.get_categories()
        
        # Сохраняем текущий файл
        current_file = bpy.data.filepath
        if current_file:
            try:
                bpy.ops.wm.save_mainfile()
            except:
                pass
        
        for category in categories:
            assets = self.get_assets_by_category(category)
            if not assets:
                continue
            
            blend_path = os.path.join(FURNITURE_DIR, f"{category}.blend")
            
            # Не пересоздаём, если уже есть
            if os.path.exists(blend_path):
                continue
            
            print(f"[APT] Создаю библиотеку: {category} ({len(assets)} объектов)")
            
            # Новая сцена
            bpy.ops.wm.read_factory_settings(use_empty=True)
            
            # Коллекция для категории
            coll = bpy.data.collections.new(category)
            bpy.context.scene.collection.children.link(coll)
            
            # Создаём все ассеты
            for asset_info in assets:
                obj = self._create_furniture(asset_info)
                if obj:
                    for col in obj.users_collection:
                        col.objects.unlink(obj)
                    coll.objects.link(obj)
                    
                    obj.asset_mark()
                    obj.asset_data.description = asset_info.description
                    obj.asset_data.tags.new(name=category)
                    for tag in asset_info.tags:
                        obj.asset_data.tags.new(name=tag)
                    
                    try:
                        bpy.ops.ed.lib_id_generate_preview({'id': obj})
                    except:
                        pass
            
            # Сохраняем
            bpy.ops.wm.save_as_mainfile(filepath=blend_path)
            print(f"[APT] ✓ Сохранено: {blend_path}")
        
        # Возвращаемся к исходному файлу
        bpy.ops.wm.read_factory_settings(use_empty=True)
        if current_file and os.path.exists(current_file):
            bpy.ops.wm.open_mainfile(filepath=current_file)

    def create_asset_catalog(self):
        """Создаёт `blender_assets.cats.txt` для Asset Browser."""
        catalog_path = os.path.join(FURNITURE_DIR, "blender_assets.cats.txt")
        
        lines = [
            "# Apartment 3D Furniture Catalog",
            "VERSION 1",
            "",
        ]
        
        import uuid
        for cat in self.get_categories():
            uid = str(uuid.uuid4())
            safe_name = cat.replace(" ", "_")
            lines.append(f"{uid}:Furniture/{safe_name}:{cat}")
        
        with open(catalog_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(lines))
        
        print(f"[APT] Каталог создан: {catalog_path}")

    def place_asset(self, asset_id, position=(0, 0, 0), rotation=0.0):
        if asset_id not in self.catalog:
            return None
        
        info = self.catalog[asset_id]
        self._ensure_collection()
        
        obj = self._create_furniture(info)
        if not obj:
            return None
        
        obj.location = (position[0], position[1], info.dimensions[2] / 2)
        obj.rotation_euler = (0, 0, rotation)
        
        for col in obj.users_collection:
            col.objects.unlink(obj)
        self._assets_collection.objects.link(obj)
        
        placed = PlacedAsset(
            asset_info=info,
            blender_object=obj,
            position=tuple(obj.location),
            rotation=(0, 0, rotation),
        )
        self.placed_assets.append(placed)
        return placed

    def check_collision(self, placed):
        collisions = []
        obj = placed.blender_object
        if not obj:
            return collisions
        
        for other in self.placed_assets:
            if other is placed or not other.blender_object:
                continue
            dist = (obj.location - other.blender_object.location).length
            min_d = (max(obj.dimensions) / 2 + max(other.blender_object.dimensions) / 2) * 0.5
            if dist < min_d:
                collisions.append(other)
        return collisions

    def remove_asset(self, placed):
        """Удаляет размещённый ассет."""
        if placed.blender_object:
            bpy.data.objects.remove(placed.blender_object, do_unlink=True)
        if placed in self.placed_assets:
            self.placed_assets.remove(placed)

    def get_all_placed_info(self):
        """Возвращает информацию о всех размещённых ассетах."""
        return [
            {
                "asset_id": p.asset_info.id,
                "name": p.asset_info.name,
                "position": list(p.position),
                "rotation": list(p.rotation),
            }
            for p in self.placed_assets
        ]

    # ═══════════════════════════════════════════════════════════════
    # ПРОЦЕДУРНАЯ ГЕОМЕТРИЯ
    # ═══════════════════════════════════════════════════════════════
    def _create_furniture(self, info):
        """Создаёт 3D-модель мебели."""
        w, d, h = info.dimensions
        category = info.category
        
        mesh = bpy.data.meshes.new(info.name)
        obj = bpy.data.objects.new(info.name, mesh)
        bm = bmesh.new()
        
        name_lower = info.name.lower()
        
        if category == "Гостиная" and "диван" in name_lower:
            self._make_sofa(bm, w, d, h, "угловой" in name_lower)
        elif category == "Спальня" and "кровать" in name_lower:
            self._make_bed(bm, w, d, h)
        elif "стул" in name_lower or "кресло" in name_lower:
            self._make_chair(bm, w, d, h, "офис" in name_lower)
        elif "стол" in name_lower:
            self._make_table(bm, w, d, h)
        elif category == "Ванная" and "ванна" in name_lower:
            self._make_bathtub(bm, w, d, h)
        elif "душ" in name_lower:
            self._make_shower(bm, w, d, h)
        elif "зеркал" in name_lower:
            self._make_mirror(bm, w, d, h)
        else:
            self._make_box(bm, w, d, h)
        
        bm.to_mesh(mesh)
        bm.free()
        
        mat = self._make_material(info)
        obj.data.materials.append(mat)
        
        bpy.context.scene.collection.objects.link(obj)
        return obj

    def _make_box(self, bm, w, d, h):
        hw, hd, hh = w / 2, d / 2, h / 2
        coords = [
            (-hw, -hd, -hh), (hw, -hd, -hh), (hw, hd, -hh), (-hw, hd, -hh),
            (-hw, -hd, hh), (hw, -hd, hh), (hw, hd, hh), (-hw, hd, hh),
        ]
        verts = [bm.verts.new(c) for c in coords]
        bm.verts.ensure_lookup_table()
        for f in [(0,1,2,3),(4,7,6,5),(0,4,5,1),(2,6,7,3),(0,3,7,4),(1,5,6,2)]:
            bm.faces.new([verts[i] for i in f])

    def _make_sofa(self, bm, w, d, h, is_corner=False):
        seat_h = h * 0.45
        seat_d = d * 0.7
        self._add_box(bm, w, seat_d, seat_h, 0, -d * 0.15, 0)
        self._add_box(bm, w, d * 0.2, h * 0.55, 0, d * 0.25, seat_h * 0.5 + h * 0.55 * 0.5)
        arm_w = 0.12
        self._add_box(bm, arm_w, seat_d, h * 0.35, -w / 2 + arm_w / 2, -d * 0.15, seat_h * 0.5)
        self._add_box(bm, arm_w, seat_d, h * 0.35, w / 2 - arm_w / 2, -d * 0.15, seat_h * 0.5)
        if is_corner:
            self._add_box(bm, d * 0.8, d * 0.7, seat_h, w / 2 + d * 0.4, 0, 0)

    def _make_bed(self, bm, w, d, h):
        self._add_box(bm, w, d, h * 0.6, 0, 0, 0)
        self._add_box(bm, w * 0.95, d * 0.95, h * 0.35, 0, 0, h * 0.6 * 0.5 + h * 0.35 * 0.5)
        self._add_box(bm, w, 0.08, h * 1.4, 0, d / 2 - 0.04, h * 0.1)

    def _make_chair(self, bm, w, d, h, is_office=False):
        seat_h = h * 0.47
        seat_thick = 0.04
        self._add_box(bm, w * 0.9, d * 0.85, seat_thick, 0, 0, seat_h)
        self._add_box(bm, w * 0.85, 0.03, h - seat_h - seat_thick, 0, -d * 0.4, seat_h + (h - seat_h) / 2)
        leg_r = 0.025
        for lx, ly in [(-w*0.38, -d*0.35), (w*0.38, -d*0.35), (-w*0.38, d*0.35), (w*0.38, d*0.35)]:
            self._add_box(bm, leg_r*2, leg_r*2, seat_h, lx, ly, seat_h/2)

    def _make_table(self, bm, w, d, h):
        top_thick = 0.04
        self._add_box(bm, w, d, top_thick, 0, 0, h - top_thick / 2)
        leg_w = 0.05
        leg_h = h - top_thick
        for lx, ly in [(-w/2+leg_w, -d/2+leg_w), (w/2-leg_w, -d/2+leg_w), (-w/2+leg_w, d/2-leg_w), (w/2-leg_w, d/2-leg_w)]:
            self._add_box(bm, leg_w, leg_w, leg_h, lx, ly, leg_h/2)

    def _make_bathtub(self, bm, w, d, h):
        wall_thick = 0.05
        self._add_box(bm, w, d, h, 0, 0, 0)
        self._add_box(bm, w - wall_thick*2, d - wall_thick*2, h - wall_thick, 0, 0, wall_thick/2)

    def _make_shower(self, bm, w, d, h):
        wall_thick = 0.03
        tray_h = 0.1
        self._add_box(bm, w, d, tray_h, 0, 0, 0)
        self._add_box(bm, wall_thick, d, h, -w/2 + wall_thick/2, 0, h/2)
        self._add_box(bm, w, wall_thick, h, 0, -d/2 + wall_thick/2, h/2)

    def _make_mirror(self, bm, w, d, h):
        self._add_box(bm, w, d, h, 0, 0, 0)

    def _add_box(self, bm, w, d, h, cx=0, cy=0, cz=0):
        hw, hd, hh = w / 2, d / 2, h / 2
        coords = [
            (cx - hw, cy - hd, cz - hh), (cx + hw, cy - hd, cz - hh),
            (cx + hw, cy + hd, cz - hh), (cx - hw, cy + hd, cz - hh),
            (cx - hw, cy - hd, cz + hh), (cx + hw, cy - hd, cz + hh),
            (cx + hw, cy + hd, cz + hh), (cx - hw, cy + hd, cz + hh),
        ]
        start_idx = len(bm.verts)
        for c in coords:
            bm.verts.new(c)
        bm.verts.ensure_lookup_table()
        v = bm.verts
        idx = start_idx
        for f in [(0,1,2,3),(4,7,6,5),(0,4,5,1),(2,6,7,3),(0,3,7,4),(1,5,6,2)]:
            bm.faces.new([v[idx + i] for i in f])

    # ═══════════════════════════════════════════════════════════════
    # МАТЕРИАЛЫ
    # ═══════════════════════════════════════════════════════════════
    def _make_material(self, info):
        category_colors = {
            'Гостиная': (0.45, 0.38, 0.32, 1),
            'Спальня': (0.78, 0.72, 0.65, 1),
            'Кухня': (0.88, 0.85, 0.82, 1),
            'Ванная': (0.93, 0.93, 0.95, 1),
            'Кабинет': (0.50, 0.40, 0.30, 1),
            'Прихожая': (0.55, 0.45, 0.35, 1),
            'Детская': (0.70, 0.75, 0.85, 1),
        }
        
        if any(t in info.tags for t in ["техника", "холодильник", "стиральная"]):
            color = (0.85, 0.85, 0.87, 1)
            metallic = 0.3
            roughness = 0.3
        elif "зеркало" in info.name.lower():
            color = (0.9, 0.9, 0.92, 1)
            metallic = 0.0
            roughness = 0.1
        else:
            color = category_colors.get(info.category, (0.5, 0.5, 0.5, 1))
            metallic = 0.0
            roughness = 0.6
        
        mat = bpy.data.materials.new(f"Mat_{info.id}")
        mat.use_nodes = True
        nodes = mat.node_tree.nodes
        nodes.clear()
        bsdf = nodes.new('ShaderNodeBsdfPrincipled')
        bsdf.inputs['Base Color'].default_value = color
        bsdf.inputs['Roughness'].default_value = roughness
        bsdf.inputs['Metallic'].default_value = metallic
        out = nodes.new('ShaderNodeOutputMaterial')
        out.location = (300, 0)
        mat.node_tree.links.new(bsdf.outputs['BSDF'], out.inputs['Surface'])
        
        return mat

    # ═══════════════════════════════════════════════════════════════
    # ИМПОРТ ПОЛЬЗОВАТЕЛЬСКИХ МОДЕЛЕЙ
    # ═══════════════════════════════════════════════════════════════
    def add_custom_asset(self, name, category, filepath, dimensions=None, description="", tags=None):
        import shutil
        aid = name.lower().replace(" ", "_")
        c = 1
        while aid in self.catalog:
            aid = f"{aid}_{c}"
            c += 1
        fn = os.path.basename(filepath)
        dest = os.path.join(FURNITURE_DIR, fn)
        if filepath != dest:
            shutil.copy2(filepath, dest)
        if not dimensions:
            dimensions = (1, 1, 1)
        info = AssetInfo(
            id=aid, name=name, category=category,
            filename=fn, dimensions=dimensions,
            description=description, tags=tags or [],
        )
        self.catalog[aid] = info
        self._save_catalog()
        return info