"""Точка входа аддона и регистрация классов/обработчиков."""

bl_info = {
    "name": "Apartment 3D Visualizer",
    "author": "Apartment3D Team",
    "version": (2, 7, 0),
    "blender": (5, 1, 0),
    "location": "View3D > Sidebar > Kvartira",
    "description": "3D визуализация плана квартиры по чертежу с библиотекой мебели",
    "category": "3D View",
}

import bpy
import sys
import os

addon_dir = os.path.dirname(os.path.abspath(__file__))
if addon_dir not in sys.path:
    sys.path.insert(0, addon_dir)

from ui.operators import OPERATOR_CLASSES
from ui.panels import PANEL_CLASSES, ApartmentProperties

from bpy.app.handlers import persistent

@persistent
def on_depsgraph_update_post(scene, depsgraph):
    """Отслеживает добавление объектов из Asset Browser по обновлениям depsgraph."""
    for update in depsgraph.updates:
        if update.is_updated_geometry:
            obj = update.id
            if hasattr(obj, 'asset_data') and obj.asset_data:
                try:
                    from core.asset_manager import AssetManager
                    am = AssetManager()
                    
                    asset_id = obj.name.split('.')[0]
                    if asset_id in am.catalog:
                        asset_info = am.catalog[asset_id]
                        from core.asset_manager import PlacedAsset
                        placed = PlacedAsset(
                            asset_info=asset_info,
                            blender_object=obj,
                            position=tuple(obj.location),
                            rotation=tuple(obj.rotation_euler),
                            scale=tuple(obj.scale)
                        )
                        am.placed_assets.append(placed)
                        print(f"[APT] Мебель размещена: {asset_info.name}")
                        
                        collisions = am.check_collision(placed)
                        if collisions:
                            print(f"[APT] ⚠ Коллизия с {len(collisions)} объектами")
                except Exception as e:
                    print(f"[APT] Ошибка отслеживания ассета: {e}")


@persistent
def on_load_post(scene):
    """Вызывается после загрузки .blend файла."""
    try:
        from core.asset_manager import AssetManager
        am = AssetManager()
        am.placed_assets = []
        for obj in bpy.data.objects:
            if obj.asset_data and obj.name.startswith(tuple(am.catalog.keys())):
                for asset_id, info in am.catalog.items():
                    if obj.name.startswith(asset_id):
                        from core.asset_manager import PlacedAsset
                        placed = PlacedAsset(
                            asset_info=info,
                            blender_object=obj,
                            position=tuple(obj.location),
                            rotation=tuple(obj.rotation_euler),
                            scale=tuple(obj.scale)
                        )
                        am.placed_assets.append(placed)
                        break
        print(f"[APT] Проект загружен, мебели: {len(am.placed_assets)}")
    except Exception as e:
        print(f"[APT] Ошибка при загрузке: {e}")

def register():
    """Активация аддона."""
    print("\n" + "="*50)
    print("Apartment 3D Visualizer v2.7 aktivirovan!")
    print("="*50 + "\n")
    
    for cls in OPERATOR_CLASSES:
        bpy.utils.register_class(cls)
    for cls in PANEL_CLASSES:
        bpy.utils.register_class(cls)
    
    bpy.types.Scene.apartment_props = bpy.props.PointerProperty(
        type=ApartmentProperties
    )
    
    bpy.app.handlers.depsgraph_update_post.append(on_depsgraph_update_post)
    bpy.app.handlers.load_post.append(on_load_post)
    
    try:
        from core.asset_manager import AssetManager
        am = AssetManager()
        am.register_asset_library()
        print("[APT] Библиотека мебели зарегистрирована в Asset Browser")
        print("[APT] Категории: Гостиная, Спальня, Кухня, Ванная, Кабинет, Прихожая, Детская")
    except Exception as e:
        print(f"[APT] Ошибка инициализации библиотеки: {e}")
    
    print("\n[APT] Аддон готов к работе!")
    print("[APT] Откройте Asset Browser (Shift+F1) для размещения мебели\n")


def unregister():
    """Деактивация аддона."""
    print("[APT] Деактивация Apartment 3D Visualizer...")
    
    if on_depsgraph_update_post in bpy.app.handlers.depsgraph_update_post:
        bpy.app.handlers.depsgraph_update_post.remove(on_depsgraph_update_post)
    if on_load_post in bpy.app.handlers.load_post:
        bpy.app.handlers.load_post.remove(on_load_post)
    
    if hasattr(bpy.types.Scene, "apartment_props"):
        del bpy.types.Scene.apartment_props
    
    for cls in reversed(PANEL_CLASSES):
        bpy.utils.unregister_class(cls)
    for cls in reversed(OPERATOR_CLASSES):
        bpy.utils.unregister_class(cls)
    
    print("[APT] Аддон деактивирован")


if __name__ == "__main__":
    register()
