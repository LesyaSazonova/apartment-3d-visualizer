bl_info = {
    "name": "Apartment 3D Visualizer",
    "author": "Apartment3D Team",
    "version": (2, 6, 0),
    "blender": (4, 3, 0),
    "location": "View3D > Sidebar > Kvartira",
    "description": "3D vizualizaciya plana kvartiry po chertezhу",
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


def register():
    for cls in OPERATOR_CLASSES:
        bpy.utils.register_class(cls)
    for cls in PANEL_CLASSES:
        bpy.utils.register_class(cls)
    bpy.types.Scene.apartment_props = bpy.props.PointerProperty(
        type=ApartmentProperties
    )
    print("Apartment 3D Visualizer v2.6 aktivirovan!")


def unregister():
    if hasattr(bpy.types.Scene, "apartment_props"):
        del bpy.types.Scene.apartment_props
    for cls in reversed(PANEL_CLASSES):
        bpy.utils.unregister_class(cls)
    for cls in reversed(OPERATOR_CLASSES):
        bpy.utils.unregister_class(cls)


if __name__ == "__main__":
    register()
