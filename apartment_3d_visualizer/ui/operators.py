import bpy
from bpy.props import StringProperty, FloatProperty, EnumProperty
from bpy.types import Operator
from bpy_extras.io_utils import ImportHelper
import os
import sys

addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if addon_dir not in sys.path:
    sys.path.append(addon_dir)


class APT_OT_LoadFloorPlan(Operator, ImportHelper):
    bl_idname = "apartment.load_floor_plan"
    bl_label = "Загрузить чертеж"
    bl_options = {'REGISTER', 'UNDO'}

    filter_glob: StringProperty(default="*.jpg;*.jpeg;*.png;*.bmp", options={'HIDDEN'})
    pixels_per_meter: FloatProperty(name="Пикс/метр", default=70.0, min=10.0, max=1000.0)
    wall_height:      FloatProperty(name="Высота стен", default=2.7, min=2.0, max=5.0)
    wall_thickness:   FloatProperty(name="Толщина стен", default=0.2, min=0.05, max=0.5)
    auto_scale: bpy.props.BoolProperty(
        name="Авто-масштаб",
        description="Определять пикселей/метр автоматически",
        default=True)

    def invoke(self, context, event):
        props = context.scene.apartment_props
        self.pixels_per_meter = props.pixels_per_meter
        self.wall_height      = props.wall_height
        self.wall_thickness   = props.wall_thickness
        return super().invoke(context, event)

    def execute(self, context):
        try:
            from core.image_processor import ImageProcessor
            from core.model_builder import ModelBuilder
            print(f"[APT] Загрузка: {self.filepath}")
            processor = ImageProcessor(pixels_per_meter=self.pixels_per_meter)
            plan_data = processor.process(self.filepath)
            if not plan_data.wall_segments:
                self.report({'WARNING'}, "Стены не найдены. Попробуйте задать px/м вручную.")
                return {'CANCELLED'}
            builder = ModelBuilder(wall_height=self.wall_height, wall_thickness=self.wall_thickness)
            builder.build_from_plan_data(plan_data)
            context.scene["apt_builder"]        = True
            context.scene["apt_wall_height"]    = self.wall_height
            context.scene["apt_wall_thickness"] = self.wall_thickness
            self.report({'INFO'},
                f"Готово: {len(plan_data.wall_segments)} стен, "
                f"{len(plan_data.openings)} проёмов ({plan_data.scale:.0f} px/m)")
            return {'FINISHED'}
        except Exception as e:
            import traceback; traceback.print_exc()
            self.report({'ERROR'}, f"Ошибка: {str(e)}")
            return {'CANCELLED'}

    def draw(self, context):
        layout = self.layout
        layout.prop(self, "auto_scale")
        row = layout.row(); row.enabled = not self.auto_scale
        row.prop(self, "pixels_per_meter")
        layout.prop(self, "wall_height")
        layout.prop(self, "wall_thickness")


class APT_OT_RenderView(Operator):
    bl_idname = "apartment.render_view"
    bl_label  = "Рендер"
    preset: EnumProperty(items=[('top','Сверху',''),('perspective','Перспективный',''),
                                  ('first_person','Изнутри','')], default='perspective')
    def execute(self, context):
        from core.renderer import SceneRenderer
        renderer = SceneRenderer()
        path = renderer.render(preset=self.preset)
        self.report({'INFO'}, f"Сохранено: {path}")
        return {'FINISHED'}


class APT_OT_RenderAllViews(Operator):
    bl_idname = "apartment.render_all_views"
    bl_label  = "Рендер всех видов"
    def execute(self, context):
        from core.renderer import SceneRenderer
        results = SceneRenderer().render_all_presets()
        self.report({'INFO'}, f"Рендеров: {len(results)}")
        return {'FINISHED'}


class APT_OT_UpdateWalls(Operator):
    bl_idname = "apartment.update_walls"
    bl_label  = "Обновить стены"
    bl_options = {'REGISTER', 'UNDO'}
    wall_height:    FloatProperty(default=2.7,  min=2.0, max=5.0)
    wall_thickness: FloatProperty(default=0.2, min=0.05, max=0.5)
    def execute(self, context):
        from core.model_builder import ModelBuilder

        prop_kind = ModelBuilder.COLUMN_OBJ_KIND

        wall_objs = [
            o for o in bpy.data.objects
            if o.type == 'MESH' and o.name.startswith("Wall_")
        ]
        column_objs = []
        seen = {id(o) for o in wall_objs}
        for o in bpy.data.objects:
            if o.type != 'MESH' or id(o) in seen:
                continue
            if o.name.startswith("Column_") or o.get(ModelBuilder.COLUMN_OBJ_PROP) == prop_kind:
                column_objs.append(o)
                seen.add(id(o))

        builder = ModelBuilder()
        builder._wall_objects = wall_objs + column_objs
        builder.wall_height = context.scene.get("apt_wall_height", 2.7)
        builder.wall_thickness = context.scene.get("apt_wall_thickness", 0.2)

        print(
            f"[APT][DBG][UPDATE_WALLS] scene_wall_h={builder.wall_height:.4f} "
            f"scene_wall_t={builder.wall_thickness:.4f} "
            f"target_wall_h={self.wall_height:.4f} target_wall_t={self.wall_thickness:.4f}"
        )

        builder.update_wall_height(self.wall_height)
        builder.update_wall_thickness(self.wall_thickness)

        context.view_layer.update()

        context.scene["apt_wall_height"] = self.wall_height
        context.scene["apt_wall_thickness"] = self.wall_thickness

        col_dims_report = []
        for co in column_objs:
            try:
                d = co.dimensions
                col_dims_report.append(f"{co.name}: dims=({d.x:.4f},{d.y:.4f},{d.z:.4f})")
            except Exception:
                col_dims_report.append(f"{co.name}: dims=?")

        print(
            f"[APT][DBG][UPDATE_WALLS] walls_updated={len(wall_objs)} columns_updated={len(column_objs)} "
            f"final_wall_h={self.wall_height:.4f} final_wall_t={self.wall_thickness:.4f}"
        )
        for line in col_dims_report[:40]:
            print(f"[APT][DBG][UPDATE_WALLS][COLUMN_DIM] {line}")
        if len(col_dims_report) > 40:
            print(f"[APT][DBG][UPDATE_WALLS][COLUMN_DIM] ... и ещё {len(col_dims_report) - 40} колонок")

        self.report({'INFO'}, "Стены обновлены")
        return {'FINISHED'}


# Настройка цвета через Principled BSDF делается по node.type: имя ноды зависит от языка интерфейса Blender.
class APT_OT_SetWallColor(Operator):
    bl_idname  = "apartment.set_wall_color"
    bl_label   = "Цвет стен"
    bl_options = {'REGISTER', 'UNDO'}
    color: bpy.props.FloatVectorProperty(
        name="Цвет стен", subtype='COLOR',
        default=(0.9, 0.9, 0.85), min=0.0, max=1.0, size=3)

    def execute(self, context):
        from core.model_builder import ModelBuilder
        ModelBuilder().set_wall_color((*self.color, 1.0))
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class APT_OT_SetFloorColor(Operator):
    bl_idname  = "apartment.set_floor_color"
    bl_label   = "Цвет пола"
    bl_options = {'REGISTER', 'UNDO'}
    color: bpy.props.FloatVectorProperty(
        name="Цвет пола", subtype='COLOR',
        default=(0.6, 0.5, 0.4), min=0.0, max=1.0, size=3)

    def execute(self, context):
        from core.model_builder import ModelBuilder
        ModelBuilder().set_floor_color((*self.color, 1.0))
        return {'FINISHED'}

    def invoke(self, context, event):
        return context.window_manager.invoke_props_dialog(self)


class APT_OT_FloorPreset(Operator):
    """Применяет процедурную текстуру пола без внешних файлов."""
    bl_idname  = "apartment.floor_preset"
    bl_label   = "Текстура пола"
    bl_options = {'REGISTER', 'UNDO'}

    preset: EnumProperty(
        name="Текстура",
        items=[
            ('PARQUET',  'Паркет / дерево', '', 'SNAP_FACE',    0),
            ('TILE',     'Плитка',          '', 'MESH_GRID',    1),
            ('CONCRETE', 'Бетон',           '', 'TEXTURE',      2),
            ('CARPET',   'Ковёр',           '', 'ALIASED',      3),
            ('MARBLE',   'Мрамор',          '', 'NODE_MATERIAL',4),
        ],
        default='PARQUET',
    )

    def execute(self, context):
        from core.model_builder import ModelBuilder
        mb = ModelBuilder()
        mb.apply_floor_preset(self.preset)
        self.report({'INFO'}, f"Текстура пола: {self.preset}")
        return {'FINISHED'}


class APT_OT_SetFloorTexture(Operator, ImportHelper):
    bl_idname  = "apartment.set_floor_texture"
    bl_label   = "Своя текстура пола"
    bl_options = {'REGISTER', 'UNDO'}
    filter_glob: StringProperty(default="*.jpg;*.jpeg;*.png;*.exr;*.tga", options={'HIDDEN'})

    def execute(self, context):
        from core.model_builder import ModelBuilder
        ModelBuilder().set_floor_texture(self.filepath)
        self.report({'INFO'}, "Текстура пола применена")
        return {'FINISHED'}


class APT_OT_SaveProject(Operator):
    bl_idname = "apartment.save_project"
    bl_label  = "Сохранить проект"
    def execute(self, context):
        from core.project_manager import ProjectManager
        from core.asset_manager import AssetManager
        pm = ProjectManager(); am = AssetManager()
        if not pm.current_project: pm.create_project("Novyj proekt")
        pm.save_project(placed_assets_info=am.get_all_placed_info())
        self.report({'INFO'}, "Проект сохранен")
        return {'FINISHED'}


class APT_OT_LoadProject(Operator):
    bl_idname    = "apartment.load_project"
    bl_label     = "Загрузить проект"
    project_path: StringProperty(subtype='DIR_PATH')
    def execute(self, context):
        from core.project_manager import ProjectManager
        data = ProjectManager().load_project(self.project_path)
        self.report({'INFO'}, f"Загружен: {data.get('name','')}")
        return {'FINISHED'}


class APT_OT_MarkSelectedAsAsset(Operator):
    bl_idname  = "apartment.mark_selected_as_asset"
    bl_label   = "Пометить как Asset"
    bl_options = {'REGISTER', 'UNDO'}
    @classmethod
    def poll(cls, context): return context.active_object is not None
    def execute(self, context):
        obj = context.active_object
        if not obj: self.report({'ERROR'}, "Нет активного объекта"); return {'CANCELLED'}
        obj.asset_mark()
        try: bpy.ops.ed.lib_id_generate_preview({'id': obj})
        except: pass
        self.report({'INFO'}, f"'{obj.name}' помечен как Asset.")
        return {'FINISHED'}


class APT_OT_AddCustomAsset(Operator, ImportHelper):
    bl_idname  = "apartment.add_custom_asset"
    bl_label   = "Импортировать мебель (.blend/.obj/.fbx)"
    bl_options = {'REGISTER', 'UNDO'}
    filter_glob: StringProperty(default="*.blend;*.obj;*.fbx", options={'HIDDEN'})
    asset_name:  StringProperty(name="Название", default="Novyj objekt")

    def execute(self, context):
        fp  = self.filepath
        ext = os.path.splitext(fp)[1].lower()
        imported = []
        try:
            if ext == '.blend':
                with bpy.data.libraries.load(fp, link=False) as (df, dt):
                    dt.objects = list(df.objects)
                for obj in dt.objects:
                    if obj: bpy.context.scene.collection.objects.link(obj); imported.append(obj)
            elif ext == '.obj':
                before = set(bpy.data.objects)
                bpy.ops.wm.obj_import(filepath=fp)
                imported = [o for o in bpy.data.objects if o not in before]
            elif ext == '.fbx':
                before = set(bpy.data.objects)
                bpy.ops.import_scene.fbx(filepath=fp)
                imported = [o for o in bpy.data.objects if o not in before]
            else:
                self.report({'ERROR'}, f"Формат не поддерживается: {ext}"); return {'CANCELLED'}
        except Exception as e:
            self.report({'ERROR'}, f"Ошибка импорта: {e}"); return {'CANCELLED'}

        for i, obj in enumerate(imported):
            if obj.type == 'MESH':
                obj.name = self.asset_name if i == 0 else f"{self.asset_name}.{i:03d}"
                obj.asset_mark()
                try: bpy.ops.ed.lib_id_generate_preview({'id': obj})
                except: pass
        self.report({'INFO'}, f"Импортировано {len(imported)} объект(ов).")
        return {'FINISHED'}

    def draw(self, context):
        self.layout.prop(self, "asset_name")


class APT_OT_OpenTexturesFolder(Operator):
    """Открывает папку textures/ аддона в системном проводнике."""
    bl_idname = "apartment.open_textures_folder"
    bl_label  = "Открыть папку textures/"

    def execute(self, context):
        import subprocess, platform
        tex_dir = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "textures"
        )
        os.makedirs(tex_dir, exist_ok=True)
        try:
            system = platform.system()
            if system == "Windows":
                os.startfile(tex_dir)
            elif system == "Darwin":
                subprocess.Popen(["open", tex_dir])
            else:
                subprocess.Popen(["xdg-open", tex_dir])
            self.report({'INFO'}, f"Открыто: {tex_dir}")
        except Exception as e:
            self.report({'WARNING'}, f"Не удалось открыть: {tex_dir}")
            print(f"[APT] Путь к папке текстур: {tex_dir}")
        return {'FINISHED'}


OPERATOR_CLASSES = [
    APT_OT_OpenTexturesFolder,
    APT_OT_LoadFloorPlan,
    APT_OT_RenderView,
    APT_OT_RenderAllViews,
    APT_OT_UpdateWalls,
    APT_OT_SetWallColor,
    APT_OT_SetFloorColor,
    APT_OT_FloorPreset,
    APT_OT_SetFloorTexture,
    APT_OT_SaveProject,
    APT_OT_LoadProject,
    APT_OT_MarkSelectedAsAsset,
    APT_OT_AddCustomAsset,
]
