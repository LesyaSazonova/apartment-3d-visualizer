import bpy
from bpy.types import Panel
from bpy.props import FloatProperty, EnumProperty, StringProperty, IntProperty
import os, sys

addon_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if addon_dir not in sys.path:
    sys.path.append(addon_dir)

_TEX_DIR = os.path.join(addon_dir, "textures")
_PRESET_FOLDER = {'PARQUET':'parquet','TILE':'tile','CONCRETE':'concrete','CARPET':'carpet','MARBLE':'marble'}
_IMG_EXTS = {'.jpg','.jpeg','.png','.exr','.tga','.hdr'}


def _has_textures(preset):
    """Проверяет, есть ли в папке пресета хотя бы один файл текстуры."""
    folder = os.path.join(_TEX_DIR, _PRESET_FOLDER.get(preset,''))
    if not os.path.isdir(folder): return False
    return any(os.path.splitext(f)[1].lower() in _IMG_EXTS for f in os.listdir(folder))


class ApartmentProperties(bpy.types.PropertyGroup):
    wall_height:    FloatProperty(name="Высота стен",  default=2.7, min=2.0, max=5.0, unit='LENGTH')
    wall_thickness: FloatProperty(name="Толщина стен", default=0.2, min=0.05,max=0.5, unit='LENGTH')
    pixels_per_meter: FloatProperty(name="Пикс/метр",  default=70., min=10., max=1000.)
    render_preset: EnumProperty(name="Вид",
        items=[('top','Сверху',''),('perspective','Перспективный',''),('first_person','Изнутри','')],
        default='perspective')
    render_samples: IntProperty(name="Сэмплы", default=128, min=16, max=4096)
    project_name:   StringProperty(name="Имя проекта", default="Novyj proekt")


class APT_PT_MainPanel(Panel):
    bl_label="Визуализация квартиры"; bl_idname="APT_PT_main"
    bl_space_type='VIEW_3D'; bl_region_type='UI'; bl_category="Kvartira"
    def draw(self,context):
        layout=self.layout
        if context.scene.get("apt_builder"): layout.label(text="Модель загружена",icon='CHECKMARK')
        else: layout.label(text="Загрузите чертёж",icon='INFO')


class APT_PT_FloorPlanPanel(Panel):
    bl_label="Чертёж"; bl_idname="APT_PT_floor_plan"
    bl_space_type='VIEW_3D'; bl_region_type='UI'; bl_category="Kvartira"; bl_parent_id="APT_PT_main"
    def draw(self,context):
        layout=self.layout; props=context.scene.apartment_props
        box=layout.box()
        box.prop(props,"pixels_per_meter"); box.prop(props,"wall_height"); box.prop(props,"wall_thickness")
        box.label(text="Авто-масштаб: рекомендуется",icon='INFO')
        row=layout.row(); row.scale_y=1.5
        row.operator("apartment.load_floor_plan",icon='FILEBROWSER')


class APT_PT_WallsPanel(Panel):
    bl_label="Стены и материалы"; bl_idname="APT_PT_walls"
    bl_space_type='VIEW_3D'; bl_region_type='UI'; bl_category="Kvartira"
    bl_parent_id="APT_PT_main"; bl_options={'DEFAULT_CLOSED'}

    @classmethod
    def poll(cls,context): return context.scene.get("apt_builder",False)

    def draw(self,context):
        layout=self.layout; props=context.scene.apartment_props
        box=layout.box(); box.label(text="Геометрия стен:",icon='MESH_CUBE')
        box.prop(props,"wall_height"); box.prop(props,"wall_thickness")
        op=box.operator("apartment.update_walls",icon='FILE_REFRESH')
        op.wall_height=props.wall_height; op.wall_thickness=props.wall_thickness
        box2=layout.box(); box2.label(text="Цвет:",icon='COLOR')
        box2.label(text="Виден в Solid и Material Preview",icon='INFO')
        row=box2.row(align=True)
        row.operator("apartment.set_wall_color",icon='SNAP_FACE',text="Цвет стен")
        row.operator("apartment.set_floor_color",icon='MESH_PLANE',text="Цвет пола")
        box3=layout.box()
        box3.label(text="Текстуры пола:",icon='TEXTURE')
        box3.label(text="Режим: Material Preview (Z)",icon='INFO')

        # Иконку берём по наличию файлов текстур в папке пресета.
        presets=[
            ('PARQUET','Паркет',   'SNAP_FACE'),
            ('TILE',   'Плитка',   'MESH_GRID'),
            ('CONCRETE','Бетон',   'TEXTURE'),
            ('CARPET', 'Ковёр',    'ALIASED'),
            ('MARBLE', 'Мрамор',   'NODE_MATERIAL'),
        ]
        grid=box3.column_flow(columns=2,align=True)
        for preset,label,icon in presets:
            has=_has_textures(preset)
            btn_label = f"\u2713 {label}" if has else label
            op=grid.operator("apartment.floor_preset",text=btn_label,icon=icon)
            op.preset=preset

        box3.separator()

        box4=layout.box()
        box4.label(text="Свои текстуры:",icon='FILEBROWSER')
        box4.label(text="Положи файлы в textures/<preset>/",icon='INFO')
        box4.operator("apartment.open_textures_folder",icon='FILE_FOLDER',
                      text="Открыть папку textures/")
        box4.operator("apartment.set_floor_texture",icon='IMAGE_DATA',
                      text="Загрузить один файл (.png/.jpg)")


class APT_PT_AssetBrowserHintPanel(Panel):
    bl_label="Мебель — Asset Browser"; bl_idname="APT_PT_assets_hint"
    bl_space_type='VIEW_3D'; bl_region_type='UI'; bl_category="Kvartira"; bl_parent_id="APT_PT_main"
    def draw(self,context):
        layout=self.layout; col=layout.column(align=True)
        b=col.box()
        b.label(text="Открыть Asset Browser:",icon='ASSET_MANAGER')
        b.label(text="  Shift+F1  или  Editor → Asset Browser")
        col.separator()
        b2=col.box()
        b2.label(text="Добавить мебель:",icon='IMPORT')
        b2.label(text="Способ А — импорт файла (.blend/.obj/.fbx):")
        b2.operator("apartment.add_custom_asset",icon='IMPORT',text="Импортировать файл")
        b2.separator()
        b2.label(text="Способ Б — из объекта в сцене:")
        b2.label(text="  Выделите объект → кнопка ниже")
        b2.operator("apartment.mark_selected_as_asset",icon='ASSET_MANAGER',text="Пометить как Asset")
        col.separator()
        b3=col.box()
        b3.label(text="Расставить мебель:",icon='HAND')
        b3.label(text="  Drag & Drop из Asset Browser в Viewport")
        b3.label(text="  G двигать · R вращать · S масштаб")


class APT_PT_RenderPanel(Panel):
    bl_label="Рендер"; bl_idname="APT_PT_render"
    bl_space_type='VIEW_3D'; bl_region_type='UI'; bl_category="Kvartira"; bl_parent_id="APT_PT_main"
    def draw(self,context):
        layout=self.layout; props=context.scene.apartment_props
        layout.prop(props,"render_preset"); layout.prop(props,"render_samples")
        row=layout.row(); row.scale_y=1.5
        op=row.operator("apartment.render_view",icon='RENDER_STILL'); op.preset=props.render_preset
        layout.operator("apartment.render_all_views",icon='RENDER_RESULT')


class APT_PT_ProjectPanel(Panel):
    bl_label="Проект"; bl_idname="APT_PT_project"
    bl_space_type='VIEW_3D'; bl_region_type='UI'; bl_category="Kvartira"
    bl_parent_id="APT_PT_main"; bl_options={'DEFAULT_CLOSED'}
    def draw(self,context):
        layout=self.layout
        layout.operator("apartment.save_project",icon='FILE_TICK')
        layout.operator("apartment.load_project",icon='FILE_FOLDER')


PANEL_CLASSES=[
    ApartmentProperties,
    APT_PT_MainPanel,
    APT_PT_FloorPlanPanel,
    APT_PT_WallsPanel,
    APT_PT_AssetBrowserHintPanel,
    APT_PT_RenderPanel,
    APT_PT_ProjectPanel,
]
