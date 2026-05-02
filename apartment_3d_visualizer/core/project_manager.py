import bpy
import json
import os
import datetime
from dataclasses import dataclass, asdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import PROJECTS_DIR


@dataclass
class ProjectMetadata:
    name: str
    created: str
    modified: str
    description: str = ""
    source_image: str = ""


class ProjectManager:
    def __init__(self):
        self.current_project = None
        self.metadata = None
        os.makedirs(PROJECTS_DIR, exist_ok=True)

    def create_project(self, name, description="", source_image=""):
        safe = "".join(c for c in name if c.isalnum() or c in ' -_').strip().replace(' ', '_')
        path = os.path.join(PROJECTS_DIR, safe)
        c = 1
        while os.path.exists(path):
            path = os.path.join(PROJECTS_DIR, f"{safe}_{c}")
            c += 1
        os.makedirs(path)
        now = datetime.datetime.now().isoformat()
        self.metadata = ProjectMetadata(name=name, created=now, modified=now, description=description, source_image=source_image)
        with open(os.path.join(path, "project.json"), 'w', encoding='utf-8') as f:
            json.dump(asdict(self.metadata), f, indent=2, ensure_ascii=False)
        self.current_project = path
        return path

    def save_project(self, project_dir=None, placed_assets_info=None):
        if not project_dir:
            project_dir = self.current_project
        if not project_dir:
            raise ValueError("Proekt ne otkryt")
        blend_path = os.path.join(project_dir, "scene.blend")
        bpy.ops.wm.save_as_mainfile(filepath=blend_path)
        if self.metadata:
            self.metadata.modified = datetime.datetime.now().isoformat()
            data = asdict(self.metadata)
            if placed_assets_info:
                data["placed_assets"] = placed_assets_info
            with open(os.path.join(project_dir, "project.json"), 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)

    def load_project(self, project_dir):
        data = {}
        mp = os.path.join(project_dir, "project.json")
        if os.path.exists(mp):
            with open(mp, 'r', encoding='utf-8') as f:
                data = json.load(f)
        bp = os.path.join(project_dir, "scene.blend")
        if os.path.exists(bp):
            bpy.ops.wm.open_mainfile(filepath=bp)
        self.current_project = project_dir
        return data

    def list_projects(self):
        projects = []
        for name in os.listdir(PROJECTS_DIR):
            pd = os.path.join(PROJECTS_DIR, name)
            if os.path.isdir(pd):
                projects.append({"name": name, "path": pd})
        return projects