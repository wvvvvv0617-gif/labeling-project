from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageDraw, ImageTk


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_CLASSES = ["object", "person", "car", "truck", "bus", "bicycle", "dog", "cat", "bird", "fruit", "plant"]
YOLO_MODEL_VERSIONS = [
    "yolov3",
    "yolov3-spp",
    "yolov4",
    "yolov5n",
    "yolov5s",
    "yolov5m",
    "yolov5l",
    "yolov5x",
    "yolov6n",
    "yolov6s",
    "yolov6m",
    "yolov6l",
    "yolov7",
    "yolov7-tiny",
    "yolov8n",
    "yolov8s",
    "yolov8m",
    "yolov8l",
    "yolov8x",
    "yolov9t",
    "yolov9s",
    "yolov9m",
    "yolov9c",
    "yolov9e",
    "yolov10n",
    "yolov10s",
    "yolov10m",
    "yolov10b",
    "yolov10l",
    "yolov10x",
    "yolov11n",
    "yolov11s",
    "yolov11m",
    "yolov11l",
    "yolov11x",
]
DIRECTION_TO_ARROW = {
    "N": "↑",
    "NE": "↗",
    "E": "→",
    "SE": "↘",
    "S": "↓",
    "SW": "↙",
    "W": "←",
    "NW": "↖",
}


class LabelingApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("YOLO Labeling Project")
        self.geometry("1400x860")
        self.minsize(1100, 700)

        self.image_paths: List[str] = []
        self.loaded_folders: List[str] = []
        self.current_index: int = 0
        self.current_annotation: Dict = {}
        self.drag_start: Optional[Tuple[int, int]] = None
        self.drag_end: Optional[Tuple[int, int]] = None
        self.preview_image: Optional[Image.Image] = None
        self.preview_photo: Optional[ImageTk.PhotoImage] = None
        self.current_image_path: Optional[str] = None
        self.current_image_size: Tuple[int, int] = (0, 0)

        self.model_var = tk.StringVar(value="yolov8n")
        self.class_name_var = tk.StringVar(value="object")
        self.class_id_var = tk.IntVar(value=0)
        self.direction_var = tk.StringVar(value="N")
        self.show_direction_var = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self) -> None:
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        left_panel = ttk.Frame(self, padding=10)
        left_panel.grid(row=0, column=0, sticky="nsew")
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        title = ttk.Label(left_panel, text="Loaded images", font=("Segoe UI", 11, "bold"))
        title.grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.listbox = tk.Listbox(left_panel, height=25, selectmode=tk.SINGLE, exportselection=False)
        self.listbox.grid(row=1, column=0, sticky="nsew")
        self.listbox.bind("<<ListboxSelect>>", self._on_image_selected)

        list_scroll = ttk.Scrollbar(left_panel, orient="vertical", command=self.listbox.yview)
        list_scroll.grid(row=1, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=list_scroll.set)

        action_frame = ttk.Frame(left_panel)
        action_frame.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(8, 0))
        action_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ttk.Button(action_frame, text="Load folders", command=self._load_folders).grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(action_frame, text="Add images", command=self._add_images).grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(action_frame, text="Remove selected", command=self._remove_selected).grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(action_frame, text="Delete file", command=self._delete_selected_file).grid(row=0, column=3, sticky="ew", padx=(4, 0))

        center_panel = ttk.Frame(self, padding=(0, 10, 10, 10))
        center_panel.grid(row=0, column=1, sticky="nsew")
        center_panel.grid_rowconfigure(0, weight=1)
        center_panel.grid_columnconfigure(0, weight=1)

        canvas_panel = ttk.Frame(center_panel)
        canvas_panel.grid(row=0, column=0, sticky="nsew")
        self.canvas = tk.Canvas(canvas_panel, bg="#ebebeb", width=700, height=650, highlightthickness=1, highlightbackground="#c0c0c0")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        right_panel = ttk.Frame(self, padding=10)
        right_panel.grid(row=0, column=2, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)

        ttk.Label(right_panel, text="Annotation settings", font=("Segoe UI", 12, "bold")).grid(row=0, column=0, sticky="w", pady=(0, 10))

        ttk.Label(right_panel, text="Object class name").grid(row=1, column=0, sticky="w")
        class_combo = ttk.Combobox(right_panel, textvariable=self.class_name_var, values=DEFAULT_CLASSES, state="normal")
        class_combo.grid(row=2, column=0, sticky="ew")
        ttk.Label(right_panel, text="Class ID").grid(row=3, column=0, sticky="w", pady=(8, 0))
        class_id_spin = ttk.Spinbox(right_panel, from_=0, to=100, textvariable=self.class_id_var, width=10)
        class_id_spin.grid(row=4, column=0, sticky="w")

        ttk.Label(right_panel, text="Model version").grid(row=5, column=0, sticky="w", pady=(12, 0))
        model_combo = ttk.Combobox(right_panel, textvariable=self.model_var, values=YOLO_MODEL_VERSIONS, state="readonly")
        model_combo.grid(row=6, column=0, sticky="ew")

        ttk.Label(right_panel, text="Direction").grid(row=7, column=0, sticky="w", pady=(12, 0))
        direction_combo = ttk.Combobox(right_panel, textvariable=self.direction_var, values=list(DIRECTION_TO_ARROW.keys()), state="readonly")
        direction_combo.grid(row=8, column=0, sticky="ew")

        ttk.Checkbutton(right_panel, text="Show direction overlay", variable=self.show_direction_var).grid(row=9, column=0, sticky="w", pady=(10, 0))

        self.annotation_info = tk.StringVar(value="No annotation yet")
        ttk.Label(right_panel, textvariable=self.annotation_info, wraplength=240, justify="left").grid(row=10, column=0, sticky="w", pady=(18, 10))

        btns = [
            ("Save annotation", self._save_annotation),
            ("Load annotation", self._load_annotation),
            ("Train dataset", self._train_dataset),
            ("Auto label (best.pt)", self._auto_label_with_best_pt),
            ("Export YOLO labels", self._export_labels),
        ]
        for row, (text, command) in enumerate(btns, start=11):
            ttk.Button(right_panel, text=text, command=command).grid(row=row, column=0, sticky="ew", pady=4)

        ttk.Separator(right_panel, orient="horizontal").grid(row=row + 1, column=0, sticky="ew", pady=(12, 10))

        folder_label = ttk.Label(right_panel, text="Loaded folders")
        folder_label.grid(row=row + 2, column=0, sticky="w")
        self.folder_listbox = tk.Listbox(right_panel, height=8, exportselection=False)
        self.folder_listbox.grid(row=row + 3, column=0, sticky="nsew")

    def _collect_images_in_folder(self, folder_path: str) -> List[str]:
        files: List[str] = []
        for entry in os.listdir(folder_path):
            full_path = os.path.join(folder_path, entry)
            if os.path.isfile(full_path) and Path(entry).suffix.lower() in IMAGE_EXTENSIONS:
                files.append(full_path)
        return sorted(files)

    def _load_folders(self) -> None:
        folders: List[str] = []
        while True:
            folder = filedialog.askdirectory(title="Select image folder")
            if not folder:
                break
            folders.append(folder)
        if not folders:
            return

        for folder in folders:
            if folder not in self.loaded_folders:
                self.loaded_folders.append(folder)
                self.folder_listbox.insert(tk.END, os.path.basename(folder) or folder)
            images = self._collect_images_in_folder(folder)
            for image in images:
                if image not in self.image_paths:
                    self.image_paths.append(image)

        if not self.image_paths:
            messagebox.showwarning("No images", "선택한 폴더에 이미지가 없습니다.")
            return

        self.listbox.delete(0, tk.END)
        for img_path in self.image_paths:
            self.listbox.insert(tk.END, os.path.basename(img_path))
        self.current_index = 0
        self._show_current_image()

    def _add_images(self) -> None:
        selected = filedialog.askopenfilenames(title="Select images", filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff")])
        if not selected:
            return

        for path in selected:
            if path not in self.image_paths:
                self.image_paths.append(path)

        self.listbox.delete(0, tk.END)
        for img_path in self.image_paths:
            self.listbox.insert(tk.END, os.path.basename(img_path))

        if self.image_paths:
            self.current_index = len(self.image_paths) - 1
            self._show_current_image()

    def _remove_selected(self) -> None:
        if not self.image_paths:
            return

        current_path = self.image_paths[self.current_index]
        self.image_paths.pop(self.current_index)
        self.listbox.delete(self.current_index)
        if not self.image_paths:
            self.canvas.delete("all")
            self.current_annotation = {}
            self.annotation_info.set("No annotation yet")
            self.current_image_path = None
            return

        if self.current_index >= len(self.image_paths):
            self.current_index = len(self.image_paths) - 1
        self._show_current_image()

        if current_path in self.current_annotation:
            self.current_annotation.pop(current_path, None)

    def _delete_selected_file(self) -> None:
        if not self.image_paths:
            return

        target_path = self.image_paths[self.current_index]
        confirm = messagebox.askyesno("Delete image", f"{os.path.basename(target_path)}를 실제 파일에서 삭제할까요?")
        if not confirm:
            return

        try:
            if os.path.exists(target_path):
                os.remove(target_path)
                json_path = self._annotation_file_path(target_path)
                txt_path = self._label_file_path(target_path)
                for path in [json_path, txt_path]:
                    if os.path.exists(path):
                        os.remove(path)
        except OSError:
            messagebox.showwarning("Delete failed", "파일 삭제에 실패했습니다.")
            return

        self._remove_selected()

    def _on_image_selected(self, event) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        self.current_index = selection[0]
        self._show_current_image()

    def _show_current_image(self) -> None:
        if not self.image_paths:
            return

        path = self.image_paths[self.current_index]
        self.current_image_path = path
        self._load_annotation_for_path(path)

        with Image.open(path) as image:
            self.preview_image = image.convert("RGBA")
            self.current_image_size = self.preview_image.size

        self._draw_image_preview()

    def _draw_image_preview(self) -> None:
        self.canvas.delete("all")
        if self.preview_image is None:
            return

        canvas_w = self.canvas.winfo_width() or 700
        canvas_h = self.canvas.winfo_height() or 650
        img_w, img_h = self.preview_image.size

        scale = min(canvas_w / img_w, canvas_h / img_h)
        target_w = max(1, int(img_w * scale))
        target_h = max(1, int(img_h * scale))
        resized = self.preview_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(resized)

        x0 = (canvas_w - target_w) // 2
        y0 = (canvas_h - target_h) // 2
        self.canvas.create_image(x0, y0, anchor="nw", image=self.preview_photo)
        self.canvas.image = self.preview_photo

        self._redraw_annotation_overlay(x0, y0, target_w, target_h)

    def _redraw_annotation_overlay(self, x0: int, y0: int, w: int, h: int) -> None:
        label = self.current_annotation.get("bbox")
        if not label:
            self.annotation_info.set("No annotation yet")
            return

        x1, y1, x2, y2 = label
        draw = self.canvas.create_rectangle(
            x0 + x1, y0 + y1, x0 + x2, y0 + y2,
            outline="#2ecc71", width=2, fill=""
        )
        self.canvas.tag_raise(draw)

        direction = self.current_annotation.get("direction", self.direction_var.get())
        show_direction = self.current_annotation.get("show_direction", True)
        if show_direction:
            arrow = DIRECTION_TO_ARROW.get(direction, "↑")
            cx = x0 + (x1 + x2) / 2
            cy = y0 + (y1 + y2) / 2
            self.canvas.create_text(cx, cy, text=arrow, fill="#ff4d4d", font=("Segoe UI", 18, "bold"))

        self.annotation_info.set(
            f"bbox: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}) | direction: {direction}"
        )

    def _get_annotation_for_current(self) -> Dict:
        return self.current_annotation

    def _on_canvas_press(self, event) -> None:
        if self.preview_image is None:
            return
        self.drag_start = (event.x, event.y)
        self.drag_end = None

    def _on_canvas_drag(self, event) -> None:
        if self.preview_image is None or self.drag_start is None:
            return
        self.drag_end = (event.x, event.y)
        self._draw_rect_preview()

    def _draw_rect_preview(self) -> None:
        if self.preview_image is None or self.drag_start is None:
            return
        self.canvas.delete("temp_rect")
        if self.drag_end is None:
            return

        x1, y1 = self.drag_start
        x2, y2 = self.drag_end
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="#ff9900", width=2, tags="temp_rect")

    def _on_canvas_release(self, event) -> None:
        if self.preview_image is None or self.drag_start is None:
            return

        if self.drag_end is None:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_start
        else:
            x1, y1 = self.drag_start
            x2, y2 = self.drag_end

        min_x, max_x = sorted((x1, x2))
        min_y, max_y = sorted((y1, y2))

        canvas_w = self.canvas.winfo_width() or 700
        canvas_h = self.canvas.winfo_height() or 650
        img_w, img_h = self.preview_image.size
        scale = min(canvas_w / img_w, canvas_h / img_h)
        target_w = max(1, int(img_w * scale))
        target_h = max(1, int(img_h * scale))
        x_offset = (canvas_w - target_w) // 2
        y_offset = (canvas_h - target_h) // 2

        normalized_x1 = max(0, min_x - x_offset)
        normalized_y1 = max(0, min_y - y_offset)
        normalized_x2 = max(0, max_x - x_offset)
        normalized_y2 = max(0, max_y - y_offset)

        box_x1 = max(0, normalized_x1 / max(1, target_w) * img_w)
        box_y1 = max(0, normalized_y1 / max(1, target_h) * img_h)
        box_x2 = max(0, normalized_x2 / max(1, target_w) * img_w)
        box_y2 = max(0, normalized_y2 / max(1, target_h) * img_h)

        if box_x1 == box_x2 or box_y1 == box_y2:
            self.drag_start = None
            self.drag_end = None
            self.canvas.delete("temp_rect")
            return

        self.current_annotation = {
            "bbox": [box_x1, box_y1, box_x2, box_y2],
            "direction": self.direction_var.get(),
            "show_direction": self.show_direction_var.get(),
        }
        self._save_annotation()
        self.drag_start = None
        self.drag_end = None
        self.canvas.delete("temp_rect")
        self._draw_image_preview()

    def _annotation_file_path(self, image_path: str) -> str:
        return str(Path(image_path).with_suffix(".json"))

    def _label_file_path(self, image_path: str) -> str:
        return str(Path(image_path).with_suffix(".txt"))

    def _load_annotation_for_path(self, image_path: str) -> None:
        json_path = self._annotation_file_path(image_path)
        self.current_annotation = {}
        if os.path.exists(json_path):
            try:
                with open(json_path, "r", encoding="utf-8") as f:
                    self.current_annotation = json.load(f)
                self.direction_var.set(self.current_annotation.get("direction", "N"))
                self.show_direction_var.set(bool(self.current_annotation.get("show_direction", True)))
                self.class_name_var.set(self.current_annotation.get("class_name", "object"))
                self.class_id_var.set(int(self.current_annotation.get("class_id", 0)))
            except Exception:
                self.current_annotation = {}
        else:
            self.direction_var.set("N")
            self.show_direction_var.set(True)
            self.class_name_var.set("object")
            self.class_id_var.set(0)

    def _save_annotation(self) -> None:
        if self.current_image_path is None:
            messagebox.showwarning("No image", "이미지를 먼저 선택해 주세요.")
            return

        image_path = self.current_image_path
        annotation = {
            "bbox": self.current_annotation.get("bbox", [0, 0, 0, 0]),
            "direction": self.direction_var.get(),
            "show_direction": self.show_direction_var.get(),
            "class_name": self.class_name_var.get(),
            "class_id": int(self.class_id_var.get()),
            "image_path": image_path,
        }
        self.current_annotation = annotation

        with open(self._annotation_file_path(image_path), "w", encoding="utf-8") as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)

        label_path = self._label_file_path(image_path)
        bbox = annotation.get("bbox", [0, 0, 0, 0])
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            img_w, img_h = self.current_image_size
            cx = ((x1 + x2) / 2) / max(img_w, 1)
            cy = ((y1 + y2) / 2) / max(img_h, 1)
            width = abs(x2 - x1) / max(img_w, 1)
            height = abs(y2 - y1) / max(img_h, 1)
            class_id = int(self.class_id_var.get())
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(f"{class_id} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n")

        self.annotation_info.set(
            f"Saved: {os.path.basename(image_path)} | class: {self.class_name_var.get()} | dir: {annotation['direction']}"
        )

    def _load_annotation(self) -> None:
        if self.current_image_path is None:
            messagebox.showwarning("No image", "이미지를 먼저 선택해 주세요.")
            return
        self._load_annotation_for_path(self.current_image_path)
        self._draw_image_preview()

    def _train_dataset(self) -> None:
        if not self.image_paths:
            messagebox.showwarning("No data", "먼저 이미지를 불러와 주세요.")
            return

        try:
            from ultralytics import YOLO
        except ModuleNotFoundError:
            messagebox.showerror("Dependency missing", "ultralytics가 설치되지 않았습니다. requirements.txt를 설치해 주세요.")
            return

        data_dir = Path.cwd() / "datasets" / "yolo_labeling"
        data_dir.mkdir(parents=True, exist_ok=True)
        imgs_dir = data_dir / "images"
        labels_dir = data_dir / "labels"
        imgs_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for image_path in self.image_paths:
            src = Path(image_path)
            target = imgs_dir / src.name
            if not target.exists():
                target.write_bytes(src.read_bytes())

            txt_path = Path(self._label_file_path(str(src)))
            if txt_path.exists():
                target_label = labels_dir / (src.stem + ".txt")
                target_label.write_text(txt_path.read_text(encoding="utf-8"), encoding="utf-8")

        yaml_path = data_dir / "data.yaml"
        yaml_path.write_text(
            "train: ./datasets/yolo_labeling/images\n"
            "val: ./datasets/yolo_labeling/images\n"
            "nc: 1\n"
            "names: ['object']\n",
            encoding="utf-8",
        )

        model_name = self.model_var.get()
        model = YOLO(f"{model_name}.pt")
        train_thread = threading.Thread(target=self._run_training, args=(model, yaml_path), daemon=True)
        train_thread.start()
        messagebox.showinfo("Training started", f"{model_name} 기반 학습을 시작합니다.\n결과는 runs/detect 폴더에 저장됩니다.")

    def _run_training(self, model, yaml_path: Path) -> None:
        model.train(data=str(yaml_path), epochs=50, imgsz=640, project="runs", name="labeling_run", exist_ok=True)

    def _auto_label_with_best_pt(self) -> None:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError:
            messagebox.showerror("Dependency missing", "ultralytics가 설치되지 않았습니다.")
            return

        best_path = filedialog.askopenfilename(title="Select best.pt", filetypes=[("YOLO model", "*.pt")])
        if not best_path:
            return

        model = YOLO(best_path)
        for image_path in self.image_paths:
            results = model(image_path, imgsz=640, conf=0.25)
            result = results[0]
            if len(result.boxes) == 0:
                continue

            labels = []
            for box in result.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                img_w = result.orig_shape[1]
                img_h = result.orig_shape[0]
                cx = ((x1 + x2) / 2) / max(img_w, 1)
                cy = ((y1 + y2) / 2) / max(img_h, 1)
                w = abs(x2 - x1) / max(img_w, 1)
                h = abs(y2 - y1) / max(img_h, 1)
                labels.append(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")

            if labels:
                txt_path = Path(image_path).with_suffix(".txt")
                txt_path.write_text("\n".join(labels) + "\n", encoding="utf-8")
                self.current_annotation = {
                    "bbox": [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)],
                    "direction": self.direction_var.get(),
                    "show_direction": self.show_direction_var.get(),
                    "image_path": image_path,
                }
                json_path = self._annotation_file_path(image_path)
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(self.current_annotation, f, ensure_ascii=False, indent=2)

        messagebox.showinfo("Auto labeling complete", "best.pt 기반 자동 라벨링이 완료되었습니다.")

    def _export_labels(self) -> None:
        if not self.image_paths:
            messagebox.showwarning("No data", "이미지를 먼저 불러와 주세요.")
            return

        export_dir = filedialog.askdirectory(title="Choose export directory")
        if not export_dir:
            return

        for image_path in self.image_paths:
            src = Path(image_path)
            txt_path = Path(self._label_file_path(str(src)))
            if txt_path.exists():
                target = Path(export_dir) / (src.stem + ".txt")
                target.write_text(txt_path.read_text(encoding="utf-8"), encoding="utf-8")

        messagebox.showinfo("Export complete", f"라벨 파일이 {export_dir}로 내보내졌습니다.")


def main() -> None:
    app = LabelingApp()
    app.mainloop()
