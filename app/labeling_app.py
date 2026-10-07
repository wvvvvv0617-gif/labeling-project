from __future__ import annotations

import json
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
DEFAULT_CLASSES = [
    "object",
    "person",
    "car",
    "truck",
    "bus",
    "bicycle",
    "dog",
    "cat",
    "bird",
    "fruit",
    "plant",
]
YOLO_MODEL_VERSIONS = [
    "yolov5n",
    "yolov5s",
    "yolov5m",
    "yolov5l",
    "yolov5x",
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
        self.title("YOLO 라벨링 도구")
        self.geometry("1480x900")
        self.minsize(1200, 760)

        self.image_paths: List[str] = []
        self.loaded_folders: List[str] = []
        self.current_index: int = -1
        self.current_annotation: Dict = {}
        self.annotations: Dict[str, Dict] = {}
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
        self.epoch_var = tk.IntVar(value=50)
        self.imgsz_var = tk.IntVar(value=640)
        self.conf_var = tk.DoubleVar(value=0.25)
        self.status_var = tk.StringVar(value="준비 완료")

        self._configure_style()
        self._build_ui()

    def _configure_style(self) -> None:
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("Card.TFrame", background="#f7f9fc")
        style.configure("Header.TLabel", background="#f7f9fc", foreground="#1f2937", font=("맑은 고딕", 13, "bold"))
        style.configure("Panel.TFrame", background="#ffffff")
        style.configure("Primary.TButton", font=("맑은 고딕", 10, "bold"))
        style.configure("Secondary.TButton", font=("맑은 고딕", 10))
        style.configure("Status.TLabel", background="#ffffff", foreground="#1f6feb", font=("맑은 고딕", 10, "bold"))

    def _build_ui(self) -> None:
        self.configure(bg="#eef3ff")
        self.grid_rowconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=1)

        left_panel = ttk.Frame(self, padding=12, style="Card.TFrame")
        left_panel.grid(row=0, column=0, sticky="nsew")
        left_panel.grid_rowconfigure(1, weight=1)
        left_panel.grid_columnconfigure(0, weight=1)

        ttk.Label(left_panel, text="폴더 목록", style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 8))
        self.folder_listbox = tk.Listbox(left_panel, height=7, exportselection=False, bg="#ffffff", bd=0, highlightthickness=1, relief="solid")
        self.folder_listbox.grid(row=1, column=0, sticky="ew")

        ttk.Label(left_panel, text="이미지 목록", style="Header.TLabel").grid(row=2, column=0, sticky="w", pady=(12, 8))
        self.listbox = tk.Listbox(left_panel, height=22, selectmode=tk.SINGLE, exportselection=False, bg="#ffffff", bd=0, highlightthickness=1, relief="solid")
        self.listbox.grid(row=3, column=0, sticky="nsew")
        self.listbox.bind("<<ListboxSelect>>", self._on_image_selected)

        list_scroll = ttk.Scrollbar(left_panel, orient="vertical", command=self.listbox.yview)
        list_scroll.grid(row=3, column=1, sticky="ns")
        self.listbox.configure(yscrollcommand=list_scroll.set)

        action_frame = ttk.Frame(left_panel, style="Card.TFrame")
        action_frame.grid(row=4, column=0, sticky="ew", pady=(12, 0))
        action_frame.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ttk.Button(action_frame, text="폴더 불러오기", command=self._load_folders, style="Primary.TButton").grid(row=0, column=0, sticky="ew", padx=(0, 4))
        ttk.Button(action_frame, text="이미지 추가", command=self._add_images, style="Secondary.TButton").grid(row=0, column=1, sticky="ew", padx=4)
        ttk.Button(action_frame, text="선택 삭제", command=self._remove_selected, style="Secondary.TButton").grid(row=0, column=2, sticky="ew", padx=4)
        ttk.Button(action_frame, text="파일 삭제", command=self._delete_selected_file, style="Secondary.TButton").grid(row=0, column=3, sticky="ew", padx=(4, 0))

        center_panel = ttk.Frame(self, padding=(0, 12, 12, 12), style="Card.TFrame")
        center_panel.grid(row=0, column=1, sticky="nsew")
        center_panel.grid_rowconfigure(0, weight=1)
        center_panel.grid_columnconfigure(0, weight=1)

        toolbar = ttk.Frame(center_panel, style="Card.TFrame")
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 8))
        toolbar.grid_columnconfigure((0, 1, 2, 3), weight=1)

        ttk.Button(toolbar, text="이전", command=self._prev_image, style="Secondary.TButton").grid(row=0, column=0, sticky="ew", padx=(0, 6))
        ttk.Button(toolbar, text="다음", command=self._next_image, style="Secondary.TButton").grid(row=0, column=1, sticky="ew", padx=6)
        ttk.Button(toolbar, text="저장", command=self._save_annotation, style="Primary.TButton").grid(row=0, column=2, sticky="ew", padx=6)
        ttk.Button(toolbar, text="불러오기", command=self._load_annotation, style="Secondary.TButton").grid(row=0, column=3, sticky="ew", padx=(6, 0))

        canvas_panel = ttk.Frame(center_panel, style="Panel.TFrame")
        canvas_panel.grid(row=1, column=0, sticky="nsew")
        self.canvas = tk.Canvas(canvas_panel, bg="#eef3ff", width=820, height=620, highlightthickness=1, highlightbackground="#d7deee")
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self._on_canvas_press)
        self.canvas.bind("<B1-Motion>", self._on_canvas_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_canvas_release)

        self.status_label = ttk.Label(center_panel, textvariable=self.status_var, style="Status.TLabel")
        self.status_label.grid(row=2, column=0, sticky="w", pady=(8, 0))

        right_panel = ttk.Frame(self, padding=12, style="Card.TFrame")
        right_panel.grid(row=0, column=2, sticky="nsew")
        right_panel.grid_columnconfigure(0, weight=1)

        ttk.Label(right_panel, text="설정", style="Header.TLabel").grid(row=0, column=0, sticky="w", pady=(0, 10))

        ttk.Label(right_panel, text="클래스 이름").grid(row=1, column=0, sticky="w")
        self.class_combo = ttk.Combobox(right_panel, textvariable=self.class_name_var, values=DEFAULT_CLASSES, state="normal")
        self.class_combo.grid(row=2, column=0, sticky="ew")

        ttk.Label(right_panel, text="클래스 ID").grid(row=3, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(right_panel, from_=0, to=100, textvariable=self.class_id_var, width=8).grid(row=4, column=0, sticky="w")

        ttk.Label(right_panel, text="YOLO 모델").grid(row=5, column=0, sticky="w", pady=(12, 0))
        ttk.Combobox(right_panel, textvariable=self.model_var, values=YOLO_MODEL_VERSIONS, state="readonly").grid(row=6, column=0, sticky="ew")

        ttk.Label(right_panel, text="기수 방향").grid(row=7, column=0, sticky="w", pady=(12, 0))
        ttk.Combobox(right_panel, textvariable=self.direction_var, values=list(DIRECTION_TO_ARROW.keys()), state="readonly").grid(row=8, column=0, sticky="ew")

        ttk.Checkbutton(right_panel, text="방향 표시 활성화", variable=self.show_direction_var).grid(row=9, column=0, sticky="w", pady=(10, 0))

        ttk.Label(right_panel, text="학습 설정").grid(row=10, column=0, sticky="w", pady=(14, 0))
        ttk.Label(right_panel, text="에폭 수").grid(row=11, column=0, sticky="w")
        ttk.Spinbox(right_panel, from_=1, to=500, textvariable=self.epoch_var, width=8).grid(row=12, column=0, sticky="w")
        ttk.Label(right_panel, text="이미지 크기").grid(row=13, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(right_panel, from_=320, to=1280, textvariable=self.imgsz_var, width=8).grid(row=14, column=0, sticky="w")
        ttk.Label(right_panel, text="신뢰도").grid(row=15, column=0, sticky="w", pady=(8, 0))
        ttk.Spinbox(right_panel, from_=0.05, to=0.99, increment=0.05, textvariable=self.conf_var, width=8).grid(row=16, column=0, sticky="w")

        self.annotation_info = tk.StringVar(value="라벨 정보 없음")
        ttk.Label(right_panel, textvariable=self.annotation_info, wraplength=260, justify="left").grid(row=17, column=0, sticky="w", pady=(18, 12))

        action_buttons = [
            ("기존 라벨 불러오기", self._load_annotation),
            ("YOLO 라벨 저장", self._save_annotation),
            ("학습 시작", self._train_dataset),
            ("best.pt 자동 라벨링", self._auto_label_with_best_pt),
            ("라벨 내보내기", self._export_labels),
        ]
        for idx, (text, cmd) in enumerate(action_buttons, start=18):
            ttk.Button(right_panel, text=text, command=cmd, style="Primary.TButton").grid(row=idx, column=0, sticky="ew", pady=4)

    def _collect_images_in_folder(self, folder_path: str) -> List[str]:
        files: List[str] = []
        for entry in sorted(os.listdir(folder_path)):
            full_path = os.path.join(folder_path, entry)
            if os.path.isfile(full_path) and Path(entry).suffix.lower() in IMAGE_EXTENSIONS:
                files.append(full_path)
        return files

    def _refresh_image_list(self) -> None:
        self.listbox.delete(0, tk.END)
        for img_path in self.image_paths:
            self.listbox.insert(tk.END, os.path.basename(img_path))
        if self.image_paths:
            if self.current_index < 0:
                self.current_index = 0
            if self.current_index >= len(self.image_paths):
                self.current_index = len(self.image_paths) - 1
            self.listbox.selection_clear(0, tk.END)
            self.listbox.selection_set(self.current_index)
            self._show_current_image()
        else:
            self.canvas.delete("all")
            self.current_annotation = {}
            self.current_image_path = None
            self.annotation_info.set("라벨 정보 없음")

    def _load_folders(self) -> None:
        folders: List[str] = []
        while True:
            folder = filedialog.askdirectory(title="이미지 폴더 선택")
            if not folder:
                break
            folders.append(folder)
        if not folders:
            return

        for folder in folders:
            if folder not in self.loaded_folders:
                self.loaded_folders.append(folder)
                self.folder_listbox.insert(tk.END, os.path.basename(folder) or folder)
            for image_path in self._collect_images_in_folder(folder):
                if image_path not in self.image_paths:
                    self.image_paths.append(image_path)

        if not self.image_paths:
            messagebox.showwarning("주의", "선택한 폴더에 이미지가 없습니다.")
            return

        self.current_index = 0
        self._refresh_image_list()
        self.status_var.set(f"{len(self.image_paths)}개 이미지 로드 완료")

    def _add_images(self) -> None:
        selected = filedialog.askopenfilenames(title="이미지 선택", filetypes=[("이미지 파일", "*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff")])
        if not selected:
            return
        for path in selected:
            if path not in self.image_paths:
                self.image_paths.append(path)
        if self.current_index < 0:
            self.current_index = 0
        self._refresh_image_list()
        self.status_var.set(f"추가됨: {len(selected)}개")

    def _remove_selected(self) -> None:
        if not self.image_paths or self.current_index < 0:
            return
        self.image_paths.pop(self.current_index)
        if self.current_index >= len(self.image_paths):
            self.current_index = max(0, len(self.image_paths) - 1)
        self._refresh_image_list()
        self.status_var.set("선택 이미지 제거 완료")

    def _delete_selected_file(self) -> None:
        if not self.image_paths or self.current_index < 0:
            return
        target = self.image_paths[self.current_index]
        if not messagebox.askyesno("삭제 확인", f"{os.path.basename(target)} 파일을 실제로 삭제할까요?"):
            return
        try:
            if os.path.exists(target):
                os.remove(target)
            for extra in [self._annotation_file_path(target), self._label_file_path(target)]:
                if os.path.exists(extra):
                    os.remove(extra)
        except OSError:
            messagebox.showwarning("삭제 실패", "파일 삭제 중 문제가 발생했습니다.")
            return
        self._remove_selected()
        self.status_var.set("파일 삭제 완료")

    def _prev_image(self) -> None:
        if not self.image_paths:
            return
        self.current_index = max(0, self.current_index - 1)
        self._refresh_image_list()

    def _next_image(self) -> None:
        if not self.image_paths:
            return
        self.current_index = min(len(self.image_paths) - 1, self.current_index + 1)
        self._refresh_image_list()

    def _on_image_selected(self, event) -> None:
        selection = self.listbox.curselection()
        if not selection:
            return
        self.current_index = selection[0]
        self._show_current_image()

    def _show_current_image(self) -> None:
        if not self.image_paths or self.current_index < 0:
            return
        path = self.image_paths[self.current_index]
        self.current_image_path = path
        self._load_annotation_for_path(path)
        try:
            with Image.open(path) as image:
                self.preview_image = image.convert("RGBA")
                self.current_image_size = self.preview_image.size
        except Exception:
            messagebox.showerror("읽기 오류", f"이미지를 불러오지 못했습니다: {path}")
            return
        self._draw_image_preview()

    def _draw_image_preview(self) -> None:
        self.canvas.delete("all")
        if self.preview_image is None:
            return
        canvas_w = self.canvas.winfo_width() or 820
        canvas_h = self.canvas.winfo_height() or 620
        img_w, img_h = self.preview_image.size
        scale = min(canvas_w / max(img_w, 1), canvas_h / max(img_h, 1))
        target_w = max(1, int(img_w * scale))
        target_h = max(1, int(img_h * scale))
        resized = self.preview_image.resize((target_w, target_h), Image.Resampling.LANCZOS)
        self.preview_photo = ImageTk.PhotoImage(resized)
        x0 = (canvas_w - target_w) // 2
        y0 = (canvas_h - target_h) // 2
        self.canvas.create_image(x0, y0, anchor="nw", image=self.preview_photo)
        self.canvas.image = self.preview_photo
        self._redraw_annotation_overlay(x0, y0)

    def _redraw_annotation_overlay(self, x0: int, y0: int) -> None:
        label = self.current_annotation.get("bbox")
        if not label:
            self.annotation_info.set("라벨 정보 없음")
            return
        x1, y1, x2, y2 = label
        self.canvas.create_rectangle(x0 + x1, y0 + y1, x0 + x2, y0 + y2, outline="#2ecc71", width=2, fill="")
        direction = self.current_annotation.get("direction", self.direction_var.get())
        if self.current_annotation.get("show_direction", True):
            arrow = DIRECTION_TO_ARROW.get(direction, "↑")
            cx = x0 + (x1 + x2) / 2
            cy = y0 + (y1 + y2) / 2
            self.canvas.create_text(cx, cy, text=arrow, fill="#ff4d4d", font=("맑은 고딕", 18, "bold"))
        self.annotation_info.set(f"bbox: ({x1:.0f}, {y1:.0f}, {x2:.0f}, {y2:.0f}) | 방향: {direction}")

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
        self.canvas.create_rectangle(x1, y1, x2, y2, outline="#ff9600", width=2, tags="temp_rect")

    def _on_canvas_release(self, event) -> None:
        if self.preview_image is None or self.drag_start is None:
            return
        x1, y1 = self.drag_start
        x2, y2 = self.drag_end if self.drag_end is not None else self.drag_start
        min_x, max_x = sorted((x1, x2))
        min_y, max_y = sorted((y1, y2))

        canvas_w = self.canvas.winfo_width() or 820
        canvas_h = self.canvas.winfo_height() or 620
        img_w, img_h = self.preview_image.size
        scale = min(canvas_w / max(img_w, 1), canvas_h / max(img_h, 1))
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
            "class_name": self.class_name_var.get(),
            "class_id": int(self.class_id_var.get()),
        }
        self.drag_start = None
        self.drag_end = None
        self.canvas.delete("temp_rect")
        self._save_annotation()
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
            messagebox.showwarning("경고", "이미지를 먼저 선택해 주세요.")
            return

        image_path = self.current_image_path
        bbox = self.current_annotation.get("bbox", [0, 0, 0, 0])
        annotation = {
            "bbox": bbox,
            "direction": self.direction_var.get(),
            "show_direction": self.show_direction_var.get(),
            "class_name": self.class_name_var.get(),
            "class_id": int(self.class_id_var.get()),
            "image_path": image_path,
        }
        self.current_annotation = annotation
        self.annotations[image_path] = annotation

        with open(self._annotation_file_path(image_path), "w", encoding="utf-8") as f:
            json.dump(annotation, f, ensure_ascii=False, indent=2)

        label_path = self._label_file_path(image_path)
        if len(bbox) == 4:
            x1, y1, x2, y2 = bbox
            img_w, img_h = self.current_image_size
            cx = ((x1 + x2) / 2) / max(img_w, 1)
            cy = ((y1 + y2) / 2) / max(img_h, 1)
            width = abs(x2 - x1) / max(img_w, 1)
            height = abs(y2 - y1) / max(img_h, 1)
            with open(label_path, "w", encoding="utf-8") as f:
                f.write(f"{int(self.class_id_var.get())} {cx:.6f} {cy:.6f} {width:.6f} {height:.6f}\n")

        self.status_var.set(f"저장 완료: {os.path.basename(image_path)}")
        self.annotation_info.set(f"클래스: {self.class_name_var.get()} | 방향: {annotation['direction']}")

    def _load_annotation(self) -> None:
        if self.current_image_path is None:
            messagebox.showwarning("경고", "이미지를 먼저 선택해 주세요.")
            return
        self._load_annotation_for_path(self.current_image_path)
        self._draw_image_preview()
        self.status_var.set("라벨 불러오기 완료")

    def _train_dataset(self) -> None:
        if not self.image_paths:
            messagebox.showwarning("주의", "이미지를 먼저 불러와 주세요.")
            return
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError:
            messagebox.showerror("의존성 오류", "ultralytics가 설치되어 있지 않습니다. requirements.txt를 설치해 주세요.")
            return

        dataset_dir = Path.cwd() / "datasets" / "labeling_dataset"
        images_dir = dataset_dir / "images"
        labels_dir = dataset_dir / "labels"
        images_dir.mkdir(parents=True, exist_ok=True)
        labels_dir.mkdir(parents=True, exist_ok=True)

        for image_path in self.image_paths:
            src = Path(image_path)
            target = images_dir / src.name
            if not target.exists():
                target.write_bytes(src.read_bytes())
            txt_path = Path(self._label_file_path(str(src)))
            if txt_path.exists():
                (labels_dir / (src.stem + ".txt")).write_text(txt_path.read_text(encoding="utf-8"), encoding="utf-8")

        yaml_path = dataset_dir / "data.yaml"
        yaml_path.write_text(
            "train: ./datasets/labeling_dataset/images\n"
            "val: ./datasets/labeling_dataset/images\n"
            "nc: 1\n"
            f"names: ['{self.class_name_var.get()}']\n",
            encoding="utf-8",
        )

        model_name = self.model_var.get()
        self.status_var.set(f"{model_name} 학습 시작")
        model = YOLO(f"{model_name}.pt")

        def run_training() -> None:
            model.train(
                data=str(yaml_path),
                epochs=int(self.epoch_var.get()),
                imgsz=int(self.imgsz_var.get()),
                project="runs",
                name="labeling_run",
                exist_ok=True,
                conf=float(self.conf_var.get()),
            )
            self.status_var.set("학습 완료")

        thread = threading.Thread(target=run_training, daemon=True)
        thread.start()
        messagebox.showinfo("학습 시작", f"{model_name} 모델로 학습을 시작합니다.\n결과는 runs 폴더에 저장됩니다.")

    def _auto_label_with_best_pt(self) -> None:
        try:
            from ultralytics import YOLO
        except ModuleNotFoundError:
            messagebox.showerror("의존성 오류", "ultralytics가 설치되어 있지 않습니다.")
            return

        best_path = filedialog.askopenfilename(title="best.pt 선택", filetypes=[("YOLO 모델", "*.pt")])
        if not best_path:
            return

        model = YOLO(best_path)
        for image_path in self.image_paths:
            results = model(image_path, imgsz=int(self.imgsz_var.get()), conf=float(self.conf_var.get()))
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
                    "class_name": self.class_name_var.get(),
                    "class_id": int(self.class_id_var.get()),
                }
                with open(self._annotation_file_path(image_path), "w", encoding="utf-8") as f:
                    json.dump(self.current_annotation, f, ensure_ascii=False, indent=2)

        self.status_var.set("자동 라벨링 완료")
        messagebox.showinfo("완료", "best.pt 기반 자동 라벨링이 완료되었습니다.")

    def _export_labels(self) -> None:
        if not self.image_paths:
            messagebox.showwarning("주의", "이미지를 먼저 불러와 주세요.")
            return
        export_dir = filedialog.askdirectory(title="라벨 저장 위치 선택")
        if not export_dir:
            return
        for image_path in self.image_paths:
            src = Path(image_path)
            txt_path = Path(self._label_file_path(str(src)))
            if txt_path.exists():
                target = Path(export_dir) / (src.stem + ".txt")
                target.write_text(txt_path.read_text(encoding="utf-8"), encoding="utf-8")
        self.status_var.set(f"라벨 내보내기 완료: {export_dir}")
        messagebox.showinfo("완료", f"라벨 파일이 {export_dir}로 내보내졌습니다.")


def main() -> None:
    app = LabelingApp()
    app.mainloop()
