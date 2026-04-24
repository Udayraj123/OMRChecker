import json
import argparse
from pathlib import Path
import cv2
import numpy as np

# Ensure project root on sys.path then try normal import, else provide fallback.
import sys, json
from pathlib import Path as _P

PROJECT_ROOT = _P(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.utils.parsing import open_template_with_defaults  # type: ignore
except Exception:
    def open_template_with_defaults(p: _P):
        try:
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise SystemExit(f"Template file not found: {p}")
        except Exception as e:
            print(f"Warning: could not parse template {p}: {e}")
            data = {}
        # Minimal defaults expected by editor
        data.setdefault("fieldBlocks", {})
        return data

# Optional: if resize util exists
try:
    from src.utils.image import ImageUtils
except Exception:
    class ImageUtils:
        @staticmethod
        def resize_util(img, target_w):
            h, w = img.shape[:2]
            if w == 0:
                return img
            scale = target_w / float(w)
            return cv2.resize(img, (int(w * scale), int(h * scale)))

HANDLE_SIZE = 8
HANDLE_COLOR = (60, 255, 255)

PRESET_BUBBLE_SETS = [
    ["A", "B", "C", "D"],
    ["A", "B", "C", "D", "E"],
    [str(i) for i in range(1, 6)],
    [str(i) for i in range(10)],
]

class SimpleTemplateEditor:
    """
    Enhanced OpenCV-based template editor.
    Interactions:
      - Left drag empty area: create new block
      - Click block: select
      - Drag inside selected block: move
      - Drag corner/edge handle: resize (updates stored bubblesGap / labelsGap)
    Keys:
      s save, q/ESC quit, d delete
      + / - add/remove bubbleValues entry
      h toggle direction
      [ / ] decrease / increase bubbleSpacing
      b cycle preset bubble sets
    Trackbars (per selected block):
      BubbleW, BubbleH, Spacing
    """
    def __init__(self, template_path: Path, image_path: Path):
        self.template_path = Path(template_path)
        self.template = open_template_with_defaults(self.template_path)
        self.img_orig = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
        if self.img_orig is None:
            raise SystemExit(f"Cannot load image: {image_path}")
        # Keep original pageDimensions if present
        page_dims = self.template.get("pageDimensions")
        target_w = page_dims[0] if page_dims else self.img_orig.shape[1]
        self.base_img = ImageUtils.resize_util(self.img_orig, target_w)
        self.winname = "OMR Template Editor"
        self.mode = "idle"  # creating | moving | resizing
        self.resize_handle = None
        self.drag_start = None
        self.current_rect = None  # (x,y,w,h) while creating
        self.selected_idx = None
        # list of (name, dict)
        self.field_blocks = list(self.template.get("fieldBlocks", {}).items())
        cv2.namedWindow(self.winname, cv2.WINDOW_NORMAL)
        cv2.setMouseCallback(self.winname, self.on_mouse)
        self._create_trackbars()

    # ---------- Trackbars ----------
    def _create_trackbars(self):
        def nothing(v): ...
        cv2.createTrackbar("BubbleW", self.winname, 20, 150, nothing)
        cv2.createTrackbar("BubbleH", self.winname, 20, 150, nothing)
        cv2.createTrackbar("Spacing", self.winname, 12, 200, nothing)

    def _sync_trackbars_from_block(self):
        if self.selected_idx is None:
            return
        _, fb = self.field_blocks[self.selected_idx]
        bw, bh = fb.get("bubbleDimensions", [20, 20])
        spacing = fb.get("bubbleSpacing", 12)
        cv2.setTrackbarPos("BubbleW", self.winname, int(bw))
        cv2.setTrackbarPos("BubbleH", self.winname, int(bh))
        cv2.setTrackbarPos("Spacing", self.winname, int(spacing))

    def _apply_trackbars_to_block(self):
        if self.selected_idx is None:
            return
        _, fb = self.field_blocks[self.selected_idx]
        bw = cv2.getTrackbarPos("BubbleW", self.winname)
        bh = cv2.getTrackbarPos("BubbleH", self.winname)
        spacing = cv2.getTrackbarPos("Spacing", self.winname)
        bw = max(bw, 4)
        bh = max(bh, 4)
        spacing = max(spacing, 4)
        fb["bubbleDimensions"] = [bw, bh]
        fb["bubbleSpacing"] = spacing

    # ---------- Drawing ----------
    def draw(self):
        self._apply_trackbars_to_block()
        canvas = cv2.cvtColor(self.base_img.copy(), cv2.COLOR_GRAY2BGR)

        for i, (name, fb) in enumerate(self.field_blocks):
            x, y, w, h = self._rect_from_block(fb)
            rect_color = (0, 180, 0) if i != self.selected_idx else (0, 0, 255)
            cv2.rectangle(canvas, (x, y), (x + w, y + h), rect_color, 2)
            cv2.putText(canvas, name, (x + 4, max(12, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, rect_color, 1)
            # bubble preview
            self._draw_bubbles(canvas, fb, selected=(i == self.selected_idx))
            # handles if selected
            if i == self.selected_idx:
                for hx, hy in self._handles(x, y, w, h):
                    cv2.rectangle(canvas, (hx, hy), (hx + HANDLE_SIZE, hy + HANDLE_SIZE), HANDLE_COLOR, -1)

        if self.current_rect is not None:
            x, y, w, h = self.current_rect
            cv2.rectangle(canvas, (x, y), (x + w, y + h), (255, 200, 0), 1)

        # instructions overlay
        overlay = [
            "s:save  q:quit  d:delete  +/-:bubble count",
            "h:toggle dir  b:cycle bubble sets  []:spacing",
        ]
        for idx, line in enumerate(overlay):
            cv2.putText(canvas, line, (8, 20 + idx * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (40, 255, 255), 1)

        cv2.imshow(self.winname, canvas)

    def _draw_bubbles(self, canvas, fb, selected=False):
        x, y, w, h = self._rect_from_block(fb)
        vals = fb.get("bubbleValues") or []
        if not vals:
            return
        bw, bh = fb.get("bubbleDimensions", [20, 20])
        spacing = fb.get("bubbleSpacing", 12)
        direction = fb.get("direction", "horizontal")
        ox = x + 4
        oy = y + 4
        for idx, _ in enumerate(vals):
            cx = ox + (bw // 2)
            cy = oy + (bh // 2)
            color = (0, 255, 255) if selected else (200, 200, 0)
            cv2.circle(canvas, (cx, cy), int(min(bw, bh) / 2 - 2), color, 1)
            cv2.putText(canvas, str(idx + 1), (cx - 4, cy + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)
            if direction == "horizontal":
                ox += bw + spacing
                if ox + bw > x + w - 4:
                    break
            else:
                oy += bh + spacing
                if oy + bh > y + h - 4:
                    break

    # ---------- Geometry ----------
    def _rect_from_block(self, fb):
        origin = fb.get("origin", [0, 0])
        x, y = int(origin[0]), int(origin[1])
        # Using stored gaps as width/height fallback
        w = int(fb.get("bubblesGap", fb.get("width", 160)))
        h = int(fb.get("labelsGap", fb.get("height", 80)))
        # Guarantee min size
        w = max(w, 30)
        h = max(h, 30)
        return x, y, w, h

    def _update_block_from_rect(self, fb, x, y, w, h):
        fb["origin"] = [int(x), int(y)]
        fb["bubblesGap"] = int(w)
        fb["labelsGap"] = int(h)

    def _handles(self, x, y, w, h):
        return [
            (x - HANDLE_SIZE // 2, y - HANDLE_SIZE // 2),  # tl
            (x + w - HANDLE_SIZE // 2, y - HANDLE_SIZE // 2),  # tr
            (x - HANDLE_SIZE // 2, y + h - HANDLE_SIZE // 2),  # bl
            (x + w - HANDLE_SIZE // 2, y + h - HANDLE_SIZE // 2),  # br
            (x + w // 2 - HANDLE_SIZE // 2, y - HANDLE_SIZE // 2),  # top
            (x + w // 2 - HANDLE_SIZE // 2, y + h - HANDLE_SIZE // 2),  # bottom
            (x - HANDLE_SIZE // 2, y + h // 2 - HANDLE_SIZE // 2),  # left
            (x + w - HANDLE_SIZE // 2, y + h // 2 - HANDLE_SIZE // 2),  # right
        ]

    def _hit_handle(self, x, y, fb):
        rx, ry, rw, rh = self._rect_from_block(fb)
        for idx, (hx, hy) in enumerate(self._handles(rx, ry, rw, rh)):
            if hx <= x <= hx + HANDLE_SIZE and hy <= y <= hy + HANDLE_SIZE:
                return idx
        return None

    # ---------- Mouse ----------
    def on_mouse(self, evt, x, y, flags, param):
        if evt == cv2.EVENT_LBUTTONDOWN:
            if self.selected_idx is not None:
                name, fb = self.field_blocks[self.selected_idx]
                handle = self._hit_handle(x, y, fb)
                if handle is not None:
                    self.mode = "resizing"
                    self.resize_handle = handle
                    self.drag_start = (x, y, *self._rect_from_block(fb))
                    return
            idx = self._hit_block(x, y)
            if idx is not None:
                self.selected_idx = idx
                self._sync_trackbars_from_block()
                self.mode = "moving"
                self.drag_start = (x, y)
            else:
                self.mode = "creating"
                self.drag_start = (x, y)
                self.current_rect = (x, y, 0, 0)

        elif evt == cv2.EVENT_MOUSEMOVE:
            if self.mode == "creating" and self.drag_start:
                sx, sy = self.drag_start
                self.current_rect = (min(sx, x), min(sy, y), abs(x - sx), abs(y - sy))
            elif self.mode == "moving" and self.selected_idx is not None and self.drag_start:
                sx, sy = self.drag_start
                dx, dy = x - sx, y - sy
                name, fb = self.field_blocks[self.selected_idx]
                ox, oy, w, h = self._rect_from_block(fb)
                self._update_block_from_rect(fb, ox + dx, oy + dy, w, h)
                self.drag_start = (x, y)
            elif self.mode == "resizing" and self.resize_handle is not None and self.drag_start:
                self._perform_resize(x, y)

        elif evt == cv2.EVENT_LBUTTONUP:
            if self.mode == "creating" and self.current_rect:
                x0, y0, w, h = self.current_rect
                if w > 10 and h > 10:
                    new_name = self._new_block_name()
                    new_block = {
                        "origin": [int(x0), int(y0)],
                        "bubblesGap": int(w),
                        "labelsGap": int(h),
                        "fieldLabels": ["q1..1"],
                        "bubbleValues": ["A", "B", "C", "D"],
                        "direction": "horizontal",
                        "bubbleDimensions": [20, 20],
                        "bubbleSpacing": 12,
                    }
                    self.field_blocks.append((new_name, new_block))
                    self.selected_idx = len(self.field_blocks) - 1
                    self._sync_trackbars_from_block()
                self.current_rect = None
            self.mode = "idle"
            self.drag_start = None
            self.resize_handle = None

    def _perform_resize(self, x, y):
        if self.selected_idx is None:
            return
        name, fb = self.field_blocks[self.selected_idx]
        sx, sy, ox, oy, ow, oh = self.drag_start
        dx = x - sx
        dy = y - sy
        rx, ry, rw, rh = ox, oy, ow, oh
        h_idx = self.resize_handle
        # 0 tl,1 tr,2 bl,3 br,4 top,5 bottom,6 left,7 right
        if h_idx == 0:  # tl
            rx = ox + dx
            ry = oy + dy
            rw = ow - dx
            rh = oh - dy
        elif h_idx == 1:  # tr
            ry = oy + dy
            rw = ow + dx
            rh = oh - dy
        elif h_idx == 2:  # bl
            rx = ox + dx
            rw = ow - dx
            rh = oh + dy
        elif h_idx == 3:  # br
            rw = ow + dx
            rh = oh + dy
        elif h_idx == 4:  # top
            ry = oy + dy
            rh = oh - dy
        elif h_idx == 5:  # bottom
            rh = oh + dy
        elif h_idx == 6:  # left
            rx = ox + dx
            rw = ow - dx
        elif h_idx == 7:  # right
            rw = ow + dx
        rw = max(rw, 30)
        rh = max(rh, 30)
        self._update_block_from_rect(fb, rx, ry, rw, rh)

    def _hit_block(self, x, y):
        for i, (_, fb) in enumerate(self.field_blocks):
            rx, ry, rw, rh = self._rect_from_block(fb)
            if rx <= x <= rx + rw and ry <= y <= ry + rh:
                return i
        return None

    # ---------- Utilities ----------
    def _new_block_name(self):
        base = "FieldBlock"
        n = 1
        existing = {name for name, _ in self.field_blocks}
        while f"{base}_{n}" in existing:
            n += 1
        return f"{base}_{n}"

    def save(self, path: Path):
        fb_obj = {name: obj for name, obj in self.field_blocks}
        new_template = dict(self.template)
        new_template["fieldBlocks"] = fb_obj
        out = Path(path).with_name(Path(path).stem + ".edited.json")
        with open(out, "w") as f:
            json.dump(new_template, f, indent=2)
        print("Saved", out)

    def _cycle_bubble_set(self, fb):
        curr = fb.get("bubbleValues") or []
        for i, preset in enumerate(PRESET_BUBBLE_SETS):
            if curr == preset:
                fb["bubbleValues"] = PRESET_BUBBLE_SETS[(i + 1) % len(PRESET_BUBBLE_SETS)]
                return
        fb["bubbleValues"] = PRESET_BUBBLE_SETS[0]

    # ---------- Main loop ----------
    def run(self):
        while True:
            self.draw()
            key = cv2.waitKey(25) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("s"):
                self.save(self.template_path)
            elif key == ord("d") and self.selected_idx is not None:
                self.field_blocks.pop(self.selected_idx)
                self.selected_idx = None
            elif key == ord("+") and self.selected_idx is not None:
                _, fb = self.field_blocks[self.selected_idx]
                vals = fb.get("bubbleValues") or []
                if not vals:
                    vals = ["A", "B"]
                else:
                    vals.append(f"X{len(vals)+1}")
                fb["bubbleValues"] = vals
            elif key == ord("-") and self.selected_idx is not None:
                _, fb = self.field_blocks[self.selected_idx]
                vals = fb.get("bubbleValues") or []
                if len(vals) > 1:
                    vals.pop()
                fb["bubbleValues"] = vals
            elif key == ord("h") and self.selected_idx is not None:
                _, fb = self.field_blocks[self.selected_idx]
                fb["direction"] = "vertical" if fb.get("direction", "horizontal") == "horizontal" else "horizontal"
            elif key == ord("[") and self.selected_idx is not None:
                _, fb = self.field_blocks[self.selected_idx]
                fb["bubbleSpacing"] = max(4, fb.get("bubbleSpacing", 12) - 2)
            elif key == ord("]") and self.selected_idx is not None:
                _, fb = self.field_blocks[self.selected_idx]
                fb["bubbleSpacing"] = min(400, fb.get("bubbleSpacing", 12) + 2)
            elif key == ord("b") and self.selected_idx is not None:
                _, fb = self.field_blocks[self.selected_idx]
                self._cycle_bubble_set(fb)

        cv2.destroyAllWindows()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--template", required=True)
    ap.add_argument("--image", required=True)
    args = ap.parse_args()
    editor = SimpleTemplateEditor(Path(args.template), Path(args.image))
    editor.run()

if __name__ == "__main__":
    main()