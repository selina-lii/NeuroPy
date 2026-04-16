"""
PromptSpace — standalone work-management UI.
Tabs: Tickets (high/mid/low priority task lists) and Hypotheses (idea map).
"""
from __future__ import annotations

import json
import os
import shutil
import uuid
import random
import datetime
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from glob import glob
from typing import List, Dict, Optional

try:
    from PIL import Image, ImageTk
    _HAS_PIL = True
except ImportError:
    _HAS_PIL = False

# Resolve data dir relative to this file (NeuroPy root / data / prompts)
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # go up from neuropy/ui/ to NeuroPy/
_DATA_DIR = os.path.join(_ROOT, 'data', 'prompts')
_IMAGES_DIR = os.path.join(_ROOT, 'data', 'images', 'random')
_UNUSED_DIR = os.path.join(_IMAGES_DIR, 'unused')
_TICKETS_PATH = os.path.join(_DATA_DIR, 'tickets.json')
_HYPOTHESES_PATH = os.path.join(_DATA_DIR, 'hypotheses.json')

os.makedirs(_DATA_DIR, exist_ok=True)
os.makedirs(_IMAGES_DIR, exist_ok=True)
os.makedirs(_UNUSED_DIR, exist_ok=True)


def _random_color() -> str:
    """Generate a random pleasant hex color."""
    hues = [
        '#e63946', '#f4a261', '#2a9d8f', '#457b9d', '#9b5de5',
        '#f15bb5', '#00bbf9', '#00f5d4', '#fee440', '#fb5607',
        '#3a86ff', '#8338ec', '#06d6a0', '#ef476f', '#ffd166',
    ]
    return random.choice(hues)


def _lighten_color(hex_color: str, factor: float = 0.85) -> str:
    """Lighten a hex color towards white by factor (0=original, 1=white)."""
    try:
        h = hex_color.lstrip('#')
        r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
        r = int(r + (255 - r) * factor)
        g = int(g + (255 - g) * factor)
        b = int(b + (255 - b) * factor)
        return f'#{r:02x}{g:02x}{b:02x}'
    except Exception:
        return '#f0f0f0'


def _make_color_image(color: str, size: int = 28) -> Optional[object]:
    """Create a solid-color PhotoImage of given size."""
    if _HAS_PIL:
        img = Image.new('RGB', (size, size), color)
        return ImageTk.PhotoImage(img)
    return None


def _load_avatar(hyp_id: str, size: int = 28) -> Optional[object]:
    """Load avatar image for a hypothesis, falling back to None."""
    path = os.path.join(_IMAGES_DIR, f'{hyp_id}.png')
    if os.path.exists(path) and _HAS_PIL:
        try:
            img = Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception:
            pass
    return None


def _assign_avatar(hyp_id: str) -> bool:
    """Move a random unused avatar to the hypothesis's named slot. Returns True if successful."""
    unused = glob(os.path.join(_UNUSED_DIR, '*.png'))
    if not unused:
        return False
    chosen = random.choice(unused)
    dest = os.path.join(_IMAGES_DIR, f'{hyp_id}.png')
    try:
        shutil.move(chosen, dest)
        return True
    except Exception:
        return False


# ---------------------------------------------------------------------------
# TicketSection
# ---------------------------------------------------------------------------

class _TicketSection(ttk.Frame):
    """One priority section with a scrollable list of editable text items."""

    def __init__(self, parent, priority: str, save_cb, settings_getter=None):
        super().__init__(parent)
        self._priority = priority
        self._save_cb = save_cb
        self._settings_getter = settings_getter  # callable → str prefix, or None
        self._items: List[dict] = []   # list of {'text_widget': ..., 'frame': ...}

        top_bar = ttk.Frame(self)
        top_bar.pack(fill=tk.X, pady=(2, 1))
        ttk.Button(top_bar, text="+ Add item", command=self._add_item).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_bar, text="Copy all",
                   command=self._copy_and_archive).pack(side=tk.LEFT, padx=2)

        # Scrollable area
        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)
        self._canvas = tk.Canvas(outer, highlightthickness=0, bg='white')
        sb = ttk.Scrollbar(outer, orient='vertical', command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._scroll_frame = ttk.Frame(self._canvas)
        self._win = self._canvas.create_window((0, 0), window=self._scroll_frame, anchor='nw')
        self._scroll_frame.bind('<Configure>', self._on_frame_configure)
        self._canvas.bind('<Configure>', self._on_canvas_configure)
        # Trackpad two-finger scroll
        self._canvas.bind('<MouseWheel>', self._on_scroll)
        self._canvas.bind('<Button-4>', self._on_scroll)
        self._canvas.bind('<Button-5>', self._on_scroll)

    def _on_frame_configure(self, _e=None):
        self._canvas.configure(scrollregion=self._canvas.bbox('all'))

    def _on_canvas_configure(self, e):
        self._canvas.itemconfig(self._win, width=e.width)

    def _on_scroll(self, event):
        if event.num == 4:
            self._canvas.yview_scroll(-1, 'units')
        elif event.num == 5:
            self._canvas.yview_scroll(1, 'units')
        else:
            self._canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

    def _add_item(self, text: str = ''):
        """Add a new ticket item (optionally pre-filled with text)."""
        row = tk.Frame(self._scroll_frame, relief='groove', borderwidth=2,
                       bg='white', padx=2, pady=2)
        row.pack(fill=tk.X, pady=2, padx=4)
        txt = tk.Text(row, height=3, wrap=tk.WORD, font=('TkDefaultFont', 9), bg='white')
        txt.pack(side=tk.LEFT, fill=tk.X, expand=True)
        if text:
            txt.insert('1.0', text)
        txt.bind('<KeyRelease>', lambda e: self._save_cb())
        btn = ttk.Button(row, text='x', width=2,
                         command=lambda r=row, t=txt: self._delete_item(r, t))
        btn.pack(side=tk.RIGHT, padx=2)
        self._items.append({'frame': row, 'text_widget': txt})
        self._on_frame_configure()
        self._save_cb()

    def _delete_item(self, frame, txt):
        self._items = [i for i in self._items if i['text_widget'] is not txt]
        frame.destroy()
        self._on_frame_configure()
        self._save_cb()

    def _copy_and_archive(self):
        texts = [i['text_widget'].get('1.0', tk.END).strip()
                 for i in self._items if i['text_widget'].get('1.0', tk.END).strip()]
        if not texts:
            messagebox.showinfo("Copy all", "No items to copy.")
            return
        formatted = '\n'.join(f"{i}. {line}" for i, line in enumerate(texts, 1))
        settings_prefix = self._settings_getter() if self._settings_getter else ''
        full_text = (settings_prefix.rstrip() + '\n\n' + formatted) if settings_prefix.strip() else formatted
        self.clipboard_clear()
        self.clipboard_append(full_text)
        # Archive
        hist_path = os.path.join(_DATA_DIR, f'history_{self._priority}.txt')
        with open(hist_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- {datetime.datetime.now().isoformat()} ---\n")
            f.write(full_text + '\n')
        # Clear items
        for item in list(self._items):
            self._delete_item(item['frame'], item['text_widget'])
        # Restore white background
        self._canvas.configure(bg='white')
        messagebox.showinfo("Copied", f"{len(texts)} item(s) copied and archived.")

    def get_texts(self) -> List[str]:
        return [i['text_widget'].get('1.0', tk.END).strip() for i in self._items]

    def load_texts(self, texts: List[str]):
        # Clear existing
        for item in list(self._items):
            item['frame'].destroy()
        self._items.clear()
        for text in texts:
            self._add_item(text)


# ---------------------------------------------------------------------------
# HypothesisCard
# ---------------------------------------------------------------------------

class _HypothesisCard(tk.Frame):
    """A card widget representing a single hypothesis."""

    def __init__(self, parent, data: dict, on_edit, on_delete, on_copy, on_select):
        bg = _lighten_color(data['color'], 0.88)
        super().__init__(parent, bg=bg, pady=2, padx=2, relief='flat')
        self._data = data
        self._photo = None

        # Left color strip
        strip = tk.Frame(self, width=5, bg=data['color'])
        strip.pack(side=tk.LEFT, fill=tk.Y)

        # Avatar — circular canvas
        avatar_canvas = tk.Canvas(self, bg=bg, width=32, height=32,
                                  highlightthickness=0)
        avatar_canvas.pack(side=tk.LEFT, padx=4)
        avatar_canvas.create_oval(2, 2, 30, 30, fill=data['color'], outline='white', width=1.5)
        # Initials inside circle
        initials = data['title'][:1].upper() if data.get('title') else '?'
        avatar_canvas.create_text(16, 16, text=initials, fill='white',
                                  font=('TkDefaultFont', 10, 'bold'))

        # Text content
        content = tk.Frame(self, bg=bg)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        title_lbl = tk.Label(content, text=data['title'], bg=bg,
                             font=('TkDefaultFont', 10, 'bold'), anchor='w')
        title_lbl.pack(fill=tk.X)
        text_lbl = tk.Label(content, text=data.get('text', ''), bg=bg,
                            font=('TkDefaultFont', 9), anchor='w', justify=tk.LEFT,
                            wraplength=220)
        text_lbl.pack(fill=tk.X)

        # Buttons (no Edit — double-click to edit)
        btn_frame = tk.Frame(self, bg=bg)
        btn_frame.pack(side=tk.RIGHT, padx=2)
        ttk.Button(btn_frame, text='Copy', width=5,
                   command=lambda: on_copy(data)).pack(pady=1)
        ttk.Button(btn_frame, text='Del', width=5,
                   command=lambda: on_delete(data)).pack(pady=1)

        # Single-click: select; double-click on title/text: edit
        for w in [self, content, strip, avatar_canvas]:
            w.bind('<Button-1>', lambda e, d=data: on_select(d))
        title_lbl.bind('<Button-1>', lambda e, d=data: on_select(d))
        text_lbl.bind('<Button-1>', lambda e, d=data: on_select(d))
        title_lbl.bind('<Double-Button-1>', lambda e, d=data: on_edit(d))
        text_lbl.bind('<Double-Button-1>', lambda e, d=data: on_edit(d))


# ---------------------------------------------------------------------------
# PromptSpace
# ---------------------------------------------------------------------------

class PromptSpace:
    """Standalone work-management window: Tickets + Hypotheses.

    Uses ``tk.Toplevel`` when a Tk root already exists (e.g. from another
    UI such as CCGReviewUI) and creates a new ``tk.Tk`` root otherwise.
    Calls ``mainloop()`` only when it owns the root, so the window stays
    open in Jupyter notebooks and standalone scripts alike.
    """

    def __init__(self):
        # Use Toplevel if a live Tk root already exists; create Tk() otherwise
        self._owns_mainloop = False
        existing = None
        try:
            existing = tk._default_root
            if existing is not None:
                existing.winfo_exists()  # throws TclError if destroyed
        except Exception:
            existing = None
        if existing is not None:
            self.root = tk.Toplevel(existing)
        else:
            self.root = tk.Tk()
            self._owns_mainloop = True
        self.root.title('PromptSpace')
        self.root.geometry('900x650')
        self._ticket_sections: Dict[str, _TicketSection] = {}
        self._hypotheses: List[dict] = []
        self._hyp_cards: Dict[str, _HypothesisCard] = {}
        self._canvas_dots: Dict[str, int] = {}   # hyp_id -> canvas item id
        self._selected_hyp_id: Optional[str] = None
        self._drag_hyp_id: Optional[str] = None
        self._drag_offset = (0, 0)
        self._card_frames: Dict[str, _HypothesisCard] = {}

        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        tickets_tab = ttk.Frame(nb)
        hyp_tab = ttk.Frame(nb)
        nb.add(tickets_tab, text='Tickets')
        nb.add(hyp_tab, text='Hypotheses')

        self._build_tickets_tab(tickets_tab)
        self._build_hypotheses_tab(hyp_tab)

        self._load_tickets()
        self._load_hypotheses()

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)

        # Raise window to front
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))

        if self._owns_mainloop:
            self.root.mainloop()

    # ------------------------------------------------------------------ Tickets

    def _build_tickets_tab(self, parent):
        sub_nb = ttk.Notebook(parent)
        sub_nb.pack(fill=tk.BOTH, expand=True)

        for priority in ('high', 'mid', 'low'):
            tab = ttk.Frame(sub_nb)
            sub_nb.add(tab, text=priority.capitalize() + ' Priority')
            sec = _TicketSection(tab, priority, self._save_tickets,
                                 settings_getter=lambda: self._settings_text.get('1.0', tk.END))
            sec.pack(fill=tk.BOTH, expand=True)
            self._ticket_sections[priority] = sec

        # Settings tab (append last)
        settings_tab = ttk.Frame(sub_nb)
        sub_nb.add(settings_tab, text='Settings')
        ttk.Label(settings_tab, text="Settings prefix (prepended to every 'Copy all'):",
                  font=('TkDefaultFont', 9)).pack(anchor='w', padx=6, pady=(6, 2))
        self._settings_text = tk.Text(settings_tab, height=10, wrap=tk.WORD,
                                      font=('TkDefaultFont', 9), bg='white')
        self._settings_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self._settings_text.bind('<KeyRelease>', lambda e: self._save_tickets())

    def _save_tickets(self):
        data = {p: sec.get_texts() for p, sec in self._ticket_sections.items()}
        data['__settings__'] = self._settings_text.get('1.0', tk.END).rstrip()
        try:
            with open(_TICKETS_PATH, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[PromptSpace] Could not save tickets: {e}")

    def _load_tickets(self):
        if not os.path.exists(_TICKETS_PATH):
            return
        try:
            with open(_TICKETS_PATH, 'r', encoding='utf-8') as f:
                data = json.load(f)
            if '__settings__' in data:
                self._settings_text.delete('1.0', tk.END)
                self._settings_text.insert('1.0', data['__settings__'])
            for priority, texts in data.items():
                if priority in self._ticket_sections:
                    self._ticket_sections[priority].load_texts(texts)
        except Exception as e:
            print(f"[PromptSpace] Could not load tickets: {e}")

    # ---------------------------------------------------------------- Hypotheses

    def _build_hypotheses_tab(self, parent):
        pw = tk.PanedWindow(parent, orient=tk.HORIZONTAL, sashwidth=5,
                             sashrelief='raised')
        pw.pack(fill=tk.BOTH, expand=True)

        # Left: card list
        left = ttk.Frame(pw)
        pw.add(left, width=320, minsize=200)

        top_bar = ttk.Frame(left)
        top_bar.pack(fill=tk.X, pady=2)
        ttk.Button(top_bar, text='+ Add Hypothesis',
                   command=self._add_hypothesis).pack(side=tk.LEFT, padx=4)

        card_outer = ttk.Frame(left)
        card_outer.pack(fill=tk.BOTH, expand=True)
        self._card_canvas = tk.Canvas(card_outer, highlightthickness=0)
        card_sb = ttk.Scrollbar(card_outer, orient='vertical',
                                 command=self._card_canvas.yview)
        self._card_canvas.configure(yscrollcommand=card_sb.set)
        card_sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._card_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._card_inner = ttk.Frame(self._card_canvas)
        self._card_win = self._card_canvas.create_window(
            (0, 0), window=self._card_inner, anchor='nw')
        self._card_inner.bind('<Configure>',
            lambda e: self._card_canvas.configure(
                scrollregion=self._card_canvas.bbox('all')))
        self._card_canvas.bind('<Configure>',
            lambda e: self._card_canvas.itemconfig(self._card_win, width=e.width))
        # Two-finger trackpad scroll for card list
        for _w in (self._card_canvas, self._card_inner):
            _w.bind('<MouseWheel>', self._on_card_scroll)
            _w.bind('<Button-4>', self._on_card_scroll)
            _w.bind('<Button-5>', self._on_card_scroll)

        # Right: scatter canvas
        right = ttk.Frame(pw)
        pw.add(right, minsize=200)
        self._scatter_zoom = 1.0   # scale factor for pinch zoom
        self._scatter = tk.Canvas(right, bg='#1e1e2e', highlightthickness=0)
        self._scatter.pack(fill=tk.BOTH, expand=True)
        self._scatter.bind('<Button-1>', self._on_canvas_click)
        self._scatter.bind('<B1-Motion>', self._on_canvas_drag)
        self._scatter.bind('<ButtonRelease-1>', self._on_canvas_release)
        # Pinch zoom (macOS trackpad sends MouseWheel with Control)
        self._scatter.bind('<MouseWheel>', self._on_scatter_scroll)
        self._scatter.bind('<Button-4>', self._on_scatter_scroll)
        self._scatter.bind('<Button-5>', self._on_scatter_scroll)
        # Rebuild scatter on canvas resize
        self._scatter.bind('<Configure>', lambda e: self.root.after(50, self._rebuild_scatter))

    def _on_card_scroll(self, event):
        if event.num == 4:
            self._card_canvas.yview_scroll(-1, 'units')
        elif event.num == 5:
            self._card_canvas.yview_scroll(1, 'units')
        else:
            self._card_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

    def _on_scatter_scroll(self, event):
        """Pinch-zoom or scroll on scatter canvas."""
        if event.state & 0x4:  # Control held = pinch zoom
            factor = 1.1 if (event.delta > 0 or event.num == 4) else 0.9
            self._scatter_zoom = max(0.2, min(5.0, self._scatter_zoom * factor))
            self._rebuild_scatter()
        # Plain scroll: ignored (canvas is not scrollable, just zoomed)

    def _rebuild_cards(self):
        for w in self._card_inner.winfo_children():
            w.destroy()
        self._card_frames.clear()
        for hyp in self._hypotheses:
            card = _HypothesisCard(
                self._card_inner, hyp,
                on_edit=self._edit_hypothesis,
                on_delete=self._delete_hypothesis,
                on_copy=self._copy_hypothesis,
                on_select=self._select_hypothesis,
            )
            card.pack(fill=tk.X, pady=2, padx=2)
            self._card_frames[hyp['id']] = card

    def _rebuild_scatter(self):
        self._scatter.delete('all')
        self._canvas_dots.clear()
        w = self._scatter.winfo_width() or 400
        h = self._scatter.winfo_height() or 400
        z = getattr(self, '_scatter_zoom', 1.0)
        r = 8   # dot radius fixed; zoom scales positions
        cx, cy = w / 2, h / 2   # zoom origin = canvas centre
        for hyp in self._hypotheses:
            fx = hyp.get('x_frac', random.random())
            fy = hyp.get('y_frac', random.random())
            # Apply zoom around canvas centre
            x = cx + (fx * w - cx) * z
            y = cy + (fy * h - cy) * z
            dot = self._scatter.create_oval(x - r, y - r, x + r, y + r,
                                             fill=hyp['color'], outline='white',
                                             width=1.5, tags=('dot', hyp['id']))
            self._canvas_dots[hyp['id']] = dot
            self._scatter.create_text(x, y + r + 8, text=hyp['title'][:15],
                                       fill='#cdd6f4', font=('TkDefaultFont', 7),
                                       tags=('label', hyp['id']))

    def _add_hypothesis(self):
        title = simpledialog.askstring('New Hypothesis', 'Title:', parent=self.root)
        if not title:
            return
        text = simpledialog.askstring('New Hypothesis', 'Description (optional):', parent=self.root) or ''
        hyp = {
            'id': str(uuid.uuid4()),
            'title': title,
            'text': text,
            'color': _random_color(),
            'x_frac': random.uniform(0.1, 0.9),
            'y_frac': random.uniform(0.1, 0.9),
        }
        _assign_avatar(hyp['id'])
        self._hypotheses.append(hyp)
        self._save_hypotheses()
        self._rebuild_cards()
        self.root.after(100, self._rebuild_scatter)   # after canvas has size

    def _edit_hypothesis(self, hyp: dict):
        new_title = simpledialog.askstring('Edit', 'Title:', initialvalue=hyp['title'], parent=self.root)
        if new_title is not None:
            hyp['title'] = new_title
        new_text = simpledialog.askstring('Edit', 'Description:', initialvalue=hyp.get('text', ''), parent=self.root)
        if new_text is not None:
            hyp['text'] = new_text
        self._save_hypotheses()
        self._rebuild_cards()
        self._rebuild_scatter()

    def _delete_hypothesis(self, hyp: dict):
        if messagebox.askyesno('Delete', f'Delete "{hyp["title"]}"?'):
            self._hypotheses = [h for h in self._hypotheses if h['id'] != hyp['id']]
            self._save_hypotheses()
            self._rebuild_cards()
            self._rebuild_scatter()

    def _copy_hypothesis(self, hyp: dict):
        n = sum(1 for h in self._hypotheses if h['title'].startswith(hyp['title']))
        new_hyp = {
            'id': str(uuid.uuid4()),
            'title': f"{hyp['title']} ({n})",
            'text': hyp.get('text', ''),
            'color': _random_color(),
            'x_frac': min(hyp.get('x_frac', 0.5) + 0.05, 0.95),
            'y_frac': min(hyp.get('y_frac', 0.5) + 0.05, 0.95),
        }
        _assign_avatar(new_hyp['id'])
        self._hypotheses.append(new_hyp)
        self._save_hypotheses()
        self._rebuild_cards()
        self.root.after(100, self._rebuild_scatter)

    def _select_hypothesis(self, hyp: dict):
        self._selected_hyp_id = hyp['id']
        # Scroll card into view
        card = self._card_frames.get(hyp['id'])
        if card:
            card.update_idletasks()
            y = card.winfo_y()
            total = self._card_inner.winfo_height()
            if total > 0:
                self._card_canvas.yview_moveto(max(0, (y - 20) / total))

    def _on_canvas_click(self, event):
        items = self._scatter.find_overlapping(event.x - 10, event.y - 10,
                                                event.x + 10, event.y + 10)
        dot_id = None
        for item in reversed(items):
            tags = self._scatter.gettags(item)
            if 'dot' in tags:
                for tag in tags:
                    if tag != 'dot':
                        dot_id = tag
                        break
            if dot_id:
                break
        if dot_id:
            self._drag_hyp_id = dot_id
            bbox = self._scatter.bbox(self._canvas_dots[dot_id])
            cx = (bbox[0] + bbox[2]) / 2
            cy = (bbox[1] + bbox[3]) / 2
            self._drag_offset = (event.x - cx, event.y - cy)
            hyp = next((h for h in self._hypotheses if h['id'] == dot_id), None)
            if hyp:
                self._select_hypothesis(hyp)
        else:
            self._drag_hyp_id = None

    def _on_canvas_drag(self, event):
        if not self._drag_hyp_id:
            return
        dot = self._canvas_dots.get(self._drag_hyp_id)
        if dot is None:
            return
        r = 8
        ox, oy = self._drag_offset
        nx = event.x - ox
        ny = event.y - oy
        self._scatter.coords(dot, nx - r, ny - r, nx + r, ny + r)
        # Move label too
        labels = self._scatter.find_withtag(self._drag_hyp_id)
        for item in labels:
            if 'label' in self._scatter.gettags(item):
                self._scatter.coords(item, nx, ny + r + 8)

    def _on_canvas_release(self, event):
        if not self._drag_hyp_id:
            return
        w = max(self._scatter.winfo_width(), 1)
        h = max(self._scatter.winfo_height(), 1)
        z = getattr(self, '_scatter_zoom', 1.0)
        dot = self._canvas_dots.get(self._drag_hyp_id)
        if dot:
            bbox = self._scatter.bbox(dot)
            px = (bbox[0] + bbox[2]) / 2
            py = (bbox[1] + bbox[3]) / 2
            # Inverse zoom to get logical fraction
            cx, cy = w / 2, h / 2
            fx = cx + (px - cx) / z
            fy = cy + (py - cy) / z
            hyp = next((h2 for h2 in self._hypotheses if h2['id'] == self._drag_hyp_id), None)
            if hyp:
                hyp['x_frac'] = max(0.0, min(1.0, fx / w))
                hyp['y_frac'] = max(0.0, min(1.0, fy / h))
                self._save_hypotheses()
        self._drag_hyp_id = None

    def _save_hypotheses(self):
        try:
            with open(_HYPOTHESES_PATH, 'w', encoding='utf-8') as f:
                json.dump(self._hypotheses, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[PromptSpace] Could not save hypotheses: {e}")

    def _load_hypotheses(self):
        if not os.path.exists(_HYPOTHESES_PATH):
            return
        try:
            with open(_HYPOTHESES_PATH, 'r', encoding='utf-8') as f:
                self._hypotheses = json.load(f)
        except Exception as e:
            print(f"[PromptSpace] Could not load hypotheses: {e}")
            self._hypotheses = []
        self._rebuild_cards()
        self.root.after(200, self._rebuild_scatter)

    def _on_close(self):
        self._save_tickets()
        self.root.destroy()
