"""
PromptSpace — standalone work-management UI.
Tabs: Tickets (high/mid/low priority task lists) and Hypotheses (idea maps).

Hypotheses support multiple named canvases with versioned save/load and
trash-only deletion (no data is ever permanently deleted).
"""
from __future__ import annotations

import json
import os
import shutil
import tempfile
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

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))  # NeuroPy root
_DATA_DIR = os.path.join(_ROOT, 'data', 'prompts')
_IMAGES_DIR = os.path.join(_ROOT, 'data', 'images', 'random')
_UNUSED_DIR = os.path.join(_IMAGES_DIR, 'unused')
_TICKETS_PATH = os.path.join(_DATA_DIR, 'tickets.json')          # legacy flat save
_TICKETS_AUTOSAVE_PATH = os.path.join(_DATA_DIR, 'tickets_autosave.json')
_HYPOTHESES_PATH = os.path.join(_DATA_DIR, 'hypotheses.json')     # legacy hypotheses
_HYP_CANVASES_DIR = os.path.join(_DATA_DIR, 'canvases')
_HYP_TRASH_DIR = os.path.join(_HYP_CANVASES_DIR, '_trash')
_HYP_AUTOSAVE = os.path.join(_HYP_CANVASES_DIR, 'autosave.json')
_HYP_SCHEMA_VERSION = 1
_HYP_AUTOSAVE_INTERVAL_MS = 5 * 60 * 1000   # 5 minutes
# Per-priority ticket directories
_TICKET_DIRS = {p: os.path.join(_DATA_DIR, f'tickets_{p}') for p in ('high', 'mid', 'low')}

for _d in (
    _DATA_DIR, _IMAGES_DIR, _UNUSED_DIR,
    _HYP_CANVASES_DIR, _HYP_TRASH_DIR,
    *_TICKET_DIRS.values(),
    *[os.path.join(d, '_trash') for d in _TICKET_DIRS.values()],
):
    os.makedirs(_d, exist_ok=True)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _detect_dark_mode(root) -> bool:
    """Return True if the current ttk theme background is dark."""
    try:
        style = ttk.Style(root)
        bg = style.lookup('TFrame', 'background') or style.lookup('.', 'background')
        if not bg:
            return False
        rgb = root.winfo_rgb(bg)
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        return lum < 32768
    except Exception:
        return False


def _random_color() -> str:
    hues = [
        '#e63946', '#f4a261', '#2a9d8f', '#457b9d', '#9b5de5',
        '#f15bb5', '#00bbf9', '#00f5d4', '#fee440', '#fb5607',
        '#3a86ff', '#8338ec', '#06d6a0', '#ef476f', '#ffd166',
    ]
    return random.choice(hues)


def _lighten_color(hex_color: str, factor: float = 0.85) -> str:
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
    if _HAS_PIL:
        img = Image.new('RGB', (size, size), color)
        return ImageTk.PhotoImage(img)
    return None


def _load_avatar(hyp_id: str, size: int = 28) -> Optional[object]:
    path = os.path.join(_IMAGES_DIR, f'{hyp_id}.png')
    if os.path.exists(path) and _HAS_PIL:
        try:
            img = Image.open(path).convert('RGB').resize((size, size), Image.LANCZOS)
            return ImageTk.PhotoImage(img)
        except Exception:
            pass
    return None


def _assign_avatar(hyp_id: str) -> bool:
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


def _atomic_write_json(path: str, data: dict):
    """Write JSON atomically using a temp file + rename."""
    dir_ = os.path.dirname(path)
    fd, tmp = tempfile.mkstemp(dir=dir_, suffix='.tmp')
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


# ---------------------------------------------------------------------------
# _TicketSection
# ---------------------------------------------------------------------------

class _TicketSection(ttk.Frame):
    """One priority section with a scrollable list of editable text items."""

    def __init__(self, parent, priority: str, save_cb, settings_getter=None, dark: bool = False):
        super().__init__(parent)
        self._priority = priority
        self._save_cb = save_cb
        self._settings_getter = settings_getter
        self._dark = dark
        self._bg = '#2b2b2b' if dark else 'white'
        self._fg = '#e0e0e0' if dark else 'black'
        self._items: List[dict] = []

        top_bar = ttk.Frame(self)
        top_bar.pack(fill=tk.X, pady=(2, 1))
        ttk.Button(top_bar, text="+ Add item", command=self._add_item).pack(side=tk.LEFT, padx=2)
        ttk.Button(top_bar, text="Copy all",
                   command=self._copy_and_archive).pack(side=tk.LEFT, padx=2)

        outer = ttk.Frame(self)
        outer.pack(fill=tk.BOTH, expand=True)
        self._canvas = tk.Canvas(outer, highlightthickness=0, bg=self._bg)
        sb = ttk.Scrollbar(outer, orient='vertical', command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=sb.set)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._scroll_frame = ttk.Frame(self._canvas)
        self._win = self._canvas.create_window((0, 0), window=self._scroll_frame, anchor='nw')
        self._scroll_frame.bind('<Configure>', self._on_frame_configure)
        self._canvas.bind('<Configure>', self._on_canvas_configure)
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
        row = tk.Frame(self._scroll_frame, relief='groove', borderwidth=2,
                       bg=self._bg, padx=2, pady=2)
        row.pack(fill=tk.X, pady=2, padx=4)
        txt = tk.Text(row, height=3, wrap=tk.WORD, font=('TkDefaultFont', 9),
                      bg=self._bg, fg=self._fg, insertbackground=self._fg)
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
        hist_path = os.path.join(_DATA_DIR, f'history_{self._priority}.txt')
        with open(hist_path, 'a', encoding='utf-8') as f:
            f.write(f"\n--- {datetime.datetime.now().isoformat()} ---\n")
            f.write(full_text + '\n')
        for item in list(self._items):
            self._delete_item(item['frame'], item['text_widget'])
        self._canvas.configure(bg=self._bg)
        messagebox.showinfo("Copied", f"{len(texts)} item(s) copied and archived.")

    def get_texts(self) -> List[str]:
        return [i['text_widget'].get('1.0', tk.END).strip() for i in self._items]

    def load_texts(self, texts: List[str]):
        for item in list(self._items):
            item['frame'].destroy()
        self._items.clear()
        for text in texts:
            self._add_item(text)


# ---------------------------------------------------------------------------
# _TicketPriorityPanel  — multi-file wrapper around _TicketSection
# ---------------------------------------------------------------------------

class _TicketPriorityPanel(ttk.Frame):
    """Multi-file ticket manager for one priority level.

    Provides named file tabs, versioned JSON save/load and trash-only deletion
    using the same logic as the hypotheses canvases.
    """

    SCHEMA_VERSION = 1

    def __init__(self, parent, priority: str, data_dir: str,
                 settings_getter, dark: bool = False):
        super().__init__(parent)
        self._priority = priority
        self._data_dir = data_dir
        self._trash_dir = os.path.join(data_dir, '_trash')
        self._autosave_path = os.path.join(data_dir, 'autosave.json')
        self._settings_getter = settings_getter
        self._dark = dark
        self._files: List[dict] = []   # [{name, section}]
        self._active_idx: int = 0

        # Top bar: file name + management + save/load
        top = ttk.Frame(self)
        top.pack(fill=tk.X, pady=(2, 1))
        ttk.Label(top, text="File:").pack(side=tk.LEFT)
        self._name_var = tk.StringVar()
        name_entry = ttk.Entry(top, textvariable=self._name_var, width=18)
        name_entry.pack(side=tk.LEFT, padx=(2, 4))
        name_entry.bind('<Return>',    lambda e: self._rename_file())
        name_entry.bind('<FocusOut>',  lambda e: self._rename_file())
        ttk.Button(top, text='+ New', command=self._add_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text='🗑 Del', command=self._delete_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text='💾', width=3, command=self._save_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(top, text='📂', width=3, command=self._load_dialog).pack(side=tk.LEFT, padx=2)

        self._nb = ttk.Notebook(self)
        self._nb.pack(fill=tk.BOTH, expand=True)
        self._nb.bind('<<NotebookTabChanged>>', self._on_tab_switch)

    # ── file management ─────────────────────────────────────────────────────

    def _next_file_name(self) -> str:
        used = set()
        for f in self._files:
            n = f['name']
            if n.startswith('New File '):
                try:
                    used.add(int(n[len('New File '):]))
                except ValueError:
                    pass
        i = 1
        while i in used:
            i += 1
        return f'New File {i}'

    def _add_file(self, name: str = None, items: list = None, switch: bool = True):
        if name is None:
            name = self._next_file_name()
        tab = ttk.Frame(self._nb)
        self._nb.add(tab, text=name[:16])
        sec = _TicketSection(tab, self._priority, self._on_content_change,
                             settings_getter=self._settings_getter, dark=self._dark)
        sec.pack(fill=tk.BOTH, expand=True)
        if items:
            sec.load_texts(items)
        self._files.append({'name': name, 'section': sec})
        if switch:
            idx = len(self._files) - 1
            self._nb.select(idx)
            self._active_idx = idx
            self._name_var.set(name)

    def _on_tab_switch(self, event=None):
        try:
            idx = self._nb.index(self._nb.select())
            if 0 <= idx < len(self._files):
                self._active_idx = idx
                self._name_var.set(self._files[idx]['name'])
        except Exception:
            pass

    def _rename_file(self):
        if not self._files:
            return
        new_name = self._name_var.get().strip()
        if not new_name or new_name == self._files[self._active_idx]['name']:
            return
        self._files[self._active_idx]['name'] = new_name
        self._nb.tab(self._active_idx, text=new_name[:16])
        self._autosave()

    def _delete_file(self):
        if len(self._files) <= 1:
            messagebox.showinfo("Delete File",
                                "Cannot delete the only file. Create a new one first.")
            return
        idx = self._active_idx
        fd = self._files[idx]
        if not messagebox.askyesno(
                "Delete File",
                f"Move file '{fd['name']}' to trash?\n"
                "Its contents are preserved and can be recovered via 📂 Load."):
            return
        # Backup to trash before removing
        ts = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        trash_path = os.path.join(self._trash_dir,
                                  f"deleted__{fd['name']}__{ts}.json")
        try:
            _atomic_write_json(trash_path, {
                'schema_version': self.SCHEMA_VERSION,
                'priority': self._priority,
                'files': [{'name': fd['name'],
                           'items': fd['section'].get_texts()}],
                'saved_at': ts,
            })
        except Exception as ex:
            print(f"[TicketPriorityPanel] trash backup failed: {ex}")
        self._nb.forget(idx)
        self._files.pop(idx)
        new_idx = max(0, idx - 1)
        self._active_idx = new_idx
        if self._files:
            self._nb.select(new_idx)
            self._name_var.set(self._files[new_idx]['name'])
        self._autosave()

    # ── save / load ──────────────────────────────────────────────────────────

    def _payload(self) -> dict:
        return {
            'schema_version': self.SCHEMA_VERSION,
            'priority': self._priority,
            'files': [{'name': f['name'], 'items': f['section'].get_texts()}
                      for f in self._files],
            'saved_at': datetime.datetime.now().isoformat(),
        }

    def _autosave(self):
        if not self._files:
            return
        try:
            _atomic_write_json(self._autosave_path, self._payload())
        except Exception as ex:
            print(f"[TicketPriorityPanel:{self._priority}] autosave failed: {ex}")

    def _on_content_change(self):
        """Triggered by _TicketSection on every keystroke/add/delete."""
        self._autosave()

    def _save_dialog(self):
        default = self._files[self._active_idx]['name'] if self._files else self._priority
        name = simpledialog.askstring(
            'Save', 'Save name:', initialvalue=default, parent=self.winfo_toplevel())
        if not name:
            return
        safe = name.replace('/', '_').replace('\\', '_')
        ts = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        path = os.path.join(self._data_dir, f"{safe}__{ts}.json")
        try:
            p = self._payload()
            p['save_name'] = name
            _atomic_write_json(path, p)
            messagebox.showinfo("Saved", f"Saved: {os.path.basename(path)}")
        except Exception as ex:
            messagebox.showerror("Save failed", str(ex))

    def _load_dialog(self):
        paths = sorted(glob(os.path.join(self._data_dir, '*.json')),
                       key=os.path.getmtime, reverse=True)
        if not paths:
            messagebox.showinfo("Load", "No saved files found.")
            return
        win = tk.Toplevel(self.winfo_toplevel())
        win.title(f"Load {self._priority.capitalize()} Priority")
        win.geometry("480x300")
        win.transient(self.winfo_toplevel())
        win.grab_set()
        ttk.Label(win, text="Select a save (replaces current files):").pack(
            anchor='w', padx=8, pady=(8, 2))
        fr = ttk.Frame(win)
        fr.pack(fill=tk.BOTH, expand=True, padx=8)
        vsb = ttk.Scrollbar(fr, orient=tk.VERTICAL)
        lb = tk.Listbox(fr, selectmode=tk.SINGLE, yscrollcommand=vsb.set,
                        font=('Courier', 9))
        vsb.config(command=lb.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        path_map: dict = {}
        for p in paths:
            base = os.path.basename(p)
            label = base
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    d = json.load(fh)
                sat = d.get('saved_at', '')[:19]
                nf  = len(d.get('files', []))
                sn  = d.get('save_name', '')
                pfx = '★ AUTOSAVE  ' if base == 'autosave.json' else ''
                label = f"{pfx}{sat}  [{nf} file(s)]  {sn or base}"
            except Exception:
                pass
            lb.insert(tk.END, label)
            path_map[lb.size() - 1] = p
        chosen: list = []

        def _ok():
            sel = lb.curselection()
            if sel:
                chosen.append(path_map.get(sel[0]))
            win.destroy()

        bf = ttk.Frame(win)
        bf.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(bf, text="Load selected", command=_ok).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bf, text="Cancel", command=win.destroy).pack(side=tk.RIGHT)
        win.wait_window(win)
        if chosen and chosen[0]:
            self._load_from_file(chosen[0])

    def _load_from_file(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as ex:
            messagebox.showerror("Load failed", str(ex))
            return
        files_raw = data.get('files', [])
        if not files_raw:
            messagebox.showinfo("Load", "File contains no ticket files.")
            return
        self._autosave()  # backup before replacing
        for _ in list(self._files):
            try:
                self._nb.forget(0)
            except Exception:
                pass
        self._files.clear()
        for fd in files_raw:
            self._add_file(name=fd.get('name', 'File'),
                           items=fd.get('items', []), switch=False)
        if self._files:
            self._nb.select(0)
            self._active_idx = 0
            self._name_var.set(self._files[0]['name'])

    # ── startup helpers ──────────────────────────────────────────────────────

    def load_from_autosave(self) -> bool:
        """Try to load from autosave.json; returns True on success."""
        if not os.path.exists(self._autosave_path):
            return False
        try:
            with open(self._autosave_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            files_raw = data.get('files', [])
            if not files_raw:
                return False
            for fd in files_raw:
                self._add_file(name=fd.get('name', 'File'),
                               items=fd.get('items', []), switch=False)
            if self._files:
                self._nb.select(0)
                self._active_idx = 0
                self._name_var.set(self._files[0]['name'])
            return True
        except Exception as ex:
            print(f"[TicketPriorityPanel:{self._priority}] autosave load failed: {ex}")
            return False

    def load_legacy_items(self, items: list):
        """Migrate a flat legacy list into a single 'Default File'."""
        if not self._files:
            self._add_file(name='Default File', items=items, switch=True)
        self._autosave()

    def ensure_one_file(self):
        """Make sure there is at least one file tab visible."""
        if not self._files:
            self._add_file(name='New File 1', switch=False)
            if self._files:
                self._nb.select(0)
                self._active_idx = 0
                self._name_var.set(self._files[0]['name'])


# ---------------------------------------------------------------------------
# _HypothesisCard
# ---------------------------------------------------------------------------

class _HypothesisCard(tk.Frame):
    """A card widget representing a single hypothesis."""

    def __init__(self, parent, data: dict, on_edit, on_delete, on_copy, on_select):
        bg = _lighten_color(data['color'], 0.88)
        super().__init__(parent, bg=bg, pady=2, padx=2, relief='flat')
        self._data = data
        self._photo = None

        strip = tk.Frame(self, width=5, bg=data['color'])
        strip.pack(side=tk.LEFT, fill=tk.Y)

        avatar_canvas = tk.Canvas(self, bg=bg, width=32, height=32, highlightthickness=0)
        avatar_canvas.pack(side=tk.LEFT, padx=4)
        avatar_canvas.create_oval(2, 2, 30, 30, fill=data['color'], outline='white', width=1.5)
        initials = data['title'][:1].upper() if data.get('title') else '?'
        avatar_canvas.create_text(16, 16, text=initials, fill='white',
                                  font=('TkDefaultFont', 10, 'bold'))

        content = tk.Frame(self, bg=bg)
        content.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        title_lbl = tk.Label(content, text=data['title'], bg=bg,
                             font=('TkDefaultFont', 10, 'bold'), anchor='w')
        title_lbl.pack(fill=tk.X)
        text_lbl = tk.Label(content, text=data.get('text', ''), bg=bg,
                            font=('TkDefaultFont', 9), anchor='w', justify=tk.LEFT,
                            wraplength=220)
        text_lbl.pack(fill=tk.X)

        btn_frame = tk.Frame(self, bg=bg)
        btn_frame.pack(side=tk.RIGHT, padx=2)
        ttk.Button(btn_frame, text='Copy', width=5,
                   command=lambda: on_copy(data)).pack(pady=1)
        ttk.Button(btn_frame, text='Del', width=5,
                   command=lambda: on_delete(data)).pack(pady=1)

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

    Hypotheses support multiple named canvases backed by versioned JSON saves.
    Saves are never deleted — stale files are moved to the ``_trash`` folder.
    """

    def __init__(self):
        self._owns_mainloop = False
        existing = None
        try:
            existing = tk._default_root
            if existing is not None:
                existing.winfo_exists()
        except Exception:
            existing = None
        if existing is not None:
            self.root = tk.Toplevel(existing)
        else:
            self.root = tk.Tk()
            self._owns_mainloop = True
        self.root.title('PromptSpace')
        self.root.geometry('900x650')
        self._dark = _detect_dark_mode(self.root)
        self._ticket_files: List[dict] = []       # [{name, high, mid, low}]
        self._active_ticket_file_idx: int = 0
        self._ticket_file_name_var = tk.StringVar()
        self._ticket_sections: Dict[str, _TicketSection] = {}
        self._ticket_sub_nb = None

        # --- Hypotheses canvas state ---
        # List of canvas data dicts: {name, hypotheses, scatter_zoom}
        self._canvases: List[dict] = []
        # Per-tab widget containers (set by _build_canvas_tab)
        self._canvas_tabs: List[dict] = []
        self._active_canvas_idx: int = 0
        # "Live" references to the active canvas's data/widgets (set by _bind_active_canvas)
        self._hypotheses: List[dict] = []
        self._scatter_zoom: float = 1.0
        self._card_canvas = None
        self._card_inner = None
        self._card_frames: Dict[str, object] = {}
        self._scatter = None
        self._canvas_dots: Dict[str, int] = {}
        self._selected_hyp_id: Optional[str] = None
        self._drag_hyp_id: Optional[str] = None
        self._drag_offset = (0, 0)
        self._autosave_after_id = None

        nb = ttk.Notebook(self.root)
        nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=4)

        tickets_tab = ttk.Frame(nb)
        hyp_tab = ttk.Frame(nb)
        nb.add(tickets_tab, text='Tickets')
        nb.add(hyp_tab, text='Hypotheses')

        self._build_tickets_tab(tickets_tab)
        self._build_hypotheses_tab(hyp_tab)

        self._load_tickets()
        self._load_canvases()   # loads versioned saves or migrates legacy hypotheses.json

        self.root.protocol('WM_DELETE_WINDOW', self._on_close)
        self.root.lift()
        self.root.attributes('-topmost', True)
        self.root.after(100, lambda: self.root.attributes('-topmost', False))

        self._schedule_autosave()

        if self._owns_mainloop:
            self.root.mainloop()

    # ================================================================ Tickets

    def _build_tickets_tab(self, parent):
        # File: row — single file selector above all priority tabs
        file_row = ttk.Frame(parent)
        file_row.pack(fill=tk.X, pady=(2, 1), padx=4)
        ttk.Label(file_row, text="File:").pack(side=tk.LEFT)
        name_entry = ttk.Entry(file_row, textvariable=self._ticket_file_name_var, width=18)
        name_entry.pack(side=tk.LEFT, padx=(2, 4))
        name_entry.bind('<Return>',   lambda e: self._rename_ticket_file())
        name_entry.bind('<FocusOut>', lambda e: self._rename_ticket_file())
        ttk.Button(file_row, text='+ New',  command=self._add_ticket_file_prompt).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_row, text='🗑 Del', command=self._delete_ticket_file).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_row, text='💾', width=3, command=self._save_ticket_files_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(file_row, text='📂', width=3, command=self._load_ticket_files_dialog).pack(side=tk.LEFT, padx=2)

        sub_nb = ttk.Notebook(parent)
        sub_nb.pack(fill=tk.BOTH, expand=True)
        self._ticket_sub_nb = sub_nb

        for priority in ('high', 'mid', 'low'):
            tab = ttk.Frame(sub_nb)
            sub_nb.add(tab, text=priority.capitalize() + ' Priority')
            section = _TicketSection(tab, priority,
                                     save_cb=self._on_ticket_content_change,
                                     settings_getter=lambda: self._settings_text.get('1.0', tk.END),
                                     dark=self._dark)
            section.pack(fill=tk.BOTH, expand=True)
            self._ticket_sections[priority] = section

        settings_tab = ttk.Frame(sub_nb)
        sub_nb.add(settings_tab, text='Settings')
        ttk.Label(settings_tab, text="Settings prefix (prepended to every 'Copy all'):",
                  font=('TkDefaultFont', 9)).pack(anchor='w', padx=6, pady=(6, 2))
        _txt_bg = '#2b2b2b' if self._dark else 'white'
        _txt_fg = '#e0e0e0' if self._dark else 'black'
        self._settings_text = tk.Text(settings_tab, height=10, wrap=tk.WORD,
                                      font=('TkDefaultFont', 9), bg=_txt_bg, fg=_txt_fg,
                                      insertbackground=_txt_fg)
        self._settings_text.pack(fill=tk.BOTH, expand=True, padx=6, pady=(0, 6))
        self._settings_text.bind('<KeyRelease>', lambda e: self._save_settings())

    def _save_settings(self):
        """Save only the settings prefix (ticket item content is autosaved per-panel)."""
        try:
            # Read existing tickets.json to preserve any legacy keys
            data: dict = {}
            if os.path.exists(_TICKETS_PATH):
                try:
                    with open(_TICKETS_PATH, 'r', encoding='utf-8') as f:
                        data = json.load(f)
                except Exception:
                    pass
            data['__settings__'] = self._settings_text.get('1.0', tk.END).rstrip()
            _atomic_write_json(_TICKETS_PATH, data)
        except Exception as e:
            print(f"[PromptSpace] Could not save settings: {e}")

    def _load_tickets(self):
        """Load settings prefix, then ticket files (autosave → legacy → empty)."""
        # Settings text
        if os.path.exists(_TICKETS_PATH):
            try:
                with open(_TICKETS_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if '__settings__' in data:
                    self._settings_text.delete('1.0', tk.END)
                    self._settings_text.insert('1.0', data['__settings__'])
            except Exception:
                pass

        # 1. New shared autosave (schema v2)
        if os.path.exists(_TICKETS_AUTOSAVE_PATH):
            try:
                with open(_TICKETS_AUTOSAVE_PATH, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                if data.get('schema_version', 1) >= 2 and data.get('files'):
                    for fd in data['files']:
                        self._ticket_files.append({
                            'name': fd.get('name', 'File'),
                            'high': fd.get('high', []),
                            'mid':  fd.get('mid',  []),
                            'low':  fd.get('low',  []),
                        })
                    active_idx = min(data.get('active_idx', 0),
                                     max(0, len(self._ticket_files) - 1))
                    self._active_ticket_file_idx = active_idx
                    if self._ticket_files:
                        self._load_ticket_file_into_sections(self._ticket_files[active_idx])
                    return
            except Exception as ex:
                print(f"[PromptSpace] tickets autosave load failed: {ex}")

        # 2. Legacy migration: per-priority autosave dirs → one combined file
        combined = {'name': 'Default File', 'high': [], 'mid': [], 'low': []}
        found_legacy = False
        for priority in ('high', 'mid', 'low'):
            old_as = os.path.join(_TICKET_DIRS[priority], 'autosave.json')
            if not os.path.exists(old_as):
                continue
            try:
                with open(old_as, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                items: list = []
                for ffile in d.get('files', []):
                    items.extend(ffile.get('items', []))
                combined[priority] = items
                found_legacy = True
            except Exception:
                pass
        if found_legacy:
            self._ticket_files.append(combined)
            self._active_ticket_file_idx = 0
            self._load_ticket_file_into_sections(combined)
            self._autosave_tickets()
            return

        # 3. Legacy flat tickets.json
        if os.path.exists(_TICKETS_PATH):
            try:
                with open(_TICKETS_PATH, 'r', encoding='utf-8') as f:
                    flat = json.load(f)
                combined = {'name': 'Default File', 'high': [], 'mid': [], 'low': []}
                for priority in ('high', 'mid', 'low'):
                    items = flat.get(priority, [])
                    if isinstance(items, list):
                        combined[priority] = items
                if any(combined[p] for p in ('high', 'mid', 'low')):
                    self._ticket_files.append(combined)
                    self._active_ticket_file_idx = 0
                    self._load_ticket_file_into_sections(combined)
                    self._autosave_tickets()
                    return
            except Exception:
                pass

        # 4. Nothing to load — start with one empty file
        self._add_ticket_file(name='New File 1', switch=True)

    # ── ticket file management ───────────────────────────────────────────────

    def _next_ticket_file_name(self) -> str:
        used = set()
        for f in self._ticket_files:
            n = f['name']
            if n.startswith('New File '):
                try:
                    used.add(int(n[len('New File '):]))
                except ValueError:
                    pass
        i = 1
        while i in used:
            i += 1
        return f'New File {i}'

    def _add_ticket_file(self, name: str = None,
                         high: list = None, mid: list = None, low: list = None,
                         switch: bool = True):
        if name is None:
            name = self._next_ticket_file_name()
        fd = {'name': name, 'high': high or [], 'mid': mid or [], 'low': low or []}
        self._ticket_files.append(fd)
        if switch:
            self._active_ticket_file_idx = len(self._ticket_files) - 1
            self._load_ticket_file_into_sections(fd)
        self._autosave_tickets()

    def _add_ticket_file_prompt(self):
        self._flush_ticket_sections_to_file()
        self._add_ticket_file()

    def _rename_ticket_file(self):
        if not self._ticket_files:
            return
        new_name = self._ticket_file_name_var.get().strip()
        if not new_name:
            return
        idx = self._active_ticket_file_idx
        if new_name == self._ticket_files[idx]['name']:
            return
        self._ticket_files[idx]['name'] = new_name
        self._autosave_tickets()

    def _delete_ticket_file(self):
        if len(self._ticket_files) <= 1:
            messagebox.showinfo("Delete File", "Cannot delete the only file.")
            return
        idx = self._active_ticket_file_idx
        fd = self._ticket_files[idx]
        if not messagebox.askyesno("Delete File",
                                   f"Delete file '{fd['name']}'?\nContents will be lost."):
            return
        self._ticket_files.pop(idx)
        new_idx = max(0, idx - 1)
        self._active_ticket_file_idx = new_idx
        self._load_ticket_file_into_sections(self._ticket_files[new_idx])
        self._autosave_tickets()

    def _flush_ticket_sections_to_file(self):
        if not self._ticket_files:
            return
        fd = self._ticket_files[self._active_ticket_file_idx]
        for priority in ('high', 'mid', 'low'):
            sec = self._ticket_sections.get(priority)
            if sec:
                fd[priority] = sec.get_texts()

    def _load_ticket_file_into_sections(self, fd: dict):
        for priority in ('high', 'mid', 'low'):
            sec = self._ticket_sections.get(priority)
            if sec:
                sec.load_texts(fd.get(priority, []))
        self._ticket_file_name_var.set(fd['name'])

    def _on_ticket_content_change(self):
        self._flush_ticket_sections_to_file()
        self._autosave_tickets()

    def _tickets_payload(self) -> dict:
        self._flush_ticket_sections_to_file()
        return {
            'schema_version': 2,
            'files': [{'name': f['name'],
                       'high': f['high'], 'mid': f['mid'], 'low': f['low']}
                      for f in self._ticket_files],
            'active_idx': self._active_ticket_file_idx,
            'saved_at': datetime.datetime.now().isoformat(),
        }

    def _autosave_tickets(self):
        if not self._ticket_files:
            return
        try:
            _atomic_write_json(_TICKETS_AUTOSAVE_PATH, self._tickets_payload())
        except Exception as ex:
            print(f"[PromptSpace] tickets autosave failed: {ex}")

    def _save_ticket_files_dialog(self):
        default = (self._ticket_files[self._active_ticket_file_idx]['name']
                   if self._ticket_files else 'tickets')
        name = simpledialog.askstring('Save Tickets', 'Save name:',
                                      initialvalue=default, parent=self.root)
        if not name:
            return
        safe = name.replace('/', '_').replace('\\', '_')
        ts = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        path = os.path.join(_DATA_DIR, f"ticketset__{safe}__{ts}.json")
        try:
            p = self._tickets_payload()
            p['save_name'] = name
            _atomic_write_json(path, p)
            messagebox.showinfo("Saved", f"Saved: {os.path.basename(path)}")
        except Exception as ex:
            messagebox.showerror("Save failed", str(ex))

    def _load_ticket_files_dialog(self):
        versioned = sorted(glob(os.path.join(_DATA_DIR, 'ticketset__*.json')),
                           key=os.path.getmtime, reverse=True)
        all_paths = ([_TICKETS_AUTOSAVE_PATH] if os.path.exists(_TICKETS_AUTOSAVE_PATH) else []) + versioned
        if not all_paths:
            messagebox.showinfo("Load", "No saved ticket files found.")
            return
        win = tk.Toplevel(self.root)
        win.title("Load Ticket Files")
        win.geometry("480x300")
        win.transient(self.root)
        win.grab_set()
        ttk.Label(win, text="Select a save (replaces all open files):").pack(
            anchor='w', padx=8, pady=(8, 2))
        fr = ttk.Frame(win)
        fr.pack(fill=tk.BOTH, expand=True, padx=8)
        vsb = ttk.Scrollbar(fr, orient=tk.VERTICAL)
        lb = tk.Listbox(fr, selectmode=tk.SINGLE, yscrollcommand=vsb.set,
                        font=('Courier', 9))
        vsb.config(command=lb.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        path_map: dict = {}
        for p in all_paths:
            base = os.path.basename(p)
            label = base
            try:
                with open(p, 'r', encoding='utf-8') as fh:
                    d = json.load(fh)
                sat = d.get('saved_at', '')[:19]
                nf  = len(d.get('files', []))
                sn  = d.get('save_name', '')
                pfx = '★ AUTOSAVE  ' if p == _TICKETS_AUTOSAVE_PATH else ''
                label = f"{pfx}{sat}  [{nf} file(s)]  {sn or base}"
            except Exception:
                pass
            lb.insert(tk.END, label)
            path_map[lb.size() - 1] = p
        chosen: list = []

        def _ok():
            sel = lb.curselection()
            if sel:
                chosen.append(path_map.get(sel[0]))
            win.destroy()

        bf = ttk.Frame(win)
        bf.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(bf, text="Load selected", command=_ok).pack(side=tk.RIGHT, padx=4)
        ttk.Button(bf, text="Cancel", command=win.destroy).pack(side=tk.RIGHT)
        win.wait_window(win)
        if chosen and chosen[0]:
            self._load_tickets_from_file(chosen[0])

    def _load_tickets_from_file(self, path: str):
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as ex:
            messagebox.showerror("Load failed", str(ex))
            return
        files_raw = data.get('files', [])
        if not files_raw:
            messagebox.showinfo("Load", "File contains no ticket files.")
            return
        self._autosave_tickets()  # backup before replacing
        self._ticket_files.clear()
        for fd in files_raw:
            self._ticket_files.append({
                'name': fd.get('name', 'File'),
                'high': fd.get('high', []),
                'mid':  fd.get('mid',  []),
                'low':  fd.get('low',  []),
            })
        active_idx = min(data.get('active_idx', 0),
                         max(0, len(self._ticket_files) - 1))
        self._active_ticket_file_idx = active_idx
        if self._ticket_files:
            self._load_ticket_file_into_sections(self._ticket_files[active_idx])

    # ============================================================== Hypotheses

    def _build_hypotheses_tab(self, parent):
        """Build the hypotheses section: name row + save/load row + canvas notebook."""
        # --- Row 1: canvas name + new/del canvas ---
        name_row = ttk.Frame(parent)
        name_row.pack(fill=tk.X, padx=4, pady=(4, 1))
        ttk.Label(name_row, text="Canvas:").pack(side=tk.LEFT)
        self._canvas_name_var = tk.StringVar(value="")
        name_entry = ttk.Entry(name_row, textvariable=self._canvas_name_var, width=22)
        name_entry.pack(side=tk.LEFT, padx=(2, 4))
        name_entry.bind('<Return>', lambda e: self._rename_canvas())
        name_entry.bind('<FocusOut>', lambda e: self._rename_canvas())
        ttk.Button(name_row, text='+ New Canvas',
                   command=self._add_canvas).pack(side=tk.LEFT, padx=2)
        ttk.Button(name_row, text='🗑 Del Canvas',
                   command=self._delete_canvas).pack(side=tk.LEFT, padx=2)

        # --- Row 2: hypothesis controls + save/load ---
        btn_row = ttk.Frame(parent)
        btn_row.pack(fill=tk.X, padx=4, pady=(1, 4))
        ttk.Button(btn_row, text='+ Add Hypothesis',
                   command=self._add_hypothesis).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text='💾', width=3,
                   command=self._save_canvases_dialog).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_row, text='📂', width=3,
                   command=self._load_canvases_dialog).pack(side=tk.LEFT, padx=2)

        # --- Canvas notebook ---
        self._hyp_nb = ttk.Notebook(parent)
        self._hyp_nb.pack(fill=tk.BOTH, expand=True, padx=4, pady=(0, 4))
        self._hyp_nb.bind('<<NotebookTabChanged>>', self._on_canvas_switch)

    # -------------------------------- canvas management

    def _next_canvas_name(self) -> str:
        """Return 'New Canvas N' where N is the next integer after existing default names."""
        used = set()
        for c in self._canvases:
            n = c['name']
            if n.startswith('New Canvas '):
                try:
                    used.add(int(n[len('New Canvas '):]))
                except ValueError:
                    pass
        i = 1
        while i in used:
            i += 1
        return f'New Canvas {i}'

    def _add_canvas(self, name: str = None, hypotheses: list = None, switch: bool = True):
        """Append a new canvas tab."""
        if name is None:
            name = self._next_canvas_name()
        if hypotheses is None:
            hypotheses = []
        idx = len(self._canvases)
        self._canvases.append({'name': name, 'hypotheses': hypotheses, 'scatter_zoom': 1.0})

        tab_frame = ttk.Frame(self._hyp_nb)
        self._hyp_nb.add(tab_frame, text=name[:18])
        tab_widgets = self._build_canvas_frame(tab_frame)
        self._canvas_tabs.append(tab_widgets)

        if switch:
            self._hyp_nb.select(idx)
            # <<NotebookTabChanged>> will fire and call _on_canvas_switch → _bind_active_canvas
            # Rebuild content manually in case the event hasn't fired yet
            self._bind_active_canvas(idx)

    def _build_canvas_frame(self, tab_frame) -> dict:
        """Build card-list + scatter layout inside tab_frame; return widget refs dict."""
        pw = tk.PanedWindow(tab_frame, orient=tk.HORIZONTAL, sashwidth=5, sashrelief='raised')
        pw.pack(fill=tk.BOTH, expand=True)

        # Left: card list
        left = ttk.Frame(pw)
        pw.add(left, width=320, minsize=200)

        card_outer = ttk.Frame(left)
        card_outer.pack(fill=tk.BOTH, expand=True)
        card_canvas = tk.Canvas(card_outer, highlightthickness=0)
        card_sb = ttk.Scrollbar(card_outer, orient='vertical', command=card_canvas.yview)
        card_canvas.configure(yscrollcommand=card_sb.set)
        card_sb.pack(side=tk.RIGHT, fill=tk.Y)
        card_canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        card_inner = ttk.Frame(card_canvas)
        card_win = card_canvas.create_window((0, 0), window=card_inner, anchor='nw')
        card_inner.bind('<Configure>',
            lambda e: card_canvas.configure(scrollregion=card_canvas.bbox('all')))
        card_canvas.bind('<Configure>',
            lambda e: card_canvas.itemconfig(card_win, width=e.width))
        for _w in (card_canvas, card_inner):
            _w.bind('<MouseWheel>', self._on_card_scroll)
            _w.bind('<Button-4>', self._on_card_scroll)
            _w.bind('<Button-5>', self._on_card_scroll)

        # Right: scatter canvas
        right = ttk.Frame(pw)
        pw.add(right, minsize=200)
        scatter = tk.Canvas(right, bg='#1e1e2e', highlightthickness=0)
        scatter.pack(fill=tk.BOTH, expand=True)
        scatter.bind('<Button-1>', self._on_canvas_click)
        scatter.bind('<B1-Motion>', self._on_canvas_drag)
        scatter.bind('<ButtonRelease-1>', self._on_canvas_release)
        scatter.bind('<MouseWheel>', self._on_scatter_scroll)
        scatter.bind('<Button-4>', self._on_scatter_scroll)
        scatter.bind('<Button-5>', self._on_scatter_scroll)
        scatter.bind('<Configure>', lambda e: self.root.after(50, self._rebuild_scatter))

        return {
            'card_canvas': card_canvas,
            'card_inner':  card_inner,
            'card_frames': {},
            'scatter':     scatter,
            'canvas_dots': {},
        }

    def _bind_active_canvas(self, idx: int):
        """Point all live references at canvas idx's data and widgets, then rebuild UI."""
        if idx < 0 or idx >= len(self._canvases):
            return
        # Flush current scatter_zoom back to previous canvas data
        if 0 <= self._active_canvas_idx < len(self._canvases):
            self._canvases[self._active_canvas_idx]['scatter_zoom'] = self._scatter_zoom

        self._active_canvas_idx = idx
        data = self._canvases[idx]
        wdg  = self._canvas_tabs[idx]

        # Rebind live instance attributes
        self._hypotheses   = data['hypotheses']
        self._scatter_zoom = data['scatter_zoom']
        self._card_canvas  = wdg['card_canvas']
        self._card_inner   = wdg['card_inner']
        self._card_frames  = wdg['card_frames']
        self._scatter      = wdg['scatter']
        self._canvas_dots  = wdg['canvas_dots']

        # Reset interaction state
        self._selected_hyp_id = None
        self._drag_hyp_id = None

        # Update name entry
        self._canvas_name_var.set(data['name'])

        # Rebuild content
        self._rebuild_cards()
        self.root.after(50, self._rebuild_scatter)

    def _on_canvas_switch(self, event=None):
        try:
            idx = self._hyp_nb.index(self._hyp_nb.select())
        except Exception:
            return
        if idx != self._active_canvas_idx:
            self._bind_active_canvas(idx)

    def _rename_canvas(self):
        """Apply the name-entry value to the active canvas."""
        if not self._canvases:
            return
        new_name = self._canvas_name_var.get().strip()
        if not new_name:
            return
        idx = self._active_canvas_idx
        self._canvases[idx]['name'] = new_name
        # Update notebook tab label
        self._hyp_nb.tab(idx, text=new_name[:18])
        self._autosave_canvases()

    def _delete_canvas(self):
        """Move the active canvas to trash (never permanently deleted)."""
        if len(self._canvases) <= 1:
            messagebox.showinfo("Delete Canvas",
                                "Cannot delete the only canvas. "
                                "Create a new canvas first.")
            return
        idx = self._active_canvas_idx
        canvas_data = self._canvases[idx]
        if not messagebox.askyesno(
                "Delete Canvas",
                f"Move canvas '{canvas_data['name']}' to trash?\n\n"
                "Its hypotheses will be saved to the trash folder and can be "
                "recovered via 📂 Load."):
            return

        # Write a versioned backup to trash before removing
        ts = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        trash_path = os.path.join(
            _HYP_TRASH_DIR,
            f"deleted__{canvas_data['name']}__{ts}.json")
        try:
            _atomic_write_json(trash_path, {
                'schema_version': _HYP_SCHEMA_VERSION,
                'canvases': [canvas_data],
                'saved_at': ts,
                'deleted_from_idx': idx,
            })
        except Exception as ex:
            print(f"[PromptSpace] trash backup failed: {ex}")

        # Remove from notebook and data
        self._hyp_nb.forget(idx)
        self._canvases.pop(idx)
        self._canvas_tabs.pop(idx)

        new_idx = max(0, idx - 1)
        self._active_canvas_idx = new_idx   # prevent flush to wrong slot in _bind_active_canvas
        self._bind_active_canvas(new_idx)
        self._autosave_canvases()

    # -------------------------------- save / load

    def _canvases_payload(self) -> dict:
        """Serialisable dict of all canvas data."""
        # Flush current scatter_zoom before snapshotting
        if 0 <= self._active_canvas_idx < len(self._canvases):
            self._canvases[self._active_canvas_idx]['scatter_zoom'] = self._scatter_zoom
        return {
            'schema_version': _HYP_SCHEMA_VERSION,
            'canvases': [
                {'name': c['name'],
                 'hypotheses': c['hypotheses'],
                 'scatter_zoom': c.get('scatter_zoom', 1.0)}
                for c in self._canvases
            ],
            'saved_at': datetime.datetime.now().isoformat(),
        }

    def _autosave_canvases(self):
        """Overwrite the autosave slot — fast crash-recovery backup."""
        if not self._canvases:
            return
        try:
            _atomic_write_json(_HYP_AUTOSAVE, self._canvases_payload())
        except Exception as ex:
            print(f"[PromptSpace] autosave failed: {ex}")

    def _schedule_autosave(self):
        """Reschedule the periodic autosave."""
        if self._autosave_after_id is not None:
            try:
                self.root.after_cancel(self._autosave_after_id)
            except Exception:
                pass
        self._autosave_after_id = self.root.after(
            _HYP_AUTOSAVE_INTERVAL_MS, self._periodic_autosave)

    def _periodic_autosave(self):
        self._autosave_canvases()
        self._schedule_autosave()

    def _save_canvases_dialog(self):
        """Prompt for a save name and write a versioned JSON file."""
        default = self._canvases[self._active_canvas_idx]['name'] if self._canvases else 'hypotheses'
        name = simpledialog.askstring(
            'Save Hypotheses', 'Save name:', initialvalue=default, parent=self.root)
        if not name:
            return
        safe = name.replace('/', '_').replace('\\', '_')
        ts = datetime.datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
        path = os.path.join(_HYP_CANVASES_DIR, f"{safe}__{ts}.json")
        try:
            payload = self._canvases_payload()
            payload['save_name'] = name
            _atomic_write_json(path, payload)
            messagebox.showinfo("Saved",
                                f"Hypotheses saved:\n  {os.path.basename(path)}")
        except Exception as ex:
            messagebox.showerror("Save failed", str(ex))

    def _load_canvases_dialog(self):
        """Show a dialog listing all saved canvas files (newest first); load chosen."""
        # Gather all saves: versioned + autosave (if present)
        versioned = sorted(
            glob(os.path.join(_HYP_CANVASES_DIR, '*.json')),
            key=os.path.getmtime, reverse=True)
        if not versioned:
            messagebox.showinfo("Load Hypotheses",
                                "No saved hypothesis files found.")
            return

        win = tk.Toplevel(self.root)
        win.title("Load Hypotheses")
        win.geometry("540x360")
        win.transient(self.root)
        win.grab_set()

        ttk.Label(win, text="Select a save to load (replaces current canvases):").pack(
            anchor='w', padx=8, pady=(8, 2))

        frame = ttk.Frame(win)
        frame.pack(fill=tk.BOTH, expand=True, padx=8)
        vsb = ttk.Scrollbar(frame, orient=tk.VERTICAL)
        lb = tk.Listbox(frame, selectmode=tk.SINGLE, yscrollcommand=vsb.set, font=('Courier', 9))
        vsb.config(command=lb.yview)
        vsb.pack(side=tk.RIGHT, fill=tk.Y)
        lb.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

        path_map = {}
        for p in versioned:
            base = os.path.basename(p)
            label = base
            try:
                with open(p, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                saved_at = d.get('saved_at', '')[:19]
                n_canvases = len(d.get('canvases', []))
                save_name = d.get('save_name', '')
                label = f"{saved_at}  [{n_canvases} canvas(es)]  {save_name or base}"
            except Exception:
                pass
            lb.insert(tk.END, label)
            path_map[lb.size() - 1] = p

        # Autosave shown specially
        if os.path.exists(_HYP_AUTOSAVE):
            try:
                with open(_HYP_AUTOSAVE, 'r', encoding='utf-8') as f:
                    d = json.load(f)
                saved_at = d.get('saved_at', '')[:19]
                n_c = len(d.get('canvases', []))
                label = f"★ AUTOSAVE  {saved_at}  [{n_c} canvas(es)]"
                # Insert at top (index 0) but it may already be in versioned list
                if _HYP_AUTOSAVE not in versioned:
                    lb.insert(0, label)
                    for k in list(path_map.keys()):
                        path_map[k + 1] = path_map[k]
                    path_map[0] = _HYP_AUTOSAVE
            except Exception:
                pass

        chosen = []

        def _ok():
            sel = lb.curselection()
            if sel:
                chosen.append(path_map.get(sel[0]))
            win.destroy()

        btn_f = ttk.Frame(win)
        btn_f.pack(fill=tk.X, padx=8, pady=6)
        ttk.Button(btn_f, text="Load selected", command=_ok).pack(side=tk.RIGHT, padx=4)
        ttk.Button(btn_f, text="Cancel", command=win.destroy).pack(side=tk.RIGHT)
        win.wait_window(win)

        if chosen and chosen[0]:
            self._load_canvases_from_file(chosen[0])

    def _load_canvases_from_file(self, path: str):
        """Load canvases from a JSON file, replacing current state."""
        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as ex:
            messagebox.showerror("Load failed", str(ex))
            return

        canvases_raw = data.get('canvases', [])
        if not canvases_raw:
            messagebox.showinfo("Load", "File contains no canvases.")
            return

        # Autosave current state before replacing
        self._autosave_canvases()

        # Tear down existing notebook tabs
        for tab in self._hyp_nb.tabs():
            self._hyp_nb.forget(tab)
        self._canvases.clear()
        self._canvas_tabs.clear()
        self._active_canvas_idx = 0

        for c in canvases_raw:
            self._add_canvas(
                name=c.get('name', 'Canvas'),
                hypotheses=c.get('hypotheses', []),
                switch=False)

        # Select first tab
        if self._canvases:
            self._hyp_nb.select(0)
            self._bind_active_canvas(0)

    def _load_canvases(self):
        """On startup: load autosave if present, else migrate legacy hypotheses.json."""
        # Prefer autosave
        if os.path.exists(_HYP_AUTOSAVE):
            try:
                with open(_HYP_AUTOSAVE, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                canvases_raw = data.get('canvases', [])
                if canvases_raw:
                    for c in canvases_raw:
                        self._add_canvas(
                            name=c.get('name', 'Canvas'),
                            hypotheses=c.get('hypotheses', []),
                            switch=False)
                    if self._canvases:
                        self._hyp_nb.select(0)
                        self._bind_active_canvas(0)
                    return
            except Exception as ex:
                print(f"[PromptSpace] autosave load failed: {ex}")

        # Migrate legacy hypotheses.json → single "Default Canvas"
        if os.path.exists(_HYPOTHESES_PATH):
            try:
                with open(_HYPOTHESES_PATH, 'r', encoding='utf-8') as f:
                    hyps = json.load(f)
                if isinstance(hyps, list) and hyps:
                    self._add_canvas(name='Default Canvas', hypotheses=hyps, switch=False)
                    if self._canvases:
                        self._hyp_nb.select(0)
                        self._bind_active_canvas(0)
                    self._autosave_canvases()
                    return
            except Exception as ex:
                print(f"[PromptSpace] legacy migrate failed: {ex}")

        # Nothing to load — start with one empty canvas
        if not self._canvases:
            self._add_canvas(switch=True)

    # -------------------------------- card list

    def _on_card_scroll(self, event):
        if self._card_canvas is None:
            return
        if event.num == 4:
            self._card_canvas.yview_scroll(-1, 'units')
        elif event.num == 5:
            self._card_canvas.yview_scroll(1, 'units')
        else:
            self._card_canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

    def _rebuild_cards(self):
        if self._card_inner is None:
            return
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

    # -------------------------------- scatter

    def _rebuild_scatter(self):
        if self._scatter is None:
            return
        self._scatter.delete('all')
        self._canvas_dots.clear()
        w = self._scatter.winfo_width() or 400
        h = self._scatter.winfo_height() or 400
        z = self._scatter_zoom
        r = 8
        cx, cy = w / 2, h / 2
        for hyp in self._hypotheses:
            fx = hyp.get('x_frac', random.random())
            fy = hyp.get('y_frac', random.random())
            x = cx + (fx * w - cx) * z
            y = cy + (fy * h - cy) * z
            dot = self._scatter.create_oval(x - r, y - r, x + r, y + r,
                                            fill=hyp['color'], outline='white',
                                            width=1.5, tags=('dot', hyp['id']))
            self._canvas_dots[hyp['id']] = dot
            self._scatter.create_text(x, y + r + 8, text=hyp['title'][:15],
                                      fill='#cdd6f4', font=('TkDefaultFont', 7),
                                      tags=('label', hyp['id']))

    def _on_scatter_scroll(self, event):
        if event.state & 0x4:
            factor = 1.1 if (event.delta > 0 or event.num == 4) else 0.9
            self._scatter_zoom = max(0.2, min(5.0, self._scatter_zoom * factor))
            # Sync zoom to canvas data
            if 0 <= self._active_canvas_idx < len(self._canvases):
                self._canvases[self._active_canvas_idx]['scatter_zoom'] = self._scatter_zoom
            self._rebuild_scatter()

    # -------------------------------- hypothesis CRUD

    def _add_hypothesis(self):
        title = simpledialog.askstring('New Hypothesis', 'Title:', parent=self.root)
        if not title:
            return
        text = simpledialog.askstring('New Hypothesis', 'Description (optional):',
                                      parent=self.root) or ''
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
        self._autosave_canvases()
        self._rebuild_cards()
        self.root.after(100, self._rebuild_scatter)

    def _edit_hypothesis(self, hyp: dict):
        new_title = simpledialog.askstring('Edit', 'Title:', initialvalue=hyp['title'],
                                           parent=self.root)
        if new_title is not None:
            hyp['title'] = new_title
        new_text = simpledialog.askstring('Edit', 'Description:', initialvalue=hyp.get('text', ''),
                                          parent=self.root)
        if new_text is not None:
            hyp['text'] = new_text
        self._autosave_canvases()
        self._rebuild_cards()
        self._rebuild_scatter()

    def _delete_hypothesis(self, hyp: dict):
        if messagebox.askyesno('Delete', f'Delete "{hyp["title"]}"?'):
            self._hypotheses[:] = [h for h in self._hypotheses if h['id'] != hyp['id']]
            self._autosave_canvases()
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
        self._autosave_canvases()
        self._rebuild_cards()
        self.root.after(100, self._rebuild_scatter)

    def _select_hypothesis(self, hyp: dict):
        self._selected_hyp_id = hyp['id']
        card = self._card_frames.get(hyp['id'])
        if card:
            card.update_idletasks()
            y = card.winfo_y()
            total = self._card_inner.winfo_height()
            if total > 0:
                self._card_canvas.yview_moveto(max(0, (y - 20) / total))

    # -------------------------------- scatter interaction

    def _on_canvas_click(self, event):
        if self._scatter is None:
            return
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
        if not self._drag_hyp_id or self._scatter is None:
            return
        dot = self._canvas_dots.get(self._drag_hyp_id)
        if dot is None:
            return
        r = 8
        ox, oy = self._drag_offset
        nx, ny = event.x - ox, event.y - oy
        self._scatter.coords(dot, nx - r, ny - r, nx + r, ny + r)
        for item in self._scatter.find_withtag(self._drag_hyp_id):
            if 'label' in self._scatter.gettags(item):
                self._scatter.coords(item, nx, ny + r + 8)

    def _on_canvas_release(self, event):
        if not self._drag_hyp_id or self._scatter is None:
            return
        w = max(self._scatter.winfo_width(), 1)
        h = max(self._scatter.winfo_height(), 1)
        z = self._scatter_zoom
        dot = self._canvas_dots.get(self._drag_hyp_id)
        if dot:
            bbox = self._scatter.bbox(dot)
            px = (bbox[0] + bbox[2]) / 2
            py = (bbox[1] + bbox[3]) / 2
            cx, cy = w / 2, h / 2
            fx = cx + (px - cx) / z
            fy = cy + (py - cy) / z
            hyp = next((h2 for h2 in self._hypotheses if h2['id'] == self._drag_hyp_id), None)
            if hyp:
                hyp['x_frac'] = max(0.0, min(1.0, fx / w))
                hyp['y_frac'] = max(0.0, min(1.0, fy / h))
                self._autosave_canvases()
        self._drag_hyp_id = None

    # -------------------------------- close

    def _on_close(self):
        self._save_settings()
        self._autosave_tickets()
        self._autosave_canvases()
        if self._autosave_after_id is not None:
            try:
                self.root.after_cancel(self._autosave_after_id)
            except Exception:
                pass
        self.root.destroy()
