"""ExportManager — CCG view export logic.

Ported from ccg_ui.py. References ui.* (AppState).
ExportDialogQt (Qt dialog frontend) is a future TODO.
"""
from __future__ import annotations
import os
import re
import numpy as np
from PIL import Image
from pyqtgraph.Qt.QtWidgets import QMessageBox, QFileDialog
from typing import TYPE_CHECKING
from neuropy.analyses.ms_connectivity import CCGDataset as _CCGDataset
from neuropy.ui.pair_selection_panel import SelectionData
if TYPE_CHECKING:
    pass

_ALL_SEGS = "all"

class ExportManager:
    """Export methods for CCGReviewUI."""

    def __init__(self, ui: "CCGReviewUI"):
        self._ui = ui

    def _export_current_view(self):
        """Export the currently displayed CCG view (including stacked/sbs) to PNG."""
        ui = self._ui
        fmt = 'png'
        if ui.ccg_data is None:
            QMessageBox.warning(None, "Export", "No plot to export.")
            return
        selected_pairs = ui._selected_pairs_from_lists()

        def _strip_any_session_pair(p):
            if p is None:
                return None
            if ui._session_any_mode and len(p) == 3:
                nk = ui._sess_mgr._nd_key_for_session_str(str(p[0]))
                if nk is not None:
                    ckey = ui._sess_mgr._type_key_for_nd(nk)
                    if ckey is not None:
                        ui._sess_mgr._bind_context_to_type_key(ckey)
                return int(p[1]), int(p[2])
            return int(p[0]), int(p[1])

        if ui._session_any_mode:
            selected_pairs = [_strip_any_session_pair(p) for p in selected_pairs]

        # If user didn't explicitly select rows, default to current pair for "Export current".
        # (Also drives preview in the dialog.)
        preview_pair = selected_pairs[0] if selected_pairs else ui._selected_pair_from_lists()
        preview_pair = _strip_any_session_pair(preview_pair)
        if preview_pair is None:
            inds = ui._current_inds()
            if inds is not None:
                preview_pair = (int(inds[0]), int(inds[1]))
        opt = ExportOptionsDialog.show(ui, fmt=fmt, preview_pair=preview_pair, selected_pairs=selected_pairs)
        if opt is None:
            return
        # Multi-export actions should go straight to a folder picker (no save-as).
        if opt.get('_action') in ('all', 'bookmarked', 'groups', 'all_groups', 'all_sessions_selected'):
            ui._export_mgr._export_pairs_from_opt(fmt=fmt, opt=opt)
            return
        # Suggest a filename from session/type/shank/pair/segment
        inds = ui._current_inds()
        if inds is not None:
            ref, tgt = int(inds[0]), int(inds[1])
        else:
            ref = tgt = None
        seg = ui.current_segment
        if seg == _ALL_SEGS:
            seg_tag = "all"
        elif seg not in ui.nav.available_segments():
            seg_tag = "custom"
        else:
            seg_tag = seg
        sess = str(getattr(getattr(ui.key, 'nd', lambda: ui.key)(), 'session', None) or getattr(ui.key, 'session', 'sess'))
        exc = getattr(ui.key, 'excitability', None)
        ct = getattr(ui.key, 'conn_type', None)
        if isinstance(ct, (tuple, list)) and len(ct) >= 2:
            _map = {'pyr': 'PYR', 'inter': 'INT', 'int': 'INT'}
            a = _map.get(str(ct[0]).lower(), str(ct[0]).upper())
            b = _map.get(str(ct[1]).lower(), str(ct[1]).upper())
            ct_str = f"{a}-{b}"
        else:
            ct_str = str(ct) if ct is not None else "any"
        type_str = f"{exc}_{ct_str}" if exc is not None else ct_str
        sh = ''
        shank_ids = getattr(ui.neurons, 'shank_ids', None)
        if shank_ids is not None and ref is not None and tgt is not None:
            try:
                sh = f"_sh{int(shank_ids[ref])}-{int(shank_ids[tgt])}"
            except Exception:
                sh = ''
        base = (f"{sess}_{type_str}{sh}_ccg_{ref}_{tgt}_{seg_tag}"
                if ref is not None else f"{sess}_{type_str}_ccg_{seg_tag}")
        path = QFileDialog.getSaveFileName(
            None,
            f"Export as {fmt.upper()}",
            f"{base}.{fmt}",
            f"{fmt.upper()} (*.{fmt});;All files (*.*)",
        )[0]

        if not path:
            return
        try:
            ui._export_mgr._export_one_view_to_path(path=path, fmt=fmt, opt=opt)
        except Exception as exc:
            QMessageBox.critical(None, "Export failed", f"Could not export:\n\n{exc}")

    def _export_one_view_to_path(self, path: str, fmt: str, opt: dict):
        """Export current view to a specific file path using overrides."""
        ui = self._ui
        old = getattr(ui, '_export_overrides', None)
        ui._export_overrides = opt
        try:
            if (not getattr(ui, '_stacked_segments', None) and
                    not getattr(ui, 'together_pairs', None) and
                    not (getattr(ui.nav, 'resolution', 'lo') == "lo_hi")):
                # Single-pair view: render directly to path, bypassing the viewer cache
                inds = ui.all_pairs_np[ui.current_pair_idx]
                seg  = ui.segment_index(ui.current_segment)
                hr   = getattr(ui.nav, 'resolution', 'lo') in ("hi", "lo_hi")
                ctx  = ui._render_engine.build_context(inds, seg, hr, None, None)
                dpi  = 300 if fmt == 'png' else None
                ui._render_engine.write_png(ctx, path, dpi=dpi)
            else:
                # Multi-view (stacked/SBS): save composite figure as-is
                ui.root.mainview.request_render()
                ui.canvas.draw()
                ui.fig.savefig(path, bbox_inches='tight', dpi=300 if fmt == 'png' else None)
        finally:
            ui._export_overrides = old

    def _export_pairs_with_handles(self, fmt: str, opt: dict,
                                     items: list[tuple], folder: str) -> None:
        """Core export loop: render each (tk_, ptr, ref, tgt) into *folder*.

        *items* is a list of 4-tuples (tk_, ptr, ref, tgt) where tk_/ptr are
        the exact Key/pointer objects from cd.ptr — no string lookup needed.
        """
        ui = self._ui
        export_segs = (opt or {}).get('export_segments', None)
        if not export_segs:
            export_segs = ["Current"]
        # normalize
        export_segs = [str(s) for s in export_segs if str(s).strip()] or ["Current"]
        subfolder_by = list((opt or {}).get('subfolder_by') or [])

        _ct_map = {'pyr': 'PYR', 'inter': 'INT', 'int': 'INT'}

        do_lores   = bool((opt or {}).get('export_lores', True))
        do_hires   = bool((opt or {}).get('export_hires', False))
        # Fall back to lo-res if neither is selected
        if not do_lores and not do_hires:
            do_lores = True
        _cs_by_sess = getattr(ui, '_custom_segments_by_session', None) or {}

        old_state = {
            'key': ui.key,
            'ccg_ptr': ui.ccg_ptr,
            'ccg_data': ui.ccg_data,
            'neurons': ui.neurons,
            'n_segments': getattr(ui, 'n_segments', 0),
            'segment_names': ui.nav.available_segments(),
            'current_pair_idx': int(getattr(ui, 'current_pair_idx', 0)),
            'current_segment': getattr(ui, 'current_segment', _ALL_SEGS),
            '_custom_segments': getattr(ui, '_custom_segments', []),
            'resolution': getattr(ui.nav, 'resolution', 'lo'),
        }

        n_ok = 0
        n_fail = 0
        fail_msgs = []
        try:
            for tk_, ptr, ref, tgt in items:
                # Point UI at this key/pointer directly — no lookup required.
                ui.key = tk_
                ui.ccg_ptr = ptr
                nd_key = tk_.nd()
                sess = str(getattr(tk_, 'session', getattr(nd_key, 'session', '')))
                ui.neurons = (ui.cd.nd.neurons_for(nd_key)
                              if getattr(ui.cd, 'nd', None) is not None else None)
                ui.n_segments = ui.cd.n_segments(tk_)
                # Every segment (whole-session 'full' + appended windows) is a dim0 label of
                # this session's array; resolve export names against that ordered label list.
                _reg_labels = ui.cd.segment_names(tk_)
                seg_indices: list[int] = []
                for export_seg in export_segs:
                    if export_seg in ('All', _ALL_SEGS):   # whole session = dim0[0]='full'
                        seg_indices.append(0)
                        continue
                    if export_seg == 'Current':
                        saved = old_state.get('current_segment', _ALL_SEGS)
                        seg_indices.append(saved if isinstance(saved, int) else ui.segment_index(saved))
                        continue
                    if export_seg in _reg_labels:
                        seg_indices.append(int(_reg_labels.index(export_seg)))
                    # else: silently skip — this session doesn't have that segment
                # de-dupe while preserving order
                _seen_seg = set()
                seg_indices = [s for s in seg_indices if not (s in _seen_seg or _seen_seg.add(s))]
                if not seg_indices:
                    continue
                if getattr(ui.nav, 'resolution', 'lo') in ("hi", "lo_hi"):
                    ui.ccg_data = (ui.cd.ccg_for(nd_key.change(resolution='highres'))
                                   if hasattr(ui.cd, 'ccg_for') else ui.ccg_data)
                else:
                    ui.ccg_data = (ui.cd.ccg_for(nd_key.change(resolution='lowres'))
                                   if hasattr(ui.cd, 'ccg_for') else ui.ccg_data)
                try:
                    ui.current_pair_idx = ui.get_pair_index((ref, tgt))
                except Exception:
                    ui.current_pair_idx = 0

                exc = getattr(tk_, 'excitability', None)
                ct = getattr(tk_, 'conn_type', None)
                if isinstance(ct, (tuple, list)) and len(ct) >= 2:
                    a = _ct_map.get(str(ct[0]).lower(), str(ct[0]).upper())
                    b = _ct_map.get(str(ct[1]).lower(), str(ct[1]).upper())
                    ct_str = f"{a}-{b}"
                else:
                    ct_str = str(ct) if ct is not None else "any"
                type_str = f"{exc}_{ct_str}" if exc is not None else ct_str
                sh = ''
                shank_ids = getattr(ui.neurons, 'shank_ids', None)
                if shank_ids is not None:
                    try:
                        sh = f"_sh{int(shank_ids[ref])}-{int(shank_ids[tgt])}"
                    except Exception:
                        pass

                def _animal_from_session(s: str) -> str:
                    try:
                        return str(s).split('_')[0]
                    except Exception:
                        return str(s)

                def _ei_folder(x) -> str:
                    s = str(x or '').strip()
                    if not s:
                        return "EI"
                    sl = s.lower()
                    if sl.startswith('e'):
                        return "E"
                    if sl.startswith('i'):
                        return "I"
                    # common: 'exc'/'inh'
                    if 'exc' in sl:
                        return "E"
                    if 'inh' in sl:
                        return "I"
                    return s

                def _conn_type_folder(ct_) -> str:
                    if isinstance(ct_, (tuple, list)) and len(ct_) >= 2:
                        a = _ct_map.get(str(ct_[0]).lower(), str(ct_[0]).upper())
                        b = _ct_map.get(str(ct_[1]).lower(), str(ct_[1]).upper())
                        return f"{a}-{b}"
                    return str(ct_ or "any")

                # Build subfolder path parts in the chosen order
                parts = []
                for k in subfolder_by:
                    if k == "conn type":
                        parts.append(_conn_type_folder(ct))
                    elif k == "excitatory/inhibitory":
                        parts.append(_ei_folder(exc))
                    elif k == "session":
                        parts.append(sess)
                    elif k == "animal":
                        parts.append(_animal_from_session(sess))

                base_dir = os.path.join(folder, *parts) if parts else folder
                try:
                    os.makedirs(base_dir, exist_ok=True)
                except Exception:
                    base_dir = folder

                def _has_hires():
                    return (hasattr(ui.cd, 'ccg_for')
                            and ui.cd.ccg_for(nd_key.change(resolution='highres')) is not None)

                def _render_one(seg, highres: bool, path: str):
                    ui.nav.set_resolution("hi" if highres else "lo")
                    if highres and _has_hires():
                        ui.ccg_data = ui.cd.ccg_for(nd_key.change(resolution='highres'))
                    else:
                        ui.ccg_data = (ui.cd.ccg_for(nd_key.change(resolution='lowres'))
                                       if hasattr(ui.cd, 'ccg_for') else ui.ccg_data)
                    old_eo = getattr(ui, '_export_overrides', None)
                    ui._export_overrides = opt
                    try:
                        ctx = ui._render_engine.build_context(
                            np.array([ref, tgt]), seg, highres, None, None)
                        dpi = 300 if fmt == 'png' else None
                        ui._render_engine.write_png(ctx, path, dpi=dpi)
                    finally:
                        ui._export_overrides = old_eo

                # Export one file per selected segment (render directly; bypass viewer cache)
                for seg_idx in seg_indices:
                    ui.current_segment = ui.nav.segment_name(int(seg_idx))
                    seg = ui.segment_index(ui.current_segment)
                    seg_label = ui.current_segment
                    if seg == 0 or seg_label == _ALL_SEGS:
                        seg_tag = "all"
                    else:
                        seg_tag = re.sub(r'[^A-Za-z0-9_\-]', '_', str(seg_label))
                    base_name = f"{sess}_{type_str}{sh}_ccg_{ref}_{tgt}_{seg_tag}"
                    if not bool((opt or {}).get('title_show_norm_details', True)):
                        _active_norms = sorted(nm.name for nm in getattr(ui, 'active_norms', set()))
                        if _active_norms:
                            base_name += '_' + '_'.join(_active_norms)
                    try:
                        if do_lores and do_hires and _has_hires():
                            # Render both resolutions and combine side by side
                            import tempfile as _tf
                            with _tf.NamedTemporaryFile(suffix='.png', delete=False) as _tlo:
                                lo_path = _tlo.name
                            with _tf.NamedTemporaryFile(suffix='.png', delete=False) as _thi:
                                hi_path = _thi.name
                            try:
                                _render_one(seg, False, lo_path)
                                _render_one(seg, True,  hi_path)
                                _lo_img = Image.open(lo_path)
                                _hi_img = Image.open(hi_path)
                                _h = max(_lo_img.height, _hi_img.height)
                                _combined = Image.new('RGB',
                                                      (_lo_img.width + _hi_img.width, _h),
                                                      (255, 255, 255))
                                _combined.paste(_lo_img, (0, 0))
                                _combined.paste(_hi_img, (_lo_img.width, 0))
                                out_path = os.path.join(base_dir, f"{base_name}_lohires.{fmt}")
                                _combined.save(out_path, dpi=(300, 300))
                            finally:
                                for _p in (lo_path, hi_path):
                                    try:
                                        os.remove(_p)
                                    except OSError:
                                        pass
                        elif do_hires and _has_hires():
                            out_path = os.path.join(base_dir, f"{base_name}_hires.{fmt}")
                            _render_one(seg, True, out_path)
                        else:
                            out_path = os.path.join(base_dir, f"{base_name}.{fmt}")
                            _render_one(seg, False, out_path)
                        n_ok += 1
                    except Exception as ex:
                        n_fail += 1
                        fail_msgs.append(f"({ref},{tgt}) seg={seg_tag}: {ex}")
        finally:
            ui.key = old_state['key']
            ui.ccg_ptr = old_state['ccg_ptr']
            ui.ccg_data = old_state['ccg_data']
            ui.neurons = old_state['neurons']
            ui.n_segments = old_state['n_segments']
            ui.current_pair_idx = old_state['current_pair_idx']
            saved_seg = old_state.get('current_segment', ui.current_segment)
            if isinstance(saved_seg, int):
                saved_seg = ui.nav.segment_name(saved_seg)
            ui.current_segment = saved_seg
            ui._custom_segments = old_state.get('_custom_segments', ui._custom_segments)
            ui.nav.set_resolution(old_state.get('resolution', 'lo'))
            try:
                ui.root.mainview.request_render()
            except Exception:
                pass

        if n_fail == 0:
            QMessageBox.information(None, "Export", f"Exported {n_ok} file(s) to:\n\n{folder}")
        else:
            msg = f"Exported {n_ok} file(s) to:\n\n{folder}\n\nFailed: {n_fail}"
            if fail_msgs:
                msg += "\n\n" + "\n".join(fail_msgs[:12])
                if len(fail_msgs) > 12:
                    msg += f"\n… ({len(fail_msgs) - 12} more)"
            QMessageBox.warning(None, "Export", msg)

    def _export_all_selected_pairs(self, fmt: str, opt: dict):
        """Export pairs listed in opt['_selected_pairs'] (current-session subset)."""
        ui = self._ui
        pairs_in = list(opt.get('_selected_pairs') or [])
        if not pairs_in:
            QMessageBox.information(None, "Export", "No pairs selected.")
            return
        folder = QFileDialog.getExistingDirectory(
            None, f"Export {len(pairs_in)} views to folder")
        if not folder:
            return

        # Build (tk_, ptr, ref, tgt) from current session only
        items = []
        for it in pairs_in:
            pair = tuple(it['pair']) if isinstance(it, dict) else tuple(it)
            try:
                ref, tgt = int(pair[0]), int(pair[1])
            except Exception:
                continue
            items.append((ui.key, ui.ccg_ptr, ref, tgt))

        ui._export_mgr._export_pairs_with_handles(fmt=fmt, opt=opt, items=items, folder=folder)

    def _collect_all_sessions_selected(self) -> list[tuple]:
        """Return (tk_, ptr, ref, tgt) for every selected pair in every session/type."""
        ui = self._ui
        sd = ui.nav.sd
        items = []
        for tk_, ptr in ui.cd.ptr.items():
            b = sd.get_selection_by_session(tk_).selections.get(tk_)
            if b is None or not b.selected:
                continue
            for pair in sorted(b.selected):
                try:
                    items.append((tk_, ptr, int(pair[0]), int(pair[1])))
                except Exception:
                    continue
        # Sort: session str → ref → tgt for stable output ordering
        items.sort(key=lambda x: (str(getattr(x[0], 'session', '')), x[2], x[3]))
        return items

    def _pair_handle_map(self) -> dict[tuple, list[tuple]]:
        """Build {(session_str, ref, tgt): [(tk_, ptr), ...]} from all cd.ptr entries.

        A pair can legitimately appear in multiple conn-type keys for the same
        session.  We keep all of them so the caller can pick the right one.
        """
        ui = self._ui
        m: dict[tuple, list] = {}
        for tk_, ptr in ui.cd.ptr.items():
            if ptr is None:
                continue
            sess = str(getattr(tk_, 'session', getattr(tk_.nd(), 'session', '')))
            inds = getattr(ptr, 'pairs', None)
            if inds is None:
                continue
            for pair in inds:
                try:
                    k = (sess, int(pair[0]), int(pair[1]))
                    m.setdefault(k, []).append((tk_, ptr))
                except Exception:
                    pass
        return m

    def _export_pairs_from_opt(self, fmt: str, opt: dict):
        """Resolve (tk_, ptr, ref, tgt) handles and export to a chosen folder."""
        ui = self._ui
        action = opt.get('_action')

        if action == 'all_sessions_selected':
            # All selected pairs across every session/type — handles come straight
            # from cd.ptr, no lookup needed.
            items = ui._export_mgr._collect_all_sessions_selected()

        elif action == 'all':
            # Pairs explicitly highlighted in the current-session listbox.
            raw = list(opt.get('_selected_pairs') or [])
            items = []
            for it in raw:
                pair = tuple(it['pair']) if isinstance(it, dict) else tuple(it)
                try:
                    items.append((ui.key, ui.ccg_ptr,
                                  int(pair[0]), int(pair[1])))
                except Exception:
                    pass

        elif action == 'bookmarked':
            # Bookmarks are per-session (current session only).
            items = []
            for p in sorted(ui.bookmarked_pairs):
                try:
                    items.append((ui.key, ui.ccg_ptr,
                                  int(p[0]), int(p[1])))
                except Exception:
                    pass

        elif action in ('groups', 'all_groups'):
            # Group data is stored as {session_str: [[ref,tgt], ...]}.
            # IMPORTANT: we must NOT “guess” a handle for (session, pair). Instead,
            # we scan each (tk_, ptr).pairs and include it only if that pair is
            # explicitly in the chosen group(s) for that session.
            if action == 'all_groups':
                gnames = [g for g in ui.groups
                          if g and not str(g).startswith('__')]
            else:
                gnames = list(opt.get('_selected_groups') or [])
                if not gnames:
                    QMessageBox.information(None, "Export", "No groups selected.")
                    return

            # Build desired pairs per session from group definitions
            want_by_sess: dict[str, set[tuple[int, int]]] = {}
            for g in gnames:
                try:
                    for sess, ref, tgt in ui.groups.forward(g):
                        want_by_sess.setdefault(str(sess), set()).add((ref, tgt))
                except Exception:
                    continue

            raw_items: list[tuple] = []
            seen: set[tuple] = set()
            found_by_sess: dict[str, set[tuple[int, int]]] = {}
            for tk_, ptr in ui.cd.ptr.items():
                if ptr is None:
                    continue
                try:
                    sess = str(getattr(tk_, 'session', tk_.nd().session))
                except Exception:
                    sess = ''
                want = want_by_sess.get(sess)
                if not want:
                    continue
                inds = getattr(ptr, 'pairs', None)
                if inds is None:
                    continue
                for pair in inds:
                    try:
                        ref, tgt = int(pair[0]), int(pair[1])
                    except Exception:
                        continue
                    if (ref, tgt) not in want:
                        continue
                    k = (id(tk_), ref, tgt)
                    if k in seen:
                        continue
                    seen.add(k)
                    raw_items.append((tk_, ptr, ref, tgt))
                    found_by_sess.setdefault(sess, set()).add((ref, tgt))

            items = sorted(
                raw_items,
                key=lambda x: (str(getattr(x[0], 'session', '')), x[2], x[3])
            )
            # Warn about any group pairs that are not present in loaded data for that session.
            missing = []
            for sess, want in want_by_sess.items():
                found = found_by_sess.get(sess, set())
                for p in sorted(want):
                    if p not in found:
                        missing.append((sess, p))
            if missing:
                preview = "\n".join([f"{s}: {p}" for s, p in missing[:15]])
                more = "" if len(missing) <= 15 else f"\n… ({len(missing) - 15} more)"
                QMessageBox.warning(None, 
                    "Export",
                    "Some pairs in the selected group(s) were not found in the loaded data and will be skipped.\n\n"
                    + preview + more
                )
        else:
            items = []

        if not items:
            QMessageBox.information(None, "Export", "No pairs to export.")
            return

        folder = QFileDialog.getExistingDirectory(
            None, f"Export {len(items)} view(s) to folder")
        if not folder:
            return

        ui._export_mgr._export_pairs_with_handles(fmt=fmt, opt=opt, items=items, folder=folder)


