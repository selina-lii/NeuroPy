# CCG

## CCGUI Display 

### Rules of UI description

1. UI components are described `t->d`: top-> down, or` l->r`: left->right. Within one line default is `l->r`.

2. {} = fill in variable; {str} str may not match var name exactly, but will be close

3. [] = describe component type or data logic, 

   e.g. [sash] [loop] [dropdown] [checkbox] [radio button] [expandable section]

### UI overview

Title bar: CCG Review (originally CCG Manual Review)

Logo: Maize CCG text over Blue square (UofM colors)

### Largest UI panel (t->d)

1. Dataset index bar: l->r:

````
Project: [dropdown0] Session: [dropdown1] Type: [dropdown2]
````

dropdown 0: loop all folder names in ui_data/projects.

dropdown 1: All, {Session_1}, ..., ... {Session_n}

dropdown2: [loop]{conn_type name: {EI} {ref_type}->{tgt_type}}

2. Hotkeys bar: (l->r) Groups: [fixed button: Del: deleted, grayed out] [arrow chip bar: loop "{hotkey}:{groupname}"] 

3. Time Slider
4. Main panel (l->r: Pair Selection, Main View, Neuron Network. See details below.)
5. Bottom info bar: Session overview

```
Significant: {n_sig}/{n_pairs} [loop]{neuron_type}:{n_neuron} Firing rate (FR): ref={}Hz, tgt={}Hz [if segment is not All]Segment FR: ref={}Hz, tgt={}Hz
```

### Menubar (l->r, each tab t->d)

#### Overview

1. Panels
   1. Pair Selection
   2. Main View (CCG) 
   3. Neuron Network
   4. Waveforms
   5. Time Slider
   6. Group Hotkeys
2. Groups
   1. Create group
   2. Manage groups
   3. Export groups
   4. Import groups
3. Selections
   1. Save selections
   2. Load selections
   3. [sash]
   4. Export PNGs
   5. [sash]
   6. Bookmark current pair
   7. Clear bookmarks
4. Modules
   1. Stats tests > Run stats test
   2. Jitter > 
      1. View queue
      2. Clear queue
   3. Simulation > New simulation
   4. Classify > Run classifier
5. Settings
   1. Settings

#### On-click behavior

##### Panels

All on clicks bring up the corresponding panel. A 'On' panel has a check mark before the label.

##### Groups

###### Create group: naming dialog (t->d)

* Group name: [input box] 
* [checkbox] Create as special group 
* Buttons "OK" and "Cancel"

###### Manage groups: complex dialog (t->d) all edits autosaved on tab swtich and close	

* [top tab panel] [loop] Entity tabs listing group names alphabetically

  Multiple rows if overflowed.

  Last item is "Special" which expands into special groups in nested panel identical to its outer panel. Special groups has no hotkey row. The pair indices are listed by session-name dividers rather than in a side tab panel.

* Group name: [input] [button:Rename]

* Hotkey (0-9/a-z): [input] [button:Set]

* Notes: 

* [scrollable expandable input box] 

* [draggable sash]

* Pairs in group:

* [side tab panel] l->r: 

  1. [indexes] name of sessions with at least pair in the group 
  2. [plain list] all pair indices in group in session formatted as "[ref tgt]"

* l->r: [button] Convert to special group [button] Delete group '{name}'

* [button] Close

###### Export groups: save dialog (t->d) ui_data/groups/groups_default.json

* Version name:
* [input box, default "groups_default"]
* [button] Save [button] Save to default [button] Cancel

###### Import groups: load dialog (t->d) ui_data/groups/groups_*.json

* Select a version to load:
* [list box] loop all files, first row is default, followed by user-saved versions, followed by [autosaved] {name}\t{datetime} 

##### Selections

###### Save selections: save dialog (t->d) ui_data/selections/{project_name}/*

* Version name:
* [input box, default datetime separated by '-']
* [button] Save [button] Save to latest [button] Cancel

Note: upon ui launch, load 'latest', saved as '{session name}__latest.json'

###### Load selections: load dialog (t->d)

same as `Import groups`

###### Export PNGs: complex dialog (t->d)

* Groups [area]

  [dual list selector] left: Available, right: selected. Options are groups.

* Color [area] 

  [table] [loop] CCG color, Baseline color, Connection strength shade color, Test window color, p-value line color, alpha threshold color.

  Each item has a [color picker] to its right. 

  A color picker has (l->r) 1. [platform or ui package default color picker] 2. name or #hex: [input box]

* Min text size: [input] pt

* [checkbox] Show legend

* X ticks [area]  (t->d)

  Use custom X ticks (ms, comma-separated): [input]

  [checkbox] Mirror to negative ticks

* Title [area] (l->r, autowrap)

  All [checkboxes]: Shanks, Neuron indices, Neuron type, Segment name, Normalization details, Session name.

* Misc [area] (t->d)

  l->r: 1. [checkbox] Adaptive test window 2. Print connection strength: [checkbox] STG [checkbox] JBSI

  l->r: Resolution: [checkbox] Lo-res [checkbox] Hi-res

* Segments to export [area] 

  [dual list selector] left: Available, right: selected. Options are segment chips.

* Subfolder by [area]

  [dual list selector, draggable] left: Available, right: Selected (drag to order). Options are: "conn type", "excitatory/inhibitory", "session", "test subject". Determines how output images are nested.

* [checkbox] Show preview

  [image] Display current pair's PNG according to the above config*

* [right aligned] (l->r) buttons. 
  * Export all
  * Export current pair
  * Save config [save dialog, ui_data/png_settings/*.json]
  * Load config [load dialog]
  * Cancel

###### Bookmark current pair: on click

###### Clear bookmarks: on click

##### Modules

###### Stats tests > Run stats test: complex dialog (t->d)

* Groups   Test type: [dropdown: "Independent t-test", "Pairwise t-test", "One-way ANOVA+Tukey", "Repeated measures ANOVA+Tukey"] Sides: [dropdown: "Two-sided", "One-sided"] [toggle button with text A>B or A<B] [checkbox] non-parametric [checkbox] log-transform

* [table] 

  column: Color, Name, Session, ConnType, Segment, Group, Data

  rows (l->r): 

  [delete x button] 

  [color picker] 

  [Name, default alphabet A, B, ...] 

  [list picker: Sessions..., label "{x} sessions or {name of session}"] 

  [dropdown: loop conn types, ref-tgt] 

  [dropdown: loop segments appended by "All segments"] 

  [list picker: Groups...] 

  [dropwdown: loop "Conn Strength", "CS norm (% change)", "CS norm (geometric)", "Ref firing rate", "Tgt firing rate", "Baseline", "Peak Width", "Peak Center"]

  A [list picker] is a button with label text and to its right a label briefing which items are selected. On-click brings up a dialog of multi-select list (ctrl, ctrl+shift) with row of buttons at the bottom: "Select all", "Select none", "Apply", "Cancel", the first two left aligned and the later two right aligned. Label text shows full name or abbreviation based on how many items are selected.

  default two rows.

  A [table] should end with a row containing "+ Add" button

* Results [area] (t->d) Display text of stats results, including p-value, input data overview...

  1. [Checkbox] Violin [cb] Show outliers [cb] Sig. brackets  W:H [input]

  2. [Image] Barplot of the groups data, if paired, connect data pairs by thin gray lines, show other formats according to check buttons

​	See current code for additional implementation details.

* buttons: Run, Export..., Load...

###### Jitter > View queue: queue dialog (t->d)

Very similar to load dialog.

* Queued tasks: 
* [list box] (empty) or queued task names
* [button] Delete selected [button] Refresh  [button, right-aligned] Close

###### Jitter > Clear queue: on click

###### Simulation > New simulation: complex dialog (t->d) [inputs]

* Name: [default str = sim1] Duration: [defualt num = 60] [dropdown: ms, s, min, hour]
* Noise (gauss /sigma): [0.0] Excess synchrony (%): [0] Synaptic delay (ms): [1.5]
* two columns: Reference neuron, Target neuron
* Each column (t->d): 
  * Nickename: 
  * [input: ref/tgt]
  * Type:
  * radio buttons: E, I, any
  * Firing rate (Hz): [default num = 5.0]
  * Burst config:
  * Burst rate (%): [default num = 0]
  * Number of spikes per burst: [default num = 3]
  * Burst interval (ms): [default num = 5.0]
* [area] simulated ccg
* buttons: Run simulation, Export simulation settings, Import simulation settings

###### Classify > Run classifier: on click, pop error "not implemented yet"

##### Settings > Settings

[side tab panel] Rows: Display, Cache, Autosave.

###### Display (t->d)

Max pairs in 'Show Together': [spinner, default=5, allow intergers 2-20] (2-20)

Minimum font size: [spinner, default=12, allow intergers 6-32] (6-32)

###### Cache (t->d)

Migrate current code

###### Autosave (t->d)

Autosave selections: [Toggle] interval: [input, default 1] [list:min/hour/day, default hour]

Autosave groups: [Toggle] interval: [input, default 1] [list:min/hour/day, default hour]

Save UI status: [Toggle]

Clear autosaved data: [button]

### Pair Selection

#### Appearance

* [top tab panel] Pair Selection, Spike Pairs

###### In Pair Selection (t->d):

* Two columns of lists 

Left:

 title: Available ({n_available_pairs})

  [scrollable list: loop " [{ref} {tgt}]" neuronal pair items. End list with grayed out portion led by sash item "--Deleted ({n_deleted})--" then loop deleted pairs. ]

Right: title: Selected ({n_selected_pairs})

  [scrollable list: loop " [{ref} {tgt}] [any tags comma separated in a square bracket]", can contain sash items "--{group name} ({n_pairs_in_group})-- (+when collapsed: '>>')"]

* [exclusive checkbox options: 1 or none] Sort by group, Sort by tag, Sort by mean, Sort by min p-val

###### In Spike Pairs (t->d):

* {n} spike pairs
* [scrollable list box: loop "{i} ref {ref_time} tgt {tgt_time} lag {+-x}ms". On click, a list item renders Plots displaying spike timing within +- 5ms window. Spike Pairs are channeled by Spike Attribution detailed later. Plots are 2 row, 1 column plots with the two coincident spikes centered and denoted by dashed redline; spikes within the window of interest are denoted by vertical lines.
* alternatively the plots display ephys traces (future feature)] 

#### Behavior

##### Selection

Selection loads all pair indices from SelectionData. If none, then load from CCGData.

Regardless of whether a pair is in Selected or Available.

Scrolling by two fingers or scrollbar.

On up/down arrows: scroll through list and show main plot. Left/right arrow does nothing within the selection panel.

On item double click: shuttle item between Available and Selected. 

On hotkey (alphanumeric):  assign/remove hotkey. advance to next pair (clamped). 

On holding shift + 1 or more hotkeys: assign/remove multiple hotkeys. *a flag is set in code to overwrite advance behavior during shift hold.

On holding ctrl/shift: regular multiselect behavior.

On del key: move to deleted.

On ctrl+b: bookmark. Bookmarked pair is red-highlighted and led by a bookmark pin icon.  

On ctrl+z: undo last action from stack.

On ctrl+y: redo last action from stack.

On ctrl+f: search bar appears with the corresponding rows highlighted. 

* "Search [input] [{nth}/{all}] [up arrow button] [down arrow button] [X close button]"

On right click: context menu (t->d):

* Move to Available/Selected

* Move to Deleted

* [sash]

* Group tag: [expandable list]: 

  * Create Group
  * [sash]
  * loop all existing groups, checkmark before already tagged groups
  * Special [expandable list: loop special groups, same logic]

* [sash]

* Show Together - stack pair cogs

* Clear 'show together' ({n} pairs) - only if has show together pairs.

* [sash]

* Pair notes - on click brings up [dialog] Tags (comma-separated): [input box] Notes: [input area] [buttons: Save, Cancel] each in its own row

  Note: If pair has non-empty pair notes, put "~" before pair tags bracket.

* Export view as PNG... - export current

* [sash] (and below only appears when Sort by group or Sort by tag)

* Collapse all groups

* Expand all groups

##### Sorting/Display

1. Default sort by index
2. Sort by tag: each group is headed by an item that collapses the group-specific list on double click.
3. Sort by group: sort by combination of all tags. Same header format.
4. Sort by mean: sort by mean plot value (largest first), no groups.
5. Sort by min p-val: sor by minimum plot p val (smallest first), no groups.

##### Coloring

1. If "Hide same channel" or "Hide same shank" is on (detailed in Neuron Network): Gray out pairs accordingly.
2. Highlight currently selected pair.
3. Having unchecked jitter result changes pair highlight.



### Main View (CCG)

#### Appearance

##### Main Plot + Waveform

Main plot supports static CCG display and dynamic configuration through multiple sections below. (t->d):

* Title "{session name} | {ref_type}-{tgt_type} | {ref_ind}->{tgt_ind} -- {segment_name} "

* Normally there is only a main Plot from plotting/ccg_plot.py.

  * if ctrl+e or Menu>Panels>Waveform is toggled, a **Waveforms** channel from plotting/probe.py appears to the right of the CCG plot within the same panel.

* Plot Toolbox (t->d):

  * Segments: [arrow chip bar: All | [loop segments] | [loop custom segments]] [button: lo|hi, CS. right aligned]

  * [expandable section] Normalization (t->d)

    * chip buttons: Ref f-rate, Tgt f-rate, Time (hr), Time (sec), CCG total area, Subtract Baseline, Same scale (pair), Same scale (session). Wrap line if too long. [button, right-aligned: Apply to data. ]

      Note: "Apply to data" triggers [save dialog] of new CCGData, all other fields identical except CCG normalized by current definition; normalization setting is documented in CCGData.

  * [expandable section] Correlogram

    * [cycle button: CCG, 3 states:  ■-solid, □-line, x-hide] [cycle button: baseline, 3 states:  ■-solid, □-line, x-hide] [|] Show ACG [cycle button: ref, 3 states:  x-hide, □-line,  ■-solid] [cycle button: tgt, 3 states:  x-hide, □-line,  ■-solid], [chip button: ref waveform]
    * ACG scale [chip button: Autoscale, False - auto match ACG to CCG scale] ref: [slider with input default num = 1.0] tgt: [slider with input default num = 1.0]  [|] Deconvolve [chip button: ref] [chip button: tgt]
    * [checkbox] Extend: {default num = 50, spin box: 5, 10, 20, 50, 100, 200, 500, 1000}ms  resolution: {default 1.0, spinner: {min_bin_size - clip to this value}, 0.1, 0.5, 1, 2, 5, 10, 20, 50, 100} ms, free entry 

  * [expandable section] Baseline & Connection Strength 

    * [checkbox: Show CS overlay] Measure: [radio buttons: STG, JBSI]
    * CS: {lo}|{hi} [chip button: non-negative]
    * Baseline: [radio buttons: Conv, Tailed, Global, Jitter] [|] Test window: [chip button: show, default True] [raised button: Adaptive - if adaptive, move test window to 0-1ms; future TODOs incoming]
    * [chip button: p] [chip button: p-corrected] - if Conv or Jitter radio button is on. [explanation of baseline. "Conv: Convolution smoothed null baseline", "Tailed: ACG deconvolution, tail-bin baseline", "Global: max bin outside of test window as baseline", "Jitter: Bootstrapped baseline from surrogate data using interval jitter" ]

  * [expandable section] Jitter

    * n=[spinner: default=100, spinner suggested values=10,20,50,100,200,500,1000; all integers] [buttons: Run Jitter, Clear, Save] Resolution: [chip button: lo, default True] [chip button: hi]

  * [expandable section] Spike Attribution

    * [raised button: Enable] Bin: [input: default num=0] [selectable list: ms, # - if "#", bin is {+-i}th relative to 0ms bin] [button: Set]

#### Features

##### Context menu

On context menu of plot (t->d):

* Export view as PNG...
* [sash]
* [expandable menu item] View values in terminal
  * CCG
  * reference ACG
  * target ACG
  * baseline
  * p-values  Note: uncorrected, corrected - print all p-vals we have 

##### Keyboard

l/r arrow: scroll plot segments (wrap);

Ctrl+R: toggle lo/hi res

##### Update plot

switch pair, switch segment, segment multi-stacking (ctrl+shift on chips, show in row={n_resolution} col=N plots), any norm toggle, CCG cycle/deconv/extend, CS/baseline/sig toggles, adaptive test window, waveforms toggle, resolution switch







### Neuron Network

#### Appearance

##### Toolbox (t->d)

* Focus: [chip button: Neuron] [input: empty] | [chip button: Pair] [input: empty] [button: Add to available]
* [expandable section: Lines]  (t->d)
  * [chip button: ON/OFF] [|] [chip button: Current pair] [|] Conn type: [4 chip buttons: P->P, P->I, I->P, I->I (the conn types here should be data-coupled with Type: from Dataset index bar, taking unique initials)] 
  * [chip button: Hide] [|] [chip buttons: Same channel, Same shank]
*  [expandable section: "Groups {"(highlighted/session/all)", if counts is on} [checkbox, right aligned: show counts] [button: clear all]] (t->d)
  * [area]
    * [area1] loop chip buttons of all groups. label is group name or "{group name} ({n_highlighted}/{n_session}/{n_all})" if counts is on.
    * [sash]
    * [area2] same format for special groups
* [expandable section: Annotations] (t->d)
  * Probe shanks [buttons, right aligned: None, All] 
  * [loop chips of all shanks, default all on]
  * [sash]
  * Zoom 
  * H scale: [H scale slider] V scale: [V scale slider] 
  * alpha: [line transparency alpha] spread: [slider: how far same-channel neurons are]
  * Annotations
  * [chip buttons: channel ids, neuron ids, pair inds]

##### Plot

**Probes**: 

* Shanks (vertical sticks) are layed out horizontally. shanks themselves are invisible, only channel locations (electrode contact points distributed on the shanks). shanks are layed out with physically realistic distance referencing the probe used by this dataset, which is already done in the code (somewhat hardcoded). 
* shanks are titled "S0"..."Sn" on top. Multiple probes are layed out distant from each other; no reaslitic distance. Channels are gray dots, although bad channels are red-outlined gray dots. 

**Neurons**: 

* Triangle for pyramidal cells and circles for interneurons (no accomodation for other neuronal types for now). 
* Multiple neurons on the same shank are scattered with a small overlap and in gradient series of gray (20% apart) to be visually distincitive. 
* Neurons are clickable, on click, it becomes the focus neuron.

**Pairwise Connections** (Edges): 

* Colored arrows from reference to target neuron. uses a red, yellow, green, blue color scheme. Arrows are clickable, on click they thicken and assumes a black outline. 
* Same-channel connection is represented by a circular arrow since they start and end at the same point; multiple same-channel connections becomes concentric circles with increasingly large radius. 
* Edges are clickable, on click, it becomes the focus pair.
* deleted pairs have gray arrows.

**Legend**: currently-on connection types and their line colors.



### All-Session View

All-session view is enabled by selecting "All", the first item in "Session:", which combines all selection data from multiple sessions. 

Most UI components stay the same, except:

* Pair Selection by default "Sort by tag" across all groups, and no pairs are shown in Available.
* Pair Selection disables "Sort by mean" and "Sort by min p-val" buttons. 
* Neuron Network plots becomes an arrow chip bar, so that user can scroll through neuron networks of all sessions.
* Time Slider plot is replaced by text explaining that multiple sessions do not have a single time slide.
* Time slider range doesn't get resolved to time, but stays as strings if given "start" "end".



### Time Slider

Time slider computes custom CCG by time-filtering behavioral variables.

#### Appearance

Time slider area spans the whole UI width. t->d:

* Time Slider - Behavioral Epochs
* [area]:
  * Theme: [dropdown: segments, loop through epochs] {x} themes available  [checkbox: Include in filter] Show: [dropdown:loop all labels in epoch, + NONE] [button: All] [button, right aligned starting from here: {Save icon}, {Load icon}, [|], [checkbox: Snap], [checkbox: Zoom-in], [checkbox: {lock icon] Lock]]
  * loop [chips: [color square] {labels in epoch}]+[chip:[gray square] NONE]
  * plot of behavioral epochs. The plot is a horizontal bar with a single x axis as time (hours). Blocks representing epoch segments are horizontally stacked. Any gaps without a label is "NONE". Use the same color scheme and see more details in time_slider.py::_draw_epochs.
    * clicking on the plot area sets two vertical blue lines that makes a box if both are present. By default (no lock or zoom-in), this sets the Start and End times, and snaps to edges if Snap is on. See below for when Zoom-in is on. Lock sets view-only.
  * CCG time range  Start:[input, default 00:00:00, formatted] End:[input,default 00:00:00, formatted] [button:Set] Name: [input, empty] Splits: [spinner, 1 or larger integers] Overlap: [input, default=0] [dropdown:%, hr, min] [button:Clear] [button:Apply to Multiple Sessions] (* start and end boxes can take "start" and "end" as input, which will parse timing to start/end timestamps)
  * (on Zoom-in:) plot of zoom window, very much the same as the previous two parts combined, except:
    * the selection box is red and filled with beige color. dashed red lines extend from two edges of selection to two edges of zoom-in view, indicating that selected range is enlarged into full UI width below. on the bottom row, "CCG time range" is replaced by "Zoom range" and only start/end/set widgets remain.

#### Behavior

* On click, Apply to Multiple Sessions: brings up dialog that allows ctrl/shift multiselection, has 'apply' 'cancel' buttons, and computes custom CCG
* On click, Load: show [side tab panel] dialog that can be indexed either by session or by customCCG name. 
  * bottom row: buttons: Load selected, Delete selected, Refresh list, Cancel
  * Data entires are "{session name} {dot} {start:end} {dot} {name}"
* Custom CCGs are uniquely identified by name and saved to data/custom_ccg in .npz.
* On click, Set runs custom CCG.
  * If a chip is on, its label is included.
  * If split/overlap is set, CCGs are generated in a sliding window.
  * if Include in filter, take all Themes that has this button on and intersect their criteria.  



## Data structures

#### Ephys curated

```
Data must-haves:
	neurons
	basepath
Strongly recommended:
	paradigm and other Epochs -> time slider
	probe info -> neurons network, waveform
Other optional data...
```

##### Session (subjects.py, Diba lab curated data)

```
File path, data, metadata:
	basepath, filePrefix, sub_name, tag
	eegfile, datfile, animal
	Probe (probegroup, best_channels, recinfo.skipped_channels)

Epoch objects:
	paradigm
	artifact
	brainstates
	Oscillations (sw, spindle, ripple, theta, theta_epochs, pbe)
	off_epochs, micro_arousals
	Behavior (maze_run, remaze_run, maze1_run, maze2_run, handling)

ProcessData objects:
	neurons, neurons_stable, neurons_iso, mua
	position, maze, remaze, maze1, maze2
```



#### Key

```
session
epoch
(ref_ind)
(target_ind)
segment
excitability
conn_type

__eq__
matches
get
remove
add
change
nd
```

#### NeuronsDataset

```
neurons
probe_info
_sessions
_conf
	name
	neuron_types
	epochs
	themes
	ch_per_shank #TODO build into subject.py
	
__init__ < _prep
_prep < _short_session_name, _load_neurons, _load_probe_info, _load_themes
_short_session_name
_load_neurons
_load_probe_info #TODO build info into subject.py
_load_themes
```

EpochSlicingConfig

```
labels
min_dur
discard
```



#### CCGDataset

```
ccg: Dict[Key, CCGData]
	key
	ccg
	ccg_null
	pval (pval_corrected)
	qval (qval_corrected)
	significant
	conf: CCGConfig
ptr: Dict[Key, CCGPointer]
	key
  inds
  selected_inds
  significant
	conf: CCGConfig
conf: CCGConfig
nd: NeuronsDataset
```



#### CCGConfig

```
      name
      conn_types (conn_types_E, conn_types_I, conn_types_flat)
      duration
      bin_size
      resolution
      conv_window
      alpha
      alpha2
      min_lag
      max_lag
      min_spkcount
      spkcount_scope
      multiple_correction
      use_accelration
      symmetrize_ccg
```



#### CCGSourceConfig

```
name
t0
t1
scope
created_from_session
sessions
n_splits
overlap_sec
filter_state
active_duration
total_time_hours
windows
firing_rates
tags
src_path
```



#### SelectionDataset (project owner, mirrors CCGDataset/cd)

```
groups: Groups                       # project-wide group memberships + metadata
_sessions: dict[session_str, SelectionData]   # lazily materialized per session
_save_dir: str
sel_for(session_str) -> SelectionData         # get-or-create
to_save_dict(session_str, type_keys, pair_to_ct, saved_at, session_label) -> dict
reset()
```

`to_save_dict` JSON keys (preserved exactly for load compat):
`session, saved_at, selections, deleted_by_type, selected, deleted, pair_tags`.
Per-type selection/deleted routed by each type-key's own session (any-session mode spans sessions).

#### SelectionData (per session)

```
_by_key: dict[str(Key), _SelectionData]   # one bucket per conn-type Key
pair_tags: dict[ct_label, {(ref,tgt): {groups,notes,tags}}]
_listed: set                           # pairs admitted into list view

# ONE live accessor — no active-type pointer here:
bucket(key_str) -> _SelectionData         # get/create the bucket for a Key
__contains__(key_str) -> bool          # does the bucket exist yet
ct_tags / reset_for_project
```

`_SelectionData = {selected, unselected, deleted}` (persistent, not a cache);
owns the state transitions `set_state(pair, state)` and `populate(all, selected, deleted)`.

No `_current_key` on SelectionData — *which conn-type is visible* lives in `AppState.key`.
`AppState.active_bucket` = `sel_data.bucket(str(self.key))` is the sole key→bucket resolver;
UI reads/writes the visible state through it (adapter exposes `active_bucket` + the legacy
`selected_inds/unselected_inds/deleted_inds` triplet delegating to it).

Owned by AppState via `nav.sd`; `nav.sel_data` = `sd.sel_for(current_session)`,
`nav.groups` = `sd.groups`. Deletions formerly in `CCGReviewUI._pair_deleted_store`
now live in each bucket's `deleted` (deleted ⊆ ptr pairs ⊆ universe, so active view == full store).




#### JitterDataset







#### Analysis

SelectionDataset → SelectionData (see above)





pyvista





## Future

### Onboarding (New)

Dialog prior to entering actual UI. Onboarding on first launch.

* Welcome to CCG ReviewUI
* buttons: Open project, Create project, Browse public datasets 

###### Open project

List dialog of folders in ui_data/, if empty, notify user they don't have any projects yet and return

###### Create project

1. Ask user to import a curated dataset. Provide options by io/ capacity. User helps determine which io/ file to use to import their data to be neuropy compatible. Cues come from which software they used to processs data (e.g. Phy) and/or what file format is (e.g. .nwb).
2. User is responsible for a correctly curated dataset, but NeuroPy should parse NeuronsDataset fields.
3. NeuroPy should overview what fields are parsed correctly, and prompt user to look for important missing fields (are they simplify misnamed that io/ didn't catch? or are they missing and user needs to update the dataset?).
4. The curated dataset is assumed to have a certain file structure, for example  (1+ subjects >) 1+ sessions > data files.
5. If OG data is missing, the project can start from halfway in the pipeline (e.g. CCGDataset, CCGPointer, SelectionData) but user will be warned certain features involving the original dataset are not available 

###### Browse public datasets

1. List box of available public datasets that are linked to web, can be pull from github (pure data repos) via per-item download button, plugged into ui_data and instantly loaded to display processed datasets.
2. Developer/lab shares public datasets, make selection data and publish them in github. Developer is responsible for maintaining integrity of public dataset.

###  Various

- Pair selection/CCG data: significance threshold slider
- Time slider: 
  - compute for all labels in theme
  - Min duration limit
- 









## Code style

Never:

* in-line import
* section banner comments

Always:

*  `__str__` at the end of the class
* 

## Glossary

```
[|] vertical sash
[arrowed chip bar] ◀ [chips] ▶ paging
[chip button] raised push toggle (not checkbox)
[cycle button] chip with arbitrary state cycle + icon
[spinbox] numeric stepper
[dataset index bar] Project | Session | Type dropdowns
[plot toolbox] sections below CCG plot
[list picker] button + summary -> multi-select dialog
[dual list selector] Available <-> Selected shuttle
```









