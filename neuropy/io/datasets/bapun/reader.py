"""Reading a Bapun-lab session directory: the .npy sidecars it carries and what each holds."""
from __future__ import annotations

import zipfile
from collections import namedtuple
from pathlib import Path

import numpy as np
import pandas as pd

from neuropy import core
from neuropy.io import BinarysignalIO, NeuroscopeIO


class ProcessData:
    def __init__(self, basepath, tag=None, use_realigned_pos=False):
        basepath = Path(basepath)
        self.basepath = basepath
        try:
            xml_files = sorted(basepath.glob("*.xml"))
            assert len(xml_files) == 1, f"Found {len(xml_files)} .xml files"
            fp = xml_files[0].with_suffix("")
            self.recinfo = NeuroscopeIO(xml_files[0])
            if self.recinfo.eeg_filename.is_file():
                self.eegfile = BinarysignalIO(
                    self.recinfo.eeg_filename,
                    n_channels=self.recinfo.n_channels,
                    sampling_rate=self.recinfo.eeg_sampling_rate,
                )
            if self.recinfo.dat_filename.is_file():
                self.datfile = BinarysignalIO(
                    self.recinfo.dat_filename,
                    n_channels=self.recinfo.n_channels,
                    sampling_rate=self.recinfo.dat_sampling_rate,
                )

        except:
            fp = basepath / basepath.name

        self.filePrefix = fp
        self.sub_name = fp.name[:4]

        self.tag = tag

        if (f := self.filePrefix.with_suffix(".animal.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.animal = core.Animal.from_dict(d)
            self.name = self.animal.name + self.animal.day

        self.probegroup = core.ProbeGroup.from_file(fp.with_suffix(".probegroup.npy"))

        # ----- epochs --------------
        # epoch_names = [
        #     "paradigm",
        #     "artifact",
        #     "brainstates",
        #     "spindle",
        #     "ripple",
        #     "theta",
        #     "pbe",
        # ]
        # for e in epoch_names:
        #     setattr(self, e, core.Epoch.from_file(fp.with_suffix(f".{e}.npy")))
        if (f := self.filePrefix.with_suffix(".best_channels.npy")).is_file():
            best_chans = namedtuple("BestChannels", ["theta", "slow_wave"])
            d = np.load(f, allow_pickle=True).item()
            self.best_channels = best_chans(d["theta"], d["slow_wave"])

        self.paradigm = core.Epoch.from_file(fp.with_suffix(".paradigm.npy"))
        self.artifact = core.Epoch.from_file(fp.with_suffix(".artifact.npy"))
        # self.brainstates = core.Epoch.from_file(fp.with_suffix(".brainstates.npy"))

        self.brainstates = core.Epoch.from_file(fp.with_suffix(".brainstates.finer.npy"))

        self.sw = core.Epoch.from_file(fp.with_suffix(".sw.npy"))
        self.spindle = core.Epoch.from_file(fp.with_suffix(".spindle.npy"))
        self.ripple = core.Epoch.from_file(fp.with_suffix(".ripple.npy"))
        self.theta = core.Epoch.from_file(fp.with_suffix(".theta.npy"))
        self.theta_epochs = core.Epoch.from_file(fp.with_suffix(".theta.epochs.npy"))
        self.pbe = core.Epoch.from_file(fp.with_suffix(".pbe.npy"))
        # self.off = core.Epoch.from_file(fp.with_suffix(".off.npy"))
        self.off_epochs = core.Epoch.from_file(fp.with_suffix(".off_epochs.npy"))
        self.micro_arousals = core.Epoch.from_file(fp.with_suffix(".micro_arousals.npy"))

        self.maze1_run = core.Epoch.from_file(fp.with_suffix(".maze1.running.npy"))
        self.maze2_run = core.Epoch.from_file(fp.with_suffix(".maze2.running.npy"))
        if not use_realigned_pos:
            self.maze_run = core.Epoch.from_file(fp.with_suffix(".maze.running.npy"))
            self.remaze_run = core.Epoch.from_file(fp.with_suffix(".remaze.running.npy"))
        elif use_realigned_pos:
            self.maze_run = core.Epoch.from_file(fp.with_suffix(".maze.running.realigned.npy"))
            self.remaze_run = core.Epoch.from_file(fp.with_suffix(".remaze.running.realigned.npy"))

        # Piezo epochs caputuring interruptions during sleep deprivations
        self.handling = core.Epoch.from_file(fp.with_suffix(".handling.npy"))

        # ---- neurons related ------------

        if (f := self.filePrefix.with_suffix(".neurons.iso.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.neurons_iso = core.Neurons.from_dict(d)

        if (f := self.filePrefix.with_suffix(".position.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.position = core.Position.from_dict(d)

        if not use_realigned_pos:
            if (f := self.filePrefix.with_suffix(".maze.linear.npy")).is_file():
                d = np.load(f, allow_pickle=True).item()
                self.maze = core.Position.from_dict(d)

            if (f := self.filePrefix.with_suffix(".remaze.linear.npy")).is_file():
                d = np.load(f, allow_pickle=True).item()
                self.remaze = core.Position.from_dict(d)
        elif use_realigned_pos:
            if (f := self.filePrefix.with_suffix(".maze.linear.realigned.npy")).is_file():
                d = np.load(f, allow_pickle=True).item()
                self.maze = core.Position.from_dict(d)

            if (f := self.filePrefix.with_suffix(".remaze.linear.realigned.npy")).is_file():
                d = np.load(f, allow_pickle=True).item()
                self.remaze = core.Position.from_dict(d)

        if (f := self.filePrefix.with_suffix(".maze1.linear.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.maze1 = core.Position.from_dict(d)

        if (f := self.filePrefix.with_suffix(".maze2.linear.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            self.maze2 = core.Position.from_dict(d)

    @property
    def delta_wave(self):
        if (f := self.filePrefix.with_suffix(".delta_wave.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    @property
    def emg(self):
        if (f := self.filePrefix.with_suffix(".emg.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Signal.from_dict(d)
        else:
            return None

    @property
    def pbe_filters(self):
        """Code in 'sd_pbe_creation.ipynb'. This data has additional columns for PBEs depicting criteria such as:
        1) is_rpl
        2) is_5units
        3) is_80percetbins
        4) is_rest
        """
        if (f := self.filePrefix.with_suffix(".pbe.filters.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    @property
    def get_pbe_filters_bool(self, is_rpl=True):
        if (f := self.filePrefix.with_suffix(".pbe.filters.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            pbe_filters = core.Epoch.from_dict(d)
            good_bool = (
                pbe_filters.is_rpl
                & pbe_filters.is_5neurons
                # & pbe_filter.is_lowtheta
                & pbe_filters.is_rest
            )

    @property
    def replay_filtered(self):
        """Contains events which satisfy the following criteria:
        1) has 1std ripple power
        2) has atleast 5 units firing
        3) ripple happens during rest
        4) contains continuous trajectory with jump distance < 40 cm

        Code cell in 'sd_replay_filters.ipynb' and was generated using '.pbe.replay.mua' file.
        """
        # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
        if (f := self.filePrefix.with_suffix(f".pbe.replay.filtered.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    @property
    def replay_filtered20jd(self):
        """Contains events which satisfy the following criteria:
        1) has 1std ripple power
        2) has atleast 5 units firing
        3) ripple happens during rest
        4) contains continuous trajectory with jump distance < 20 cm

        Code cell in 'sd_replay_filters.ipynb' and was generated using '.pbe.replay.mua' file.
        """
        # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
        if (f := self.filePrefix.with_suffix(f".pbe.replay.filtered.jumpthresh20.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    @property
    def replay_filtered1h(self):
        """Contains events which satisfy the following criteria:
        1) has 1std ripple power
        2) has atleast 5 units firing
        3) ripple happens during rest
        4) contains continuous trajectory

        Code cell in 'sd_replay_filters.ipynb' and was generated using '.pbe.replay.mua' file.
        """
        # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
        if (f := self.filePrefix.with_suffix(".pbe.replay.filtered1h.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    @property
    def replay_pbe_mua(self):
        # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
        if (f := self.filePrefix.with_suffix(".pbe.replay.mua.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    @property
    def replay_pbe_mua_column(self):
        # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
        if (f := self.filePrefix.with_suffix(".pbe.replay.mua.column.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    # @property
    # def replay_pbe_mua_column_max(self):
    #     # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
    #     if (
    #         f := self.filePrefix.with_suffix(".pbe.replay.mua.column_cycle.maxjd.npy")
    #     ).is_file():
    #         d = np.load(f, allow_pickle=True).item()
    #         return core.Epoch.from_dict(d)

    # @property
    # def replay_wcorr(self):
    #     # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
    #     if (f := self.filePrefix.with_suffix(".pbe.wcorr.npy")).is_file():
    #         d = np.load(f, allow_pickle=True).item()
    #         return core.Epoch.from_dict(d)

    # @property
    # def replay_radon(self):
    #     # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
    #     if (f := self.filePrefix.with_suffix(".pbe.radon.npy")).is_file():
    #         d = np.load(f, allow_pickle=True).item()
    #         return core.Epoch.from_dict(d)

    @property
    def replay_radon_mua(self):
        # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
        if (f := self.filePrefix.with_suffix(".pbe.radon.mua.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    # @property
    # def replay_wcorr_mua(self):
    #     # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
    #     if (f := self.filePrefix.with_suffix(".pbe.wcorr.mua.npy")).is_file():
    #         d = np.load(f, allow_pickle=True).item()
    #         return core.Epoch.from_dict(d)

    # @property
    # def replay_spearman(self):
    #     # if (f := self.filePrefix.with_suffix(".replay_pbe.npy")).is_file():
    #     if (f := self.filePrefix.with_suffix(".pbe.replay.spearman.npy")).is_file():
    #         d = np.load(f, allow_pickle=True).item()
    #         return core.Epoch.from_dict(d)

    @property
    def remaze_replay_pbe(self):
        if (f := self.filePrefix.with_suffix(".remaze_replay_pbe.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Epoch.from_dict(d)

    @property
    def neurons(self):
        # it is relatively heavy on memory hence loaded only while required
        if (f := self.filePrefix.with_suffix(".neurons.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Neurons.from_dict(d)

    @property
    def neurons_stable(self):
        # it is relatively heavy on memory hence loaded only while required
        if (f := self.filePrefix.with_suffix(".neurons.stable.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Neurons.from_dict(d)

    @property
    def mua(self):
        if (f := self.filePrefix.with_suffix(".mua.npy")).is_file():
            d = np.load(f, allow_pickle=True).item()
            return core.Mua.from_dict(d)

    def get_zt_1h(self, include_pre=True, include_maze=True, pre_length=2.5):
        post = self.paradigm["post"].flatten()
        # post_starts = np.array([0, 4, 5]) * 3600 + post[0]
        post_starts = np.array([0, 1, 2, 3, 4, 5]) * 3600 + post[0]
        post_stops = post_starts + 3600

        # labels = ["0-1", "4-5", "5-6"]
        labels = ["0-1", "1-2", "2-3", "3-4", "4-5", "5-6"]

        if include_maze:
            maze = self.paradigm["maze"].flatten()
            post_starts = np.insert(post_starts, 0, maze[0])
            post_stops = np.insert(post_stops, 0, maze[1])
            labels = ["MAZE"] + labels

        if include_pre:
            pre = self.paradigm["pre"].flatten()
            pre = [np.max([pre[0], pre[1] - pre_length * 3600]), pre[1]]

            post_starts = np.insert(post_starts, 0, pre[0])
            post_stops = np.insert(post_stops, 0, pre[1])
            labels = ["PRE"] + labels

        return core.Epoch.from_array(post_starts, post_stops, labels)

    def get_zt_epochs(self, include_pre=True, include_maze=True):
        post = self.paradigm["post"].flatten()
        # zts = np.array([0, 2.5, 5])
        # post_starts = zts * 3600 + post[0]
        # post_stops = post_starts + 2.5 * 3600
        zts = np.arange(0, 8, 2.5) * 3600 + post[0]
        post_starts, post_stops = zts[:-1], zts[1:]

        labels = ["0-2.5", "2.5-5", "5-7.5"]

        if include_maze:
            maze = self.paradigm["maze"].flatten()
            post_starts = np.insert(post_starts, 0, maze[0])
            post_stops = np.insert(post_stops, 0, maze[1])
            labels = ["MAZE"] + labels

        if include_pre:
            pre = self.paradigm["pre"].flatten()
            pre = [np.max([pre[0], pre[1] - 2.5 * 3600]), pre[1]]

            post_starts = np.insert(post_starts, 0, pre[0])
            post_stops = np.insert(post_stops, 0, pre[1])
            labels = ["PRE"] + labels

        return core.Epoch.from_array(post_starts, post_stops, labels)

    def get_sliding_zt_epochs(
        self, window=900, slideby=None, include_pre=True, include_maze=True
    ):
        post = self.paradigm["post"].flatten()
        post_dur = (post[1] - post[0]) / 3600

        if slideby is None:
            slideby = window

        if self.tag == "NSD_HM":
            post_starts = np.arange(0, 2.7 * 3600 + window, slideby) + post[0]
        elif self.tag == "NSD_GM":
            post_starts = (
                np.arange(0, (post_dur - 0.3) * 3600 + window, slideby) + post[0]
            )
        else:
            post_starts = np.arange(0, 7.5 * 3600 + window, slideby) + post[0]

        post_stops = post_starts + window
        post_mids = (post_starts + post_stops) / 2
        labels = [np.round((t - post[0]) / 3600, 2) for t in post_mids]

        if include_maze:
            maze = self.paradigm["maze"].flatten()
            post_starts = np.insert(post_starts, 0, maze[0])
            post_stops = np.insert(post_stops, 0, maze[1])
            labels = ["MAZE"] + labels

        if include_pre:
            pre = self.paradigm["pre"].flatten()
            pre = [np.max([pre[0], pre[1] - 2.5 * 3600]), pre[1]]

            post_starts = np.insert(post_starts, 0, pre[0])
            post_stops = np.insert(post_stops, 0, pre[1])
            labels = ["PRE"] + labels

        return core.Epoch.from_array(post_starts, post_stops, labels)

    @property
    def data_table(self):
        files = [
            "paradigm",
            "artifact",
            "brainstates",
            "spindle",
            "ripple",
            "theta",
            "pbe",
            "neurons",
            "position",
            "maze.linear",
            "re-maze.linear",
            "maze1.linear",
            "maze2.linear",
        ]

        df = pd.DataFrame(columns=files)
        is_exist = []
        for file in files:
            if self.filePrefix.with_suffix(f".{file}.npy").is_file():
                is_exist.append(True)
            else:
                is_exist.append(False)

        df.loc[0] = is_exist
        df.insert(0, "session", self.filePrefix.name)

        return df

    def save_data(d, f):
        np.save(f, arr=d)

    def create_time_machine(self, suffix):
        files_id = [
            "animal",
            "paradigm",
            "artifact",
            "brainstates",
            "probegroup",
            "neurons",
            "neurons.stable",
            "ripple",
            "pbe",
            "position",
            "maze.linear",
            "maze.running",
            "remaze.linear",
            "remaze.running",
            "pbe.filters",
            "pbe.replay.filtered",
            "pbe.replay.mua",
            "pbe.replay.mua.column",
            "pbe.replay.mua.column_cycle.maxjd",
            "pbe.wcorr",
            "pbe.radon",
            "pbe.radon.mua",
            "pbe.wcorr.mua",
            "mua",
            "off_epochs",
        ]
        files = [self.filePrefix.with_suffix(f".{_}.npy") for _ in files_id]

        suffix = f".{suffix}.zip"
        zip_filename = self.filePrefix.with_suffix(suffix)
        with zipfile.ZipFile(zip_filename, "w") as zipF:
            for file in files:
                if file.is_file():
                    zipF.write(file, file.name, compress_type=zipfile.ZIP_DEFLATED)

            print("files has been compressed")

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.recinfo.source_file.name})\n"


def data_table(sessions: list):
    df = []
    for sess in sessions:
        df.append(sess.data_table)

    return pd.concat(df, ignore_index=True)


