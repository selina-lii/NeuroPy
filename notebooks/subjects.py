from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from neuropy import core
from neuropy.io import BinarysignalIO, NeuroscopeIO
from scipy.ndimage import gaussian_gradient_magnitude
from scipy import stats
import matplotlib as mpl
from neuropy import plotting
from collections import namedtuple
import zipfile


class SdFig:
    def __init__(self) -> None:
        self.common_settings = dict(
            fontsize=5, axis_lw=0.8, tick_size=2, constrained_layout=False
        )

    def fig1(self, nrows=9, ncols=12):
        fig = plotting.Fig(nrows, ncols, **self.common_settings)
        return fig

    def fig_supp(self, nrows=8, ncols=8, **kwargs):
        fig = plotting.Fig(nrows, ncols, **self.common_settings, **kwargs)
        return fig

    def fig1_supp(self, nrows=8, ncols=8):
        fig = plotting.Fig(nrows, ncols, **self.common_settings)
        return fig

    def fig2(self, nrows=8, ncols=10):
        fig = plotting.Fig(nrows, ncols, **self.common_settings)
        return fig

    def fig2_supp(self, nrows=8, ncols=8):
        fig = plotting.Fig(nrows, ncols, **self.common_settings)
        return fig

    def fig3(self, nrows=14, ncols=6):
        fig = plotting.Fig(nrows, ncols, **self.common_settings)
        return fig

    def fig4(self, nrows=6, ncols=5):
        fig = plotting.Fig(nrows, ncols, **self.common_settings)
        return fig


def get_statannot_ranksum():
    from statannotations.stats.StatTest import StatTest

    custom_long_name = "Wilcoxon_ranksum"
    custom_short_name = "Wilcoxon_ranksum"
    custom_func = stats.ranksums
    custom_test = StatTest(custom_func, custom_long_name, custom_short_name)
    return custom_test


def adjust_lightness(color, amount=0.5):
    import colorsys

    import matplotlib.colors as mc

    try:
        c = mc.cnames[color]
    except:
        c = color
    c = colorsys.rgb_to_hls(*mc.to_rgb(c))
    c = colorsys.hls_to_rgb(c[0], max(0, min(1, amount * c[1])), c[2])
    return mc.to_hex(c)


def colors_sd(amount=1):
    return [
        adjust_lightness("#424242", amount=amount),
        adjust_lightness("#eb4034", amount=amount),
    ]


def colors_sd_light(amount=1):
    return [
        adjust_lightness("#707070", amount=amount),
        adjust_lightness("#f18179", amount=amount),
    ]


def colors_tn(amount=1):
    return [
        adjust_lightness("#e9cc2b", amount=amount),
        adjust_lightness("#12d399", amount=amount),
    ]


def colors_rs(amount=1):
    return [adjust_lightness("#5599ff", amount=amount)]


colors_sleep = {
    "AW": "k",
    "QW": "k",
    "REM": "k",
    "NREM": "k",
}


colors_sleep_old = {
    "nrem": "#a3a3a3",
    "rem": "#a3a3a3",
    "quiet": "#a3a3a3",
    "active": "#a3a3a3",
}

hypno_kw = dict(labels_order=["NREM", "REM", "QW", "AW"], colors=colors_sleep)


lineplot_kw = dict(
    marker="o",
    err_style="bars",
    linewidth=1,
    legend=None,
    mew=0.2,
    markersize=2,
    err_kws=dict(elinewidth=1, zorder=-1, capsize=1),
)

errorbar_kw = dict(
    marker="o",
    capsize=1,
    elinewidth=1,
    mec="w",
    markersize=2,
    linewidth=1,
    mew=0.2,
)


def boxplot_kw(color, lw=1):
    return dict(
        showfliers=False,
        linewidth=lw,
        boxprops=dict(facecolor="none", edgecolor=color),
        showcaps=True,
        capprops=dict(color=color),
        medianprops=dict(color=color, lw=lw),
        whiskerprops=dict(color=color),
    )


stat_kw = dict(
    text_format="star",
    loc="inside",
    # verbose=True,
    fontsize=mpl.rcParams["axes.labelsize"],
    line_width=0.5,
    line_height=0.01,
    text_offset=0.2,
    # line_offset=0.2,
    # line_offset_to_group=0.9,
    pvalue_thresholds=[[0.05, "*"], [1, "ns"]]
    # pvalue_format={'star':[[0.05, "*"],[1, "ns"]]},
    # pvalue_format= {'correction_format': '{star} ({suffix})',
    #                           'fontsize': 'small',
    #                           'pvalue_format_string': '{:.3e}',
    #                           'show_test_name': True,
    #                         #   'simple_format_string': '{:.2f}',
    #                           'text_format': 'star',
    #                           'pvalue_thresholds': [
    #                               [1e-4, "*"],
    #                               [1e-3, "*"],
    #                               [1e-2, "*"],
    #                               [0.05, "*"],
    #                               [1, "ns"]]
    #                           },
    # color= 'r',
    # line_offset_to_box=0.2,
    # use_fixed_offset=True,
)

sns_boxplot_kw = dict(
    linewidth=0.8,
    palette=colors_sd(1),
    saturation=1,
    showfliers=False,
    # linewidth=lw,
    boxprops=dict(edgecolor="k"),
    showcaps=True,
    capprops=dict(color="k"),
    medianprops=dict(color="k"),
    whiskerprops=dict(color="k"),
)

sns_violin_kw = dict(
    palette=colors_sd(1),
    saturation=1,
    linewidth=0.4,
    cut=True,
    split=False,
    inner="box",
    showextrema=False,
    # showmeans=True,
)


fig_folder = Path("/home/nkinsky/Documents/figures/")
fig_root = Path("/home/nkinsky/Documents/figures/")
figpath_sd = fig_root / "sleep_deprivation"
figpath_tn = fig_root / "two_novel"


# ProcessData now lives in the package, so scripts can read these sessions without the notebook.
from neuropy.io.datasets.bapun import ProcessData


# minimally processed
class Group:
    tag = None
    import os
    basedir = Path(os.path.expanduser("~/Documents/ms_synchrony/bapun"))
    # basedir = Path("/data/Clustering/sessions/")

    def _process(self, rel_path, use_relaligned_pos=False):
        return [ProcessData(self.basedir / rel_path, self.tag, use_relaligned_pos)]

    def data_exist(self):
        self.allsess

# open field
class Of(Group):
    tag = "OF"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratJday4
            + self.ratKday4
            + self.ratNday4
            + self.ratUday5
        )
        return pipelines

    @property
    def ratJday4(self):
        return self._process("RatJ/Day4/")

    @property
    def ratKday4(self):
        return self._process("RatK/Day4/")

    @property
    def ratNday4(self):
        return self._process("RatN/Day4/")

    @property
    def ratUday5(self):
        return self._process("RatU/RatUDay5OpenfieldSD/")

# sleep deprivation
class Sd(Group):
    tag = "SD"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratJday1
            + self.ratKday1
            + self.ratNday1
            + self.ratSday3
            + self.ratRday2
            + self.ratUday1
            + self.ratUday4
            + self.ratVday2
        )
        return pipelines

    @property
    def mua_sess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratJday1
            + self.ratKday1
            + self.ratNday1
            + self.ratSday3
            + self.ratRday2
            + self.ratUday1
            + self.ratUday4
            + self.ratVday2
        )
        return pipelines

    @property
    def ripple_sess(self):
        pipelines: List[ProcessData] = (
            self.ratJday1
            + self.ratKday1
            + self.ratNday1
            + self.ratSday3
            + self.ratRday2
            + self.ratUday1
            + self.ratUday4
            + self.ratVday2
        )
        return pipelines

    @property
    def brainstates_sess(self):
        pipelines: List[ProcessData] = (
            self.ratJday1
            + self.ratKday1
            + self.ratNday1
            + self.ratSday3
            + self.ratRday2
            + self.ratUday4
        )
        return pipelines

    @property
    def pf_sess(self):
        pipelines: List[ProcessData] = (
            self.ratKday1
            + self.ratNday1
            + self.ratSday3
            + self.ratUday1
            + self.ratUday4
            + self.ratVday2
            + self.ratRday2
        )
        return pipelines

    @property
    def bilateral(self):
        pipelines: List[ProcessData] = (
            self.ratSday3 + self.ratUday1 + self.ratUday4 + self.ratVday2
        )
        return pipelines

    @property
    def remaze(self):
        pipelines: List[ProcessData] = (
            self.ratSday3 + self.ratUday1 + self.ratUday4 + self.ratVday2 + self.ratRday2
        )

        return pipelines

    @property
    def remaze_realign(self):
        """use position data that has been re-aligned using the correlation between
        speed and theta power from an electrode in or above the cell layer (not in radiatum, where theta power is
        NOT correlated with speed, see Kennedy et al. (2022) J.Neuro: https://doi.org/10.1523/JNEUROSCI.0987-21.2022"""
        pipelines: List[ProcessData] = (
                self.ratSday3re + self.ratUday1re + self.ratUday4re + self.ratVday2re + self.ratRday2re
        )
        return pipelines

    @property
    def handling_data_sess(self):
        pipelines: List[ProcessData] = self.ratUday1 + self.ratUday4 + self.ratVday2
        return pipelines

    @property
    def ratJday1(self):
        return self._process("RatJ/Day1/")

    @property
    def ratKday1(self):
        return self._process("RatK/Day1/")

    @property
    def ratNday1(self):
        return self._process("RatN/Day1/")

    @property
    def ratSday3(self):
        return self._process("RatS/Day3SD/")

    @property
    def ratSday3re(self):
        return self._process("RatS/Day3SD/", use_relaligned_pos=True)

    @property
    def ratRday2(self):
        return self._process("RatR/Day2SD")

    @property
    def ratRday2re(self):
        return self._process("RatR/Day2SD", use_relaligned_pos=True)

    @property
    def ratUday1(self):
        return self._process("RatU/RatUDay1SD")

    @property
    def ratUday1re(self):
        return self._process("RatU/RatUDay1SD", use_relaligned_pos=True)

    @property
    def ratUday4(self):
        return self._process("RatU/RatUDay4SD")

    @property
    def ratUday4re(self):
        return self._process("RatU/RatUDay4SD", use_relaligned_pos=True)

    @property
    def ratVday2(self):
        return self._process("RatV/RatVDay2SD/")

    @property
    def ratVday2re(self):
        return self._process("RatV/RatVDay2SD/", use_relaligned_pos=True)

    # @property
    # def ratUday5(self):
    #     path = "/data/Clustering/sessions/RatU/RatUDay5OpenfieldSD/"
    #     return [ProcessData(path)]

    @property
    def utkuAG_day1(self):
        path = "Utku/AG_2019-12-22_SD_day1/"
        return [ProcessData(path)]

    @property
    def utkuAG_day2(self):
        path = "Utku/AG_2019-12-26_SD_day2/"
        return [ProcessData(path)]

    def __add__(self, other):
        pipelines: List[ProcessData] = self.allsess + other.allsess
        return pipelines

    @staticmethod
    def color(amount=1):
        # return adjust_lightness("#df670c", amount=amount)
        # return adjust_lightness("#f06292", amount=amount)
        # return adjust_lightness("#ff0000", amount=amount)
        return adjust_lightness("#ff8080", amount=amount)

    @staticmethod
    def rs_color(amount=1):
        return adjust_lightness("#00B8D4", amount=amount)

# non-sleep deprivation
class Nsd(Group):
    tag = "NSD"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratJday2
            + self.ratKday2
            + self.ratNday2
            + self.ratSday2
            + self.ratRday1
            + self.ratUday2
            + self.ratVday1
            + self.ratVday3
        )
        return pipelines

    @property
    def mua_sess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratJday2
            + self.ratKday2
            + self.ratNday2
            + self.ratSday2
            + self.ratRday1
            + self.ratUday2
            + self.ratVday1
            + self.ratVday3
        )
        return pipelines

    @property
    def ripple_sess(self):
        pipelines: List[ProcessData] = (
            self.ratJday2
            + self.ratKday2
            + self.ratNday2
            + self.ratSday2
            + self.ratRday1
            + self.ratUday2
            + self.ratVday1
            + self.ratVday3
        )
        return pipelines

    @property
    def brainstates_sess(self):
        pipelines: List[ProcessData] = (
            self.ratJday2 + self.ratKday2 + self.ratNday2 + self.ratSday2 + self.ratUday2
        )
        return pipelines

    @property
    def pf_sess(self):
        pipelines: List[ProcessData] = (
            self.ratKday2
            + self.ratNday2
            + self.ratSday2
            + self.ratUday2
            + self.ratVday1
            + self.ratVday3
        )
        return pipelines

    @property
    def bilateral(self):
        pipelines: List[ProcessData] = (
            self.ratSday2 + self.ratUday2 + self.ratVday1 + self.ratVday3
        )
        return pipelines

    @property
    def remaze(self):
        pipelines: List[ProcessData] = (
                self.ratSday2 + self.ratUday2 + self.ratVday1 + self.ratVday3
        )
        return pipelines

    @property
    def remaze_realign(self):
        """use position data that has been re-aligned using the correlation between
                speed and theta power from an electrode in or above the cell layer (not in radiatum, where theta power is
                NOT correlated with speed, see Kennedy et al. (2022) J.Neuro: https://doi.org/10.1523/JNEUROSCI.0987-21.2022"""
        pipelines: List[ProcessData] = (
                self.ratSday2 + self.ratUday2re + self.ratVday1re + self.ratVday3re
        )
        return pipelines

    @property
    def ratJday2(self):
        return self._process("RatJ/Day2/")

    @property
    def ratKday2(self):
        return self._process("RatK/Day2/")

    @property
    def ratNday2(self):
        return self._process("RatN/Day2/")

    @property
    def ratSday2(self):
        return self._process("RatS/Day2NSD/")

    @property
    def ratRday1(self):
        return self._process("RatR/Day1NSD/")

    @property
    def ratUday2(self):
        return self._process("RatU/RatUDay2NSD/")

    @property
    def ratUday2re(self):
        return self._process("RatU/RatUDay2NSD/", use_relaligned_pos=True)

    @property
    def ratVday1(self):
        return self._process("RatV/RatVDay1NSD/")

    @property
    def ratVday1re(self):
        return self._process("RatV/RatVDay1NSD/", use_relaligned_pos=True)

    @property
    def ratVday3(self):
        return self._process("RatV/RatVDay3NSD")

    @property
    def ratVday3re(self):
        return self._process("RatV/RatVDay3NSD", use_relaligned_pos=True)

    def __add__(self, other):
        pipelines: List[ProcessData] = self.allsess + other.allsess
        return pipelines

    @staticmethod
    def color(amount=1):
        # return adjust_lightness("#815bcd", amount=amount)
        # return adjust_lightness("#424242", amount=amount)
        return adjust_lightness("#bdbdbd", amount=amount)

# two novel mazes
class Tn(Group):
    tag = "TN"
    paths = [
        "/data/Clustering/sessions/RatJ/Day3/",
        "/data/Clustering/sessions/RatK/Day3/",
        "/data/Clustering/sessions/RatN/Day3/",
    ]

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = self.ratKday3 + self.ratSday5 + self.ratUday3
        return pipelines

    # @property
    # def ratJday3(self):
    #     return self._process("RatJ/RatJDay3TwoNovel")

    @property
    def ratKday3(self):
        # path = "/data/Clustering/sessions/RatK/Day3/"
        return self._process("RatK/RatKDay3TwoNovel")

    @property
    def ratNday3(self):
        return self._process("RatN/Day3")

    @property
    def ratSday5(self):
        path = "/data/Clustering/sessions/RatS/Day5TwoNovel/"
        return [ProcessData(path)]

    @property
    def ratUday3(self):
        return self._process("RatU/RatUDay3TwoNovel")


class NsdHiro(Group):
    tag = "NSD_HM"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratRoyMaze1
            # + self.ratRoyMaze2
            # + self.ratRoyMaze3
            + self.ratTedMaze1
            + self.ratTedMaze2
            + self.ratTedMaze3
            + self.ratKevinMaze1
        )
        return pipelines

    @property
    def ratRoyMaze1(self):
        return self._process("Hiro/RoyMaze1")

    # @property
    # def ratRoyMaze2(self):
    #     return self._process("Hiro/RoyMaze2")

    # @property
    # def ratRoyMaze3(self):
    #     return self._process("Hiro/RoyMaze3")

    @property
    def ratTedMaze1(self):
        return self._process("Hiro/TedMaze1")

    @property
    def ratTedMaze2(self):
        return self._process("Hiro/TedMaze2")

    @property
    def ratTedMaze3(self):
        return self._process("Hiro/TedMaze3")

    @property
    def ratKevinMaze1(self):
        return self._process("Hiro/KevinMaze1")


class NsdGrosmark(Group):
    tag = "NSD_GM"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratAchilles_10252013
            + self.ratAchilles_11012013
            + self.ratBuddy_06272013
            + self.ratCicero_09172014
            + self.ratGatsby_08282013
        )
        return pipelines

    @property
    def ratAchilles_10252013(self):
        return self._process("GrosmarkReclusteredData/Achilles_10252013")

    @property
    def ratAchilles_11012013(self):
        return self._process("GrosmarkReclusteredData/Achilles_11012013")

    @property
    def ratBuddy_06272013(self):
        return self._process("GrosmarkReclusteredData/Buddy_06272013")

    @property
    def ratCicero_09172014(self):
        return self._process("GrosmarkReclusteredData/Cicero_09172014")

    @property
    def ratGatsby_08282013(self):
        return self._process("GrosmarkReclusteredData/Gatsby_08282013")


class NsdOld(Group):
    tag = "NSD_Old"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = self.rat_2022_06_24
        return pipelines

    @property
    def ratB1_2022_06_24(self):
        return self._process("UtkuOldAnimals/RatB1/RatB1_2022-06-24_NSD_CA1_24Hrs")

    @property
    def ratB2_2022_05_28(self):
        return self._process("UtkuOldAnimals/RatB2/RatB2_2022-05-28_NSD_CA1_24hrs")


class SdOld(Group):
    tag = "SD_Old"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = self.rat_2022_06_24
        return pipelines

    @property
    def ratB1_2022_06_27(self):
        return self._process("UtkuOldAnimals/RatB1/RatB1_2022-06-27_SD_CA1_24Hrs")

# rolipram
class SdRol(Group):
    tag = "SD_ROL"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratN_2019_10_19
            + self.ratA14_2020_02_26
            + self.ratU_2021_08_09
            + self.ratMR10_2021_08_23
        )
        return pipelines

    @property
    def ratN_2019_10_19(self):
        return self._process("rolipram/BGN_2019-10-19_SDROL")

    @property
    def ratA14_2020_02_26(self):
        return self._process("rolipram/A14_2020-02-26_SDROL")

    @property
    def ratU_2021_08_09(self):
        return self._process("rolipram/BGU_2021-08-09_SDROL")

    @property
    def ratMR10_2021_08_23(self):
        return self._process("rolipram/MR10_2021-08-23_SDROL")

# saline
class SdSal(Group):
    tag = "SD_SAL"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = (
            self.ratN_2019_10_21
            + self.ratA14_2020_02_23
            + self.ratU_2021_08_11
            + self.ratMR10_2021_08_21
        )

        return pipelines

    @property
    def ratN_2019_10_21(self):
        return self._process("rolipram/BGN_2019-10-21_SDSAL")

    @property
    def ratA14_2020_02_23(self):
        return self._process("rolipram/A14_2020-02-23_SDPBS")

    @property
    def ratU_2021_08_11(self):
        return self._process("rolipram/BGU_2021-08-11_SDSAL")

    @property
    def ratMR10_2021_08_21(self):
        return self._process("rolipram/MR10_2021-08-21_SDPBS")


class SimData(Group):
    tag = "Sim"

    @property
    def allsess(self):
        pipelines: List[ProcessData]
        pipelines = self.ratSim1

        return pipelines

    @property
    def ratSim1(self):
        return self._process("Simulated/RatSim1")

# processed data aggregated across all animal
class GroupData:
    __slots__ = (
        "path",
        "add_zt",
        "swa_examples",
        "brainstates_proportion",
        "ripple_psd",
        "ripple_examples",
        "ripple_rate",
        "ripple_rate_statewise_blocks",
        "ripple_total_duration",
        "ripple_features",
        "ripple_features_1h_blocks",
        "ripple_features_normalized",
        "ripple_autocorr",
        "ripple_bootstrap_session_ripples",
        "ripple_normalized_bootstrap_session_ripples",
        "ripple_1h_blocks_bootstrap_session_ripples",
        "ripple_rate_bootstrap_session",
        "ripple_rate_post5h_trend",
        "ripple_rate_post5h_trend_bootstrap",
        "ripple_features_brainstate",
        "ripple_NREM_bootstrap_session_ripples",
        "ripple_WK_bootstrap_session_ripples",
        "pbe_rate",
        "pbe_total_duration",
        "candidate_PBE_duration",
        "candidate_PBE_duration_bootstrap",
        "frate_zscore",
        "frate_post_chunks",
        "frate_post_chunks_statewise",
        "frate_1h_blocks",
        "frate_1h_blocks_bootstrap",
        # "frate_post_chunks_nrem_qw",
        "frate_post_chunks_zscore",
        "frate_ratio_nsd_vs_sd",
        "frate_interneuron_around_Zt5",
        "frate_change_1vs5",
        "frate_change_pre_to_post",
        "frate_pre_to_maze_quantiles_in_POST",
        "frate_pre_to_maze_quantiles_in_POST_shuffled",
        "frate_in_ripple",
        "frate_blocks_bootstrap_session_neurons",
        "frate_blocks_WK_bootstrap_session_neurons",
        "frate_blocks_NREM_bootstrap_session_neurons",
        "frate_IQR_blocks_bootstrap_session_neurons",
        "frate_ShapiroWilk_statistic_blocks_bootstrap",
        "frate_STD_blocks_bootstrap_session_neurons",
        "frate_Kurtosis_blocks_bootstrap_session_neurons",
        "frate_in_ripple_blocks_bootstrap_session_neurons",
        "pairwise_correlations_NREM",
        "pairwise_correlations_WAKE",
        "pairwise_correlations_aligned_by_NREM_onset",
        "pairwise_correlations_aligned_by_WAKE",
        "ei_ratio",
        "ev_pooled",
        "ev_in_chunks",
        "ev_brainstates",
        "ev_1h_blocks",
        "ev_1h_blocks_bootstrap",
        "ev_bootstrap_session",
        "ev_bootstrap_pairs",
        "ev_bootstrap_session_pairs",
        "ev_bootstrap_session_mean",
        "ev_bootstrap_pairs_mean",
        "ev_bootstrap_session_pairs_mean",
        "ev_NSD_WK_sliding_bootstrap_session_pairs",
        "ev_NSD_NREM_sliding_bootstrap_session_pairs",
        "ev_SD_WK_sliding_bootstrap_session_pairs",
        "ev_aligned_by_NREM_onset",
        "ev_NREM_bootstrap",
        "ev_mean_aligned_by_NREM_onset",
        "ev_aligned_by_WAKE",
        "ev_mean_aligned_by_WAKE",
        "pf_norm_tuning",
        "replay_examples",
        "replay_continuous_events",
        "replay_sig_frames",
        "replay_wcorr",
        "replay_wcorr_mua",
        "replay_radon",
        "replay_radon_mua",
        "replay_jumpdist",
        "replay_jumpdist_mua",
        "replay_re_maze_score",
        "replay_post_score",
        "replay_pos_distribution",
        "replay_re_maze_position_distribution",
        "continuous_replay_PBE_duration",
        "continuous_replay_PBE_duration_bootstrap",
        "continuous_replay_proportion_bootstrap",
        "continuous_replay_number",
        "continuous_replay_number_bootstrap",
        "continuous_replay_bias_blocks",
        "replay_continuous_events_1h_blocks",
        "continuous_replay_proportion_1h_blocks_bootstrap",
        "continuous_replay_number_1h_blocks",
        "continuous_replay_number_1h_blocks_bootstrap",
        "candidate_replay_number",
        "candidate_replay_number_bootstrap",
        "remaze_ev_example",
        "remaze_ev_on_POST_example",
        "remaze_ev",
        "remaze_temporal_bias",
        "remaze_maze_paircorr",
        "remaze_first5_paircorr",
        "remaze_first5_subsample",
        "remaze_first5_bootstrap",
        "remaze_last5_paircorr",
        "remaze_corr_across_session",
        "remaze_activation_of_maze",
        "remaze_temporal_bias_com_correlation_across_session",
        "remaze_ensemble_corr_across_sess",
        "remaze_ensemble_activation_across_sess",
        "remaze_ev_on_zt0to5",
        "remaze_ev_on_POST_pooled",
        "post_first5_last5_paircorr",
        "off_rate",
        "off_mean_duration",
        "off_rate_bootstrap_session",
        "nrem_duration_NREM",
        "nrem_duration_aligned_by_nrem_onset",
        "wake_duration_aligned_by_WAKE",
        "ev_tc_NREM",
        "ev_tc_WAKE",
        "ev_tc_linear_high_NREM",
        "ev_tc_linear_WAKE",
        "ev_slopes_high_NREM_bootstrap",
        "ev_slopes_WAKE_bootstrap",
        "ev_mean_tc_aligned_by_NREM_onset",
        "ev_mean_tc_aligned_by_WAKE",
        "delta_wave_rate",
        "delta_wave_amp_blocks",
        "delta_wave_rate_bootstrap",
        "ev_goodness_fit_NREM",
        "ev_goodness_fit_WAKE",
    )

    def __init__(self, add_zt: bool = True) -> None:
        self.path = Path("/home/selinali/Documents/ms_synchrony/sessions")
        # self.path = Path("/home/nkinsky/Documents/sleep_deprivation/ProcessedData")
        self.add_zt = add_zt
        # for f in self.path.iterdir():
        #     setattr(self, f.name, self.load(f.stem))

    def save(self, d, fp):
        if isinstance(d, pd.DataFrame):
            d = d.to_dict()
        data = {"data": d}
        np.save(self.path / fp, data)
        print(f"{fp} saved")

    def load(self, fp):
        data = np.load(self.path / f"{fp}.npy", allow_pickle=True).item()
        try:
            data["data"] = pd.DataFrame(data["data"])
            if self.add_zt:
                data["data"] = add_zt_str(data["data"])
        except:
            pass
        return data

    def __getattr__(self, name: str):
        return self.load(name)["data"]


sd = Sd()
nsd = Nsd()
of = Of()
tn = Tn()
sdrol = SdRol()


def mua_sess():
    return nsd.mua_sess + sd.mua_sess


# placefield sessions
def pf_sess():
    sessions = nsd.pf_sess + sd.pf_sess
    print(f"#Sessions = {len(sessions)}")
    return sessions


def ripple_sess():
    return nsd.ripple_sess + sd.ripple_sess


def remaze_sess():
    return nsd.remaze + sd.remaze


def remaze_realign_sess():
    return nsd.remaze_realign + sd.remaze_realign


def bilateral_sess():
    return nsd.bilateral + sd.bilateral


def add_zt_str(df: pd.DataFrame, zt_key="zt", epoch_str=("0-2.5", "2.5-5", "5-7.5")):
    """Fix zt strings to prepend ZT"""
    for epoch_name in epoch_str:
        df.loc[df[zt_key] == epoch_name, zt_key] = f"ZT {epoch_name}"

    return df


def sess_name_fix(str_to_fix):
    """Make capitalized names compatible with NSD and SD attribute names"""
    letter = str_to_fix[3]
    day = str_to_fix[7]
    return f"rat{letter}day{day}"

