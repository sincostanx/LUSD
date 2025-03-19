import os
import pandas as pd
from scipy.integrate import simps
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path
import pickle

sns.set_theme(style="whitegrid")

def convert_name_to_reorgrnize(index):
    prefix, suffix = index.split("-")

    if suffix == "input": filename = os.path.join(f"{prefix}_1")
    elif suffix == "output1": filename = os.path.join(f"{prefix}_inde_2")
    elif suffix == "output2": filename = os.path.join(f"{prefix}_inde_3")
    return filename

METRICS = ["l1", "l2", "clip_i", "clip_t", "clip_r"]
METRICS_REPORT = ["clip_t", "l1", "clip_i"]
ALL_CAT = ["add", "change"]

class ScoreSummary():
    def __init__(self, root, num_min_samples=30):
        self.root = root
        self.num_min_samples = num_min_samples
        self._load_category_df()
        self.initialize()

    def initialize(self):
        self.summary_df = []
        self.methods = []
        self.num_pass_thresholds = {"":{}, "add":{}, "change":{}}
        self.l1_thresholds = {"":{}, "add":{}, "change":{}}
        self.clipI_thresholds = {"":{}, "add":{}, "change":{}}
        self.dino_thresholds = {"":{}, "add":{}, "change":{}}
        self.l1_thresholds_std = {"":{}, "add":{}, "change":{}}
        self.clipI_thresholds_std = {"":{}, "add":{}, "change":{}}
        self.dino_thresholds_std = {"":{}, "add":{}, "change":{}}
        self.raw_pass_thresholds = {"":{}, "add":{}, "change":{}}

    def _load_category_df(self):
        path = os.path.join("magicbrush/magicbrush_category.csv")
        category_df = pd.read_csv(path)
        category_df["idx"] = category_df["idx"].apply(convert_name_to_reorgrnize)
        category_df["category"] = category_df["category"].apply(lambda x: "change" if x == "remove" else x)
        self.category_df = category_df

    def add_results(self, method):
        # 1. gather scores for all samples
        scores = {}
        for metric in METRICS:
            path = os.path.join(self.root, f"{method}/{metric}.pickle")
            with open(path, "rb") as f:
                x = pickle.load(f)
                gen_paths = [val[0] for val in x["image_pair"]]
                gt_paths = [val[1] for val in x["image_pair"]]
                scores[metric] = x["score"]

        df = pd.DataFrame({"gen_path": gen_paths, "gt_path": gt_paths})
        for metric in METRICS:
            df[metric] = scores[metric]
        
        df["idx"] = df["gen_path"].apply(lambda x: str(Path(x).stem))
        df = pd.merge(df, self.category_df, on="idx", how="inner")

        # 2. calculate <l1, clip-t, clip-i>
        row = {"case": method}

        for metric in METRICS_REPORT:
            row.update({f"{metric}": df[metric].mean()})

        for metric in METRICS_REPORT:
            num_samples = 0
            for cat in ALL_CAT:
                num_samples += len(df[df["category"] == cat])
                row.update({f"{metric}_{cat}": df[df["category"] == cat][metric].mean()})

            assert num_samples == len(df)

        # 3. compute <clip-auc, l1*, clip-i*>
        MAX_CLIP_THRESHOLD = 41
        CLIP_THRESHOLDS = [1 + 0.01 * k for k in range(MAX_CLIP_THRESHOLD)]
        BACKGROUND_CLIPR_MAX = 1.22
        BACKGROUND_CLIPR_MAX = round((BACKGROUND_CLIPR_MAX - 1) * 100 + 1)

        # print(MAX_CLIP_THRESHOLD, BACKGROUND_CLIPR_MAX)

        def update_rows(vfunc, cat=""):
            total_samples = len(df) if cat == "" else len(df[df["category"] == cat[1:]])
            self.raw_pass_thresholds[cat[1:]][method] = np.array([len(df[vfunc(df, threshold)]) for threshold in CLIP_THRESHOLDS])
            self.num_pass_thresholds[cat[1:]][method] = np.array([len(df[vfunc(df, threshold)]) for threshold in CLIP_THRESHOLDS]) / total_samples
            self.l1_thresholds[cat[1:]][method] = [df[vfunc(df, threshold)]["l1"].mean() for threshold in CLIP_THRESHOLDS]
            self.clipI_thresholds[cat[1:]][method] = [df[vfunc(df, threshold)]["clip_i"].mean() for threshold in CLIP_THRESHOLDS]

            row.update({f"clip_auc{cat}": simps(self.num_pass_thresholds[cat[1:]][method], CLIP_THRESHOLDS)})
            row.update({f"l1*{cat}": simps(self.l1_thresholds[cat[1:]][method][:BACKGROUND_CLIPR_MAX], CLIP_THRESHOLDS[:BACKGROUND_CLIPR_MAX])})
            row.update({f"clip_i*{cat}": simps(self.clipI_thresholds[cat[1:]][method][:BACKGROUND_CLIPR_MAX], CLIP_THRESHOLDS[:BACKGROUND_CLIPR_MAX])})

        # 3.1 total
        vfunc = lambda df, threshold: df["clip_r"] > threshold
        update_rows(vfunc)
        
        # 3.2 add
        vfunc = lambda df, threshold: (df["clip_r"] > threshold) & (df["category"] == "add")
        update_rows(vfunc, cat="_add")

        vfunc = lambda df, threshold: (df["clip_r"] > threshold) & (df["category"] == "change")
        update_rows(vfunc, cat="_change")

        self.summary_df.append(row)
        self.methods.append(method)

    def summarize(self, methods=None):
        if methods is None: methods = self.methods
        temp_df = pd.DataFrame(self.summary_df)
        mask = temp_df["case"].isin(methods)
        return temp_df[mask].round(3)

    def plot_clip_r(self, ax, methods=None, xlim=None):
        if methods is None: methods = self.methods
        labels = methods

        for i, case in enumerate(methods):
            case_name = Path(case).stem
            print(case_name)

            line_color = "black" if labels[i] == "Ours" else None
            num_data_points = len(self.num_pass_thresholds[""][case_name])
            CLIP_THRESHOLDS = [1 + 0.01 * k for k in range(num_data_points)]
            sns.lineplot(
                x=CLIP_THRESHOLDS,
                y=self.num_pass_thresholds[""][case_name],
                label=labels[i],
                color=line_color,
                ax=ax
            )

        ax.legend()
        ax.set_xlabel("CLIP-R")
        ax.set_ylabel("Ratio of successful edits")
        
        if xlim is not None: ax.set_xlim(xlim)