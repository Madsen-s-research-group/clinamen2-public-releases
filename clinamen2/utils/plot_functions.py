"""Plot functions for CMA-ES results.
"""
import pathlib
from numbers import Number
from typing import Tuple

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

from clinamen2.utils.file_handling import CMAFileHandler


class CMAPlotHelper:
    """Setup figure parameters and plot common results.

    Args:
        label: Label identifying the evolution and its results.
        input_dir: Directory to read data from.
        handler: Object to access the stored evolution and generations.
        generation_bounds: First and last generation to include. Count starts
            at 1, because generation 0 is the founder.
            If generation_bounds[1] is passed in with value -1, the last
            generation is read from the evolution element "last_gen" if
            present.
        losses: Loss of all read individuals.
        generations: Indices of all read generations.
        step_sizes: Step size of AlgorithmState.
        output_dir: Directory to save files to. Will be created if it does not
            exist. Default is None.
        information_list: List of dictionaries with additional information per
            generation.
        additional_info_types: Contains keys identifying additional information
            and the corresponding types as values for all additional
            information that is of type numpy.ndarray, list of Numbers or
            Number.
        figsize: Default figure size to be used when no alternative is passed
            to a plotting function.
        colors: Dictionary of colors to be used in the plots.
        fontsizes: Dictionary of fontsizes to be used in the plots.
    """

    def __init__(
        self,
        label: str,
        input_dir: pathlib.Path,
        generation_bounds: Tuple[int, int],
        output_dir: pathlib.Path = None,
        figsize: Tuple[float, float] = (10, 8),
        colors: dict = {
            "dark_line": "#440154",
            "medium_line": "#31688e",
            "light_line": "#6ece58",
            "dark_area": "#3e4989",
            "medium_area": "#1f9e89",
            "light_area": "#b5de2b",
            "highlight": "#fde725",
        },
        fontsizes: dict = {
            "title": 36,
            "ax_label": 28,
            "tick_param": 24,
            "legend": 24,
        },
        json=True,
    ):
        self.json = json
        self.colors = colors
        self.fontsizes = fontsizes
        self.label = label
        self.handler = CMAFileHandler(label=label, target_dir=input_dir)
        file_format = "json" if json else "dill"
        self.evolution = self.handler.load_evolution(file_format=file_format)
        try:
            self.last_gen = self.evolution[3]["last_gen"]
        except KeyError:
            self.last_gen = 0
        if generation_bounds[1] == -1 and self.last_gen > 0:
            generation_bounds = (generation_bounds[0], self.last_gen)
        self.generation_bounds = generation_bounds
        (
            self.losses,
            self.generations,
            self.step_sizes,
            self.information_list,
        ) = self.get_data_from_generations()
        self.figsize = figsize
        self.output_dir = output_dir if output_dir is not None else input_dir
        if output_dir is not None:
            output_dir.mkdir(parents=True, exist_ok=True)
        self.additional_info_types = self.classify_additional_information()

    def get_data_from_generations(self):
        """Parse generations with handler and return data."""
        file_format = "json" if self.json else "dill"
        losses = []
        generations = []
        step_sizes = []
        information_list = []
        for g in range(
            self.generation_bounds[0], self.generation_bounds[1] + 1
        ):
            try:
                (
                    current_state,
                    _,
                    loss,
                    _,
                    information,
                ) = self.handler.load_generation(
                    generation=g, label=self.label, file_format=file_format
                )
                losses.append(loss)
                generations.append(g)
                step_sizes.append(current_state.step_size)
                information_list.append(information)
            except FileNotFoundError:
                pass  # not every generation must have been saved
        return (
            np.asarray(losses),
            np.asarray(generations),
            np.asarray(step_sizes),
            information_list,  # list of dictionaries
        )

    def classify_additional_information(self):
        """Parse additional information and classify."""
        # check which keys are available
        try:
            additional_keys = self.information_list[0]["information"][0].keys()
            print(f"Found additional keys {additional_keys}")
        except IndexError:
            additional_keys = []
        except KeyError:
            additional_keys = []
        if len(additional_keys) == 0:
            return None

        # for each key try and collect the information - limited
        #   np.ndarray or list (if list then list of Numbers) or Number
        additional_information_types = {}
        for key in additional_keys:
            info = self.information_list[0]["information"][0][key]
            if isinstance(info, np.ndarray):
                additional_information_types[key] = np.ndarray
            elif isinstance(info, list):
                if isinstance(info[0], Number):
                    additional_information_types[key] = list
            elif isinstance(info, Number):
                additional_information_types[key] = Number
            else:
                try:
                    info = float(info)
                    additional_information_types[key] = Number
                except:
                    pass

        print(f"resulting in {additional_information_types}")
        return additional_information_types

    def plot_additional_information_per_generation(
        self,
        key: str = None,
        ax: matplotlib.axis.Axis = None,
        generation_bounds: Tuple[int, int] = None,
        figsize: Tuple[float, float] = None,
        y_units: str = "",
        save_fig: bool = False,
        fig_type: str = "pdf",
    ):
        """Plot additional information depending on its type.

        Args:
            key: Key that identifies the additional information. Supercedes
                index if both given. Default is None.
            ax: Main ax of figure. To be twinned if present. Default is None.
            generation_bounds: First and last generation to include. Count
                starts at 1, because generation 0 is the founder.
            figsize: Default figure size to be used when no alternative is
                passed to a plotting function.
            y_units: Units of loss to be added to y_label. Default is empty
                string.
            save_fig: If True the figure will be saved to file. Default is
                False.
            fig_type: File type to be saved. Default is 'pdf'.

        Returns:
            Optional: Label for plot legend.
        """
        if key not in self.additional_info_types:
            raise KeyError("Key does not identify plotable data.")

        if ax is None:
            if generation_bounds is None:
                generation_bounds = self.generation_bounds
            if figsize is None:
                figsize = self.figsize

            fig = plt.figure(figsize=figsize)
            ax = fig.add_subplot(111)
            ax.xaxis.set_tick_params(labelsize=self.fontsizes["ax_label"])
            ax.set_xlabel(r"Generation", fontsize=self.fontsizes["ax_label"])
            ax.yaxis.set_tick_params(labelsize=self.fontsizes["ax_label"])
            ax.set_ylabel(
                r"" + key + y_units, fontsize=self.fontsizes["ax_label"]
            )

            generation_index = np.argwhere(
                np.greater_equal(self.generations, generation_bounds[0])
                & np.less_equal(self.generations, generation_bounds[1]),
            ).flatten()
            selected_generations = self.generations[
                np.greater_equal(self.generations, generation_bounds[0])
                & np.less_equal(self.generations, generation_bounds[1])
            ]

            x = selected_generations

            if self.additional_info_types[key] == Number:
                data = np.asarray(
                    [
                        [
                            float(d[key])
                            for d in self.information_list[i]["information"]
                        ]
                        for i in generation_index
                    ]
                )
            data = data[generation_index]
            for i in range(data.shape[1]):
                ax.scatter(
                    x,
                    # np.arange(data.shape[0]) + 1,
                    data[:, i],
                    color="black",
                    marker=".",
                )

            data_mean = data.mean(axis=1)
            lns = ax.plot(
                x, data_mean, color=self.colors["dark_line"], label="mean"
            )
            data_min = data.min(axis=1)
            lns = ax.plot(
                x, data_min, color=self.colors["medium_line"], label="min"
            )
            ax.fill_between(
                x,
                data_min,
                data_mean,
                color=self.colors["medium_area"],
                alpha=0.25,
            )
            data_max = data.max(axis=1)
            lns = ax.plot(
                x, data_max, color=self.colors["light_line"], label="max"
            )
            ax.fill_between(
                x,
                data_mean,
                data_max,
                color=self.colors["medium_area"],
                alpha=0.25,
            )

            plt.title(
                f"Information with key '{key}' per generation.",
                fontsize=self.fontsizes["title"],
            )
            plt.legend(fontsize=self.fontsizes["legend"])

            if save_fig:
                plt.savefig(
                    self.output_dir / (self.label + f"_{key}." + fig_type),
                    format=fig_type,
                )

    def plot_stepsize_ax2(
        self,
        ax: matplotlib.axis.Axis,
        x: npt.ArrayLike,
        selected_generations: npt.ArrayLike,
        y_units: str = r"",  # / \si{\angstrom}",
    ) -> matplotlib.axis.Axis:
        """Plot stepsize as twin of given axis.

        Args:
            ax: Main ax of figure. To be twinned.
            x: Values for x-axis.
            selected_generations: Generations to be plotted.
            y_units: Units of loss to be added to y_label.
                Default is r' / \si{\angstrom}'.
        """
        ax2 = ax.twinx()
        ax2.set_ylabel(
            r"$\sigma$" + y_units,
            fontsize=self.fontsizes["ax_label"],
            color=self.colors["medium_line"],
        )
        ax2.yaxis.set_tick_params(
            labelsize=self.fontsizes["tick_param"],
            color=self.colors["medium_line"],
        )
        ln = ax2.plot(
            x,
            self.step_sizes[selected_generations],
            color=self.colors["medium_line"],
            linestyle="dashed",
            label=r"$\sigma$",
        )
        ax2.tick_params(axis="y", colors=self.colors["medium_line"])

        return ln

    def plot_mean_loss_per_generation(
        self,
        generation_bounds: Tuple[int, int] = None,
        figsize: Tuple[float, float] = None,
        show_sigma_e: bool = True,
        sigma_e_mult: int = 3,
        show_min_e: bool = False,
        show_sigma: bool = False,
        show_legend: bool = True,
        y_units: str = "",
        y_lim: Tuple[float] = None,
        ref_val: float = None,
        save_fig: bool = False,
        fig_type: str = "pdf",
    ) -> None:
        """Plot mean loss per generation.

        Args:
            generations: Number of generations to be included.
            generation_bounds: First and last generation to include. Count
                starts at 1, because generation 0 is the founder.
            figsize: Default figure size to be used when no alternative is
                passed to a plotting function.
            show_sigma_e: If True, sigma_e_mult * std deviation is plotted
                additionally around the mean. Default is True.
            sigma_e_mult: If sigma_e is shown, this defines the width.
            show_min_e: If True, the minimum loss within each generation will
                be plotted. Default is False.
            show_sigma: If True, plot step size on axis y2. Default is False.
            show_legend: If True, plot a legend. Default is True.
            y_units: Units of loss to be added to y_label. Default is empty
                string.
            y_lim: Tuple of y_min and y_max to restrict the plot.
                Default is None.
            ref_val: Value to be plotted as a reference line. Default is None.
            save_fig: If True the figure will be saved to file. Default is
                False.
            fig_type: File type to be saved. Default is 'pdf'.
        """
        if generation_bounds is None:
            generation_bounds = self.generation_bounds
        if figsize is None:
            figsize = self.figsize
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.xaxis.set_tick_params(labelsize=self.fontsizes["ax_label"])
        ax.set_xlabel(r"Generation", fontsize=self.fontsizes["ax_label"])
        ax.yaxis.set_tick_params(labelsize=self.fontsizes["ax_label"])
        ax.set_ylabel(r"Loss" + y_units, fontsize=self.fontsizes["ax_label"])

        selected_generations = np.asarray(
            self.generations[
                np.greater_equal(self.generations, generation_bounds[0])
                & np.less_equal(self.generations, generation_bounds[1])
            ]
        )

        generation_index = np.argwhere(
            np.greater_equal(self.generations, generation_bounds[0])
            & np.less_equal(self.generations, generation_bounds[1]),
        ).flatten()
        x = np.asarray(selected_generations)
        losses = self.losses[generation_index]

        loss_mean = losses.mean(axis=1)
        if show_min_e:
            loss_min = losses.min(axis=1)
        lns = ax.plot(
            x, loss_mean, color=self.colors["dark_line"], label="mean"
        )

        if show_sigma_e:
            loss_std = losses.std(axis=1)
            ax.fill_between(
                x,
                loss_mean,
                loss_mean + sigma_e_mult * loss_std,
                color=self.colors["highlight"],
                alpha=0.75,
            )
            lns += ax.plot(
                x,
                loss_mean + sigma_e_mult * loss_std,
                color=self.colors["light_line"],
                label=f"{sigma_e_mult}" + r"$\sigma_{E}$",
            )
            ax.fill_between(
                x,
                loss_mean,
                loss_mean - sigma_e_mult * loss_std,
                color=self.colors["highlight"],
                alpha=0.75,
            )
            ax.plot(
                x,
                loss_mean - sigma_e_mult * loss_std,
                color=self.colors["light_line"],
            )

        if show_min_e:
            lns += [
                ax.scatter(
                    x,
                    loss_min,
                    marker="x",
                    color=self.colors["dark_line"],
                    label="min",
                )
            ]

        if show_sigma:
            lns += self.plot_stepsize_ax2(
                ax=ax, x=x, selected_generations=generation_index
            )

        if y_lim is not None:
            ax.set_ylim(y_lim)

        if ref_val is not None:
            lns += [
                ax.axhline(
                    y=ref_val,
                    linestyle="dotted",
                    color="black",
                    alpha=0.85,
                    label="ref",
                )
            ]

        if show_legend:
            labs = [l.get_label() for l in lns]
            ax.legend(lns, labs, fontsize=self.fontsizes["legend"])

        plt.title(
            f"Mean loss per generation ({self.label})",
            fontsize=self.fontsizes["title"],
        )

        if save_fig:
            plt.savefig(
                self.output_dir / (self.label + "_mean_loss." + fig_type),
                format=fig_type,
            )

        return fig, ax

    def plot_loss_boxplots(
        self,
        generations: list,
        figsize: Tuple[float, float] = None,
        y_units: str = "",
        color_by_key: bool = False,
        color_key: str = "",
        save_fig: bool = False,
        fig_type: str = "pdf",
    ) -> None:
        """Plot loss of gen.

        Args:
            generations: Index of generations to be plotted. More than five will
                not produce a satisfactory result.
            figsize: Default figure size to be used when no alternative is
                passed to a plotting function.
            y_units: Units of loss to be added to y_label. Default is empty
                string.
            color_key: If True, the scatter plot will be colored according
                to the data in 'key'. Default is False.
            save_fig: If True the figure will be saved to file. Default is
                False.
            fig_type: File type to be saved. Default is 'pdf'.
        """
        if figsize is None:
            figsize = self.figsize

        # check if all generations are available in the data
        for g in generations:
            if g not in (self.generations):
                raise ValueError(f"Generation {g} not found.")

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.xaxis.set_tick_params(labelsize=self.fontsizes["ax_label"])
        ax.set_xlabel(r"Generation", fontsize=self.fontsizes["ax_label"])
        # ax.xaxis.set_ticks(generations)
        ax.xaxis.set_ticklabels([str(g) for g in generations])
        ax.yaxis.set_tick_params(labelsize=self.fontsizes["ax_label"])
        ax.set_ylabel(r"Loss" + y_units, fontsize=self.fontsizes["ax_label"])

        data = []
        for i, g in enumerate(generations):
            index_of_g = np.argwhere(np.asarray(self.generations) == g)
            data.append(self.losses[index_of_g].flatten())
        _ = ax.boxplot(
            data,
            patch_artist=True,
            boxprops=dict(facecolor=self.colors["light_area"], alpha=0.3),
            zorder=-1,
        )
        if color_by_key:
            coloring = []
            for g in generations:
                index_of_g = int(
                    np.argwhere(np.asarray(self.generations) == g).flatten()[0]
                )
                if self.additional_info_types[color_key] == Number:
                    color_data = np.asarray(
                        [
                            [
                                float(d[color_key])
                                for d in self.information_list[
                                    index_of_g
                                ][  # g - 1][
                                    "information"
                                ]
                            ]
                        ]
                    )
                    coloring.append(color_data)

            for i, d in enumerate(data):
                scatter = ax.scatter(
                    [i + 1] * d.shape[0],
                    d,
                    alpha=1.0,
                    cmap="plasma",
                    c=coloring[i],
                )
            cbar = plt.colorbar(scatter)
            cbar.ax.tick_params(labelsize=self.fontsizes["tick_param"])
            cbar.set_label(label=color_key, size=self.fontsizes["ax_label"])

        else:
            for i, d in enumerate(data):
                _ = ax.scatter(
                    [i + 1] * d.shape[0],
                    d,
                    color=self.colors["dark_area"],
                    alpha=1.0,
                )

        plt.title(
            f"Loss of individuals per generations {generations}",
            fontsize=self.fontsizes["title"],
        )

        if save_fig:
            plt.savefig(
                self.output_dir / (self.label + "_loss_boxplots." + fig_type),
                format=fig_type,
            )

    def plot_loss_with_errorbars(
        self,
        generation: int,
        key: str,
        figsize: Tuple[float, float] = None,
        errorbars: bool = True,
        y_units: str = "",
        save_fig: bool = False,
        fig_type: str = "pdf",
    ) -> None:
        """Plot loss of gen.

        Args:
            generation: Index of generation.
            key: Identifies the additional information to be plotted.
            figsize: Default figure size to be used when no alternative is
                passed to a plotting function.
            errorbars: If True, errorbars are plotted.
            y_units: Units of loss to be added to y_label.Default is empty
                string.
            save_fig: If True the figure will be saved to file. Default is
                False.
            fig_type: File type to be saved. Default is 'pdf'.
        """
        # check if the generation is available in the data
        if generation not in (self.generations):
            raise ValueError(f"Generation {generation} not found.")
        index_of_g = int(
            np.argwhere(np.asarray(self.generations) == generation)
        )
        loss = self.losses[index_of_g]

        if errorbars:
            try:
                if self.additional_info_types[key] == Number:
                    extra_data = np.asarray(
                        [
                            [
                                float(d[key])
                                for d in self.information_list[index_of_g][
                                    "information"
                                ]
                            ]
                        ]
                    )
            except KeyError as exc:
                raise KeyError(
                    f"Key {key} not present in additional information."
                ) from exc

        if figsize is None:
            figsize = self.figsize

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.xaxis.set_tick_params(labelsize=self.fontsizes["ax_label"])
        ax.set_xlabel(r"Individual", fontsize=self.fontsizes["ax_label"])
        ax.yaxis.set_tick_params(labelsize=self.fontsizes["ax_label"])
        ax.set_ylabel(r"Loss" + y_units, fontsize=self.fontsizes["ax_label"])

        x = np.arange(loss.shape[0])
        ax.set_xticks(x)

        ax.scatter(x, loss, c=self.colors["dark_area"], s=50)
        if errorbars:
            ax.errorbar(
                x,
                loss,
                yerr=extra_data,
                fmt="none",
                capsize=10,
                ecolor=self.colors["medium_area"],
            )

        title = f"Individual loss in generation {generation}"
        if errorbars:
            title += " with errorbars."
        plt.title(
            title,
            fontsize=self.fontsizes["title"],
        )

        if save_fig:
            plt.savefig(
                self.output_dir
                / (self.label + "_loss_with_errorbars." + fig_type),
                format=fig_type,
            )

    def plot_loss_per_generation(
        self,
        generation_bounds: Tuple[int, int] = None,
        figsize: Tuple[float, float] = None,
        y_units: str = "",
        y_lim: Tuple[float, float] = None,
        show_legend: bool = True,
        ref_val: float = None,
        kwargs: dict = None,
        save_fig: bool = False,
        fig_type: str = "pdf",
    ) -> None:
        """Plot loss per individual per generation.

        Args:
            generations: Number of generations to be included.
            generation_bounds: First and last generation to include. Count
                starts at 1, because generation 0 is the founder.
            figsize: Default figure size to be used when no alternative is
                passed to a plotting function.
            show_legend: If True, plot a legend. Default is True.
            ref_val: Value to be plotted as a reference line. Default is None.
            y_units: Units of loss to be added to y_label. Default is empty
                string.
            y_lim: Limit to restrict y axis. Default is None.
            save_fig: If True the figure will be saved to file. Default is
                False.
            fig_type: File type to be saved. Default is 'pdf'.
            kwargs: Dicitionary with additional keywords.
        """
        if generation_bounds is None:
            generation_bounds = self.generation_bounds
        if figsize is None:
            figsize = self.figsize
        if kwargs is None:
            kwargs = {}
        if "marker" not in kwargs.keys():
            kwargs["marker"] = "x"
        if "alpha" not in kwargs.keys():
            kwargs["alpha"] = 0.5

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.xaxis.set_tick_params(labelsize=self.fontsizes["ax_label"])
        ax.set_xlabel(r"Generation", fontsize=self.fontsizes["ax_label"])
        ax.yaxis.set_tick_params(labelsize=self.fontsizes["ax_label"])
        ax.set_ylabel(r"Loss" + y_units, fontsize=self.fontsizes["ax_label"])

        selected_generations = self.generations[
            np.greater_equal(self.generations, generation_bounds[0])
            & np.less_equal(self.generations, generation_bounds[1])
        ]
        generation_index = np.argwhere(
            np.greater_equal(self.generations, generation_bounds[0])
            & np.less_equal(self.generations, generation_bounds[1]),
        ).flatten()
        losses = self.losses[generation_index]
        x = np.tile(
            selected_generations,
            (losses.shape[1], 1),
        ).T

        ax.scatter(
            x,
            losses,
            color=self.colors["medium_line"],
            # label="all",
            **kwargs,
        )

        ax.scatter(
            x[:, 0],
            losses.max(axis=1),
            marker=".",
            color=self.colors["dark_line"],
            label="max",
        )

        ax.scatter(
            x[:, 0],
            losses.min(axis=1),
            marker=".",
            color=self.colors["light_line"],
            label="min",
        )

        if ref_val is not None:
            ax.axhline(
                y=ref_val,
                linestyle="dotted",
                color="black",
                alpha=0.85,
                label="ref",
            )

        if y_lim is not None:
            ax.set_ylim(y_lim)

        plt.title(
            f"Loss per generation ({self.label})",
            fontsize=self.fontsizes["title"],
        )

        if show_legend:
            plt.legend(fontsize=self.fontsizes["legend"])

        if save_fig:
            plt.savefig(
                self.output_dir / (self.label + "_loss." + fig_type),
                format=fig_type,
            )

    def plot_additional_per_individual(
        self,
        key: str = None,
        generation_bounds: Tuple[int, int] = None,
        figsize: Tuple[float, float] = None,
        y_units: str = "",
        show_legend: bool = True,
        ref_val: float = None,
        kwargs: dict = None,
        save_fig: bool = False,
        fig_type: str = "pdf",
    ) -> None:
        """Plot additional info per individual per generation.

        Args:
            key: Key that identifies the additional information.
                Supercedes index if both given. Default is None.
            generations: Number of generations to be included.
            generation_bounds: First and last generation to include. Count
                starts at 1, because generation 0 is the founder.
            figsize: Default figure size to be used when no alternative is
                passed to a plotting function.
            show_legend: If True, plot a legend. Default is True.
            ref_val: Value to be plotted as a reference line. Default is None.
            y_units: Units of loss to be added to y_label. Default is empty
                string.
            save_fig: If True the figure will be saved to file. Default  is
                False.
            fig_type: File type to be saved. Default is 'pdf'.
            kwargs: Dicitionary with additional keywords.
        """
        if generation_bounds is None:
            generation_bounds = self.generation_bounds
        if figsize is None:
            figsize = self.figsize
        if kwargs is None:
            kwargs = {}
        if "marker" not in kwargs.keys():
            kwargs["marker"] = "x"
        if "alpha" not in kwargs.keys():
            kwargs["alpha"] = 0.5

        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
        ax.xaxis.set_tick_params(labelsize=self.fontsizes["ax_label"])
        ax.set_xlabel(r"Generation", fontsize=self.fontsizes["ax_label"])
        ax.yaxis.set_tick_params(labelsize=self.fontsizes["ax_label"])
        ax.set_ylabel(f"{key}" + y_units, fontsize=self.fontsizes["ax_label"])

        selected_generations = self.generations[
            np.greater_equal(self.generations, generation_bounds[0])
            & np.less_equal(self.generations, generation_bounds[1])
        ]

        if self.additional_info_types[key] == Number:
            data = np.asarray(
                [
                    [
                        float(d[key])
                        for d in self.information_list[i]["information"]
                    ]
                    for i in range(self.generations.shape[0])
                ]
            )
        data = data[selected_generations - 1]

        x = np.tile(
            selected_generations,
            (data.shape[1], 1),
        ).T

        ax.scatter(
            x,
            data,
            color=self.colors["medium_line"],
            # label="all",
            **kwargs,
        )

        ax.scatter(
            x[:, 0],
            data.max(axis=1),
            marker=".",
            color=self.colors["dark_line"],
            label="max",
        )

        ax.scatter(
            x[:, 0],
            data.min(axis=1),
            marker=".",
            color=self.colors["light_line"],
            label="min",
        )

        if ref_val is not None:
            ax.axhline(
                y=ref_val,
                linestyle="dotted",
                color="black",
                alpha=0.85,
                label="ref",
            )

        plt.title(
            f"{key} per generation ({self.label})",
            fontsize=self.fontsizes["title"],
        )

        if show_legend:
            plt.legend(fontsize=self.fontsizes["legend"])

        if save_fig:
            plt.savefig(
                self.output_dir / (self.label + f"_{key}." + fig_type),
                format=fig_type,
            )
