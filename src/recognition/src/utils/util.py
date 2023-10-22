import numpy as np

def adjust_loss_alpha(alpha, epoch, factor=0.9, loss_type="ce_family", loss_rate_decay="lrdv1", dataset_type="imagenet"):
    """Early Decay Teacher"""
    epoch+=4
    if dataset_type == "imagenet":
        if loss_rate_decay == "lrdv1":
            return alpha * (factor ** (epoch // 3))
        else:  # lrdv2
            if "ce" in loss_type or "kd" in loss_type:
                return 0 if epoch <= 3 else alpha * (factor ** (epoch // 3))
            else:
                return alpha * (factor ** (epoch // 3))
    else:  # cifar
        if loss_rate_decay == "lrdv1":
            return alpha
        else:  # lrdv2
            if epoch >= 160:
                exponent = 2
            elif epoch >= 60:
                exponent = 1
            else:
                exponent = 0
            if "ce" in loss_type or "kd" in loss_type:
                return 0 if epoch <= 60 else alpha * (factor**exponent)
            else:
                return alpha * (factor**exponent)
            

def plot(
    x,
    y,
    ax=None,
    title=None,
    draw_legend=True,
    draw_centers=False,
    draw_cluster_labels=False,
    colors=None,
    legend_kwargs=None,
    label_order=None,
    **kwargs
):
    import matplotlib

    if ax is None:
        _, ax = matplotlib.pyplot.subplots(figsize=(10, 10))

    if title is not None:
        ax.set_title(title)

    plot_params = {"alpha": kwargs.get("alpha", 0.9), "s": kwargs.get("s", 1)}

    # Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}

    point_colors = list(map(colors.get, y))

    ax.scatter(x[:, 0], x[:, 1], c=point_colors, rasterized=False, **plot_params)

    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)

        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )

        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ax.text(
                    centers[idx, 0],
                    centers[idx, 1] + 2.2,
                    label,
                    fontsize=kwargs.get("fontsize", 6),
                    horizontalalignment="center",
                )

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")

    if draw_legend:
        legend_handles = [
            matplotlib.lines.Line2D(
                [],
                [],
                marker="s",
                color="w",
                markerfacecolor=colors[yi],
                ms=13,
                alpha=1,
                linewidth=0,
                label=yi,
                markeredgecolor="k",
            )
            for yi in classes
        ]
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(0.9, 0.5), frameon=False, fontsize=15)
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)

    return ax
