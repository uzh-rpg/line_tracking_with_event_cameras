import cv2
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np


def visualizeEventsWithImage(events, image_path, height, width):
    """Visualize events projected onto the last image"""

    if len(events) == 0:
        return

    image_events = getEventsImage(events, height, width)
    image_superimposed = getSuperImposedImage(image_events, image_path)

    plt.rcParams["figure.figsize"] = (24, 18)
    plt.imshow(image_superimposed)
    plt.waitforbuttonpress()
    plt.cla()


def visualizeClustersWithImage(clusters, image_path, height, width):
    """Visualize clustered events projected onto the last image"""

    if len(clusters) == 0:
        return

    image_clusters = getClustersImage(clusters, height, width)
    image_superimposed = getSuperImposedImage(image_clusters, image_path)

    plt.rcParams["figure.figsize"] = (24, 18)
    plt.imshow(image_superimposed)
    addClusterIdsToPlot(clusters)
    plt.waitforbuttonpress()
    plt.cla()


def visualizeClustersWithImageAndPoint(clusters, point, t, image_path, height, width):
    if len(clusters) == 0:
        return

    image_clusters = getClustersImage(clusters, height, width)
    image_superimposed = getSuperImposedImage(image_clusters, image_path)

    plt.rcParams["figure.figsize"] = (24, 18)
    plt.scatter(point[0], point[1], color="green", s=50, marker="o")
    plt.imshow(image_superimposed)
    addClusterIdsToPlot(clusters)
    plt.title("Clusters at t = " + str(t))
    plt.waitforbuttonpress()
    plt.cla()


def visualizeLinesWithImageAndPoint(lines, point, t, image_path, height, width):
    events = []
    for line_id in lines:
        events.extend(lines[line_id].events)

    image_events = getEventsImage(events, height, width)
    image_superimposed = getSuperImposedImage(image_events, image_path)

    plt.rcParams["figure.figsize"] = (24, 18)
    plt.scatter(point[0], point[1], color="orange", s=60, marker="o")
    plt.imshow(image_superimposed)
    addLinesToPlot(lines)
    plt.title("Lines at t = " + str(t))
    plt.cla()


def visualizeEventsWithImageAndPoint(
    events, point, t, image_path, height, width, title="Events with Image And Point"
):
    image_events = getEventsImage(events, height, width)
    image_superimposed = getSuperImposedImage(image_events, image_path)

    plt.rcParams["figure.figsize"] = (24, 18)
    plt.scatter(point[0], point[1], color="orange", s=60, marker="o")
    plt.imshow(image_superimposed)
    plt.title(title + ", t = " + str(t))
    plt.waitforbuttonpress()
    plt.cla()


def visualizeChainWithImage(chain, image_path, height, width):

    image_chain = getEventsImage(chain, height, width)
    image_superimposed = getSuperImposedImage(image_chain, image_path)

    plt.rcParams["figure.figsize"] = (24, 18)
    plt.imshow(image_superimposed)
    plt.waitforbuttonpress()
    plt.cla()


def visualizeClustersAndLinesWithImage(clusters, lines, t, image_path, height, width):

    image_clusters = getClustersImage(clusters, height, width)
    image_superimposed = getSuperImposedImage(image_clusters, image_path)

    plt.rcParams["figure.figsize"] = (24, 18)
    plt.imshow(image_superimposed)
    addClusterIdsToPlot(clusters)
    addLinesToPlot(lines)
    plt.title("Clusters and Lines at t = " + str(t))
    plt.xlim([0, width])
    plt.ylim([height, 0])
    plt.waitforbuttonpress()
    plt.cla()


def visualizeChainAndLinesWithImage(chain, lines, t, image_path, height, width):

    image_chain = getEventsImage(chain, height, width)
    image_superimposed = getSuperImposedImage(image_chain, image_path)

    plt.rcParams["figure.figsize"] = (24, 18)
    plt.imshow(image_superimposed)
    addLinesToPlot(lines)
    plt.title("Chain and Lines at t = " + str(t))
    plt.waitforbuttonpress()
    plt.cla()


def visualizeUnasignedEventsAndChainAndLinesWithImage(
    unasigned_events, chain, lines, t, image_path, height, width
):
    image_unasigned_events = getEventsImagePolarityColored(
        unasigned_events, height, width
    )
    image_superimposed = getSuperImposedImage(image_unasigned_events, image_path)

    plt.rcParams["figure.figsize"] = (24, 18)
    plt.imshow(image_superimposed)
    addChainToPlot(chain)
    addLinesToPlot(lines)
    plt.title("Chain and Lines at t = " + str(t))
    plt.waitforbuttonpress()
    plt.cla()


def visualizeClustersAndLinesWithImagePolarity(
    clusters, lines, pol, t, image_path, height, width
):
    clusters_pol = {}
    for cluster_id in clusters:
        cluster = clusters[cluster_id]
        if cluster.pol == pol:
            clusters_pol[cluster_id] = cluster

    lines_pol = {}
    for line_id in lines:
        line = lines[line_id]
        if line.pol == pol:
            lines_pol[line_id] = line

    visualizeClustersAndLinesWithImage(
        clusters_pol, lines_pol, t, image_path, height, width
    )


def visualizeClustersAndLinesWithImagePolaritySeparated(
    clusters, lines, t, image_path, height, width
):

    if len(clusters) == 0:
        return

    clusters_pos = {}
    clusters_neg = {}

    for cluster_id in clusters:
        cluster = clusters[cluster_id]
        if cluster.pol == 1:
            clusters_pos[cluster_id] = cluster
        else:
            clusters_neg[cluster_id] = cluster

    image_clusters_pos = getClustersImage(clusters_pos, height, width)
    image_clusters_neg = getClustersImage(clusters_neg, height, width)

    lines_pos = {}
    lines_neg = {}

    for line_id in lines:
        line = lines[line_id]
        if line.pol == 1:
            lines_pos[line_id] = line
        else:
            lines_neg[line_id] = line

    image_superimposed_pos = getSuperImposedImage(image_clusters_pos, image_path)
    image_superimposed_neg = getSuperImposedImage(image_clusters_neg, image_path)

    fig = plt.gcf()
    ax_pos = fig.add_subplot(121)
    ax_neg = fig.add_subplot(122)

    # fig, (ax_pos, ax_neg) = plt.subplots(1, 2)
    fig.suptitle("Clusters and Lines at t = " + str(t))
    plt.rcParams["figure.figsize"] = (24, 18)
    ax_pos.imshow(image_superimposed_pos)
    ax_neg.imshow(image_superimposed_neg)
    ax_pos.set_adjustable("datalim")
    ax_neg.set_adjustable("datalim")
    ax_pos.set_aspect(1)
    ax_neg.set_aspect(1)
    addClusterIdsToAxis(clusters_pos, ax_pos)
    addClusterIdsToAxis(clusters_neg, ax_neg)
    addLinesToAxis(lines_pos, ax_pos)
    addLinesToAxis(lines_neg, ax_neg)
    plt.waitforbuttonpress()
    plt.cla()


def visualizeLinesIndividually(lines, image_path, height, width):
    if len(lines) == 0:
        return

    for line_id in lines:
        visualizeLineWithImage(lines[line_id], line_id, image_path, height, width)


def visualizeLineIdWithImage(lines, line_id, image_path, heigth, width):
    if len(lines) == 0:
        return

    visualizeLineWithImage(lines[line_id], line_id, image_path, heigth, width)


def visualizeLineWithImage(line, line_id, image_path, height, width):
    image_events = getEventsImage(line.events, height, width)
    image_superimposed = getSuperImposedImage(image_events, image_path)
    plt.rcParams["figure.figsize"] = (24, 18)
    plt.imshow(image_superimposed)
    addLineToPlot(line, line_id)
    plt.waitforbuttonpress()
    plt.cla()


def visualizeLine3D(lines, line_id, image_path, t):
    line = lines[line_id]
    events = np.stack(line.events, axis=0)
    x = np.linspace(np.min(events[:, 0]), np.max(events[:, 0]), 5)
    y = np.linspace(np.min(events[:, 1]), np.max(events[:, 1]), 5)
    xx, yy = np.meshgrid(x, y)
    d = -line.cog.dot(line.normal)
    z = (-line.normal[0] * xx - line.normal[1] * yy - d) * 1.0 / line.normal[2]

    # fig = plt.figure()
    ax = plt.axes(projection="3d")
    ax.scatter3D(events[:, 0], events[:, 1], events[:, 2], c="seagreen", s=15)
    ax.plot_surface(xx, yy, z, alpha=0.3, color="b")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")

    max_range = np.array([x.max() - x.min(), y.max() - y.min()]).max()
    Xb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][0].flatten() + 0.5 * (
        x.max() + x.min()
    )
    Yb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][1].flatten() + 0.5 * (
        y.max() + y.min()
    )
    Zb = 0.5 * max_range * np.mgrid[-1:2:2, -1:2:2, -1:2:2][2].flatten() + 0.5 * (
        z.max() + z.min()
    )
    # Comment or uncomment following both lines to test the fake bounding box:
    for xb, yb, zb in zip(Xb, Yb, Zb):
        ax.plot([xb], [yb], [zb], "w")

    # add image
    image_frame = cv2.cvtColor(cv2.imread(image_path, 0), cv2.COLOR_GRAY2RGB) / 255
    image_frame = np.dstack(
        (image_frame, np.full((image_frame.shape[0], image_frame.shape[1]), 0.5))
    )
    x_im = np.arange(0, image_frame.shape[1])
    y_im = np.arange(0, image_frame.shape[0])
    xx_im, yy_im = np.meshgrid(x_im, y_im)
    zz_im = np.full(xx_im.shape, t)
    ax.plot_surface(
        xx_im,
        yy_im,
        zz_im,
        rstride=4,
        cstride=4,
        facecolors=image_frame,
    )

    ax.invert_yaxis()
    ax.view_init(90, 90)
    plt.show()


def visualizeLines3D(lines, t):
    colors_idx_shuffeled = np.arange(len(lines))
    np.random.shuffle(colors_idx_shuffeled)
    colors = cm.rainbow(np.linspace(0, 1, len(lines)))
    colors = colors[colors_idx_shuffeled]
    ax = plt.axes(projection="3d")

    for i, line_id in enumerate(lines):
        line = lines[line_id]
        color = colors[i]
        events = np.stack(line.events, axis=0)
        ax.scatter3D(events[:, 0], events[:, 1], events[:, 2], c=color, s=25)

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.invert_yaxis()

    plt.show()


def visualizeLinesAndClusters3D(lines, clusters, t, height, width):

    # events
    event_alpha = 0.5

    # line colors
    colors_idx_shuffeled = np.arange(len(lines))
    np.random.shuffle(colors_idx_shuffeled)
    colors = cm.rainbow(np.linspace(0, 1, len(lines)))
    colors = colors[colors_idx_shuffeled]
    ax = plt.axes(projection="3d")

    # cluster color
    cluster_color = [180.0 / 255, 180.0 / 255, 180.0 / 255]

    # testing
    line_color = [1.0, 0.0, 0.0]
    line_not_init_color = [51.0 / 255, 153.0 / 255, 255.0 / 255]

    for i, line_id in enumerate(lines):
        line = lines[line_id]
        color = colors[i]
        events = np.stack(line.events, axis=0)
        # line endpoints
        point_1 = line.point[:2] - line.line_direction * line.length / 2
        point_2 = line.point[:2] + line.line_direction * line.length / 2

        if line.initializing:
            l_color = line_not_init_color
        else:
            l_color = line_color

        ax.scatter3D(
            events[:, 0],
            events[:, 1],
            events[:, 2],
            color=l_color,
            alpha=event_alpha,
            s=30,
        )

        label = "Line " + str(line_id)
        ax.plot(
            [point_1[0], point_2[0]],
            [point_1[1], point_2[1]],
            zs=t,
            zdir="z",
            label=label,
            color=l_color,
            linewidth=5,
        )

    for cluster_id in clusters:
        cluster = clusters[cluster_id]
        events = np.stack(cluster.events, axis=0)
        ax.scatter3D(
            events[:, 0],
            events[:, 1],
            events[:, 2],
            color=cluster_color,
            alpha=event_alpha,
            s=25,
        )

    ax.legend()
    ax.set_xlim(0, width)
    ax.set_ylim(0, height)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.invert_yaxis()

    plt.show()


def visualizeEvents3D(events, t):

    ax = plt.axes(projection="3d")

    color_pos = [255.0 / 255, 0.0 / 255, 0.0 / 255]
    color_neg = [0.0 / 255, 0.0 / 255, 0.0 / 255]

    events = np.stack(events, axis=0)
    events_pos = events[events[:, 3] == 1]
    events_neg = events[events[:, 3] == 0]

    ax.scatter3D(
        events_pos[:, 0], events_pos[:, 1], events_pos[:, 2], color=color_pos, s=30
    )

    ax.scatter3D(
        events_neg[:, 0], events_neg[:, 1], events_neg[:, 2], color=color_neg, s=30
    )

    ax.set_xlim(0, 180)
    ax.set_ylim(0, 240)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("t")
    ax.invert_yaxis()

    plt.show()


def visualizeLines(lines, image_path, height, width):
    if len(lines) == 0:
        return

    events = []
    for line_id in lines:
        events.extend(lines[line_id].events)
    image_events = getEventsImage(events, height, width)
    image_superimposed = getSuperImposedImage(image_events, image_path)
    plt.rcParams["figure.figsize"] = (24, 18)
    plt.imshow(image_superimposed)
    addLinesToPlot(lines)
    plt.waitforbuttonpress()
    plt.cla()


def addClusterIdsToPlot(clusters):
    for cluster_id in clusters:
        spatial_mean = np.mean(np.array(clusters[cluster_id].events)[:, :2], axis=0)
        plt.text(
            spatial_mean[0],
            spatial_mean[1],
            str(cluster_id),
            fontsize=12,
            fontweight="bold",
            color="red",
        )


def addClusterIdsToAxis(clusters, ax):
    for cluster_id in clusters:
        spatial_mean = np.mean(np.array(clusters[cluster_id].events)[:, :2], axis=0)
        ax.text(
            spatial_mean[0],
            spatial_mean[1],
            str(cluster_id),
            fontsize=12,
            fontweight="bold",
            color="red",
        )


def addLineToPlot(line, line_id):
    end_point_1 = line.point[:2] + line.line_direction * (line.length / 2)
    end_point_2 = line.point[:2] - line.line_direction * (line.length / 2)

    if line.pol == 1:
        text_color = "lightsteelblue"
        line_color = "royalblue"
    else:
        text_color = "lightcoral"
        line_color = "firebrick"

    plt.text(
        line.point[0],
        line.point[1],
        str(line_id),
        fontsize=14,
        fontweight="bold",
        color=text_color,
    )
    plt.plot(
        [end_point_1[0], end_point_2[0]],
        [end_point_1[1], end_point_2[1]],
        linewidth=3,
        color=line_color,
    )
    plt.plot(end_point_1[0], end_point_1[1], color=line_color, linewidth=7)
    plt.plot(end_point_2[0], end_point_2[1], color=line_color, linewidth=7)


def addLineToPlotColor(line, line_id, line_color, text_color):

    end_point_1 = line.point[:2] + line.line_direction * (line.length / 2)
    end_point_2 = line.point[:2] - line.line_direction * (line.length / 2)

    plt.text(
        line.point[0],
        line.point[1],
        str(line_id),
        fontsize=14,
        fontweight="bold",
        color=text_color,
    )
    plt.plot(
        [end_point_1[0], end_point_2[0]],
        [end_point_1[1], end_point_2[1]],
        linewidth=3,
        color=line_color,
    )
    plt.plot(end_point_1[0], end_point_1[1], color=line_color, linewidth=7)
    plt.plot(end_point_2[0], end_point_2[1], color=line_color, linewidth=7)


def addLineToAxis(line, line_id, ax):
    end_point_1 = line.point[:2] + line.line_direction * (line.length / 2)
    end_point_2 = line.point[:2] - line.line_direction * (line.length / 2)

    if line.pol == 0:
        text_color = "lightsteelblue"
        line_color = "royalblue"
    else:
        text_color = "lightcoral"
        line_color = "firebrick"

    ax.text(
        line.point[0],
        line.point[1],
        str(line_id),
        fontsize=14,
        fontweight="bold",
        color=text_color,
    )
    ax.plot(
        [end_point_1[0], end_point_2[0]],
        [end_point_1[1], end_point_2[1]],
        linewidth=3,
        color=line_color,
    )
    ax.plot(end_point_1[0], end_point_1[1], color=line_color, linewidth=7)
    ax.plot(end_point_2[0], end_point_2[1], color=line_color, linewidth=7)


def addChainToPlot(chain):
    chain_x = [e[0] for e in chain]
    chain_y = [e[1] for e in chain]
    plt.scatter(chain_x, chain_y, color="green", s=35, marker="o")


def addLinesToPlot(lines):
    for line_id in lines:
        addLineToPlot(lines[line_id], line_id)


def addInitializedLinesToPlot(lines):
    for line_id in lines:
        if not lines[line_id].initializing:
            addLineToPlot(lines[line_id], line_id)


def addAllLinesToPlot(lines):
    for line_id in lines:

        # initializing lines
        if lines[line_id].initializing:
            line_color = "grey"
            text_color = "lightgrey"
            addLineToPlotColor(lines[line_id], line_id, line_color, text_color)

        # hibernating lines
        elif lines[line_id].hibernate:
            line_color = "sandybrown"
            text_color = "peachpuff"
            addLineToPlotColor(lines[line_id], line_id, line_color, text_color)
        # normal lines
        else:
            line_color = "firebrick"
            text_color = "lightcoral"
            addLineToPlotColor(lines[line_id], line_id, line_color, text_color)


def addLinesToImage(image, lines):

    for line_id in lines:
        line = lines[line_id]
        if line.pol == 0:
            color = (0, 0, 255)
        else:
            color = (255, 0, 0)

        end_point_1 = tuple(
            (line.point[:2] + line.line_direction * (line.length / 2)).astype(int)
        )
        end_point_2 = tuple(
            (line.point[:2] - line.line_direction * (line.length / 2)).astype(int)
        )
        image = cv2.line(image, end_point_1, end_point_2, color, 1)


def addLinesToAxis(lines, ax):
    for line_id in lines:
        addLineToAxis(lines[line_id], line_id, ax)


def getSuperImposedImage(image_events, image_path):
    image_frame = cv2.cvtColor(cv2.imread(image_path, 0), cv2.COLOR_GRAY2RGB)
    image_frame[image_events != 0] = 0

    image_superimposed = cv2.addWeighted(image_frame, 1, image_events, 1, 0)
    return image_superimposed


def getEventsImagePolarityColored(events, height, width):

    image_events_both = np.zeros((height, width, 3))
    color = []
    events = np.stack(events, axis=0)
    events_pos = events[events[:, 3] == 1]
    events_neg = events[events[:, 3] == 0]

    events_both = [events_neg, events_pos]

    for i, events in enumerate(events_both):
        if i == 0:
            color = [1, 0, 0]
        else:
            color = [0, 0, 1]
        image_events = np.zeros(height * width, dtype=np.uint8)
        np.add.at(
            image_events, (events[:, 0] + events[:, 1] * width).astype("int32"), 1
        )
        image_events = image_events.reshape((height, width))

        image_events = (
            np.stack(
                [
                    image_events * color[0],
                    image_events * color[1],
                    image_events * color[2],
                ],
                axis=2,
            )
            * 255
        )
        image_events_both += image_events

    return image_events_both.astype("uint8")


def getEventsImage(events, height, width):
    events = np.stack(events, axis=0)
    color = [203.0 / 255, 255.0 / 255, 229.0 / 255]
    image_events = np.zeros(height * width, dtype=np.uint8)
    np.add.at(image_events, (events[:, 0] + events[:, 1] * width).astype("int32"), 1)
    image_events = image_events.reshape((height, width))

    image_events = (
        np.stack(
            [
                image_events * color[0],
                image_events * color[1],
                image_events * color[2],
            ],
            axis=2,
        )
        * 255
    )

    return image_events.astype("uint8")


def getClustersImage(clusters, height, width):

    image_clusters = np.zeros((height, width, 3))

    colors_idx_shuffeled = np.arange(len(clusters))
    np.random.shuffle(colors_idx_shuffeled)
    colors = cm.rainbow(np.linspace(0, 1, len(clusters)))
    colors = colors[colors_idx_shuffeled]

    for i, cluster_id in enumerate(clusters):
        events = np.stack(clusters[cluster_id].events, axis=0)
        color = colors[i]
        image_cluster = np.zeros(height * width, dtype=np.uint8)
        np.add.at(
            image_cluster, (events[:, 0] + events[:, 1] * width).astype("int32"), 1
        )
        image_cluster = image_cluster.reshape((height, width))

        image_cluster = (
            np.stack(
                [
                    image_cluster * color[0],
                    image_cluster * color[1],
                    image_cluster * color[2],
                ],
                axis=2,
            )
            * 255
        )
        image_clusters += image_cluster

    image_clusters = image_clusters.astype("uint8")

    return image_clusters


def writeLinesAndImage(lines, pol, image_path, path):
    image_frame = cv2.cvtColor(cv2.imread(image_path, 0), cv2.COLOR_GRAY2RGB)

    lines_pol = {}
    for line_id in lines:
        line = lines[line_id]
        # if line.pol == pol:
        lines_pol[line_id] = line

    addLinesToImage(image_frame, lines_pol)

    cv2.imwrite(path, image_frame)


def writeInitializedLinesAndImage(lines, image_path, path, height, width):
    image_frame = cv2.cvtColor(cv2.imread(image_path, 0), cv2.COLOR_GRAY2RGB)

    plt.imshow(image_frame)
    addInitializedLinesToPlot(lines)
    plt.xlim([0, width])
    plt.ylim([height, 0])
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.cla()


def writeAllLinesAndImage(lines, image_path, path, height, width):
    image_frame = cv2.cvtColor(cv2.imread(image_path, 0), cv2.COLOR_GRAY2RGB)

    plt.imshow(image_frame)
    addAllLinesToPlot(lines)
    plt.xlim([0, width])
    plt.ylim([height, 0])
    plt.axis("off")
    plt.savefig(path, bbox_inches="tight")
    plt.cla()
