import sys

import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from tsp_generator import *
from plotting import *


def rotate(angle) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def solve():
    def cart(a: np.ndarray, b: np.ndarray) -> [float, float]:
        """
        Calculate cartesian equivalent of a vector equation

        :param a: point a (x,y)
        :param b: direction vector b(x,y)
        :return: [m,c] for y=mx+c
        """
        return [(m := b[1] / b[0]), a[1] - m * a[0]]

    n = 0
    colours = ["coral", "lime", "teal", "orchid"]
    with open("tests/8.csv", 'r') as f:
        for line in f:
            print("\re", end="")
            plt.xlim(0, 1.4)
            plt.ylim(0, 1.4)
            points: np.ndarray = np.array([float(i) for i in line.split(',')]).reshape((8, 2))
            # np.random.shuffle(points)

            # elastic band bounding
            #
            bounding = np.array([points[np.argmin(points[:, 0])]])
            if np.equal(points[np.argmin(points[:, 1])], bounding[0]).all():
                points = np.dot(points, np.array([[0, -1], [1, 0]]))
                bounding = np.dot(bounding, np.array([[0, -1], [1, 0]]))
            min_point = bounding[0]
            points_to_right = points[np.greater(points[:, 0], min_point[0])]
            bounding = np.append(bounding, [points_to_right[
                np.argmin(np.divide(*np.flip(np.subtract(points_to_right, min_point), axis=1).transpose()))
            ]], axis=0)
            theta = np.arctan(np.divide(*np.flip(np.subtract(bounding[-1], min_point))))
            points = np.dot(points, rotate(theta))
            bounding = np.dot(bounding, rotate(theta))
            min_point = bounding[-1]
            while np.sum(np.square(bounding[0] - min_point)) > 1e-10:
                if np.less(np.abs(np.max(points[:, 0]) - min_point[0]), 1e-10).all():
                    points = np.dot(points, np.array([[0, -1], [1, 0]]))
                    min_point = np.dot(min_point, np.array([[0, -1], [1, 0]]))
                    bounding = np.dot(bounding, np.array([[0, -1], [1, 0]]))
                points_to_right = points[np.greater(points[:, 0], min_point[0])]
                bounding = np.append(bounding, [points_to_right[
                    np.argmin(np.divide(*np.flip(np.subtract(points_to_right, min_point), axis=1).transpose()))
                ]], axis=0)
                theta = np.arctan(np.divide(*np.flip(np.subtract(bounding[-1], min_point))))
                min_point = bounding[-1]
                points = np.dot(points, rotate(theta))
                bounding = np.dot(bounding, rotate(theta))
                min_point = np.dot(min_point, rotate(theta))
            plt.plot(*np.append(points, [points[0]], axis=0).transpose(), 'ro--')
            plt.plot(*bounding.transpose(), 'b-')
            bounding = bounding[:-1]


            # for i in range(0, 360, 10):
            #     theta = (i * np.pi / 180)
            #     rot_matrix = np.array([[np.cos(theta), -np.sin(theta)],
            #                            [np.sin(theta), np.cos(theta)]])
            #     bounding_indexes = np.append(bounding_indexes, [np.argmin(np.dot(points, rot_matrix)[:, 1])])
            # bounding_indexes = bounding_indexes[np.append([True], np.not_equal(np.diff(bounding_indexes), 0))]
            # points = decentral_normalised(points)
            # bounding = points[[int(i) for i in bounding_indexes]]
            # plt.plot(*np.append(bounding, [bounding[0]], axis=0).transpose(), 'co-')

            # _, points = generate(9)
            # points = np.array([[0.2757311, 0.92799509],
            #                    [0.43148596, 0.89791586],
            #                    [0.93685708, 0.99717645],
            #                    [0.84154063, 0.79369014],
            #                    [0.52026162, 0.65797271],
            #                    [0.35079133, 0.07908241],
            #                    [0.14903261, 0.10842341],
            #                    [0.35224857, 0.55725443],
            #                    [0.29073447, 0.7492436],
            #                    [0.2757311, 0.92799509]])
            # Shortest path
            # print(points)

            # Getting bounding points
            # sort_y = points[np.argsort(points[:, 1])]
            # sort_x = points[np.argsort(points[:, 0])]
            # points_new = np.array([[0, 0]])
            # bounding = np.array([sort_x[0], sort_y[0], sort_x[-1], sort_y[-1]])
            # # bounding = np.array([bounding_all[-1]])
            # # for i in bounding_all:
            # #     if i not in bounding:
            # #         bounding = np.append(bounding, [i], axis=0)
            # for i in points:
            #     if np.not_equal(bounding, i).all():
            #         points_new = np.append(points_new, [i], axis=0)
            # points = points_new[1:]
            # del points_new
            #
            # vectors = np.diff(np.append(bounding, [bounding[0]], axis=0), axis=0)
            # vectors = np.divide(vectors.transpose(), np.sqrt(np.abs(np.sum(np.square(vectors), axis=1)))).transpose()
            # vectors = np.subtract(vectors, np.array([vectors[-1], *vectors[:-1]]))
            #
            # for i, n in enumerate([ax1, ax2, ax3, ax4]):
            #     n.plot(*np.append(bounding, [bounding[0]], axis=0).transpose(), 'ko-')
            #     start = bounding[i]
            #     end = bounding[i] + (vectors[i] * 2)
            #     n.plot([start[0], end[0]], [start[1], end[1]], '--', color=colours[i])
            #     start = bounding[(i + 1) % 4]
            #     end = bounding[(i + 1) % 4] + (vectors[(i + 1) % 4] * 2)
            #     n.plot([start[0], end[0]], [start[1], end[1]], '--', color=colours[i])
            #
            # func_coeffs = np.array([cart(np.flip(bounding[0]), np.flip(vectors[0])),
            #                         cart(bounding[1], vectors[1]),
            #                         cart(np.flip(bounding[2]), np.flip(vectors[2])),
            #                         cart(bounding[3], vectors[3])])
            # regions = np.array([np.add(np.multiply(points[:, 1], func_coeffs[0, 0]), func_coeffs[0, 1]) < points[:, 0],
            #                     np.add(np.multiply(points[:, 0], func_coeffs[1, 0]), func_coeffs[1, 1]) < points[:, 1],
            #                     np.sign(func_coeffs[2, 0]) * np.add(np.multiply(points[:, 1], func_coeffs[2, 0]),
            #                                                         func_coeffs[2, 1]) > np.sign(
            #                         func_coeffs[2, 0]) * points[:, 0],
            #                     np.add(np.multiply(points[:, 0], func_coeffs[3, 0]), func_coeffs[3, 1]) < points[:, 1]])
            # points_in_region = np.array([np.logical_and(regions[0], np.logical_not(regions[1])),
            #                              np.logical_and(regions[1], np.logical_not(regions[2])),
            #                              np.logical_and(regions[2], regions[3]),
            #                              np.logical_and(np.logical_not(regions[3]), np.logical_not(regions[0]))])

            # for i, (ax, pts) in enumerate(zip([ax1, ax2, ax3, ax4], points_in_region)):
            #     ax.plot(*np.array([points[n] for n, v in enumerate(pts) if v]).transpose(), 'o', color=colours[i])

            plt.show()
            # sys.exit(0)


if __name__ == "__main__":
    solve()

# if __name__ == '__main__':
#     point_num = 7
#     points = np.ndarray((1, 3))
#     for n in range(10):
#         _, test = generate(point_num)
#         plt.title(f"{point_num} | {n}")
#         plt.xlim((0, 1))
#         plt.ylim((0, 1))
#         test = test.transpose()
#         plt.plot(*test, '-')
#         plt.plot(*test, 'o')
#         print(test[0], test[1])
#         plt.show()
# test = test.transpose()
# test[0] = np.divide(test[0] - np.min(test[0]), np.max(test[0]) - np.min(test[0]))
# test[1] = np.divide(test[1] - np.min(test[1]), np.max(test[1]) - np.min(test[1]))
# test = central_normalised(test).transpose()
# for i in range(len(test) - 1):
#     points = np.append(points, np.ndarray([np.dot()]).transpose(), axis=0)
# print(f'\r{n + 1:>4}/1000 iterations', end='')
# print()

# Gaussian KDE
# fig, ax = plt.subplots(1, 1)
# assert isinstance(fig, plt.Figure)
# assert isinstance(ax, plt.Axes)
# normalised_density_plot(ax, points)
#
# plt.show()
# fig.savefig(f'gaussian_kde_{point_num}.png', dpi=500)
# test = np.divide(test - )
# for i in range(10):
#     _, test = generate(8)
#     normals = np.divide(*np.flip(np.diff(test, axis=0), axis=1).transpose())
#     fig, ax = plt.subplots(1, 1)
#     assert isinstance(fig, plt.Figure)
#     assert isinstance(ax, plt.Axes)
#     ax.set_xlim(0, 1)
#     ax.set_ylim(0, 1)
#     ax.plot(*test.transpose(), 'o-')
#     mean = np.mean(test, axis=0)
#     for n in test:
#         ax.plot([n[0], mean[0]], [n[1], mean[1]], 'k--', linewidth=1)
#     ax.plot(*mean, 'o')
#     # plt.show()
#     fig.savefig(f'examples/eg8_{i:0>3}.png')
#     plt.close()
#     print(f'\r{i + 1:>3}/100', end='')
