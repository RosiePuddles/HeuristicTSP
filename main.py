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

            plt.show()


if __name__ == "__main__":
    solve()
