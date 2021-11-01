import csv
import sys

import matplotlib.pyplot as plt
import numpy as np


def rotate(angle) -> np.ndarray:
    return np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])


def solve(points: np.ndarray):
    """
    Function to heuristically solves a metric TSP problem

    :param points: flattened points in the range [-1,1]
    :return:
    """
    plt.xlim(-1.4, 1.4)
    plt.ylim(-1.4, 1.4)
    points = points.reshape((len(points) // 2, 2))
    # Optional randomisation/shuffling of the original set
    # np.random.shuffle(points)

    # elastic band bounding
    left = np.argmin(points[:, 0])
    points_above = points[np.greater(points[:, 1], points[left][1])]
    first = np.argmax(np.divide(*np.flip(np.subtract(points_above, points[left]), axis=1).transpose()))
    theta = np.arctan(np.divide(*np.flip(np.subtract(points[first], points[left])))) - np.pi / 2
    points = np.dot(points, rotate(theta))
    bounding = np.array([left])
    theta_all = np.array([])
    rotated = False
    while left != first:
        if np.argmin(points[:, 1]) == left:
            rotated = True
            points = np.dot(points, np.array([[0, -1], [1, 0]]))
        left_point = points[left]
        indexes = np.less(points[:, 1], left_point[1])
        points_below = points[indexes]
        bounding = np.append(bounding, [
            np.where(indexes)[0][
                np.argmin(np.divide(*np.flip(np.subtract(points_below, left_point), axis=1).transpose()))
            ]], axis=0)
        theta = np.arctan(np.divide(*np.flip(np.subtract(points[bounding[-1]], left_point)))) + np.pi / 2
        theta_all = np.append(theta_all, [theta]) + (np.pi / 2 if rotated else 0)
        rotated = False
        left = bounding[-1]
        points = np.dot(points, rotate(theta))
    plt.plot(*np.append(points, [points[0]], axis=0).transpose(), 'ro--')
    plt.plot(*points[bounding].transpose(), 'b-')

    plt.show()
    sys.exit(0)


if __name__ == "__main__":
    with open("tests/8.csv", "r") as f:
        read = csv.reader(f)
        for row in read:
            solve(np.array([float(i) for i in row]))
