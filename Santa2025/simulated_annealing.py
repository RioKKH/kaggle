#!/usr/bin/env python

import datetime
import copy
import time
import math
import os
import random
from decimal import Decimal, getcontext

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.patches as mpatches

from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
from shapely import affinity, touches
from shapely.geometry import Polygon
from shapely.ops import unary_union
from shapely.strtree import STRtree
from abc import ABC, abstractmethod

pd.set_option("display.float_format", "{:.12f}".format)

# Set precision for Decimal
getcontext().prec = 25  # Decimal 精度25桁を使用して数値誤差を防ぐ
scale_factor = Decimal(
    "1e15"
)  # スケールファクター 1e15で座標を整数化して計算精度を向上

# Build the index of the submission, in the format:
#   <trees_in_problem>_<tree_index>
index = [f"{n:03d}_{t}" for n in range(1, 201) for t in range(n)]


class ChristmasTree:
    """Represetns a single, rotatable Christmas tree of a fixed size."""

    def __init__(self, center_x="0", center_y="0", angle="0"):
        """Initializes the Christmas tree with a specific position and rotation."""
        self.center_x = Decimal(center_x)
        self.center_y = Decimal(center_y)
        self.angle = Decimal(angle)

        trunk_w = Decimal("0.15")  # 幹の幅
        trunk_h = Decimal("0.2")  # 幹の高さ
        base_w = Decimal("0.7")  # 底辺の幅
        mid_w = Decimal("0.4")  # 中段の幅
        top_w = Decimal("0.25")  # 上段の幅
        tip_y = Decimal("0.8")  # 頂点の高さ
        tier_1_y = Decimal("0.5")
        tier_2_y = Decimal("0.25")
        base_y = Decimal("0.0")
        trunk_bottom_y = -trunk_h

        initial_polygon = Polygon(
            [
                # Start at Tip
                (Decimal("0.0") * scale_factor, tip_y * scale_factor),
                # Right side - Top Tier
                (top_w / Decimal("2") * scale_factor, tier_1_y * scale_factor),
                (top_w / Decimal("4") * scale_factor, tier_1_y * scale_factor),
                # Right side - Middle Tier
                (mid_w / Decimal("2") * scale_factor, tier_2_y * scale_factor),
                (mid_w / Decimal("4") * scale_factor, tier_2_y * scale_factor),
                # Right side - Bottom Tier
                (base_w / Decimal("2") * scale_factor, base_y * scale_factor),
                # Right Trunk
                (trunk_w / Decimal("2") * scale_factor, base_y * scale_factor),
                (trunk_w / Decimal("2") * scale_factor, trunk_bottom_y * scale_factor),
                # Left Trunk
                (
                    -(trunk_w / Decimal("2")) * scale_factor,
                    trunk_bottom_y * scale_factor,
                ),
                (-(trunk_w / Decimal("2")) * scale_factor, base_y * scale_factor),
                # Left side - Bottom Tier
                (-(base_w / Decimal("2")) * scale_factor, base_y * scale_factor),
                # Left side - Middle Tier
                (-(mid_w / Decimal("4")) * scale_factor, tier_2_y * scale_factor),
                (-(mid_w / Decimal("2")) * scale_factor, tier_2_y * scale_factor),
                # Left side - Top Tier
                (-(top_w / Decimal("4")) * scale_factor, tier_1_y * scale_factor),
                (-(top_w / Decimal("2")) * scale_factor, tier_1_y * scale_factor),
            ]
        )
        rotated = affinity.rotate(initial_polygon, float(self.angle), origin=(0, 0))
        self.polygon = affinity.translate(
            rotated,
            xoff=float(self.center_x * scale_factor),
            yoff=float(self.center_y * scale_factor),
        )

    def get_params(self):
        return self.center_x, self.center_y, self.angle

    def set_params(self, center_x, center_y, angle):
        self.__init__(str(center_x), str(center_y), str(angle))

    def clone(self):
        """
        Create a deep copy of the tree.
        """
        return ChristmasTree(str(self.center_x), str(self.center_y), str(self.angle))


class Problem(ABC):
    """
    Abstract class for an optimization problem.
    SA algorithm works with any Problem without knowing the details of a specific task.
    """

    @abstractmethod
    def get_initial_state(self):
        """Returns the initial state of the problem."""
        pass

    @abstractmethod
    def perturb(self, state):
        """Modifies the state of the problem.
        Returns (new_state, old_state_for_rollback).
        """
        pass

    @abstractmethod
    def evaluate(self, state):
        """
        Calculates the value of the function for the state.
        Lower = better (minimization).
        """
        pass

    @abstractmethod
    def is_valid(self, state):
        """Checks if the state is valie."""
        pass

    @abstractmethod
    def clone_state(self, state):
        """Creates a copy of the state."""
        pass


class SingleBlockProblem(Problem):
    """
    Base class for a single grid block problem.
    Can be used to build more complex problems (e.g., two-block).
    State is a dictionary with pair and grid parameters:
    {
        'pair': [tree0, tree1], # Pair of trees (unit cell)
        'a': Decimal,           # Grid step in X
        'b': Decimal,           # Grid step in Y
    }
    """

    def __init__(
        self,
        initial_pair,
        ncols,
        nrows,
        initial_a,
        initial_b,
        position_delta=0.002,
        angle_delta=1.0,
        delta_t=0.002,
    ):
        """
        initial_pair: list of 2 trees (base pair)
        ncols, nrows: grid dimensions
        initial_a, initial_b: initial values of a and b (required parameters)
        """
        self.initial_pair = [tree.clone() for tree in initial_pair]
        self.ncols = ncols
        self.nrows = nrows
        self.position_delta = Decimal(str(position_delta))
        self.angle_delta = Decimal(str(angle_delta))
        self.delta_t = Decimal(str(delta_t))
        # Save initial a and b
        self.initial_a = Decimal(str(initial_a))
        self.initial_b = Decimal(str(initial_b))

    def get_initial_state(self):
        """Returns initial state: pair and initial a, b."""
        pair_trees = [tree.clone() for tree in self.initial_pair]

        return {
            "pair": pair_trees,
            "a": self.initial_a,
            "b": self.initial_b,
        }

    def translate(self, state, offset_x=Decimal("0"), offset_y=Decimal("0")):
        """
        Builds a grid from a pair with steps a and b.
        offset_x, offset_y: offset of the entire block
        Returns a list of all trees in the grid.
        """
        pair = state["pair"]
        a = state["a"]
        b = state["b"]

        grid_trees = []
        for row in range(self.nrows):
            for col in range(self.ncols):
                for tree in pair:
                    # Offset each tree from the pair
                    new_x = tree.center_x + Decimal(col) * a + offset_x
                    new_y = tree.center_y + Decimal(row) * b + offset_y
                    grid_trees.append(
                        ChristmasTree(str(new_x), str(new_y), str(tree.angle))
                    )

        return grid_trees

    def perturb(self, state):
        """
        Modifies state: either pair parameters or a, b.
        Returns (new_state, rollback_data).
        """
        # Choose change type: 0-1 = tree in pair, 2 = a, 3 = b, 4 = rotate_all
        move_type = random.randint(0, 4)

        if move_type < 2:
            # Modify one tree in the pair
            tree_idx = move_type
            tree = state["pair"][tree_idx]
            old_params = tree.get_params()

            dx = Decimal(
                str(
                    random.uniform(
                        -float(self.position_delta), float(self.position_delta)
                    )
                )
            )
            dy = Decimal(
                str(
                    random.uniform(
                        -float(self.position_delta), float(self.position_delta)
                    )
                )
            )
            dangle = Decimal(
                str(random.uniform(-float(self.angle_delta), float(self.angle_delta)))
            )
            new_x = tree.center_x + dx
            new_y = tree.center_y + dy
            new_angle = (Decimal(str(tree.angle)) + dangle) % 360
            tree.set_params(new_x, new_y, new_angle)

            return state, ("tree", tree_idx, old_params)

        elif move_type == 2:
            # Modify a
            old_a = state["a"]
            da = Decimal(str(random.uniform(-float(self.delta_t), float(self.delta_t))))
            new_a = old_a + old_a * da
            if new_a <= 0:
                new_a = old_a
            state["a"] = new_a
            return state, ("a", old_a)

        elif move_type == 3:
            # Modify b
            old_b = state["b"]
            db = Decimal(str(random.uniform(-float(self.delta_t), float(self.delta_t))))
            new_b = old_b + old_b * db
            if new_b <= 0:
                new_b = old_b
            state["b"] = new_b
            return state, ("b", old_b)

        else:  # move_type == 4
            # Rotate all trees in the pair by one angle
            old_angles = [Decimal(str(tree.angle)) for tree in state["pair"]]
            dangle = Decimal(
                str(random.uniform(-float(self.angle_delta), float(self.angle_delta)))
            )
            for tree in state["pair"]:
                new_angle = (Decimal(str(tree.angle)) + dangle) % 360
                tree.set_params(tree.center_x, tree.center_y, new_angle)
            return state, ("rotate_all", old_angles)

    def rollback(self, state, rollback_data):
        """Rolls back the state change."""
        if rollback_data is None:
            return

        if rollback_data[0] == "tree":
            _, tree_idx, old_params = rollback_data
            state["pair"][tree_idx].set_params(*old_params)
        elif rollback_data[0] == "a":
            _, old_a = rollback_data
            state["a"] = old_a
        elif rollback_data[0] == "b":
            _, old_b = rollback_data
            state["b"] = old_b
        elif rollback_data[0] == "rotate_all":
            _, old_angles = rollback_data
            for i, tree in enumerate(state["pair"]):
                tree.set_params(tree.center_x, tree.center_y, old_angles[i])

    def evaluate(self, state, offset_x=Decimal("0"), offset_y=Decimal("0")):
        """Calculates the bounding square side for the block (lower = better)."""
        grid_trees = self.translate(state, offset_x, offset_y)
        if not grid_trees:
            return Decimal("0")
        xys = np.concatenate(
            [np.asarray(t.polygon.exterior.xy).T / 1e15 for t in grid_trees]
        )
        min_x, min_y = xys.min(axis=0)
        max_x, max_y = xys.max(axis=0)
        width = Decimal(str(max_x - min_x))
        height = Decimal(str(max_y - min_y))
        return max(width, height)

    def is_valid(self, state, offset_x=Decimal("0"), offset_y=Decimal("0")):
        """Checks if there are no overlaps in the block."""
        grid_trees = self.translate(state, offset_x, offset_y)
        if len(grid_trees) <= 1:
            return True
        for i, tree1 in enumerate(grid_trees):
            for j, tree2 in enumerate(grid_trees):
                if i < j:
                    if tree1.polygon.intersects(
                        tree2.polygon
                    ) and not tree1.polygon.touches(tree2.polygon):
                        return False
        return True

    def clone_state(self, state):
        """Creates a copy of the state."""
        return {
            "pair": [tree.clone() for tree in state["pair"]],
            "a": Decimal(str(state["a"])),
            "b": Decimal(str(state["b"])),
        }


class TreePackingProblem(SingleBlockProblem):


def generate_weighted_angle():
    """
    Generates a random angle with a distribution weighted by abs(sin(2*angle)).
    This helps place more trees in corners, and makes the packing less round.
    """
    while True:
        angle = random.uniform(0, 2 * math.pi)
        if random.uniform(0, 1) < abs(math.sin(2 * angle)):
            return angle


def initialize_trees(num_trees, existing_trees=None):
    """
    This builds a simple, greedy starting configuration, by using the previous n-tree
    placements, and adding more tree for the (n+1)-tree configuration. We place a tree
    fairly far away at a (weighte) random angle, and the bring it closer to the center
    until it overlaps. Then we back it up until it no longer overlaps.

    You can easily modify this code to build each n-tree configuration completely from
    scratch.
    """
    if num_trees == 0:
        return [], Decimal("0")

    if existing_trees is None:
        placed_trees = []
    else:
        placed_trees = list(existing_trees)

    num_to_add = num_trees - len(placed_trees)

    if num_to_add > 0:
        unplaced_trees = [
            ChristmasTree(angle=random.uniform(0, 360)) for _ in range(num_to_add)
        ]
        if (
            not placed_trees
        ):  # Only place the first tree at origin if starting from scratch
            placed_trees.append(unplaced_trees.pop(0))

        for tree_to_place in unplaced_trees:
            placed_polygons = [p.polygon for p in placed_trees]
            tree_index = STRtree(placed_polygons)

            best_px = None
            best_py = None
            min_radius = Decimal("Infinity")

            # This loop tries 10 random starting attempts and keeps the best one
            for _ in range(10):
                # The new tree starts at a position 20 from the center,
                # at a random vector angle.
                angle = generate_weighted_angle()
                vx = Decimal(str(math.cos(angle)))
                vy = Decimal(str(math.sin(angle)))

                # Move towards center along the vector in steps of 0.5 until collision
                radius = Decimal("20.0")
                step_in = Decimal("0.5")

                collision_found = False
                while radius >= 0:
                    px = radius * vx
                    py = radius * vy

                    candidate_poly = affinity.translate(
                        tree_to_place.polygon,
                        xoff=float(px * scale_factor),
                        yoff=float(py * scale_factor),
                    )

                    # Looking for nearby objects
                    possible_indices = tree_index.query(candidate_poly)
                    # This is the collision detection step
                    if any(
                        (
                            candidate_poly.intersects(placed_polygons[i])
                            and not candidate_poly.touches(placed_polygons[i])
                        )
                        for i in possible_indices
                    ):
                        collision_found = True
                        break
                    radius -= step_in

                # back up in steps of 0.05 until it no longer has a collision.
                if collision_found:
                    step_out = Decimal("0.05")
                    while True:
                        radius += step_out
                        px = radius * vx
                        py = radius * vy

                        candidate_poly = affinity.translate(
                            tree_to_place.polygon,
                            xoff=float(px * scale_factor),
                            yoff=float(py * scale_factor),
                        )

                        possible_indices = tree_index.query(candidate_poly)
                        if not any(
                            (
                                candidate_poly.intersects(placed_polygons[i])
                                and not candidate_poly.touches(placed_polygons[i])
                            )
                            for i in possible_indices
                        ):
                            break
                else:
                    # No collision found even at the center. Place it at the center.
                    radius = Decimal("0")
                    px = Decimal("0")
                    py = Decimal("0")

                if radius < min_radius:
                    min_radius = radius
                    best_px = px
                    best_py = py

            tree_to_place.center_x = best_px
            tree_to_place.center_y = best_py
            tree_to_place.polygon = affinity.translate(
                tree_to_place.polygon,
                xoff=float(tree_to_place.center_x * scale_factor),
                yoff=float(tree_to_place.center_y * scale_factor),
            )
            placed_trees.append(tree_to_place)  # Add the newly placed tree to the list

    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds

    minx = Decimal(bounds[0]) / scale_factor
    miny = Decimal(bounds[1]) / scale_factor
    maxx = Decimal(bounds[2]) / scale_factor
    maxy = Decimal(bounds[3]) / scale_factor

    width = maxx - minx
    height = maxy - miny

    # this forces a square bounding using the largest side
    side_length = max(width, height)

    return placed_trees, side_length


def plot_results(side_length, placed_trees, num_trees):
    """
    Plots the arrangement of trees and the bounding square.
    """
    _, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.viridis([i / num_trees for i in range(num_trees)])

    all_polygons = [t.polygon for t in placed_trees]
    bounds = unary_union(all_polygons).bounds

    for i, tree in enumerate(placed_trees):
        # Rescale for plotting
        x_scaled, y_scaled = tree.polygon.exterior.xy
        x = [Decimal(val) / scale_factor for val in x_scaled]
        y = [Decimal(val) / scale_factor for val in y_scaled]
        ax.plot(x, y, color=colors[i])
        ax.fill(x, y, alpha=0.5, color=colors[i])

    minx = Decimal(bounds[0]) / scale_factor
    miny = Decimal(bounds[1]) / scale_factor
    maxx = Decimal(bounds[2]) / scale_factor
    maxy = Decimal(bounds[3]) / scale_factor

    width = maxx - minx
    height = maxy - miny

    square_x = minx if width >= height else minx - (side_length - width)
    square_y = miny if height >= width else miny - (side_length - height)
    bounding_square = Rectangle(
        (float(square_x), float(square_y)),
        float(side_length),
        float(side_length),
        fill=False,
        edgecolor="red",
        linewidth=2,
        linestyle="--",
    )
    ax.add_patch(bounding_square)

    padding = 0.5
    ax.set_xlim(
        float(square_x - Decimal(str(padding))),
        float(square_x + side_length + Decimal(str(padding))),
    )
    ax.set_ylim(
        float(square_y - Decimal(str(padding))),
        float(square_y + side_length + Decimal(str(padding))),
    )
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    plt.title(f"{num_trees} Trees: {side_length:.12f}")
    plt.show()
    plt.close()


tree_data = []
current_placed_trees = []

for n in range(200):
    # Pass the current_placed_trees to initialize_trees
    current_placed_trees, side = initialize_trees(
        n + 1, existing_trees=current_placed_trees
    )
    if (n + 1) % 10 == 0:
        plot_results(side, current_placed_trees, n + 1)
    for tree in current_placed_trees:
        tree_data.append([tree.center_x, tree.center_y, tree.angle])

cols = ["x", "y", "deg"]
submission = pd.DataFrame(index=index, columns=cols, data=tree_data).rename_axis("id")

for col in cols:
    submission[col] = submission[col].astype(float).round(decimals=6)

# To ensure everything is kept as a string, prepend an 's'
for col in submission.columns:
    submission[col] = "s" + submission[col].astype("string")
submission.to_csv("sample_submission.csv")
