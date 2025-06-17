import numpy as np
from typing import Dict, Tuple, Any

class HexGrid:
    """
    A class to represent a hexagonal grid using cube coordinates.

    Attributes
    ----------
    radius : int
        The radius of the hexagonal grid (number of layers around the center).
    tiles : dict
        Dictionary storing cube coordinates as keys and tile identifiers as values.
    deltas : list
        Cube coordinate deltas for moving to each of the 6 neighboring hexes.
    vecs : np.ndarray
        Basis vectors for the hex grid in 2D space.
    """
    deltas = [[1, 0, -1], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1], [1, -1, 0]]
    a0 = 0
    a1 = np.pi / 3
    a2 = -np.pi / 3
    vecs = np.array([
        [np.sqrt(3) * np.cos(a0), np.sin(a0) / np.sqrt(3)],
        [np.sqrt(3) * np.cos(a1), np.sin(a1) / np.sqrt(3)],
        [np.sqrt(3) * np.cos(a2), np.sin(a2) / np.sqrt(3)]
    ])

    def __init__(self, radius: int) -> None:
        """
        Initialize a hexagonal grid with a specified radius.

        Parameters
        ----------
        radius : int
            The radius of the hexagonal grid, determining the number of hexagonal layers around the center.
        """
        self.radius: int = radius
        self.tiles: Dict[Tuple[int, int, int], str] = {(0, 0, 0): "0"}
        for r in range(radius):
            a = 0
            b = -r
            c = +r
            for j in range(6):
                num_of_hexagons_in_edge = r
                for _ in range(num_of_hexagons_in_edge):
                    a = a + self.deltas[j][0]
                    b = b + self.deltas[j][1]
                    c = c + self.deltas[j][2]
                    # Assign a color or identifier to each tile based on its coordinates
                    self.tiles[(a, b, c)] = str((a - c) % 3)

    def coords(self) -> np.ndarray:
        """
        Calculate and return the 2D coordinates of the hexagonal grid tiles.

        Returns
        -------
        np.ndarray
            Array of 2D coordinates for each tile in the grid.
        """
        tiles = np.array([a for a in self.tiles.keys()])
        coords = np.dot(tiles, self.vecs)
        return coords

    def rotations(self) -> np.ndarray:
        """
        Retrieve the rotation associated with each tile in the hexagonal grid.

        Returns
        -------
        np.ndarray
            Array of rotation values for each tile.
        """
        colors = np.array([self.tiles[a] for a in self.tiles.keys()])
        return colors * 2 * np.pi / 3 

    def show(self) -> None:
        """
        Display a visual representation of the hexagonal grid in the console.

        This method prints the grid layout to the console, representing each tile with its associated identifier.
        The grid is displayed as a 2D array of characters, where each tile is represented by 0, 1, or 2 based on its rotation (color).
        The grid is padded with dots (.) to represent empty spaces outside the hexagonal structure.
        """
        # Compute row and column indices for all tiles
        row_indices = [self.radius - 1 - b for (a, b, c) in self.tiles.keys()]
        col_indices = [a - c + (2 * (self.radius - 1)) for (a, b, c) in self.tiles.keys()]
        min_row, max_row = min(row_indices), max(row_indices)
        min_col, max_col = min(col_indices), max(col_indices)

        # Initialize grid with correct size
        grid: list[list[str]] = []
        for _ in range(max_row - min_row + 1):
            grid.append(["."] * (max_col - min_col + 1))

        # Place tiles using offset
        for (a, b, c), tile in self.tiles.items():
            row = self.radius - 1 - b - min_row
            col = a - c + (2 * (self.radius - 1)) - min_col
            grid[row][col] = tile

        map_string = ""
        for row in grid:
            map_string += "".join(row) + "\n"
        print(map_string)

if __name__ == "__main__":
    print("Demo: HexGrid")
    grid = HexGrid(3)
    print("Coordinates:")
    print(grid.coords())
    for layers in range(1, 5):
        grid = HexGrid(layers)
        print("Grid visualization:")
        grid.show()