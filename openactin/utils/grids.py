import numpy as np
from typing import Dict, Tuple, Any, Optional, Union
from abc import ABC, abstractmethod

class BaseGrid(ABC):
    """
    Abstract base class for grid structures.

    Attributes
    ----------
    tiles : dict
        Dictionary storing coordinates as keys and tile identifiers as values.
    """
    def __init__(self) -> None:
        self.tiles: Dict[Any, str] = {}

    @abstractmethod
    def coords(self) -> np.ndarray:
        """
        Calculate and return the 2D coordinates of the grid tiles.

        Returns
        -------
        np.ndarray
            Array of 2D coordinates for each tile in the grid.
        """
        pass

    @abstractmethod
    def colors(self) -> np.ndarray:
        """
        Retrieve the color or identifier associated with each tile in the grid.

        Returns
        -------
        np.ndarray
            Array of color identifiers for each tile.
        """
        pass

    @abstractmethod
    def show(self) -> None:
        """
        Display a visual representation of the grid in the console.
        """
        pass

    def rotations(self) -> np.ndarray:
        """
        Retrieve the rotation (in radians) associated with each tile in the grid.
        By default, returns zeros (no rotation). Should be overridden by subclasses.

        Returns
        -------
        np.ndarray
            Array of rotation values for each tile.
        """
        return np.zeros(len(self.tiles))

class HexGrid(BaseGrid):
    """
    A class to represent a hexagonal grid using cube coordinates.

    Attributes
    ----------
    radius : int
        The radius of the hexagonal grid (number of layers around the center).
    tiles : dict
        Dictionary storing cube coordinates as keys and tile identifiers as values.
    deltas : list[list[int]]
        Cube coordinate deltas for moving to each of the 6 neighboring hexes.
    vecs : np.ndarray
        Basis vectors for the hex grid in 2D space.
    """
    deltas: list[list[int]] = [[1, 0, -1], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1], [1, -1, 0]]
    a0: float = 0
    a1: float = np.pi / 3
    a2: float = -np.pi / 3
    vecs: np.ndarray = np.array([
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
        super().__init__()
        self.radius: int = radius
        self.tiles: Dict[Tuple[int, int, int], str] = {(0, 0, 0): "0"}
        for r in range(radius+1):
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
        tiles = np.array(list(self.tiles.keys()))
        coords = np.dot(tiles, self.vecs)
        return coords

    def colors(self) -> np.ndarray:
        """
        Retrieve the color or identifier associated with each tile in the hexagonal grid.

        Returns
        -------
        np.ndarray
            Array of color identifiers for each tile.
        """
        colors = np.array([self.tiles[a] for a in self.tiles.keys()])
        return colors

    def rotations(self) -> np.ndarray:
        """
        Retrieve the rotation (in radians) associated with each tile in the hexagonal grid.

        Returns
        -------
        np.ndarray
            Array of rotation values for each tile.
        """
        colors = np.array([int(self.tiles[a]) for a in self.tiles.keys()])
        return colors * 2 * np.pi / 3

    def show(self) -> None:
        """
        Display a visual representation of the hexagonal grid in the console.

        This method prints the grid layout to the console, representing each tile with its associated identifier.
        The grid is displayed as a 2D array of characters, where each tile is represented by 0, 1, or 2 based on its rotation (color).
        The grid is padded with dots (.) to represent empty spaces outside the hexagonal structure.
        """
        row_indices = [self.radius - 1 - b for (a, b, c) in self.tiles.keys()]
        col_indices = [a - c + (2 * (self.radius - 1)) for (a, b, c) in self.tiles.keys()]
        min_row, max_row = min(row_indices), max(row_indices)
        min_col, max_col = min(col_indices), max(col_indices)
        grid: list[list[str]] = []
        for _ in range(max_row - min_row + 1):
            grid.append(["."] * (max_col - min_col + 1))
        for (a, b, c), tile in self.tiles.items():
            row = self.radius - 1 - b - min_row
            col = a - c + (2 * (self.radius - 1)) - min_col
            grid[row][col] = tile
        map_string = ""
        for row in grid:
            map_string += "".join(row) + "\n"
        print(map_string)

class SquareGrid(BaseGrid):
    """
    A class to represent a square lattice grid.

    Attributes
    ----------
    size : int
        The length of one side of the square lattice (grid is size x size).
    tiles : dict
        Dictionary storing (row, col) coordinates as keys and tile identifiers as values.
    """
    def __init__(self, size: int) -> None:
        """
        Initialize a square lattice grid with a specified size and assign 4 colors in a repeating pattern.

        Parameters
        ----------
        size : int
            The length of one side of the square lattice (grid is size x size).
        """
        super().__init__()
        self.size: int = size
        self.tiles: Dict[Tuple[int, int], str] = {}
        for row in range(-size, size+1):
            for col in range(-size, size+1):
                # Assign colors in a repeating pattern of 4
                color = (col%2 - 2*(row%2)) % 4
                self.tiles[(row, col)] = str(color)

    def coords(self) -> np.ndarray:
        """
        Calculate and return the 2D coordinates of the square lattice tiles.

        Returns
        -------
        np.ndarray
            Array of 2D coordinates for each tile in the grid.
        """
        tiles = np.array(list(self.tiles.keys()))
        return tiles

    def colors(self) -> np.ndarray:
        """
        Retrieve the color or identifier associated with each tile in the square lattice.

        Returns
        -------
        np.ndarray
            Array of color identifiers for each tile.
        """
        colors = np.array([self.tiles[a] for a in self.tiles.keys()])
        return colors

    def rotations(self) -> np.ndarray:
        """
        Retrieve the rotation (in radians) associated with each tile in the square lattice.
        There are 4 types of rotations (0, 1, 2, 3) corresponding to multiples of 2π/4.

        Returns
        -------
        np.ndarray
            Array of rotation values for each tile.
        """
        colors = np.array([int(self.tiles[a]) for a in self.tiles.keys()])
        return colors * 2 * np.pi / 4

    def show(self) -> None:
        """
        Display a visual representation of the square lattice in the console.

        This method prints the grid layout to the console, representing each tile with its associated identifier.
        The grid is displayed as a 2D array of characters.
        """
        grid = [[self.tiles[(row, col)] for col in range(-self.size, self.size+1)] for row in range(-self.size, self.size+1)]
        map_string = ""
        for row in grid:
            map_string += "".join(row) + "\n"
        print(map_string)

if __name__ == "__main__":
    print("Demo: HexGrid")
    grid = HexGrid(1)
    print("Coordinates:")
    print(grid.coords())
    print("Colors:")
    print(grid.colors())
    print("Rotations (radians):")
    print(grid.rotations())
    grid.show()
    print("Demo: SquareGrid")
    lattice = SquareGrid(1)
    print("Coordinates:")
    print(lattice.coords())
    print("Colors:")
    print(lattice.colors())
    print("Rotations (radians):")
    print(lattice.rotations())
    lattice.show()
