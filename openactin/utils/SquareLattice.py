import numpy as np
from typing import Dict, Tuple

class SquareLattice:
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
        self.size: int = size
        self.tiles: Dict[Tuple[int, int], str] = {}
        for row in range(-size, size+1):
            for col in range(-size, size+1):
                #color = (int(2*((col % 2)-0.5))*row%2+col)%4
                color = (col%2 - 2*(row%2)) % 4
                #color = ((2 * (col % 2) - 1) * row + col) % 4
                self.tiles[(row, col)] = str(color)  # Assign colors in a repeating pattern of 4

    def coords(self) -> np.ndarray:
        """
        Calculate and return the 2D coordinates of the square lattice tiles.

        Returns
        -------
        np.ndarray
            Array of 2D coordinates for each tile in the grid.
        """
        tiles = np.array([a for a in self.tiles.keys()])
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
    print("Demo: SquareLattice")
    lattice = SquareLattice(3)
    print("Coordinates:")
    print(lattice.coords())
    print("Grid visualization:")
    lattice.show()
