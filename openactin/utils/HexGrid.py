import numpy as np

class HexGrid():
    # Cube coordinate deltas for moving to each of the 6 neighboring hexes
    deltas = [[1, 0, -1], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1], [1, -1, 0]]

    # Angles for generating basis vectors for the hex grid (used for visualization or conversion)
    a0 = 0
    a1 = np.pi / 3
    a2 = -np.pi / 3

    # Basis vectors for the hex grid in 2D space
    vecs = np.array([[np.sqrt(3) * np.cos(a0), np.sin(a0) / np.sqrt(3)],
                     [np.sqrt(3) * np.cos(a1), np.sin(a1) / np.sqrt(3)],
                     [np.sqrt(3) * np.cos(a2), np.sin(a2) / np.sqrt(3)]])
    
    def __init__(self, radius):
        self.radius = radius
        self.tiles = {(0, 0, 0): "0"}
        for r in range(radius):
            a = 0
            b = -r
            c = +r
            for j in range(6):
                num_of_hexagons_in_edge = r
                for i in range(num_of_hexagons_in_edge):
                    a = a + self.deltas[j][0]
                    b = b + self.deltas[j][1]
                    c = c + self.deltas[j][2]
                    print(a,b,c)
                    self.tiles[a, b, c] = str((a-c)%3)

    def coords(self):
        tiles = np.array([a for a in self.tiles.keys()])
        coords = np.dot(tiles, self.vecs)
        return coords

    def colors(self):
        colors = np.array([self.tiles[a] for a in self.tiles.keys()])
        return colors

    def show(self):
        # Compute row and column indices for all tiles
        row_indices = [self.radius - 1 - b for (a, b, c) in self.tiles.keys()]
        col_indices = [a - c + (2 * (self.radius - 1)) for (a, b, c) in self.tiles.keys()]
        min_row, max_row = min(row_indices), max(row_indices)
        min_col, max_col = min(col_indices), max(col_indices)

        # Initialize grid with correct size
        l = []
        for y in range(max_row - min_row + 1):
            l.append(["."] * (max_col - min_col + 1))

        # Place tiles using offset
        for (a, b, c), tile in self.tiles.items():
            row = self.radius - 1 - b - min_row
            col = a - c + (2 * (self.radius - 1)) - min_col
            l[row][col] = self.tiles[a, b, c]

        mapString = ""
        for y in range(len(l)):
            for x in range(len(l[y])):
                mapString += l[y][x]
            mapString += "\n"
        print(mapString)

if __name__ == "__main__":
    print("Demo: HexGrid")
    
    grid = HexGrid(3)
    print("Coordinates:")
    print(grid.coords())
    
    for layers in range(1, 5):
        grid = HexGrid(layers)
        print("Grid visualization:")
        grid.show()