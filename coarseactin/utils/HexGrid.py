import numpy as np
class HexGrid():
    deltas = [[1, 0, -1], [0, 1, -1], [-1, 1, 0], [-1, 0, 1], [0, -1, 1], [1, -1, 0]]
    a0 = 0
    a1 = np.pi / 3
    a2 = -np.pi / 3
    vecs = np.array([[np.sqrt(3) * np.cos(a0), np.sin(a0) / np.sqrt(3)],
                     [np.sqrt(3) * np.cos(a1), np.sin(a1) / np.sqrt(3)],
                     [np.sqrt(3) * np.cos(a2), np.sin(a2) / np.sqrt(3)]])

    def __init__(self, radius):
        self.radius = radius
        self.tiles = {(0, 0, 0): "X"}
        for r in range(radius):
            a = 0
            b = -r
            c = +r
            for j in range(6):
                num_of_hexas_in_edge = r
                for i in range(num_of_hexas_in_edge):
                    a = a + self.deltas[j][0]
                    b = b + self.deltas[j][1]
                    c = c + self.deltas[j][2]
                    self.tiles[a, b, c] = "X"

    def coords(self):
        tiles = np.array([a for a in self.tiles.keys()])
        coords = np.dot(tiles, self.vecs)
        return coords

    def show(self):
        l = []
        for y in range(20):
            l.append([])
            for x in range(60):
                l[y].append(".")
        for (a, b, c), tile in self.tiles.items():
            l[self.radius - 1 - b][a - c + (2 * (self.radius - 1))] = self.tiles[a, b, c]
        mapString = ""
        for y in range(len(l)):
            for x in range(len(l[y])):
                mapString += l[y][x]
            mapString += "\n"
        print(mapString)