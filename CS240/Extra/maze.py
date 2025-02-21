import numpy as np
import heapq


class Maze:
    def __init__(self, arr, parent, move, cost, depth):
        self.arr = arr
        self.parent = parent
        self.move = move
        self.cost = cost
        self.depth = depth

    def __lt__(self, other):
        return self.cost < other.cost


def heuristic(arr, destArr):
    curr = -1
    dest = -1
    for i in range(len(arr)):
        if arr[i] == "*":
            curr = i
        if destArr[i] == "*":
            dest = i
    cX = curr // 5
    cY = curr % 5
    dX = dest // 5
    dY = dest % 5

    return abs(cX - dX) + abs(cY - dY)


def movement(arr, move):
    newarr = [ele for ele in arr]
    pos = arr.index("*")
    if pos < 5 and move == -5:
        return False, arr
    if pos >= 20 and move == 5:
        return False, arr
    if pos % 5 == 0 and move == -1:
        return False, arr
    if pos % 5 == 4 and move == 1:
        return False, arr
    newpos = pos + move
    if arr[newpos] == "1":
        return False, arr
    else:
        newarr[pos], newarr[newpos] = newarr[newpos], newarr[pos]
    return True, newarr


def a_star(initial, final):

    moves = {5, -5, 1, -1}

    initialarr = [item for sublist in initial for item in sublist]
    finalarr = [item for sublist in final for item in sublist]

    openlist = []
    closelist = set()

    heapq.heappush(openlist, Maze(initialarr, None, None, 0, 0))

    while openlist:
        
        currMaze = heapq.heappop(openlist)
        
        if currMaze.arr == finalarr:
            # backtrack
            movesTaken = []
            arrays = []
            while currMaze.parent != None:
                movesTaken.append(currMaze.move)
                arrays.append(currMaze.arr)
                currMaze = currMaze.parent
            return movesTaken, arrays
        
        closelist.add(tuple(currMaze.arr))

        for move in moves:
            
            status, newarr = movement(currMaze.arr, move)
            
            if status:
                if tuple(newarr) in closelist:
                    continue
                else:
                    newstate = Maze(
                        newarr,
                        currMaze,
                        move,
                        currMaze.depth + heuristic(newarr, finalarr) + 1,
                        currMaze.depth + 1,
                    )
                    heapq.heappush(openlist, newstate)
    return [], []


init = [
    ["*", "0", "1", "0", "0"],
    ["1", "0", "1", "0", "1"],
    ["0", "0", "0", "0", "1"],
    ["1", "1", "1", "1", "1"],
    ["0", "0", "0", "1", "0"],
]

fina = [
    ["0", "0", "1", "0", "0"],
    ["1", "0", "1", "0", "1"],
    ["0", "0", "0", "0", "1"],
    ["1", "1", "1", "1", "1"],
    ["0", "0", "0", "1", "*"],
]

movesTaken, arrays = a_star(init, fina)

for array in arrays[::-1]:
    rows, cols = 5, 5
    matrix = [array[i * cols : (i + 1) * cols] for i in range(rows)]
    
    for row in matrix:
        print(" ".join(row))  # Print each row neatly
    print("\n" + "-" * 10 + "\n")  # Add separator for readability
