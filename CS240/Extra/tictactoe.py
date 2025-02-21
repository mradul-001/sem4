import heapq


class State:

    def __init__(self, arr, parent, move, fValue, depth):
        self.arr = arr
        self.parent = parent
        self.move = move
        self.fValue = fValue
        self.depth = depth

    def __lt__(self, other):
        return self.fValue < other.fValue


# Generate moves based on the turn and the current status of the array
def generateMoves(arr, turn):
    moves = []
    for i in range(3):
        for j in range(3):
            if arr[i][j] == " ":
                moves.append([turn, i * 3 + j])
    return moves


def checkDest(arr, player):

    # Check for column wins
    col = any(arr[0][i] == arr[1][i] == arr[2][i] == player for i in range(3))

    # Check for row wins
    row = any(arr[i][0] == arr[i][1] == arr[i][2] == player for i in range(3))

    # Check for diagonal wins
    dia = (arr[0][0] == arr[1][1] == arr[2][2] == player) or (
        arr[0][2] == arr[1][1] == arr[2][0] == player
    )

    return col or row or dia


# Total number of tiles unfilled
def heuristic(arr):
    sum = 0
    for i in range(3):
        for j in range(3):
            if arr[i][j] != " ":
                sum += 1
    return sum


def aStar(iniArr):

    openlist = []
    closelist = set()
    noOfNodes = 0

    heapq.heappush(openlist, State(iniArr, None, None, heuristic(iniArr), 0))

    while openlist:

        # Expanding the node
        currstate = heapq.heappop(openlist)
        noOfNodes += 1

        # If we found the state with computer's victory,
        # return the moves taken to reach that state

        if checkDest(currstate.arr, "O"):
            # Backtrack to find the moves taken
            movesTaken = []
            while currstate.move != None:
                movesTaken.append(currstate.move)
                currstate = currstate.parent
            return movesTaken, noOfNodes

        # Add the node to close list
        closelist.add(tuple(map(tuple, currstate.arr)))

        # Check whether to move "X" or "O"
        if currstate.move is None or currstate.move[0] == "O":
            allMoves = generateMoves(currstate.arr, "X")
        else:
            allMoves = generateMoves(currstate.arr, "O")

        # Get the neighbours of the current state
        for move in allMoves:
            newarr = [row[:] for row in currstate.arr]
            xPos, yPos = divmod(move[1], 3)
            newarr[xPos][yPos] = move[0]

            if tuple(map(tuple, newarr)) in closelist:
                continue
            else:
                newstate = State(
                    newarr,
                    currstate,
                    move,
                    currstate.depth + 1 + heuristic(newarr),
                    currstate.depth + 1,
                )
            heapq.heappush(openlist, newstate)

    return [], noOfNodes



# Author: ChatGPT
def printGrid(arr):
    for i, row in enumerate(arr):
        print(" | ".join(row))
        if i < 2:
            print("-" * 9)



# Author: ChatGPT
def test():

    iniArr = [[" ", "X", " "], [" ", " ", " "], [" ", "O", " "]]
    movesTaken, noOfNodes = aStar(iniArr=iniArr)

    if movesTaken == []:
        print()
        print("Victory not possible!")
        print()
        return

    print()
    print("Initial Grid:")
    printGrid(iniArr)
    print()

    for move in reversed(movesTaken):
        value, pos = move
        xPos, yPos = divmod(pos, 3)
        iniArr[xPos][yPos] = value
        print(f"Move: {value} at ({xPos}, {yPos})")
        print("------------------")
        printGrid(iniArr)
        print()


    print("Final Winning Grid:")
    print("------------------")
    printGrid(iniArr)
    print()
    print("Number of nodes expanded:", noOfNodes)
    print()
    return

test()
