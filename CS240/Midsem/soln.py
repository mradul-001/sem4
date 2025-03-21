import heapq

# ---------------------------------------------------------------------------
# Summary of the solution:
# 
#     - Two heuristics proposed:
#         - One is the position of tthe first queen on the y-axis
#         - Other is the maximum absolute differene between the y-positions of 
#           the queens places till now
# 
#     - Cost of each insertion is taken as 1. So g() value of any node can 
#       be written as:
#         - g(child) = g(parent) + 1
#     
#     - h0: Baseline heuristic
#     - h1: Position of first queen heuristic
#     - h2: Max abs differene between position of queens placed till now
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# State is defined in the following way:
    # arr       : the array consisting of Y-coordinates of the 8 queens in order/
    # parent    : the parent state of the current sttate in the searh tree
    # move      : the move which led to the current state
    # cost      : the cost (or the f-value) for the current node
    # depth     : depth of the current node in the search tree

class State:

    def __init__(self, arr, parent, cost, depth):
        self.arr     = arr
        self.parent  = parent
        self.cost    = cost
        self.depth   = depth
    
    def __lt__(self, other):
        return self.cost < other.cost
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Operation: function to fill a new row in the board
def fillRow(arr, num, lastFilledRow):

    # Make the copy of the array    
    temp = arr[:]
    temp[lastFilledRow + 1] = num
    
    # If it is the first insertion, let it happen
    if lastFilledRow == -1:
        return True, temp

    # CHeck if the insertion is valid
    for i in range(lastFilledRow + 1):
        if (temp[i] == num):
            return False, arr
        if abs((temp[i] - num) / (i - (lastFilledRow + 1))) == 1:
            return False, arr
    
    return True, temp
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Heuristic 1: The column number of 1st queen
# It is a monotone heuristic, since the value of heuristic for children 
# is just parent's heuristic value + 1 (increment only because of cost of insertion)
def posQueen(arr):
    return arr[0]
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Heuristic 2: Maximum diffferene between the y positions of tthe queens
# This heuristic is monotone since the maximum differene in the new board is 
# guarenteed to be at least as big as it was in parent node 
def prevDiff(arr, lastFilledRow):
    maxdiff = -1
    for i in range(lastFilledRow):
        maxdiff = max(maxdiff, abs(arr[i] - arr[i+1]))
    return maxdiff
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def a_star(start, heuristic):

    openlist  = []          # priority queue
    closelist = set()       # set
    heapq.heappush(openlist, State(start, None, 0, 0))
    nodeExpanded = 0

    while openlist:
        
        # Node expanded here
        currstate = heapq.heappop(openlist)
        nodeExpanded += 1

        # Find in which row to insert next queen
        lastFilledRow = -1
        allpositive = True

        for i in currstate.arr:
            if i < 0:
                lastFilledRow = currstate.arr.index(i) - 1
                allpositive = False

        # If all rows are filled, return the valid array
        if allpositive:
            return currstate.arr, nodeExpanded
        
        # Add the current node to close list
        closelist.add(tuple(currstate.arr))

        # Get the new board after valid insertions and put them in open list
        for i in range(8):
            
            # NOTE: Cost of inserting a new queen is taken as 1, heuristics are defined above as functions
            status, newvalidarr = fillRow(currstate.arr, i, lastFilledRow)

            if status:
    
                if tuple(newvalidarr) in closelist:
                    continue

                if heuristic == "TD":
                    newstate = State(
                                    newvalidarr, 
                                    currstate, 
                                    currstate.depth + 1 + posQueen(newvalidarr),
                                    currstate.depth + 1     # g = depth
                                )
                elif heuristic == "DF":
                    newstate = State(
                                    newvalidarr, 
                                    currstate, 
                                    currstate.depth + 1 + prevDiff(newvalidarr, lastFilledRow),
                                    currstate.depth + 1     # g = depth
                                )                    
                else:
                    newstate = State(
                                    newvalidarr, 
                                    currstate, 
                                    currstate.depth + 1,
                                    currstate.depth + 1     # g = depth
                                )

                # Insert the new node in open list
                heapq.heappush(openlist, newstate)

    return [], nodeExpanded
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
def test():

    startArr = [-1] * 8     # Do not insert anything att start
    finalArr, _ = a_star(startArr, "baseline")
    print()
    print("- One valid configuration of the chess board is: ")
    print(" ".ljust(2) + "---------------")
    for i in finalArr:
        print(" ".ljust(2), end="")
        for j in range(8):
            if j == i:
                print("o", end = " ")
            else:
                print("x", end = " ")
        print()
    print(" ".ljust(2) + "---------------")
    print("- Note: 'o' represent the queen.")
    print()
    print()
    print("----------------- Results -----------------")
    print(" - Number of nodes expanded with h0:", a_star(startArr,"baseline")[1])
    print(" - Number of nodes expanded with h1:", a_star(startArr, "TD")[1])
    print(" - Number of nodes expanded with h2:", a_star(startArr, "DF")[1])
    print()
    
    return

# ---------------------------------------------------------------------------

test()