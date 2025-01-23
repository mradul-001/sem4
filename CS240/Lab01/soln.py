import numpy as np
from collections import deque
import heapq
from typing import List, Tuple, Set, Dict

"""
Do not import any other package unless allowed by te TAs in charge of the lab.
Do not change the name of any of the functions below.
"""


# Helper class


# class that tracks each state, the parent of that state, the move that led to that state, the depth in search tree and the cost (f = g + h)
class BoardState:

    def __init__(self, stateArray, parent, move, depth, cost):
        self.stateArray = stateArray
        self.parent = parent
        self.move = move
        self.depth = depth
        self.cost = cost

    def __lt__(self, other):
        return self.cost < other.cost


# Helper functions


# function to shift the blank, given the "move"
def movement(stateArray, move, blank):
    newBoard = stateArray[:]
    newBlank = blank + moves[move]
    newBoard[blank], newBoard[newBlank] = (
        newBoard[newBlank],
        newBoard[blank],
    )
    return newBoard


# manhattan heuristic
def manhattan(stateArray, final):
    manhattanDistance = 0
    for i in range(9):
        if stateArray[i] != 0:
            a, b = divmod(i, 3)
            c, d = divmod(np.where(final == stateArray[i])[0][0], 3)
            manhattanDistance += abs(a - c) + abs(b - d)
    return manhattanDistance


# displaced tiles heuristic
def displacedTiles(state, final):
    tileCount = 0
    for i in range(9):
        if state[i] != 0:
            a, b = divmod(i, 3)
            c, d = divmod(np.where(final == state[i])[0][0], 3)
            if a != c or b != d:
                tileCount += 1
    return tileCount


# different possibly allowed movements
moves = {"U": -3, "D": 3, "L": -1, "R": 1}


def bfs(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int]:
    """
    Implement Breadth-First Search algorithm to solve 8-puzzle problem.

    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array.
                            Example: np.array([[1, 2, 3], [4, 0, 5], [6, 7, 8]])
                            where 0 represents the blank space
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array.
                          Example: np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

    Returns:
        Tuple[List[str], int]: A tuple containing:
            - List of moves to reach the goal state. Each move is represented as
              'U' (up), 'D' (down), 'L' (left), or 'R' (right), indicating how
              the blank space should move
            - Number of nodes expanded during the search

    Example return value:
        (['R', 'D', 'R'], 12) # Means blank moved right, down, right; 12 nodes were expanded

    """

    # reshaping initial and goal states as 1d lists
    initial = list(initial.reshape(1, 9)[0])
    goalState = list(goal.reshape(1, 9)[0])

    openList = []  # open list
    closeList = set()  # cloased list
    nodeCount = 0

    # pushing the initial state on the open list
    heapq.heappush(openList, BoardState(initial, None, None, 0, 0))

    # checking till the open list is empty
    while openList:

        currentState = heapq.heappop(openList)
        nodeCount += 1

        if currentState.stateArray == goalState:
            # backtracking to find the optimal path
            movesTaken = []
            while currentState.move is not None:
                movesTaken.insert(0, currentState.move)
                currentState = currentState.parent
            return (movesTaken, nodeCount)

        # add the state to close list
        closeList.add(tuple(currentState.stateArray))
        posBlank = currentState.stateArray.index(0)

        # exploring all the possible moves and put the states in open list
        for move in moves:
            if move == "U" and posBlank < 3:
                continue
            if move == "D" and posBlank > 5:
                continue
            if move == "L" and posBlank % 3 == 0:
                continue
            if move == "R" and posBlank % 3 == 2:
                continue

            # make the movement
            newBoard = movement(currentState.stateArray, move, posBlank)

            if tuple(newBoard) in closeList:
                continue

            # add the state in open list
            newState = BoardState(
                newBoard,
                currentState,
                move,
                currentState.depth + 1,
                currentState.depth + 1,
            )
            heapq.heappush(openList, newState)

    return ([], 0)


def dfs(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int]:
    """
    Implement Depth-First Search algorithm to solve 8-puzzle problem.

    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array

    Returns:
        Tuple[List[str], int]: A tuple containing:
            - List of moves to reach the goal state
            - Number of nodes expanded during the search
    """

    # reshaping initial and goal states as 1d lists
    initial = list(initial.reshape(1, 9)[0])
    goalState = list(goal.reshape(1, 9)[0])

    openList = []
    closeList = set()
    nodeCount = 0

    # pushing the initial state on the open list
    heapq.heappush(openList, BoardState(initial, None, None, 0, float("inf")))

    while openList:

        currentState = heapq.heappop(openList)
        nodeCount += 1

        if currentState.stateArray == goalState:
            # backtracking to find the optimal part
            movesTaken = []
            while currentState.move is not None:
                movesTaken.insert(0, currentState.move)
                currentState = currentState.parent
            return (movesTaken, nodeCount)

        closeList.add(tuple(currentState.stateArray))

        posBlank = currentState.stateArray.index(0)

        # explore all the moves
        for move in moves:
            if move == "U" and posBlank < 3:
                continue
            if move == "D" and posBlank > 5:
                continue
            if move == "L" and posBlank % 3 == 0:
                continue
            if move == "R" and posBlank % 3 == 2:
                continue

            newBoard = movement(currentState.stateArray, move, posBlank)

            if tuple(newBoard) in closeList:
                continue

            newState = BoardState(
                newBoard,
                currentState,
                move,
                currentState.depth + 1,
                1 / (currentState.depth + 1),
            )
            heapq.heappush(openList, newState)

    return ([], 0)


def dijkstra(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int, int]:
    """
    Implement Dijkstra's algorithm to solve 8-puzzle problem.

    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array

    Returns:
        Tuple[List[str], int, int]: A tuple containing:
            - List of moves to reach the goal state
            - Number of nodes expanded during the search
            - Total cost of the path for transforming initial into goal configuration

    """

    # reshaping initial and goal states as 1d lists
    initial = list(initial.reshape(1, 9)[0])
    goalState = list(goal.reshape(1, 9)[0])

    openList = []
    closeList = set()

    nodeCount = 0

    heapq.heappush(openList, BoardState(initial, None, None, 0, 0))

    while openList:

        currentState = heapq.heappop(openList)
        nodeCount += 1

        if currentState.stateArray == goalState:
            # backtracking to find the optimal part
            movesTaken = []
            while currentState.move is not None:
                movesTaken.insert(0, currentState.move)
                currentState = currentState.parent
            return (movesTaken, nodeCount, len(movesTaken))

        closeList.add(tuple(currentState.stateArray))
        posBlank = currentState.stateArray.index(0)

        # exploring all the moves
        for move in moves:
            if move == "U" and posBlank < 3:
                continue
            if move == "D" and posBlank > 5:
                continue
            if move == "L" and posBlank % 3 == 0:
                continue
            if move == "R" and posBlank % 3 == 2:
                continue

            newBoard = movement(currentState.stateArray, move, posBlank)

            if tuple(newBoard) in closeList:
                continue

            newState = BoardState(
                newBoard,
                currentState,
                move,
                currentState.depth + 1,
                currentState.depth + 1,
            )
            heapq.heappush(openList, newState)

    return ([], 0, 0)


def astar_dt(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int, int]:
    """
    Implement A* Search with Displaced Tiles heuristic to solve 8-puzzle problem.

    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array

    Returns:
        Tuple[List[str], int, int]: A tuple containing:
            - List of moves to reach the goal state
            - Number of nodes expanded during the search
            - Total cost of the path for transforming initial into goal configuration


    """

    # reshaping initial and goal states as 1d lists
    initial = list(initial.reshape(1, 9)[0])
    goalState = list(goal.reshape(1, 9)[0])

    # TODO: Implement this function
    openList = []
    closeList = set()
    nodeCount = 0

    heapq.heappush(
        openList, BoardState(initial, None, None, 0, displacedTiles(initial, goal))
    )

    while openList:

        currentState = heapq.heappop(openList)
        nodeCount += 1

        if currentState.stateArray == goalState:
            # backtracking to find the optimal part
            movesTaken = []
            while currentState.move is not None:
                movesTaken.insert(0, currentState.move)
                currentState = currentState.parent
            return (movesTaken, nodeCount, len(movesTaken))

        closeList.add(tuple(currentState.stateArray))
        posBlank = currentState.stateArray.index(0)

        # exploring all the moves
        for move in moves:
            if move == "U" and posBlank < 3:
                continue
            if move == "D" and posBlank > 5:
                continue
            if move == "L" and posBlank % 3 == 0:
                continue
            if move == "R" and posBlank % 3 == 2:
                continue

            newBoard = movement(currentState.stateArray, move, posBlank)

            if tuple(newBoard) in closeList:
                continue

            newState = BoardState(
                newBoard,
                currentState,
                move,
                currentState.depth + 1,
                currentState.depth + 1 + displacedTiles(newBoard, goal),
            )
            heapq.heappush(openList, newState)

    return ([], 0, 0)


def astar_md(initial: np.ndarray, goal: np.ndarray) -> Tuple[List[str], int, int]:
    """
    Implement A* Search with Manhattan Distance heuristic to solve 8-puzzle problem.

    Args:
        initial (np.ndarray): Initial state of the puzzle as a 3x3 numpy array
        goal (np.ndarray): Goal state of the puzzle as a 3x3 numpy array

    Returns:
        Tuple[List[str], int, int]: A tuple containing:
            - List of moves to reach the goal state
            - Number of nodes expanded during the search
            - Total cost of the path for transforming initial into goal configuration
    """

    # reshaping initial and goal states as 1d lists
    initial = list(initial.reshape(1, 9)[0])
    goalState = list(goal.reshape(1, 9)[0])

    # TODO: Implement this function
    openList = []
    closeList = set()

    nodeCount = 0

    heapq.heappush(
        openList, BoardState(initial, None, None, 0, manhattan(initial, goal))
    )

    while openList:

        currentState = heapq.heappop(openList)
        nodeCount += 1

        if currentState.stateArray == goalState:
            # backtracking to find the optimal part
            movesTaken = []
            while currentState.move is not None:
                movesTaken.insert(0, currentState.move)
                currentState = currentState.parent
            return (movesTaken, nodeCount, len(movesTaken))

        closeList.add(tuple(currentState.stateArray))

        posBlank = currentState.stateArray.index(0)

        for move in moves:
            if move == "U" and posBlank < 3:
                continue
            if move == "D" and posBlank > 5:
                continue
            if move == "L" and posBlank % 3 == 0:
                continue
            if move == "R" and posBlank % 3 == 2:
                continue

            newBoard = movement(currentState.stateArray, move, posBlank)

            if tuple(newBoard) in closeList:
                continue

            newState = BoardState(
                newBoard,
                currentState,
                move,
                currentState.depth + 1,
                currentState.depth + 1 + manhattan(newBoard, goal),
            )
            heapq.heappush(openList, newState)

    return ([], 0, 0)


# Example test case to help verify your implementation
if __name__ == "__main__":

    # Example puzzle configuration
    initial_state = np.array([[1, 2, 3], [4, 0, 5], [6, 7, 8]])
    goalState = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 0]])

    # Test each algorithm
    # print("Testing BFS...")
    bfs_moves, bfs_expanded = bfs(initial_state, goalState)
    # print(f"BFS Solution: {bfs_moves}")
    # print(f"Nodes expanded: {bfs_expanded}")

    # print("\nTesting DFS...")
    dfs_moves, dfs_expanded = dfs(initial_state, goalState)
    # print(f"DFS Solution: {dfs_moves}")
    # print(f"Nodes expanded: {dfs_expanded}")

    # print("\nTesting Dijkstra...")
    dijkstra_moves, dijkstra_expanded, dijkstra_cost = dijkstra(
        initial_state, goalState
    )
    # print(f"Dijkstra Solution: {dijkstra_moves}")
    # print(f"Nodes expanded: {dijkstra_expanded}")
    # print(f"Total cost: {dijkstra_cost}")

    # print("\nTesting A* with Displaced Tiles...")
    dt_moves, dt_expanded, dt_fscore = astar_dt(initial_state, goalState)
    # print(f"A* (DT) Solution: {dt_moves}")
    # print(f"Nodes expanded: {dt_expanded}")
    # print(f"Total cost: {dt_fscore}")

    # print("\nTesting A* with Manhattan Distance...")
    md_moves, md_expanded, md_fscore = astar_md(initial_state, goalState)
    # print(f"A* (MD) Solution: {md_moves}")
    # print(f"Nodes expanded: {md_expanded}")
    # print(f"Total cost: {md_fscore}")
