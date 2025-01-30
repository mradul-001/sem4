import heapq
import json
from typing import List, Tuple


class State:
    def __init__(self, curValues, pastState, cost, gstar):
        self.curValues = curValues
        self.pastState = pastState
        self.cost = cost
        self.gstar = gstar

    def __lt__(self, other):
        return self.cost < other.cost


def check_valid(
    state: list, max_missionaries: int, max_cannibals: int
) -> bool:  # 10 marks
    """
    Graded
    Check if a state is valid. State format: [m_left, c_left, boat_position].
    """
    cLeft = state[1]
    mLeft = state[0]
    leftValidity = cLeft <= mLeft or mLeft == 0
    rightValidity = (max_cannibals - cLeft) <= (max_missionaries - mLeft) or (
        max_missionaries - mLeft
    ) == 0
    return leftValidity and rightValidity


def get_neighbours(
    state: list, max_missionaries: int, max_cannibals: int
) -> List[list]:  # 10 marks
    """
    Graded
    Generate all valid neighbouring states.
    """
    mLeft = state[0]
    cLeft = state[1]
    bPosn = state[2]

    moves = [
        (1, 0),
        (0, 1),
        (1, 1),
        (2, 0),
        (0, 2),
    ]

    nextStateList = []

    if bPosn == 1:
        for dm, dc in moves:
            newmLeft, newcLeft, newbPosn = mLeft - dm, cLeft - dc, 0
            if (
                0 <= newmLeft <= max_missionaries
                and 0 <= newcLeft <= max_cannibals
                and check_valid(
                    [newmLeft, newcLeft, newbPosn], max_missionaries, max_cannibals
                )
            ):
                nextStateList.append([newmLeft, newcLeft, newbPosn])

    else:
        for dm, dc in moves:
            newmLeft, newcLeft, newbPosn = mLeft + dm, cLeft + dc, 1
            if (
                0 <= newmLeft <= max_missionaries
                and 0 <= newcLeft <= max_cannibals
                and check_valid(
                    [newmLeft, newcLeft, newbPosn], max_missionaries, max_cannibals
                )
            ):
                nextStateList.append([newmLeft, newcLeft, newbPosn])

    return nextStateList


def gstar(state: list, new_state: list) -> int:  # 5 marks
    """
    Graded
    The weight of the edge between state and new_state, this is the number of people on the boat.
    """
    m1, c1, bPos1 = state
    m2, c2, bPos2 = new_state
    return abs(m2 - m1) + abs(c2 - c1)


def h1(state: list) -> int:  # 3 marks
    """
    Graded
    h1 is the number of people on the left bank.
    """
    mLeft, cLeft, bPos = state
    return mLeft + cLeft


def h2(state: list) -> int:  # 3 marks
    """
    Graded
    h2 is the number of missionaries on the left bank.
    """
    mLeft, cLeft, bPos = state
    return mLeft


def h3(state: list) -> int:  # 3 marks
    """
    Graded
    h3 is the number of cannibals on the left bank.
    """
    mLeft, cLeft, bPos = state
    return cLeft


def h4(state: list) -> int:  # 3 marks
    """
    Graded
    Weights of missionaries is higher than cannibals.
    h4 = missionaries_left * 1.5 + cannibals_left
    """
    mLeft, cLeft, bPos = state
    return mLeft * 1.5 + cLeft


def h5(state: list) -> int:  # 3 marks
    """
    Graded
    Weights of missionaries is lower than cannibals.
    h5 = missionaries_left + cannibals_left*1.5
    """
    mLeft, cLeft, bPos = state
    return mLeft + 1.5 * cLeft


def astar_h1(
    init_state: list, final_state: list, max_missionaries: int, max_cannibals: int
) -> Tuple[List[list], bool]:  # 28 marks
    """
    Graded
    Implement A* with h1 heuristic.
    This function must return path obtained and a boolean which says if the heuristic chosen satisfes Monotone restriction property while exploring or not.
    """

    openList = []
    closeList = {}
    monotone = True

    heapq.heappush(openList, State(init_state, None, h1(init_state), 0))

    while len(openList) > 0:

        currentState = heapq.heappop(openList)

        # if (
        #     tuple(currentState.curValues) in closeList
        #     and closeList[tuple(currentState.curValues)] <= currentState.gstar
        # ):
        #     continue

        closeList[tuple(currentState.curValues)] = currentState.gstar

        if currentState.curValues == final_state:
            path = []
            while currentState is not None:
                path.append(currentState.curValues)
                currentState = currentState.pastState
            return (path[::-1], monotone)

        for nextState in get_neighbours(
            currentState.curValues, max_missionaries, max_cannibals
        ):
            newgStar = currentState.gstar + gstar(currentState.curValues, nextState)
            newCost = newgStar + h1(nextState)

            if h1(currentState.curValues) > gstar(currentState.curValues, nextState) + h1(nextState):
                monotone = False

            if (
                tuple(nextState) not in closeList
                or closeList[tuple(nextState)] > newgStar
            ):
                heapq.heappush(
                    openList,
                    State(nextState, currentState, newCost, newgStar),
                )

    return ([], monotone)


def astar_h2(
    init_state: list, final_state: list, max_missionaries: int, max_cannibals: int
) -> Tuple[List[list], bool]:  # 8 marks
    """
    Graded
    Implement A* with h2 heuristic.
    """

    openList = []
    closeList = {}
    monotone = True

    heapq.heappush(openList, State(init_state, None, h2(init_state), 0))

    while len(openList) > 0:

        currentState = heapq.heappop(openList)

        if (
            tuple(currentState.curValues) in closeList
            and closeList[tuple(currentState.curValues)] <= currentState.gstar
        ):
            continue

        closeList[tuple(currentState.curValues)] = currentState.gstar

        if currentState.curValues == final_state:
            path = []
            while currentState is not None:
                path.append(currentState.curValues)
                currentState = currentState.pastState
            return (path[::-1], monotone)

        for nextState in get_neighbours(
            currentState.curValues, max_missionaries, max_cannibals
        ):
            newgStar = currentState.gstar + gstar(currentState.curValues, nextState)
            newCost = newgStar + h2(nextState)

            if h2(currentState.curValues) > gstar(currentState.curValues, nextState) + h2(nextState):
                monotone = False

            if (
                tuple(nextState) not in closeList
                or closeList[tuple(nextState)] > newgStar
            ):
                heapq.heappush(
                    openList,
                    State(nextState, currentState, newCost, newgStar),
                )

    return ([], monotone)


def astar_h3(
    init_state: list, final_state: list, max_missionaries: int, max_cannibals: int
) -> Tuple[List[list], bool]:  # 8 marks
    """
    Graded
    Implement A* with h3 heuristic.
    """

    openList = []
    closeList = {}
    monotone = True

    heapq.heappush(openList, State(init_state, None, h3(init_state), 0))

    while len(openList) > 0:

        currentState = heapq.heappop(openList)

        if (
            tuple(currentState.curValues) in closeList
            and closeList[tuple(currentState.curValues)] <= currentState.gstar
        ):
            continue

        closeList[tuple(currentState.curValues)] = currentState.gstar

        if currentState.curValues == final_state:
            path = []
            while currentState is not None:
                path.append(currentState.curValues)
                currentState = currentState.pastState
            return (path[::-1], monotone)

        for nextState in get_neighbours(
            currentState.curValues, max_missionaries, max_cannibals
        ):
            newgStar = currentState.gstar + gstar(currentState.curValues, nextState)
            newCost = newgStar + h3(nextState)

            if h3(currentState.curValues) > gstar(currentState.curValues, nextState) + h3(nextState):
                monotone = False

            if (
                tuple(nextState) not in closeList
                or closeList[tuple(nextState)] > newgStar
            ):
                heapq.heappush(
                    openList,
                    State(nextState, currentState, newCost, newgStar),
                )

    return ([], monotone)


def astar_h4(
    init_state: list, final_state: list, max_missionaries: int, max_cannibals: int
) -> Tuple[List[list], bool]:  # 8 marks
    """
    Graded
    Implement A* with h4 heuristic.
    """

    openList = []
    closeList = {}

    heapq.heappush(openList, State(init_state, None, h4(init_state), 0))

    while len(openList) > 0:

        currentState = heapq.heappop(openList)

        if (
            tuple(currentState.curValues) in closeList
            and closeList[tuple(currentState.curValues)] <= currentState.gstar
        ):
            continue

        closeList[tuple(currentState.curValues)] = currentState.gstar

        if currentState.curValues == final_state:
            path = []
            while currentState is not None:
                path.append(currentState.curValues)
                currentState = currentState.pastState
            return (path[::-1], False)

        for nextState in get_neighbours(
            currentState.curValues, max_missionaries, max_cannibals
        ):
            newgStar = currentState.gstar + gstar(currentState.curValues, nextState)
            newCost = newgStar + h4(nextState)


            if (
                tuple(nextState) not in closeList
                or closeList[tuple(nextState)] > newgStar
            ):
                heapq.heappush(
                    openList,
                    State(nextState, currentState, newCost, newgStar),
                )

    return ([], False)


def astar_h5(
    init_state: list, final_state: list, max_missionaries: int, max_cannibals: int
) -> Tuple[List[list], bool]:  # 8 marks
    """
    Graded
    Implement A* with h5 heuristic.
    """

    openList = []
    closeList = {}

    heapq.heappush(openList, State(init_state, None, h5(init_state), 0))

    while len(openList) > 0:

        currentState = heapq.heappop(openList)

        if (
            tuple(currentState.curValues) in closeList
            and closeList[tuple(currentState.curValues)] <= currentState.gstar
        ):
            continue

        closeList[tuple(currentState.curValues)] = currentState.gstar

        if currentState.curValues == final_state:
            path = []
            while currentState is not None:
                path.append(currentState.curValues)
                currentState = currentState.pastState
            return (path[::-1], False)

        for nextState in get_neighbours(
            currentState.curValues, max_missionaries, max_cannibals
        ):
            newgStar = currentState.gstar + gstar(currentState.curValues, nextState)
            newCost = newgStar + h5(nextState)


            if (
                tuple(nextState) not in closeList
                or closeList[tuple(nextState)] > newgStar
            ):
                heapq.heappush(
                    openList,
                    State(nextState, currentState, newCost, newgStar),
                )

    return ([], False)


def print_solution(solution: List[list], max_mis, max_can):
    """
    Prints the solution path.
    """
    if not solution:
        print("No solution exists for the given parameters.")
        return

    print("\nSolution found! Number of steps:", len(solution) - 1)
    print("\nLeft Bank" + " " * 20 + "Right Bank")
    print("-" * 50)

    for state in solution:
        if state[-1]:
            boat_display = "(B) " + " " * 15
        else:
            boat_display = " " * 15 + "(B) "

        print(
            f"M: {state[0]}, C: {state[1]}  {boat_display}"
            f"M: {max_mis-state[0]}, C: {max_can-state[1]}"
        )


def print_mon(ism: bool):
    """
    Prints if the heuristic function is monotone or not.
    """
    if ism:
        print("-" * 10)
        print("|Monotone|")
        print("-" * 10)
    else:
        print("-" * 14)
        print("|Not Monotone|")
        print("-" * 14)


def main():
    try:
        testcases = [{"m": 3, "c": 3}]

        for case in testcases:
            max_missionaries = case["m"]
            max_cannibals = case["c"]

            init_state = [max_missionaries, max_cannibals, 1]  # initial state
            final_state = [0, 0, 0]  # final state

            if not check_valid(init_state, max_missionaries, max_cannibals):
                print(f"Invalid initial state for case: {case}")
                continue

            path_h1, ism1 = astar_h1(
                init_state, final_state, max_missionaries, max_cannibals
            )
            path_h2, ism2 = astar_h2(
                init_state, final_state, max_missionaries, max_cannibals
            )
            path_h3, ism3 = astar_h3(
                init_state, final_state, max_missionaries, max_cannibals
            )
            path_h4, ism4 = astar_h4(
                init_state, final_state, max_missionaries, max_cannibals
            )
            path_h5, ism5 = astar_h5(
                init_state, final_state, max_missionaries, max_cannibals
            )
            print_solution(path_h1, max_missionaries, max_cannibals)
            print_mon(ism1)
            print("-" * 50)
            print_solution(path_h2, max_missionaries, max_cannibals)
            print_mon(ism2)
            print("-" * 50)
            print_solution(path_h3, max_missionaries, max_cannibals)
            print_mon(ism3)
            print("-" * 50)
            print_solution(path_h4, max_missionaries, max_cannibals)
            print_mon(ism4)
            print("-" * 50)
            print_solution(path_h5, max_missionaries, max_cannibals)
            print_mon(ism5)
            print("=" * 50)

    except KeyError as e:
        print(f"Missing required key in test case: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
