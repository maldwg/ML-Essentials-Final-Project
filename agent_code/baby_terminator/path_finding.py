import heapq
import numpy as np

# Define the 4 possible movement directions (no diagonal movements allowed)
DIRECTIONS = [(0, 1), (0, -1), (1, 0), (-1, 0)]
ACTIONS = ["UP", "DOWN", "RIGHT", "LEFT"]

def heuristic(a, b):
    """
    Compute the Manhattan distance between two points a and b.

    :param a: Tuple of coordinates for the first point.
    :param b: Tuple of coordinates for the second point.

    :return: int: The Manhattan distance between the two points.
    """
    # Calculate Manhattan distance
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar(start, goal, field):
    """
    A* pathfinding algorithm.

    :param start: Tuple of coordinates for the start point.
    :param goal: Tuple of coordinates for the goal point.
    :param field: 2D numpy array representing the field.

    :return: List of tuples representing the path from start to goal, or None if no path is found.
    """
    # Create a grid of points
    x, y = np.meshgrid(np.arange(field.shape[0]), np.arange(field.shape[1]))
    grid = np.vstack([x.ravel(), y.ravel()]).T

    # Initialize the open set with the start node
    open_set = []
    heapq.heappush(open_set, (0, start))
    
    # Initialize came_from and cost dictionaries
    came_from = {}
    g_score = {tuple(point): float("inf") for point in grid}
    g_score[start] = 0
    f_score = {tuple(point): float("inf") for point in grid}
    f_score[start] = heuristic(start, goal)

    while open_set:
        # Pop the node with the lowest f_score value
        _, current = heapq.heappop(open_set)

        # Check if the goal has been reached
        if current == goal:
            path = [current]
            while current in came_from:
                current = came_from[current]
                path.append(current)
            return path[::-1]

        # Explore neighbors
        for dx, dy in DIRECTIONS:
            neighbor = current[0] + dx, current[1] + dy

            # Skip neighbors that are out of bounds
            if (
                neighbor[0] < 0
                or neighbor[0] >= field.shape[0]
                or neighbor[1] < 0
                or neighbor[1] >= field.shape[1]
            ):
                continue

            # Skip neighbors that are obstacles or unwalkable
            if field[neighbor] == -1 or field[neighbor] == 1:
                continue

            # Compute tentative_g_score as the cost from start to the neighbor through current
            tentative_g_score = g_score[current] + 1  # Assuming each step cost is 1

            # Update path and scores if a better path is found
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = tentative_g_score + heuristic(neighbor, goal)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    return None  # No path found
