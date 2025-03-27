import heapq
from constants import OBSTACLE_COST # Assuming heuristic is simple enough to keep inline or move utils later

def heuristic(a, b):
    """Calculates the Manhattan distance heuristic between two grid points (a, b)."""
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

def astar_pathfinding(grid, start_node, end_node):
    """
    Finds the shortest path between two nodes on a grid using the A* algorithm.
    Considers terrain costs defined in the grid.

    Args:
        grid (list[list[int]]): 2D list representing the grid costs.
        start_node (tuple[int, int]): The starting grid coordinates (x, y).
        end_node (tuple[int, int]): The target grid coordinates (x, y).

    Returns:
        list[tuple[int, int]] or None: Path as list of coords, or None if no path.
    """
    if not grid or not start_node or not end_node: return None
    rows, cols = len(grid), len(grid[0])
    if not (0 <= start_node[0] < cols and 0 <= start_node[1] < rows and \
            0 <= end_node[0] < cols and 0 <= end_node[1] < rows):
        return None
    if grid[start_node[1]][start_node[0]] == OBSTACLE_COST or \
       grid[end_node[1]][end_node[0]] == OBSTACLE_COST:
        return None

    neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0)] # 4-way movement
    # neighbors = [(0, 1), (0, -1), (1, 0), (-1, 0), (1, 1), (1, -1), (-1, 1), (-1, -1)] # 8-way

    close_set = set()
    came_from = {}
    gscore = {start_node: 0}
    fscore = {start_node: heuristic(start_node, end_node)}
    oheap = []
    heapq.heappush(oheap, (fscore[start_node], start_node))

    while oheap:
        current = heapq.heappop(oheap)[1]
        if current == end_node:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            return path[::-1]

        close_set.add(current)
        for i, j in neighbors:
            neighbor = current[0] + i, current[1] + j
            if not (0 <= neighbor[0] < cols and 0 <= neighbor[1] < rows):
                continue

            neighbor_cost = grid[neighbor[1]][neighbor[0]]
            # Add cost for diagonal movement if using 8-way
            move_cost = neighbor_cost # Base cost is terrain cost
            # if abs(i) == 1 and abs(j) == 1: # If diagonal
            #    move_cost = neighbor_cost * 1.414 # Approx sqrt(2)

            if neighbor_cost == OBSTACLE_COST or neighbor in close_set:
                continue

            tentative_g_score = gscore[current] + move_cost # Use move_cost here

            if tentative_g_score < gscore.get(neighbor, float('inf')):
                came_from[neighbor] = current
                gscore[neighbor] = tentative_g_score
                fscore[neighbor] = tentative_g_score + heuristic(neighbor, end_node)
                heapq.heappush(oheap, (fscore[neighbor], neighbor))
    return None