from collections import defaultdict, deque
from multiprocessing import Pool

def can_order_assignments(args):
    N, assignments, K = args
    
    graph = defaultdict(list)
    in_degree = defaultdict(int)
    
    for i in range(N):
        left_var, right_vars = assignments[i]
        for var in right_vars:
            graph[var].append(left_var)
            in_degree[left_var] += 1

    queue = deque([i for i in range(N) if in_degree[i] == 0])
    sorted_order = []
    
    while queue:
        node = queue.popleft()
        sorted_order.append(node)
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)

    if len(sorted_order) != N:
        return False

    used_in_rhs = set()
    count = 0
    for i in range(N):
        left_var, right_vars = assignments[sorted_order[i]]
        if left_var in used_in_rhs:
            count += 1
        used_in_rhs.update(right_vars)
        if count > K:
            return False

    return True

def parallel_processing(examples):
    with Pool() as pool:
        results = pool.map(can_order_assignments, examples)
    return results

examples = [
    # Пример 1: x0 = x1 + x2, x1 = x2, x2 = x3, x3 = 5
    (4, [(0, [1, 2]), (1, [2]), (2, [3]), (3, [])], 1),
    # Пример 2: x0 = x1, x1 = x2, x2 = x0
    (3, [(0, [1]), (1, [2]), (2, [0])], 0),
    # Пример 3: x0 = x1 + x2, x1 = x2 + x3, x2 = x3 + x4, x3 = x4, x4 = 10  
    (5, [(0, [1, 2]), (1, [2, 3]), (2, [3, 4]), (3, [4]), (4, [])], 2),
    # Пример 4: x0 = x1 + x2 + x3, x1 = x2 + x3, x2 = x3 + x4, x3 = x4 + x5, x4 = x5, x5 = 15
    (6, [(0, [1, 2, 3]), (1, [2, 3]), (2, [3, 4]), (3, [4, 5]), (4, [5]), (5, [])], 3),
    # Пример 5: x0 = x1, x1 = x2, x2 = x3, x3 = x0
    (4, [(0, [1]), (1, [2]), (2, [3]), (3, [0])], 1),
    # Пример 6: x0 = x1, x1 = x2, x2 = x3, x3 = 20
    (4, [(0, [1]), (1, [2]), (2, [3]), (3, [])], 0),
    # Пример 7: x0 = x1 + x2, x1 = x2, x2 = 30
    (3, [(0, [1, 2]), (1, [2]), (2, [])], 1),
    # Пример 8: x0 = x1, x1 = x0 + x2, x2 = x3, x3 = x4, x4 = 40
    (5, [(0, [1]), (1, [0, 2]), (2, [3]), (3, [4]), (4, [])], 2),
    # Пример 9: x0 = x1 + x2, x1 = x2 + x3, x2 = x3, x3 = x0
    (4, [(0, [1, 2]), (1, [2, 3]), (2, [3]), (3, [0])], 1),
    # Пример 10: x0 = x1 + x2, x1 = x2 + x3, x2 = x3 + x4, x3 = x4 + x5, x4 = x5 + x0, x5 = x1 + x2 + x3
    (6, [(0, [1, 2]), (1, [2, 3]), (2, [3, 4]), (3, [4, 5]), (4, [5, 0]), (5, [1, 2, 3])], 4)
]

results = parallel_processing(examples)
for i, result in enumerate(results):
    print(f"Пример {i+1}: {'Можно' if result else 'Нельзя'}")

