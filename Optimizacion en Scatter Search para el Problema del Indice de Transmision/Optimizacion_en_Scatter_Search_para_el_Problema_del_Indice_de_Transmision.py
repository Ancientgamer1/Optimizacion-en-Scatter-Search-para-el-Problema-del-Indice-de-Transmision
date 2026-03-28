import random
import time

# -----------------------------------
# GRAFO (sin pesos, como Tabu Search)
# -----------------------------------
graph = {
    'A': ['B', 'C', 'E', 'F'],
    'B': ['A', 'C', 'D', 'E', 'F'],
    'C': ['A', 'B', 'D', 'E', 'F'],
    'D': ['B', 'C', 'E', 'G'],
    'E': ['A', 'B', 'C', 'D', 'F', 'G'],
    'F': ['A', 'B', 'C', 'D', 'E', 'G'],
    'G': ['D', 'E', 'F', 'H'],
    'H': ['G', 'I'],
    'I': ['H', 'J'],
    'J': ['I', 'K'],
    'K': ['J', 'L'],
    'L': ['K', 'M'],
    'M': ['L', 'N'],
    'N': ['M', 'O'],
    'O': ['N', 'P'],
    'P': ['O', 'Q'],
    'Q': ['P', 'R'],
    'R': ['Q']
}

# -----------------------------------
# GENERACIÓN DE CAMINOS (diversificación)
# -----------------------------------
def random_path(graph, start, goal):
    path = [start]
    visited = set([start])
    current = start

    while current != goal:
        neighbors = [n for n in graph[current] if n not in visited]

        if not neighbors:
            return None  # Camino inválido

        next_node = random.choice(neighbors)
        path.append(next_node)
        visited.add(next_node)
        current = next_node

    return path


# -----------------------------------
# COSTO (longitud del camino)
# -----------------------------------
def path_cost(path):
    return len(path) - 1

# -----------------------------------
# METRICA DE DIVERSIDAD (aristas)
# -----------------------------------

def path_distance(p1, p2):
    edges1 = set((p1[i], p1[i+1]) for i in range(len(p1)-1))
    edges2 = set((p2[i], p2[i+1]) for i in range(len(p2)-1))
    return len(edges1 ^ edges2)

# -----------------------------------
# MEJORA LOCAL (tipo Tabu simplificado)
# -----------------------------------
def improve_path(graph, path):
    # Intenta acortar el camino eliminando ciclos
    improved = []
    seen = set()

    for node in path:
        if node in seen:
            break
        improved.append(node)
        seen.add(node)

    return improved


# -----------------------------------
# COMBINACIÓN DE CAMINOS
# -----------------------------------
def combine_paths(graph, p1, p2, goal):
    # Precaucion en caso de que los caminos sean muy cortos
    if min(len(p1), len(p2)) < 2:
        return None

    # Tomar prefijo de p1 y continuar con p2 si es válido
    cut = random.randint(1, min(len(p1), len(p2)) - 1)

    new_path = p1[:cut]

    current = new_path[-1]
    visited = set(new_path)

    # Intentar continuar con p2
    for node in p2:
        if node not in visited and node in graph[current]:
            new_path.append(node)
            visited.add(node)
            current = node

        if current == goal:
            return new_path

    # Fallback ? completar con random
    tail = random_path(graph, current, goal)
    if tail:
        return new_path + tail[1:]

    return None


# -----------------------------------
# GENERAR SOLUCIONES INICIALES
# -----------------------------------
def generate_initial_population(graph, start, goal, size, max_attempts=1000):
    population = []
    attempts = 0

    while len(population) < size and attempts < max_attempts:
        attempts += 1
        path = random_path(graph, start, goal)
        if path:
            population.append((path_cost(path), path))

    if len(population) < size:
        # return whatever was found and log a hint
        print(f"[WARN] Only generated {len(population)}/{size} initial solutions for {start}->{goal} after {attempts} attempts")
    return population


# -----------------------------------
# CONSTRUIR REFSET
# -----------------------------------
def build_refset(population, size):
    if not population:
        return []

    population.sort(key=lambda x: x[0])  # Ordenar por costo

    size = min(size, len(population))
    elite_size = size // 2
    diverse_size = size - elite_size

    # -----------------------------
    # 1. Mejores soluciones
    # -----------------------------
    refset = population[:elite_size]

    # -----------------------------
    # 2. Selección diversa
    # -----------------------------
    candidates = population[elite_size:]

    while len(refset) < size and candidates:
        best_candidate = None
        max_distance = -1

        for cand in candidates:
            path_cand = cand[1]

            # Distancia mínima a refset
            # Si refset esta vacia, tratamos la distancia como infinita y agarramos un candidato
            if refset:
                min_dist = min(path_distance(path_cand, r[1]) for r in refset)
            else:
                min_dist = float('inf')

            if min_dist > max_distance:
                max_distance = min_dist
                best_candidate = cand
        
        if best_candidate is None:
            break

        refset.append(best_candidate)
        candidates.remove(best_candidate)

    return refset


# -----------------------------------
# SCATTER SEARCH PRINCIPAL
# -----------------------------------
def scatter_search(graph, start, goal, pop_size=20, ref_size=6, iterations=20):

    # Diversificación
    population = generate_initial_population(graph, start, goal, pop_size)

    # Si no hay soluciones iniciales, retorna un sentinel
    if not population:
        return (float('inf'), None)

    # RefSet
    refset = build_refset(population, ref_size)

    if not refset:
        return (float('inf'), None)

    for _ in range(iterations):

        new_solutions = []

        # Combinación sistemática
        for i in range(len(refset)):
            for j in range(i + 1, len(refset)):

                p1 = refset[i][1]
                p2 = refset[j][1]

                new_path = combine_paths(graph, p1, p2, goal)

                if new_path:
                    new_path = improve_path(graph, new_path)
                    new_solutions.append((path_cost(new_path), new_path))

        # Actualización RefSet
        refset.extend(new_solutions)
        refset = build_refset(refset, ref_size)

    return min(refset, key=lambda x: x[0])


# -----------------------------------
# TODOS LOS PARES (igual que Tabu)
# -----------------------------------
def scatter_all_pairs(graph):
    nodes = list(graph.keys())
    all_paths = {}

    for s in nodes:
        all_paths[s] = {}
        for t in nodes:
            if s == t:
                all_paths[s][t] = [s]
            else:
                _, path = scatter_search(graph, s, t)
                all_paths[s][t] = path

    return all_paths


# -----------------------------------
# ÍNDICE DE TRANSMISIÓN (igual Tabu)
# -----------------------------------
def transmission_index(graph, all_paths):

    edge_load = {}

    # Inicializar
    for u in graph:
        for v in graph[u]:
            edge_load[frozenset([u, v])] = 0

    # Contar uso
    for s in all_paths:
        for t in all_paths[s]:
            if s == t:
                continue

            path = all_paths[s][t]

            for i in range(len(path) - 1):
                edge = frozenset([path[i], path[i + 1]])
                edge_load[edge] += 1

    return max(edge_load.values())


# -----------------------------------
# MAIN
# -----------------------------------
if __name__ == "__main__":

    print("STARTING:", __file__)
    t0 = time.time()

    # Quick check: run a single, small search first to ensure code completes quickly
    single = scatter_search(graph, 'A', 'H', pop_size=10, ref_size=4, iterations=5)
    print("Camino mas corto:", single)

    # Compute routing for all pairs (no per-pair progress prints)
    print("Computing routing for all pairs (this may take time)...")
    nodes = list(graph.keys())

    all_paths = {}
    for s in nodes:
        all_paths[s] = {}
        for t in nodes:
            if s == t:
                all_paths[s][t] = [s]
            else:
                # smaller parameters to avoid long runs
                res_cost, res_path = scatter_search(graph, s, t, pop_size=10, ref_size=4, iterations=5)
                all_paths[s][t] = res_path

    pi = transmission_index(graph, all_paths)
    print("Indice de Transmision (Scatter Search):", pi)
    print("Elapsed (s):", round(time.time() - t0, 2))