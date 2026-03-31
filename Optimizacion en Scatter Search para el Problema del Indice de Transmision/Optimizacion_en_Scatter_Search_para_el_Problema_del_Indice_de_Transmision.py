import random
import time
import os
from collections import deque

# =========================================================
# LECTURA DEL CASO DE PRUEBA
# =========================================================
def read_testcase(path):
    with open(path, 'r') as f:
        lines = [ln.strip() for ln in f if ln.strip() and not ln.startswith('#')]

    pop, ref, it = map(int, lines[0].split())

    graph = {}
    for ln in lines[1:]:
        parts = ln.split()
        graph[parts[0]] = parts[1:]

    return graph, pop, ref, it


# =========================================================
# BFS CON CACHE (evita recomputación)
# =========================================================
bfs_cache = {}

def bfs_path(graph, start, goal):
    if (start, goal) in bfs_cache:
        return bfs_cache[(start, goal)]

    if start == goal:
        return [start]

    q = deque([start])
    parent = {start: None}

    while q:
        u = q.popleft()

        for v in graph[u]:
            if v not in parent:
                parent[v] = u

                if v == goal:
                    path = [v]
                    while parent[path[-1]] is not None:
                        path.append(parent[path[-1]])
                    path = list(reversed(path))
                    bfs_cache[(start, goal)] = path
                    return path

                q.append(v)

    bfs_cache[(start, goal)] = None
    return None


# =========================================================
# MUESTREO DE PARES (reduce O(n²) → O(k))
# =========================================================
def smart_sample_pairs(graph, k=100):

    nodes = list(graph.keys())
    pairs = set()

    # -------------------------------------------------
    # 1. PARES LEJANOS (usando BFS)
    # -------------------------------------------------
    for s in nodes:

        # calcular distancias desde s
        visited = {s: 0}
        queue = [s]

        while queue:
            u = queue.pop(0)

            for v in graph[u]:
                if v not in visited:
                    visited[v] = visited[u] + 1
                    queue.append(v)

        # ordenar por distancia (descendente)
        far_nodes = sorted(visited.items(), key=lambda x: -x[1])

        # tomar algunos lejanos
        for t, _ in far_nodes[:3]:
            if s != t:
                pairs.add((s, t))

    # -------------------------------------------------
    # 2. ASEGURAR COBERTURA (cada nodo participa)
    # -------------------------------------------------
    for s in nodes:
        t = random.choice(nodes)
        if s != t:
            pairs.add((s, t))

    # -------------------------------------------------
    # 3. COMPLETAR CON ALEATORIOS
    # -------------------------------------------------
    while len(pairs) < k:
        s = random.choice(nodes)
        t = random.choice(nodes)

        if s != t:
            pairs.add((s, t))

    return list(pairs)


# =========================================================
# ÍNDICE DE TRANSMISIÓN (PARCIAL - ARISTAS)
# =========================================================
def transmission_index_partial(routing):

    edge_load = {}

    for (s, t), path in routing.items():

        if not path:
            continue

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]

            # arista no dirigida (sin frozenset)
            edge = (u, v) if u < v else (v, u)

            edge_load[edge] = edge_load.get(edge, 0) + 1

    return max(edge_load.values()) if edge_load else 0


# =========================================================
# GENERAR ENRUTAMIENTO INICIAL (solo pares muestreados)
# =========================================================
def generate_routing(graph, pairs):
    routing = {}

    for (s, t) in pairs:
        routing[(s, t)] = bfs_path(graph, s, t)

    return routing


# =========================================================
# PERTURBACIÓN (mejora local)
# =========================================================
def perturb_routing(graph, routing, pairs, intensity=5):

    for _ in range(intensity):
        s, t = random.choice(pairs)

        new_path = bfs_path(graph, s, t)

        if new_path:
            routing[(s, t)] = new_path

    return routing


# =========================================================
# DISTANCIA ENTRE SOLUCIONES (diversidad)
# =========================================================
def routing_distance(r1, r2, sample_size=20):

    keys = list(r1.keys())
    dist = 0

    for _ in range(sample_size):
        k = random.choice(keys)

        if r1[k] != r2[k]:
            dist += 1

    return dist


# =========================================================
# GENERAR POBLACIÓN INICIAL
# =========================================================
def generate_initial_population(graph, pairs, size):

    population = []

    for _ in range(size):
        routing = generate_routing(graph, pairs)
        routing = perturb_routing(graph, routing, pairs)

        cost = transmission_index_partial(routing)

        population.append((cost, routing))

    return population


# =========================================================
# BUILD REFSET (calidad + diversidad)
# =========================================================
def build_refset(population, size):

    population.sort(key=lambda x: x[0])

    elite_size = size // 2
    refset = population[:elite_size]
    candidates = population[elite_size:]

    while len(refset) < size and candidates:

        best = None
        max_dist = -1

        for cand in candidates:
            dist = min(routing_distance(cand[1], r[1]) for r in refset)

            if dist > max_dist:
                max_dist = dist
                best = cand

        if not best:
            break

        refset.append(best)
        candidates.remove(best)

    return refset


# =========================================================
# COMBINACIÓN DE ENRUTAMIENTOS
# =========================================================
def combine_routings(r1, r2):

    new = {}

    for k in r1:
        new[k] = r1[k] if random.random() < 0.5 else r2[k]

    return new


# =========================================================
# SCATTER SEARCH PRINCIPAL
# =========================================================
def scatter_search(graph, pop_size, ref_size, iterations, pair_sample=50):

    nodes = list(graph.keys())
    pairs = smart_sample_pairs(graph, pair_sample)

    population = generate_initial_population(graph, pairs, pop_size)
    refset = build_refset(population, ref_size)

    best_cost = float('inf')
    stagnation = 0

    for _ in range(iterations):

        new_solutions = []

        for i in range(len(refset)):
            for j in range(i+1, len(refset)):

                r1 = refset[i][1]
                r2 = refset[j][1]

                new_r = combine_routings(r1, r2)
                new_r = perturb_routing(graph, new_r, pairs)

                cost = transmission_index_partial(new_r)

                new_solutions.append((cost, new_r))

        refset.extend(new_solutions)
        refset = build_refset(refset, ref_size)

        current_best = refset[0][0]

        if current_best < best_cost:
            best_cost = current_best
            stagnation = 0
        else:
            stagnation += 1

        if stagnation >= 5:
            break

    return min(refset, key=lambda x: x[0])


# =========================================================
# EXPANSIÓN A TODOS LOS PARES (evaluación exacta)
# =========================================================
def expand_routing_to_all_pairs(graph, partial_routing):

    nodes = list(graph.keys())
    full_routing = {}

    for s in nodes:
        for t in nodes:

            if s == t:
                full_routing[(s, t)] = [s]
                continue

            if (s, t) in partial_routing:
                full_routing[(s, t)] = partial_routing[(s, t)]
            else:
                full_routing[(s, t)] = bfs_path(graph, s, t)

    return full_routing


# =========================================================
# ÍNDICE DE TRANSMISIÓN COMPLETO (EXACTO)
# =========================================================
def transmission_index_full(routing):

    edge_load = {}

    for (s, t), path in routing.items():

        if not path:
            continue

        for i in range(len(path) - 1):
            u, v = path[i], path[i+1]

            edge = (u, v) if u < v else (v, u)

            edge_load[edge] = edge_load.get(edge, 0) + 1

    return max(edge_load.values()), edge_load


# =========================================================
# MAIN
# =========================================================
if __name__ == "__main__":

    file = next((f for f in os.listdir() if f.endswith(".txt")), None)

    if not file:
        print("No input file found")
        exit()

    graph, pop, ref, it = read_testcase(file)

    print("\nArchivo:", file)
    print("Parámetros:", pop, ref, it)
    print("Nodos:", len(graph))

    t0 = time.time()

    # 🔵 FASE 1: optimización rápida
    best_cost, best_partial = scatter_search(
        graph,
        pop_size=pop,
        ref_size=ref,
        iterations=it,
        pair_sample=50
    )

    print("\n[FASE 1] Mejor costo aproximado:", best_cost)

    # 🟢 FASE 2: evaluación exacta
    print("\n[FASE 2] Expandiendo a todos los pares...")

    full_routing = expand_routing_to_all_pairs(graph, best_partial)

    print("[FASE 2] Calculando índice exacto...")

    real_cost, _ = transmission_index_full(full_routing)

    print("\nÍndice de transmisión REAL:", real_cost)
    print("Tiempo total:", round(time.time() - t0, 2), "s")