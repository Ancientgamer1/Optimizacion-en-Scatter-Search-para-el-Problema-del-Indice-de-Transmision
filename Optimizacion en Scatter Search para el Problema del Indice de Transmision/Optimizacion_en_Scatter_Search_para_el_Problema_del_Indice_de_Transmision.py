import random
import time
import numpy as np
import math
import heapq
from collections import defaultdict, deque
from typing import List, Tuple

# =========================
# PARAMETROS GLOBALES
# =========================
ALPHA_DEFAULT = 0.5
BETA_DEFAULT = 0.3
IMPROVE_PROB = 0.35
path_cache = {}
MAX_PATH_CACHE = 50000
PATH_EDGE_CACHE = {}
MAX_PATH_EDGE_CACHE = 200000
VERBOSE = False

# =========================
# UTILIDADES
# =========================
def normalize_edge(u, v):
    return tuple(sorted((u, v)))

def is_valid_solution(solution):
    return solution is not None and solution.routing and len(solution.routing) > 0

def real_transmission_index(solution):
    return max(solution.edge_load.values()) if solution.edge_load else 0

def get_real_index(solution):
    if not hasattr(solution, "real_index"):
        solution.real_index = None

    if solution.real_index is None:
        solution.real_index = real_transmission_index(solution)

    return solution.real_index

def path_to_edges(path):

    key = tuple(path)

    cached = PATH_EDGE_CACHE.get(key)

    if cached is not None:
        return cached

    edges = frozenset(normalize_edge(path[i], path[i + 1]) for i in range(len(path) - 1))

    PATH_EDGE_CACHE[key] = edges

    if len(PATH_EDGE_CACHE) > MAX_PATH_EDGE_CACHE:
        PATH_EDGE_CACHE.clear()

    return edges

def edge_set_from_path(path):
    return {normalize_edge(path[i], path[i + 1]) for i in range(len(path) - 1)}

def top_congested_edges(solution, limit=10):
    return sorted(solution.edge_load, key=solution.edge_load.get, reverse=True)[:limit]

def congestion_signature(solution, limit=10):
    loads = sorted(solution.edge_load.values(), reverse=True)[:limit]
    return tuple(loads)

def improves_congestion(old_sig, new_sig):
    return new_sig < old_sig

# =========================
# GRAFO
# =========================
class Graph:
    def __init__(self):
        self.adj = defaultdict(list)

    def add_edge(self, u, v):
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)

    def nodes(self):
        return list(self.adj.keys())

    def neighbors(self, u):
        return self.adj[u]

# =========================
# SOLUCIÓN
# =========================
class Solution:
    def __init__(self, routing, graph, total_pairs=1):
        self.routing = routing
        self.graph = graph
        self.total_pairs = total_pairs

        self.edge_load = defaultdict(int)
        self.total_load = 0
        self.sum_sq_load = 0
        self.max_load = 0
        self.version = 0
        self.edge_to_pairs = defaultdict(set)
        self.real_index = None

        self._initialize_edge_load()
        self.cost = self.compute_cost()

    def copy(self):

        new = Solution.__new__(Solution)

        # Compartidos inicialmente
        new.routing = {k: v.copy() for k,v in self.routing.items()}
        new.edge_to_pairs = {e: pairs.copy() for e, pairs in self.edge_to_pairs.items()}

        # Copias ligeras
        new.edge_load = self.edge_load.copy()

        new.graph = self.graph
        new.total_pairs = self.total_pairs

        new.total_load = self.total_load
        new.sum_sq_load = self.sum_sq_load
        new.max_load = self.max_load
        new.real_index = self.real_index

        new.cost = self.cost
        new.version = self.version

        # Flags copy-on-write
        new._routing_shared = True
        new._pairs_shared = False

        return new

    def _initialize_edge_load(self):
        # Calcular carga inicial y mapeo de aristas a pares
        for (u, v), path in self.routing.items():
            # Solo contar carga si el camino es válido
            for i in range(len(path) - 1):

                e = normalize_edge(path[i], path[i+1])

                self.edge_load[e] += 1

                # Mapeo de arista a pares que la usan, para facilitar mejoras posteriores
                self.edge_to_pairs[e].add((u, v))
        
        # Calcular métricas de carga
        for load in self.edge_load.values():

            self.total_load += load
            self.sum_sq_load += load * load
            self.max_load = max(self.max_load, load)

    def update_pair(self, u, v, new_path):
        # Copy-on-write para evitar copias completas al actualizar pares individuales
        if getattr(self, "_routing_shared", False):
            self.routing = self.routing.copy()
            self._routing_shared = False

        if getattr(self, "_pairs_shared", False):
            self.edge_to_pairs = {e: pairs.copy() for e, pairs in self.edge_to_pairs.items()}
            self._pairs_shared = False

        old_path = self.routing[(u, v)]
        
        max_might_have_dropped = False

        # Remover carga de camino antiguo
        for i in range(len(old_path) - 1):
            e = normalize_edge(old_path[i], old_path[i + 1])

            old = self.edge_load[e]

            # Si esta arista tenía el máximo,
            # quizá haya que recalcular después
            if old == self.max_load:
                max_might_have_dropped = True

            new = old - 1
            self.edge_load[e] = new

            self.edge_to_pairs[e].discard((u, v))

            # Actualizar métricas incrementales
            self.sum_sq_load += new**2 - old**2
            self.total_load -= 1

            # Limpiar aristas con carga 0
            if new == 0:
                del self.edge_load[e]

                if not self.edge_to_pairs[e]:
                    del self.edge_to_pairs[e]

        # Agregar carga de nuevo camino
        for i in range(len(new_path) - 1):
            e = normalize_edge(new_path[i], new_path[i + 1])

            old = self.edge_load.get(e, 0)
            new = old + 1

            self.edge_load[e] = new

            if e not in self.edge_to_pairs:
                self.edge_to_pairs[e] = set()

            self.edge_to_pairs[e].add((u, v))

            # Actualizar métricas incrementales
            self.sum_sq_load += new**2 - old**2
            self.total_load += 1

            # Actualizar máximo si crece
            if new > self.max_load:
                self.max_load = new

        self.routing[(u, v)] = new_path

        # Si el máximo podría haber bajado, recalcularlo completamente (caso raro)
        if max_might_have_dropped:
            self.max_load = max(self.edge_load.values())

        self.cost = self.compute_cost()
        self.real_index = None
        self.version += 1

    def compute_cost(self):
        if not self.edge_load:
            return float('inf')

        # solo desempate
        return self.sum_sq_load

# =========================
# LECTURA
# =========================
def read_graph(filename):
    g = Graph()
    
    with open(filename) as f:
        lines = f.readlines()

    # Parámetros de Scatter Search
    pop_size, refset_size, iterations = map(int, lines[0].split())

    # Construcción del grafo. Soporta nodos no numéricos y múltiples vecinos por línea.
    for line in lines[1:]:
        parts = line.split()
        u = parts[0]
        for v in parts[1:]:
            g.add_edge(u, v)

    return g, pop_size, refset_size, iterations

# =========================
# PRECOMPUTACION DE CAMNINOS
# =========================
def bfs_all(graph, start):
    parent = {start: None}
    queue = deque([start])

    while queue:
        u = queue.popleft()
        for v in graph.neighbors(u):
            if v not in parent:
                parent[v] = u
                queue.append(v)
    return parent


def build_path(parent, v):
    path = []
    while v is not None:
        path.append(v)
        v = parent[v]
    return path[::-1]


def precompute_all_pairs(graph):
    all_paths = {}
    nodes = graph.nodes()

    for i, u in enumerate(nodes):
        parent = bfs_all(graph, u)

        for v in nodes[i+1:]:
            all_paths[(u, v)] = build_path(parent, v)

    return all_paths

# =========================
# SHORTEST PATH EVITANDO ARISTAS PROHIBIDAS (BFS)
# =========================
def shortest_path_avoiding(graph, source, target, forbidden_edges):

    # Usar orden canónico
    key = (source, target, frozenset(forbidden_edges))

    # Buscar en cache
    if key in path_cache:
        return path_cache[key]

    queue = deque([source])
    parent = {source: None}

    while queue:

        u = queue.popleft()

        for v in graph.neighbors(u):

            edge = normalize_edge(u, v)

            if edge in forbidden_edges:
                continue

            if v not in parent:
                parent[v] = u

                if v == target:
                    path = build_path(parent, target)

                    # Guardar en cache
                    path_cache[key] = path

                    # Limitar tamaño
                    if len(path_cache) > MAX_PATH_CACHE:
                        path_cache.clear()

                    return path

                queue.append(v)

    # También cachear fallo
    path_cache[key] = None

    if len(path_cache) > MAX_PATH_CACHE:
        path_cache.clear()

    return None

def randomized_shortest_path(graph, source, target):

    queue = deque([source])
    parent = {source: None}

    while queue:
        u = queue.popleft()

        neighbors = list(graph.neighbors(u))
        random.shuffle(neighbors)

        for v in neighbors:

            if v not in parent:
                parent[v] = u

                if v == target:
                    return build_path(parent, target)

                queue.append(v)

    return None

def build_base_solution(graph, all_paths):
    routing = {pair: path for pair, path in all_paths.items()}

    return Solution(routing, graph, total_pairs=len(all_paths))

# =========================
# GENERAR SOLUCIÓN (TODOS LOS PARES)
# =========================
def build_initial_solution(graph, all_paths):

    routing = {}

    pairs = list(all_paths.keys())
    n_pairs = len(pairs)

    # Mucho más barato que modificar 65% de todos los pares
    max_alternatives = min(2500, n_pairs // 12)

    alt_pairs = set(random.sample(pairs, min(max_alternatives, n_pairs)))

    for (u, v), base_path in all_paths.items():

        if (u, v) not in alt_pairs:
            routing[(u, v)] = base_path
            continue

        if len(base_path) < 3:
            routing[(u, v)] = base_path
            continue

        # Probar pocas alternativas, no BFS masivo
        best_path = base_path
        best_score = float("inf")

        attempts = 2

        for _ in range(attempts):

            idx = random.randint(0, len(base_path) - 2)

            forbidden = {normalize_edge(base_path[idx], base_path[idx + 1])}

            alt = shortest_path_avoiding(graph, u, v, forbidden)

            if not alt:
                continue

            # preferir caminos no demasiado largos
            score = len(alt)

            if score < best_score:
                best_score = score
                best_path = alt

        routing[(u, v)] = best_path

    return Solution(routing, graph, total_pairs=len(all_paths))

def sampled_population_distances(population, sample_size=120):

    n = len(population)

    if n < 2:
        return [0]

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]

    sample = random.sample(all_pairs, min(sample_size, len(all_pairs)))

    return [solution_distance(population[i], population[j]) for i, j in sample]


# =========================
# PATH RELINKING
# =========================
def path_relinking(solA, solB):

    if random.random() < 0.5:
        solA, solB = solB, solA

    child = solA.copy()

    differing_pairs = [
        p for p in solA.routing
        if solA.routing[p] != solB.routing[p]
    ]

    if not differing_pairs:
        return child

    critical_edges = set(top_congested_edges(solA, limit=10))

    critical_pairs = [
        p for p in differing_pairs
        if path_to_edges(solA.routing[p]) & critical_edges
    ]

    non_critical_pairs = [
        p for p in differing_pairs
        if p not in critical_pairs
    ]

    random.shuffle(non_critical_pairs)

    if critical_pairs:
        differing_pairs = critical_pairs + non_critical_pairs[:len(critical_pairs)]
    else:
        differing_pairs = non_critical_pairs

    MAX_SCORING_PAIRS = 5000

    if len(differing_pairs) > MAX_SCORING_PAIRS:
        differing_pairs = random.sample(differing_pairs, MAX_SCORING_PAIRS)

    scored_pairs = []

    for pair in differing_pairs:

        pathA = solA.routing[pair]
        pathB = solB.routing[pair]

        edgesA = path_to_edges(pathA)
        edgesB = path_to_edges(pathB)

        score = 0

        for e in edgesA - edgesB:
            score += (
                solA.edge_load.get(e, 0)
                + 0.3 * solB.edge_load.get(e, 0)
            )

        for e in edgesB - edgesA:
            score -= 0.15 * solA.edge_load.get(e, 0)

        scored_pairs.append((score, pair))

    effective_d = len(scored_pairs)

    if effective_d == 0:
        return child

    low = max(120, effective_d // 8)
    high = min(700, effective_d // 4)

    if low > high:
        low = high

    take_size = random.randint(low, high)

    if take_size <= 0:
        return child

    top_pool = heapq.nlargest(min(4 * take_size, len(scored_pairs)), scored_pairs)

    selected = random.sample(top_pool, min(take_size, len(top_pool)))

    for _, pair in selected:
        child.update_pair(pair[0], pair[1], solB.routing[pair])

    # =================================================
    # REPARACIÓN CONGESTIVA LIGERA DENTRO DE PR
    # =================================================
    # Objetivo:
    # permitir que PR también ataque algunas aristas calientes,
    # sin convertirlo en una perturbación grande.
    # Esto ayuda a que PR no dependa tanto de perturb_solution.
    # =================================================

    hot_edges = sorted(
        child.edge_load,
        key=child.edge_load.get,
        reverse=True
    )[:8]

    hot_edges_set = set(hot_edges)

    candidate_pairs = set()

    for e in hot_edges:
        candidate_pairs.update(child.edge_to_pairs.get(e, set()))

    candidate_pairs = list(candidate_pairs)
    random.shuffle(candidate_pairs)

    MAX_PR_REPAIR_MOVES = 12
    repair_moves = 0

    for (u, v) in candidate_pairs:

        if repair_moves >= MAX_PR_REPAIR_MOVES:
            break

        old_path = child.routing[(u, v)]
        old_edges = path_to_edges(old_path)

        forbidden = old_edges & hot_edges_set

        if not forbidden:
            continue

        old_real = child.max_load
        old_cost = child.cost

        new_path = shortest_path_avoiding(child.graph, u, v, forbidden)

        if not new_path or new_path == old_path:
            continue

        child.update_pair(u, v, new_path)

        new_real = child.max_load

        if (new_real < old_real or (new_real == old_real and child.cost < old_cost)):
            repair_moves += 1
        else:
            child.update_pair(u, v, old_path)

    return child


# =========================
# IMPROVE 
# =========================
def improve(sol, graph, global_edge_load, k=7, max_moves=50):

    # Combina congestión actual + memoria histórica
    score = {
        e: sol.edge_load[e] + 0.4 * global_edge_load.get(e, 0)
        for e in sol.edge_load
    }

    critical = sorted(score, key=score.get, reverse=True)[:10]

    critical_set = set(critical)

    # Aristas actualmente en el máximo real
    max_edges = [e for e in sol.edge_load if sol.edge_load[e] == sol.max_load]

    candidate_pairs = set()

    # Prioridad 1: pares que usan aristas con carga máxima
    for e in max_edges:
        candidate_pairs.update(sol.edge_to_pairs.get(e, set()))

    # Prioridad 2: pares que usan aristas críticas según memoria actual/global
    if len(candidate_pairs) < max_moves:
        for e in critical:
            candidate_pairs.update(sol.edge_to_pairs.get(e, set()))

    candidates = list(candidate_pairs)
    random.shuffle(candidates)
    candidates = candidates[:max_moves]

    accepted_changes = 0
    MAX_ACCEPTED = min(8, max_moves)

    for (u, v) in candidates:

        old_path = sol.routing[(u, v)]
        old_edges = path_to_edges(old_path)

        local_forbidden = old_edges & critical_set

        if not local_forbidden:
            continue

        old_real = sol.max_load
        old_cost = sol.cost
        old_sig = congestion_signature(sol, limit=10)

        new_path = shortest_path_avoiding(graph, u, v, local_forbidden)

        if not new_path or new_path == old_path:
            continue

        sol.update_pair(u, v, new_path)

        new_real = sol.max_load
        new_sig = congestion_signature(sol, limit=10)

        accept = False

        if new_real < old_real:
            accept = True

        elif new_real == old_real and sol.cost < old_cost:
            accept = True

        elif new_real <= old_real + 1 and improves_congestion(old_sig, new_sig):
            accept = True

        if accept:
            accepted_changes += 1

            if accepted_changes >= MAX_ACCEPTED:
                break
        else:
            sol.update_pair(u, v, old_path)

    return sol

def perturb_solution(solution, graph, num_changes=5):

    new_sol = solution.copy()

    hot_edges = sorted(new_sol.edge_load, key=new_sol.edge_load.get, reverse=True)[:10]

    candidate_pairs = set()

    for e in hot_edges:
        candidate_pairs.update(new_sol.edge_to_pairs.get(e, set()))

    if not candidate_pairs:
        candidate_pairs = set(new_sol.routing.keys())

    chosen_pairs = random.sample(
        list(candidate_pairs),
        min(num_changes, len(candidate_pairs))
    )

    changes = 0

    for (u, v) in chosen_pairs:

        old_path = new_sol.routing[(u, v)]

        forbidden = {e for e in hot_edges if e in path_to_edges(old_path)}

        if not forbidden:
            continue

        new_path = shortest_path_avoiding(graph, u, v, forbidden)

        if not new_path or new_path == old_path:
            continue

        new_sol.update_pair(u, v, new_path)
        changes += 1

    print(
        f"Perturbación: {changes} pares modificados "
        f"(de {len(candidate_pairs)} conflictivos)"
    )

    return new_sol

def generate_initial_variant(base_solution, graph, num_changes=800):

    new_sol = base_solution.copy()

    pairs = list(new_sol.routing.keys())

    chosen_pairs = random.sample(pairs, min(num_changes, len(pairs)))

    changes = 0

    for (u, v) in chosen_pairs:

        old_path = new_sol.routing[(u, v)]

        if len(old_path) < 2:
            continue

        old_edges = list(path_to_edges(old_path))

        forbidden = {random.choice(old_edges)}

        new_path = shortest_path_avoiding(graph, u, v, forbidden)

        if not new_path or new_path == old_path:
            continue

        new_sol.update_pair(u, v, new_path)
        changes += 1

    return new_sol

# =========================
# DISTANCIA ENTRE SOLUCIONES
# =========================
def solution_distance(sol1, sol2, threshold=None):

    diff = 0

    for k in sol1.routing:
        if sol1.routing[k] != sol2.routing[k]:
            diff += 1

            if threshold is not None and diff >= threshold:
                return diff

    return diff


def min_distance_to_set(candidate, solutions, threshold=None):

    best = float("inf")

    for sol in solutions:
        d = solution_distance(candidate, sol, threshold=threshold)

        if d < best:
            best = d

        if threshold is not None and best < threshold:
            return best

    return best

# =========================
# REFSET (ELITE + DIVERSIDAD)
# =========================
def build_refset(population, refset_size):

    population = sorted(population, key=lambda s: (get_real_index(s), s.cost))

    n_best = refset_size // 2
    best_part = population[:n_best]

    remaining = population[n_best:]
    diverse_part = []

    while len(best_part) + len(diverse_part) < refset_size:

        ref_members = best_part + diverse_part

        # Sample pequeño para acelerar
        sample_size = min(20, len(remaining))
        candidates = random.sample(remaining, sample_size)

        best_candidate = None
        best_score = -1

        for cand in candidates:

            min_dist = min( solution_distance(cand, s) for s in ref_members)

            cand_real = get_real_index(cand)

            same_real_count = sum(get_real_index(s) == cand_real for s in ref_members)

            # Penalizar exceso de mismo real_index
            score = min_dist - 5 * same_real_count

            if score > best_score:
                best_score = score
                best_candidate = cand

        if best_candidate is None:
            break

        diverse_part.append(best_candidate)
        remaining.remove(best_candidate)

    return best_part + diverse_part


# =========================
# SCATTER SEARCH
# =========================
def scatter_search(graph, pop_size, refset_size, iterations):

    print("Precomputando caminos...")
    all_paths = precompute_all_pairs(graph)

    print("Construyendo población...")

    base_solution = build_base_solution(graph, all_paths)
    population = [base_solution]

    for _ in range(pop_size - 1):

        r = random.random()

        if r < 0.50:
            changes = 1200
        elif r < 0.85:
            changes = 2200
        else:
            changes = 3500

        population.append(
            generate_initial_variant(base_solution, graph, num_changes=changes))

    print("Construyendo RefSet...")
    refset = build_refset(population, refset_size)

    best = min(refset, key=lambda s: (get_real_index(s), s.cost)).copy()

    print(
        f"Iteración 1: "
        f"real={get_real_index(best)} "
        f"cost={best.cost:.4f}"
    )

    # Memoria global
    global_edge_load = {e: load for e, load in best.edge_load.items()}

    # Parámetros de control adaptativo
    stagnation = 0
    diversifications_without_improvement = 0
    perturbation_used = False
    MAX_DIVERSIFICATIONS = max(3, min(6, int(math.log2(len(graph.nodes()))) - 3))
    max_no_improve = max(8, len(graph.nodes()) // 20)

    MAX_PR_PAIRS = max(8, min(20, refset_size * 2))
    VERBOSE_PR = False
    VERBOSE_FILTERS = False

    weak_child_streak = 0
    strategic_cooldown = 0
    best_real_stagnation = 0

    for it in range(iterations):

        print(f"\n--- Iteración {it+1} ---")

        new_solutions = []

        # ==========================================
        # THRESHOLD DINÁMICO PARA DIVERSIDAD EN POBLACIÓN
        # ==========================================

        population_distances = sampled_population_distances(population, sample_size=120)

        median_population_dist = np.median(population_distances)

        POP_DISTANCE_THRESHOLD = max(80, int(median_population_dist * 0.04))

        print(
            f"Median Population distance: "
            f"{int(median_population_dist)} | "
            f"POP_DISTANCE_THRESHOLD: {POP_DISTANCE_THRESHOLD}"
        )

        # =================================================
        # DISTANCIA MÍNIMA ADAPTATIVA PARA PATH RELINKING
        # =================================================

        distances_refset = [
            solution_distance(refset[i], refset[j])
            for i in range(len(refset))
            for j in range(i + 1, len(refset))
        ]

        median_refset_dist = np.median(distances_refset)

        MIN_PR_DISTANCE = 1500

        print(
            "Median RefSet distance:",
            int(median_refset_dist),
            "| MIN_PR_DISTANCE:",
            MIN_PR_DISTANCE
        )

        # =================================================
        # PATH RELINKING
        # =================================================

        all_pairs = [(i,j) for i in range(len(refset)) for j in range(i+1,len(refset))]

        random.shuffle(all_pairs)

        pairs = all_pairs[: min(MAX_PR_PAIRS, len(all_pairs))]

        for (i, j) in pairs:

            A = refset[i]
            B = refset[j]

            d = solution_distance(A, B)

            if d < MIN_PR_DISTANCE:
                continue

            child = path_relinking(A, B)

            child_dist_A = solution_distance(child, A, threshold=POP_DISTANCE_THRESHOLD)
            child_dist_B = solution_distance(child, B, threshold=POP_DISTANCE_THRESHOLD)

            if VERBOSE_PR:
                print(
                    "PR parent dist:",
                    d,
                    "child-A:",
                    child_dist_A,
                    "child-B:",
                    child_dist_B
                )

            if (child_dist_A < POP_DISTANCE_THRESHOLD and child_dist_B < POP_DISTANCE_THRESHOLD):
                continue

            child_real_before = get_real_index(child)
            parent_real = min(get_real_index(A), get_real_index(B))

            # Improve fuerte solo si el hijo ya es prometedor
            if child_real_before <= parent_real + 8:
                child = improve(child, graph, global_edge_load, k=7, max_moves=15)
            else:
                child = improve(child, graph, global_edge_load, k=5, max_moves=6)

            # Filtrar hijos antes de considerar inserción, para evitar saturar población con soluciones similares o peores
            child_real = get_real_index(child)
            parent_real = min(get_real_index(A), get_real_index(B))
            parent_cost = min(A.cost, B.cost)

            ACCEPT_MARGIN = 0.02   # 2% de margen de aceptación para soluciones con mismo real_index pero peor costo

            # Firma de congestion para comparar patrones de congestión, no solo el máximo
            parent_sig = min(congestion_signature(A, limit=10), congestion_signature(B, limit=10))

            child_sig = congestion_signature(child, limit=10)

            if child_real < parent_real:
                new_solutions.append(child)

            elif child_real == parent_real:
                if child.cost <= parent_cost * 1.05:
                    new_solutions.append(child)

            elif child_real <= parent_real + 80:
                if improves_congestion(parent_sig, child_sig) or child.cost <= parent_cost * 1.03:
                    new_solutions.append(child)

        pr_children_count = len(new_solutions)
        strategic_children_added = 0

        MIN_USEFUL_CHILDREN = max(4, refset_size // 2)

        # =================================================
        # PERTURBACIÓN ESTRATÉGICA CONTROLADA
        # =================================================

        if (best_real_stagnation >= 4 and strategic_cooldown == 0):

            print("Pocos hijos útiles reales, agregando perturbación estratégica...")

            weak_child_streak = 0
            strategic_cooldown = 4

            quality_half = refset_size // 2

            parent = min(refset[:quality_half], key=lambda s: (get_real_index(s), s.cost))

            strategic_child = perturb_solution(parent, graph, num_changes=max(40, int(0.0002 * len(parent.routing))))

            strategic_child = improve(strategic_child, graph, global_edge_load, k=5, max_moves=8)

            parent_real = get_real_index(parent)
            child_real = get_real_index(strategic_child)

            if child_real < get_real_index(best):
                new_solutions.insert(0, strategic_child)
                strategic_children_added += 1

            elif (child_real == parent_real and strategic_child.cost < parent.cost):
                new_solutions.append(strategic_child)
                strategic_children_added += 1

        # Ordenar por calidad (real_index) y luego por costo para priorizar mejores soluciones
        new_solutions.sort(key=lambda s: (get_real_index(s), s.cost))

        MAX_CHILDREN_KEEP = max(6, refset_size)

        new_solutions = new_solutions[:MAX_CHILDREN_KEEP]

        print("Hijos PR/Improve:", pr_children_count)
        print("Hijos perturbados:", strategic_children_added)
        print("Hijos totales:", len(new_solutions))

        refset_replacements = 0
        MAX_REFSET_REPLACEMENTS = max(2, refset_size // 3)

        # =================================================
        # INSERTAR EN POBLACIÓN
        # =================================================

        inserted = 0
        accepted_children = []
        MAX_INSERTIONS_PER_ITER = max(2, refset_size // 3)

        for child in new_solutions:

            if inserted >= MAX_INSERTIONS_PER_ITER:
                break

            child_real = get_real_index(child)
            best_real = get_real_index(best)

            if child_real < best_real:

                population.append(child)
                accepted_children.append(child)
                inserted += 1

                quality_half = refset_size // 2

                worst_quality_idx = max(range(quality_half), key=lambda i: (get_real_index(refset[i]), refset[i].cost))

                refset[worst_quality_idx] = child
                refset_replacements += 1

                continue

            MIN_POP_DISTANCE = max(150, int(median_population_dist * 0.08))
            MAX_ACCEPT_GAP = max(60, best_real // 35)

            if child_real > best_real + MAX_ACCEPT_GAP:
                continue

            # ---------------------------------
            # Threshold adaptativo
            # ---------------------------------
            if child_real < best_real:
                effective_threshold = int(MIN_POP_DISTANCE * 0.7)

            elif child_real <= best_real + 2:
                effective_threshold = int(MIN_POP_DISTANCE * 0.85)

            else:
                effective_threshold = MIN_POP_DISTANCE

            # ---------------------------------
            # Distancia real a población
            # ---------------------------------
            min_dist = min_distance_to_set(child, population, threshold=effective_threshold)

            # ---------------------------------
            # FILTRO DE ACEPTACIÓN
            # ---------------------------------

            if child_real < best_real:
                pass

            elif child_real <= best_real + 3:
                if min_dist < int(effective_threshold * 0.75):
                    continue

            else:
                if min_dist < effective_threshold:
                    continue

            # ---------------------------------
            # Evitar duplicados entre hijos nuevos
            # ---------------------------------
            too_close = (min_distance_to_set(child, accepted_children, threshold=effective_threshold) < effective_threshold)

            if too_close:
                continue

            if VERBOSE_FILTERS:
                print(
                    "Min dist:",
                    min_dist,
                    "Threshold:",
                    effective_threshold
                )

            population.append(child)
            accepted_children.append(child)
            inserted += 1

            # ---------------------------------
            # 2. Intentar meterlo al RefSet
            # ---------------------------------
            quality_half = refset_size // 2

            child_real = get_real_index(child)

            # =================================================
            # 1. Intentar entrar en mitad de calidad
            # =================================================
            worst_quality_idx = max(
                range(quality_half),
                key=lambda i: (get_real_index(refset[i]), refset[i].cost)
            )

            worst_quality = refset[worst_quality_idx]

            best_refset_real = min(get_real_index(s) for s in refset)

            if VERBOSE_FILTERS:
                print(
                    "RefSet insert:",
                    "child =", child_real,
                    "best_ref =", best_refset_real,
                    "worst_quality =", get_real_index(worst_quality)
                )

            MAX_REAL_GAP = max(2, best_refset_real // 200)

            better_quality = (child_real <= best_refset_real + MAX_REAL_GAP
                and
                (child_real < get_real_index(worst_quality)
                or
                (child_real == get_real_index(worst_quality)
                and child.cost < worst_quality.cost)
                )
            )

            if better_quality and refset_replacements < MAX_REFSET_REPLACEMENTS:
                refset[worst_quality_idx] = child
                refset_replacements += 1
            else:
                # =================================================
                # 2. Intentar entrar en mitad diversa
                # =================================================
                min_child_dist = min(solution_distance(child, refset[i]) for i in range(quality_half, refset_size))

                diversity_scores = [
                    min(solution_distance(refset[i], refset[j]) for j in range(refset_size) if j != i)
                    for i in range(quality_half, refset_size)
                ]

                worst_div_local = min(range(len(diversity_scores)), key=lambda k: diversity_scores[k])

                worst_div_idx = quality_half + worst_div_local
                worst_div_score = diversity_scores[worst_div_local]

                if (min_child_dist > worst_div_score and child_real <= best_refset_real + MAX_REAL_GAP and refset_replacements < MAX_REFSET_REPLACEMENTS):
                    refset[worst_div_idx] = child
                    refset_replacements += 1

        print("Hijos insertados desde candidatos totales:", inserted)
        print(
            "Reemplazos en RefSet:",
            refset_replacements,
            "/",
            MAX_REFSET_REPLACEMENTS
        )

        # =================================================
        # CONTROL PARA NO ABUSAR DE PERTURBACIÓN
        # =================================================

        no_effective_insertion = (inserted == 0 and refset_replacements == 0)

        if no_effective_insertion:
            weak_child_streak += 1
        else:
            weak_child_streak = 0

        if strategic_cooldown > 0:
            strategic_cooldown -= 1

        # Conservar mejores soluciones en población para mantener diversidad y evitar saturar con soluciones similares
        population = sorted(population, key=lambda s: (get_real_index(s), s.cost))[:pop_size]

        # =================================================
        # ACTUALIZAR BEST
        # =================================================

        current_best = min(refset, key=lambda s: (get_real_index(s), s.cost))

        improved = False

        if (get_real_index(current_best) < get_real_index(best)):
            improved = True
        elif (get_real_index(current_best) == get_real_index(best) and current_best.cost < best.cost):
            improved = True

        if improved:

            best = current_best.copy()
            old_best_real = get_real_index(best)

            stagnation = 0
            diversifications_without_improvement = 0

            if get_real_index(best) < old_best_real:
                best_real_stagnation = 0
            else:
                best_real_stagnation += 1

            print(f"Iter {it+1}: Mejor = {best.cost:.4f}")
            print(f"Real index: {best.real_index}")
            print(sorted(best.edge_load.values())[:10])
            print(sorted(best.edge_load.values(), reverse=True)[:10])

            # Actualizar memoria global
            for e in global_edge_load:
                global_edge_load[e] *= 0.9

            for e, load in best.edge_load.items():
                global_edge_load[e] = (
                    global_edge_load.get(e, 0)
                    + 0.1 * load
                )

        else:
            stagnation += 1
            best_real_stagnation += 1
            print("Sin mejora.")

        # =================================================
        # DIVERSIFICACIÓN
        # =================================================

        if stagnation >= max_no_improve:

            print("Diversificando población...")

            diversifications_without_improvement += 1

            # Reemplazar peores 25%
            num_replace = pop_size // 4

            for i in range(num_replace):
                population[-(i+1)] = (build_initial_solution(graph, all_paths))

            print("Reconstruyendo RefSet...")

            refset = build_refset(population, refset_size)

            stagnation = 0
            perturbation_used = False

            print("Diversificaciones sin mejora: ", diversifications_without_improvement)

        # =================================================
        # CORTE
        # =================================================

        if (diversifications_without_improvement >= MAX_DIVERSIFICATIONS):
            print(
                "Demasiadas diversificaciones "
                "sin mejora. Terminando."
            )
            break
    
    for i in range(20):

        print(f"\nMejora final {i+1}/20 con perturbación...")

        p = perturb_solution(best, graph, num_changes=100)

        p = improve(p, graph, global_edge_load, k=5, max_moves=20)

        print(get_real_index(p))


    return best


# =========================
# MAIN
# =========================
if __name__ == "__main__":

    start = time.time()

    graph, pop, refset, iters = read_graph("test8.txt")

    print("========================Parámetros========================")
    print(f"Nodos: {len(graph.nodes())} | Población: {pop} | RefSet: {refset} | Iteraciones: {iters}")

    best = scatter_search(graph, pop, refset, iters)

    real_index = get_real_index(best)
    total_load = sum(best.edge_load.values())

    end = time.time()

    print("========================Resultados========================")
    print(f"Índice de Transmisión aproximado (real): {real_index}")
    print(f"Costo heurístico: {best.cost:.4f}")
    print(f"Carga total: {total_load}")
    print(f"Numero de pares: {len(best.routing)}")
    print(f"Tiempo de ejecución: {end - start:.4f} segundos")