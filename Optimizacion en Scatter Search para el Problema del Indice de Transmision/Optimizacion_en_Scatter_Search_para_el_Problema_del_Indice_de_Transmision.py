import sys
import random
import time
import numpy as np
import math
import heapq
from collections import defaultdict, deque


# =============================================================================
# PARÁMETROS GLOBALES
# =============================================================================

# Construcción base balanceada: entre K caminos mínimos aleatorios elige el menos
# congestionado. Genérica (no asume estructura), acerca el punto de partida al óptimo.
BALANCED_BASE_K = 10

# Cache de caminos BFS evitando aristas prohibidas: (origen, destino, aristas_prohibidas)
path_cache = {}
MAX_PATH_CACHE = 50000

# Cache de conjuntos de aristas por camino: clave = tupla del camino
PATH_EDGE_CACHE = {}
MAX_PATH_EDGE_CACHE = 200000

# =============================================================================
# PARÁMETROS DE ENTONACIÓN (perillas ajustables)
# =============================================================================
# Estas son las constantes fijas que regulan el comportamiento del algoritmo.
# Los valores que escalan con el tamaño del problema (número de iteraciones,
# refset_size, nodos, medianas de distancia) se derivan dentro de las funciones,
# ya que son adaptativos por diseño y no perillas para ajustar a mano.

# --- Construcción de la población inicial ---
# num_changes de las variantes según un sorteo: 50% suave, 35% medio, 15% agresivo
VARIANT_CHANGES = (1200, 2200, 3500)
VARIANT_PROBS = (0.50, 0.85)

# --- RefSet (élite + diversidad) ---
REFSET_MIN_QUALITY_DIST = 150      # distancia mínima forzada en la mitad de calidad
REFSET_RELAXED_QUALITY_DIST = 80   # umbral relajado si no hay suficientes diversas
REFSET_SAMPLE_SIZE = 20            # candidatas muestreadas por ronda en la mitad diversa
REFSET_SAME_REAL_PENALTY = 5       # penalización por repetir índice real en el RefSet

# --- Path Relinking ---
PR_MAX_MOVES = 100                 # tope de movimientos (cambios de ruta) por hijo
PR_PRE_CAP_FACTOR = 3              # pre-filtro de candidatos = PR_MAX_MOVES * factor
PR_CRITICAL_EDGE_SLACK = 2         # aristas "críticas": congestión >= (máximo - slack)
PR_RELIEF_GLOBAL_WEIGHT = 0.3      # peso de la memoria histórica en el alivio
PR_COST_GLOBAL_WEIGHT = 0.1        # peso de la memoria histórica en el costo de entrada
PR_TOLERANCE_DIVISOR = 100         # tolerancia sobre el techo del mejor padre, la base (ceiling // divisor)

# --- Búsqueda local (improve) ---
IMPROVE_GLOBAL_WEIGHT = 0.4        # peso de la congestión histórica al puntuar aristas
IMPROVE_MAX_ACCEPTED = 20          # cambios aceptados como máximo por pasada

# --- Memoria histórica de congestión (promedio móvil exponencial) ---
MEMORY_DECAY = 0.9                 # factor de envejecimiento de la información antigua
MEMORY_INCREMENT = 0.1             # peso de la congestión reciente

# --- Umbrales adaptativos del ciclo principal (coeficientes fijos) ---
POP_DIST_FACTOR = 0.04             # POP_DISTANCE_THRESHOLD = mediana_pob * factor
POP_DIST_FLOOR = 80
MIN_PR_DIST_FACTOR = 0.15          # MIN_PR_DISTANCE = mediana_refset * factor
MIN_PR_DIST_FLOOR = 200
POP_DEDUP_THRESHOLD = 30           # dedup de hijos contra el RefSet
STRATEGIC_COOLDOWN = 4             # iteraciones de enfriamiento tras una perturbación
CONVERGENCE_RATIO_THRESHOLD = 0.15 # por debajo de esto se reemplaza el 50% de la población



# =============================================================================
# UTILIDADES
# =============================================================================

def normalize_edge(u, v):
    """Retorna la arista en orden canónico para evitar duplicados (u,v) vs (v,u)."""
    return tuple(sorted((u, v)))


def real_transmission_index(solution):
    """Índice de transmisión real: congestión máxima sobre todas las aristas."""
    return max(solution.edge_load.values()) if solution.edge_load else 0


def get_real_index(solution):
    """Retorna el índice real cacheado en la solución, calculándolo si es necesario."""
    if solution.real_index is None:
        solution.real_index = real_transmission_index(solution)
    return solution.real_index


def path_to_edges(path):
    """
    Convierte un camino a un frozenset de aristas normalizadas.
    Usa cache para evitar recalcular caminos ya procesados.
    """
    key = tuple(path)

    cached = PATH_EDGE_CACHE.get(key)
    if cached is not None:
        return cached

    edges = frozenset(normalize_edge(path[i], path[i + 1]) for i in range(len(path) - 1))

    PATH_EDGE_CACHE[key] = edges
    if len(PATH_EDGE_CACHE) > MAX_PATH_EDGE_CACHE:
        PATH_EDGE_CACHE.clear()

    return edges


def congestion_signature(solution, limit=10):
    """Firma de congestión: tupla de las congestiones más altas, usada para comparar estados."""
    loads = sorted(solution.edge_load.values(), reverse=True)[:limit]
    return tuple(loads)


def improves_congestion(old_sig, new_sig):
    # True si la nueva firma de congestiones es lexicográficamente menor (menos congestión arriba).
    return new_sig < old_sig


# =============================================================================
# GRAFO
# =============================================================================

class Graph:
    """Grafo no dirigido representado con listas de adyacencia."""
    def __init__(self):
        self.adj = defaultdict(list)

    def add_edge(self, u, v):
        # Inserta la arista en ambos sentidos (no dirigido), evitando duplicados.
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)

    def nodes(self):
        return list(self.adj.keys())

    def neighbors(self, u):
        return self.adj[u]


# =============================================================================
# LECTURA DEL GRAFO
# =============================================================================

def read_graph(filename):
    """
    Lee el grafo y los parámetros de Scatter Search desde un archivo.
    Primera línea: pop_size refset_size iterations
    Líneas siguientes: nodo vecino1 vecino2 ...
    """
    g = Graph()

    with open(filename) as f:
        lines = f.readlines()

    # Primera línea: parámetros de Scatter Search
    pop_size, refset_size, iterations = map(int, lines[0].split())

    # Resto: cada línea es "nodo vecino1 vecino2 ..." -> se añaden las aristas
    for line in lines[1:]:
        parts = line.split()
        u = parts[0]
        for v in parts[1:]:
            g.add_edge(u, v)

    return g, pop_size, refset_size, iterations


# =============================================================================
# SOLUCIÓN
# =============================================================================

class Solution:
    """
    Representa una asignación de caminos a todos los pares origen-destino.

    Mantiene las congestiones de las aristas de forma incremental: update_pair() actualiza
    solo las aristas afectadas sin recalcular todo desde cero.

    Atributos clave:
        routing:       dict (u,v) -> lista de nodos del camino
        edge_load:     dict arista -> número de caminos que la usan
        edge_to_pairs: dict arista -> conjunto de pares que la usan
        max_load:      congestión máxima actual (= índice real de transmisión)
        real_index:    cache del índice real (None si está desactualizado)
    """

    def __init__(self, routing, graph):
        self.routing = routing
        self.graph = graph

        # Estadísticas de congestión (se mantienen incrementalmente en update_pair)
        self.edge_load = defaultdict(int)
        self.load_count = defaultdict(int)  # histograma: congestión -> nº de aristas con esa congestión
        self.sum_sq_load = 0
        self.max_load = 0
        self.edge_to_pairs = defaultdict(set)
        self.real_index = None
        # Procedencia para la atribución de mejoras (asignados por el ciclo principal)
        self._origin = "pr"
        self._pr_real_before = None
        self._pr_real_after = None

        # Poblar las estadísticas y el costo a partir del routing recibido
        self._initialize_edge_load()
        self.cost = self.compute_cost()

    def copy(self):
        """
        Clon de la solución con copia perezosa (copy-on-write).
        routing y edge_to_pairs se comparten por referencia y solo se duplican
        cuando update_pair los modifica por primera vez; un clon que nunca se
        muta no paga el costo O(P·L) de duplicar el enrutamiento.
        Las congestiones (edge_load, load_count) y los escalares sí se copian de una
        vez para poder evaluar el clon de forma independiente.
        graph se comparte (es inmutable).
        """
        new = Solution.__new__(Solution)

        # Compartidos hasta la primera mutación (update_pair los materializa)
        new.routing = self.routing
        new.edge_to_pairs = self.edge_to_pairs
        new.edge_load = self.edge_load.copy()
        new.load_count = self.load_count.copy()

        new.graph = self.graph

        new.sum_sq_load = self.sum_sq_load
        new.max_load = self.max_load
        new.real_index = self.real_index
        new.cost = self.cost
        # La procedencia se reinicia en cada clon (se re-asigna al generarlo)
        new._origin = "pr"
        new._pr_real_before = None
        new._pr_real_after = None

        new._routing_shared = True
        new._pairs_shared = True

        return new

    def _initialize_edge_load(self):
        """Calcula congestiones iniciales recorriendo todos los caminos del routing."""
        # Fase 1: contar cuántas cadenas cruzan cada arista (= vector de congestiones)
        for (u, v), path in self.routing.items():
            for i in range(len(path) - 1):
                e = normalize_edge(path[i], path[i + 1])
                self.edge_load[e] += 1
                self.edge_to_pairs[e].add((u, v))

        # Fase 2: agregar el vector en escalares (suma, suma de cuadrados, máximo, histograma)
        for load in self.edge_load.values():
            self.sum_sq_load += load * load
            self.load_count[load] += 1
            self.max_load = max(self.max_load, load)

    def update_pair(self, u, v, new_path):
        """
        Reemplaza el camino del par (u,v) actualizando las congestiones de forma incremental.
        Solo toca las aristas del camino viejo y del nuevo, no recalcula todo.
        Recalcula max_load solo si el máximo podría haber bajado (caso poco frecuente).
        """
        if getattr(self, "_routing_shared", False):
            self.routing = self.routing.copy()
            self._routing_shared = False

        if getattr(self, "_pairs_shared", False):
            self.edge_to_pairs = {e: pairs.copy() for e, pairs in self.edge_to_pairs.items()}
            self._pairs_shared = False

        old_path = self.routing[(u, v)]
        max_might_have_dropped = False

        # Restar congestión del camino viejo
        for i in range(len(old_path) - 1):
            e = normalize_edge(old_path[i], old_path[i + 1])
            old = self.edge_load[e]

            if old == self.max_load:
                max_might_have_dropped = True

            new = old - 1
            self.edge_load[e] = new
            self.edge_to_pairs[e].discard((u, v))
            self.sum_sq_load += new ** 2 - old ** 2

            self.load_count[old] -= 1
            if new > 0:
                self.load_count[new] += 1

            if new == 0:
                del self.edge_load[e]
                if not self.edge_to_pairs[e]:
                    del self.edge_to_pairs[e]

        # Sumar congestión del nuevo camino
        for i in range(len(new_path) - 1):
            e = normalize_edge(new_path[i], new_path[i + 1])
            old = self.edge_load.get(e, 0)
            new = old + 1

            self.edge_load[e] = new

            if e not in self.edge_to_pairs:
                self.edge_to_pairs[e] = set()
            self.edge_to_pairs[e].add((u, v))

            self.sum_sq_load += new ** 2 - old ** 2

            if old > 0:
                self.load_count[old] -= 1
            self.load_count[new] += 1

            if new > self.max_load:
                self.max_load = new

        self.routing[(u, v)] = new_path

        if max_might_have_dropped:
            # Baja max_load a la siguiente cubeta no vacía — O(descenso), no O(E)
            while self.max_load > 0 and self.load_count.get(self.max_load, 0) == 0:
                self.max_load -= 1

        self.cost = self.compute_cost()
        self.real_index = None

    def compute_cost(self):
        """
        Costo heurístico: suma de cuadrados de las congestiones.
        Se usa como desempate cuando dos soluciones tienen el mismo índice real.
        Minimizar esta suma distribuye la congestión más uniformemente.
        """
        if not self.edge_load:
            return float('inf')
        return self.sum_sq_load


# =============================================================================
# PRECOMPUTACIÓN DE CAMINOS
# =============================================================================

def bfs_all(graph, start):
    """BFS desde start. Retorna diccionario nodo -> padre para reconstruir caminos."""
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
    """Reconstruye el camino desde la raíz hasta v usando el diccionario de padres."""
    path = []
    while v is not None:
        path.append(v)
        v = parent[v]
    return path[::-1]


def precompute_all_pairs(graph):
    """
    Precomputa caminos más cortos para todos los pares dirigidos (u, v) con u ≠ v.
    Usa rutas dirigidas: C(u,v) y C(v,u) son independientes, permitiendo mayor
    flexibilidad para balancear la congestión y alcanzar el óptimo teórico.
    """
    all_paths = {}
    nodes = graph.nodes()

    # Un solo BFS por origen cubre los caminos mínimos a todos sus destinos
    for u in nodes:
        parent = bfs_all(graph, u)
        for v in nodes:
            if v != u:
                all_paths[(u, v)] = build_path(parent, v)

    return all_paths


# =============================================================================
# CAMINO MÁS CORTO EVITANDO ARISTAS (BFS CON CACHÉ)
# =============================================================================

def shortest_path_avoiding(graph, source, target, forbidden_edges):
    """
    BFS desde source hasta target evitando forbidden_edges.
    Resultado cacheado por (source, target, aristas_prohibidas).
    Retorna None si no existe camino.
    """
    key = (source, target, frozenset(forbidden_edges))

    # Hit de cache: reutiliza el resultado ya calculado para esta consulta
    if key in path_cache:
        return path_cache[key]

    queue = deque([source])
    parent = {source: None}

    while queue:
        u = queue.popleft()
        for v in graph.neighbors(u):
            edge = normalize_edge(u, v)
            if edge in forbidden_edges:   # arista bloqueada: no se expande por aquí
                continue
            if v not in parent:
                parent[v] = u
                if v == target:           # llegó al destino: reconstruye, cachea y retorna
                    path = build_path(parent, target)
                    path_cache[key] = path
                    if len(path_cache) > MAX_PATH_CACHE:
                        path_cache.clear()
                    return path
                queue.append(v)

    # Sin camino posible evitando esas aristas: cachea el fallo también
    path_cache[key] = None
    if len(path_cache) > MAX_PATH_CACHE:
        path_cache.clear()

    return None


# =============================================================================
# CONSTRUCCIÓN DE SOLUCIONES
# =============================================================================

def build_balanced_base(graph, all_paths, K=10):
    """
    Construcción balanceada genérica del enrutamiento inicial.

    Para cada par (s,t) genera K caminos mínimos aleatorios (descendiendo desde t
    por el gradiente de distancias del BFS de s) y se queda con el que menos
    congestiona las aristas ya congestionadas. A diferencia de asignar directamente el
    árbol BFS único —que concentraría la congestión—, este método la distribuye entre
    los múltiples caminos mínimos equivalentes, bajando el índice de partida.

    Es GENÉRICO: no asume hipercubo ni ninguna estructura, solo usa caminos
    mínimos, por lo que aplica a cualquier grafo conexo.
    """
    adj = {n: list(graph.neighbors(n)) for n in graph.nodes()}
    edge_load = defaultdict(int)
    routing = {}

    # Agrupar destinos por origen para reusar un único BFS de niveles por origen
    by_src = defaultdict(list)
    for (u, v) in all_paths:
        by_src[u].append(v)

    for s, targets in by_src.items():
        # BFS de niveles desde s: distancia de s a cada nodo (para el descenso por gradiente)
        dist = {s: 0}
        q = deque([s])
        while q:
            x = q.popleft()
            for y in adj[x]:
                if y not in dist:
                    dist[y] = dist[x] + 1
                    q.append(y)

        for t in targets:
            # Prueba K caminos mínimos y se queda con el menos congestionado
            best_path, best_max = None, None
            for _ in range(K):
                # Camino mínimo aleatorio: desde t hacia s por gradiente de dist
                path = [t]
                cur = t
                while cur != s:
                    cands = [y for y in adj[cur] if dist.get(y, -1) == dist[cur] - 1]
                    cur = random.choice(cands)
                    path.append(cur)
                path.reverse()
                # Congestión máxima que provocaría este camino sobre las aristas ya congestionadas
                mx = max((edge_load[normalize_edge(path[i], path[i + 1])]
                          for i in range(len(path) - 1)), default=0)
                if best_max is None or mx < best_max:
                    best_max, best_path = mx, path
            # Fija el mejor camino y acumula su congestión para influir en los siguientes pares
            routing[(s, t)] = best_path
            for i in range(len(best_path) - 1):
                edge_load[normalize_edge(best_path[i], best_path[i + 1])] += 1

    return Solution(routing, graph)


def build_initial_solution(graph, all_paths):
    """
    Genera una solución inicial diversa modificando aleatoriamente un subconjunto
    de caminos base. Para cada par seleccionado, prohíbe una arista aleatoria
    del camino original y busca el camino más corto alternativo.
    """
    routing = {}
    pairs = list(all_paths.keys())
    n_pairs = len(pairs)

    # Selecciona un subconjunto acotado de pares a los que se les cambiará la ruta
    max_alternatives = min(4000, n_pairs // 8)
    alt_pairs = set(random.sample(pairs, min(max_alternatives, n_pairs)))

    for (u, v), base_path in all_paths.items():
        # Los pares no seleccionados (o demasiado cortos) conservan su camino base
        if (u, v) not in alt_pairs or len(base_path) < 3:
            routing[(u, v)] = base_path
            continue

        # Prohíbe una arista aleatoria del camino base y busca una alternativa mínima
        idx = random.randint(0, len(base_path) - 2)
        forbidden = {normalize_edge(base_path[idx], base_path[idx + 1])}
        alt = shortest_path_avoiding(graph, u, v, forbidden)
        routing[(u, v)] = alt if alt else base_path

    return Solution(routing, graph)



def generate_initial_variant(base_solution, graph, num_changes=800):
    """
    Genera una variante de la solución base modificando num_changes pares.
    Más agresivo que build_initial_solution: usado para poblar la población inicial
    con soluciones estructuralmente variadas.
    """
    new_sol = base_solution.copy()
    pairs = list(new_sol.routing.keys())
    chosen_pairs = random.sample(pairs, min(num_changes, len(pairs)))

    for (u, v) in chosen_pairs:
        old_path = new_sol.routing[(u, v)]
        if len(old_path) < 2:
            continue

        # Prohíbe una arista aleatoria del camino actual y reencamina si hay alternativa distinta
        old_edges = list(path_to_edges(old_path))
        forbidden = {random.choice(old_edges)}
        new_path = shortest_path_avoiding(graph, u, v, forbidden)

        if new_path and new_path != old_path:
            new_sol.update_pair(u, v, new_path)

    return new_sol


# =============================================================================
# DISTANCIA ENTRE SOLUCIONES
# =============================================================================

def solution_distance(sol1, sol2, threshold=None):
    """
    Distancia estructural: número de pares con rutas distintas.
    Si threshold no es None, termina anticipadamente al alcanzar ese valor
    (útil para verificar si la distancia supera o no un umbral sin calcular todo).
    """
    diff = 0

    for k in sol1.routing:
        if sol1.routing[k] != sol2.routing[k]:
            diff += 1
            if threshold is not None and diff >= threshold:
                return diff

    return diff


def min_distance_to_set(candidate, solutions, threshold=None):
    """
    Distancia mínima de candidate a cualquier solución del conjunto.
    Termina anticipadamente si encuentra una distancia menor que threshold.
    """
    best = float("inf")

    for sol in solutions:
        d = solution_distance(candidate, sol, threshold=threshold)
        if d < best:
            best = d
        if threshold is not None and best < threshold:
            return best

    return best


def sampled_population_distances(population, sample_size=120):
    """
    Estima la distribución de distancias dentro de la población
    muestreando sample_size pares aleatorios.
    """
    n = len(population)
    if n < 2:
        return [0]

    all_pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    sample = random.sample(all_pairs, min(sample_size, len(all_pairs)))

    return [solution_distance(population[i], population[j]) for i, j in sample]


# =============================================================================
# REFSET (ÉLITE + DIVERSIDAD)
# =============================================================================

def build_refset(population, refset_size):
    """
    Construye el RefSet dividido en dos mitades:

    Mitad de calidad (n_best): mejores soluciones con diversidad mínima forzada
    (MIN_QUALITY_DIST=150); si no hay suficientes, se relaja el umbral (≥80) y en
    último caso sin restricción.

    Mitad diversa: selección greedy de las más distantes al RefSet actual, dentro de
    un margen de calidad (MAX_DIV_GAP) respecto al mejor.

    Así PR dispone de soluciones buenas y estructuralmente distintas a la vez.
    """
    population = sorted(population, key=lambda s: (get_real_index(s), s.cost))

    n_best = refset_size // 2

    # --- Mitad de calidad con diversidad forzada ---
    quality_part = [population[0]]
    for sol in population[1:]:
        if len(quality_part) >= n_best:
            break

        min_dist = min(
            solution_distance(sol, s, threshold=REFSET_MIN_QUALITY_DIST)
            for s in quality_part
        )
        if min_dist >= REFSET_MIN_QUALITY_DIST:
            quality_part.append(sol)

    # Relleno con distancia relajada si no hay suficientes soluciones diversas
    if len(quality_part) < n_best:
        for sol in population:
            if sol not in quality_part:
                min_d = min(solution_distance(sol, s, threshold=REFSET_RELAXED_QUALITY_DIST) for s in quality_part)
                if min_d >= REFSET_RELAXED_QUALITY_DIST:
                    quality_part.append(sol)
            if len(quality_part) >= n_best:
                break

    # Último recurso: relleno sin restricción de diversidad
    if len(quality_part) < n_best:
        for sol in population:
            if len(quality_part) >= n_best:
                break
            if sol not in quality_part:
                quality_part.append(sol)

    # --- Mitad diversa ---
    remaining = [s for s in population if s not in quality_part]
    diverse_part = []

    best_real_in_refset = get_real_index(quality_part[0])
    MAX_DIV_GAP = max(200, best_real_in_refset // 5)

    while len(quality_part) + len(diverse_part) < refset_size:
        ref_members = quality_part + diverse_part

        sample_size = min(REFSET_SAMPLE_SIZE, len(remaining))
        if sample_size == 0:
            break

        candidates = random.sample(remaining, sample_size)

        best_candidate = None
        best_score = -1

        for cand in candidates:
            cand_real = get_real_index(cand)

            # Descartar soluciones demasiado peores que el mejor del RefSet
            if cand_real > best_real_in_refset + MAX_DIV_GAP:
                continue

            # Puntaje: distancia mínima al RefSet penalizada por repetir índice real
            min_dist = min(solution_distance(cand, s, threshold=2000) for s in ref_members)
            same_real_count = sum(get_real_index(s) == cand_real for s in ref_members)
            score = min_dist - REFSET_SAME_REAL_PENALTY * same_real_count

            if score > best_score:
                best_score = score
                best_candidate = cand

        if best_candidate is None:
            break

        diverse_part.append(best_candidate)
        remaining.remove(best_candidate)

    return quality_part + diverse_part


# =============================================================================
# PATH RELINKING
# =============================================================================

def path_relinking(solA, solB, global_edge_load):
    """
    Genera un hijo combinando las rutas de solA y solB.

    Parte del mejor padre (base) y adopta pares del guía en orden de beneficio neto
    de congestión (relief al salir de aristas del base − cost al entrar las del guía),
    aceptando cada cambio solo si no supera el techo del peor padre + tolerancia.
    """
    if get_real_index(solA) <= get_real_index(solB):
        base, guide = solA, solB
    else:
        base, guide = solB, solA

    child = base.copy()

    # Solo se consideran los pares que cruzan aristas críticas (en el máximo o cerca):
    # cambiar la ruta de un par que no toca el máximo no puede bajar el índice real.
    max_load_val = child.max_load
    critical_edges = [e for e in child.edge_load if child.edge_load[e] >= max_load_val - PR_CRITICAL_EDGE_SLACK]

    # Cuenta de incidencias en aristas críticas: proxy barato de relevancia.
    # Un par que cruza más aristas críticas tiene más palanca sobre el máximo.
    pair_count = defaultdict(int)
    for e in critical_edges:
        for p in base.edge_to_pairs.get(e, set()):
            pair_count[p] += 1

    differing_pairs = [p for p in pair_count if base.routing[p] != guide.routing[p]]

    if not differing_pairs:
        return child

    MAX_PR_MOVES = PR_MAX_MOVES
    PRE_CAP = MAX_PR_MOVES * PR_PRE_CAP_FACTOR  # filtro barato antes del costoso pair_edges/scoring

    # Pre-filtro por incidencia antes de la costosa puntuación (si difieren mucho, D ~ decenas de miles).
    if len(differing_pairs) > PRE_CAP:
        differing_pairs = sorted(
            differing_pairs, key=lambda p: pair_count[p], reverse=True
        )[:PRE_CAP]

    # Precomputar aristas base y guía solo para el conjunto pre-capeado
    pair_edges = {}
    for pair in differing_pairs:
        pair_edges[pair] = (
            path_to_edges(child.routing[pair]),
            path_to_edges(guide.routing[pair]),
        )

    parent_ceiling = get_real_index(base)
    tolerance = max(1, parent_ceiling // PR_TOLERANCE_DIVISOR)

    def compute_score(pair):
        exclusive_out = pair_edges[pair][0] - pair_edges[pair][1]
        exclusive_in  = pair_edges[pair][1] - pair_edges[pair][0]
        relief = sum(child.edge_load.get(e, 0) + PR_RELIEF_GLOBAL_WEIGHT * global_edge_load.get(e, 0) for e in exclusive_out)
        cost   = sum(child.edge_load.get(e, 0) + PR_COST_GLOBAL_WEIGHT * global_edge_load.get(e, 0) for e in exclusive_in)
        return relief - cost

    # Tope final a los de mayor beneficio: acota la propagación de la repuntuación (que sería
    # O(D²) en soluciones balanceadas) y descarta movimientos marginales o dañinos.
    initial_scores = {pair: compute_score(pair) for pair in differing_pairs}
    if len(differing_pairs) > MAX_PR_MOVES:
        differing_pairs = sorted(
            differing_pairs, key=lambda p: initial_scores[p], reverse=True
        )[:MAX_PR_MOVES]

    remaining = set(differing_pairs)

    # Índice arista -> pares que la usan, para la repuntuación parcial acotada por MAX_PR_MOVES.
    edge_to_remaining = defaultdict(set)
    for pair in remaining:
        for e in pair_edges[pair][0] | pair_edges[pair][1]:
            edge_to_remaining[e].add(pair)

    cached_score = {pair: initial_scores[pair] for pair in remaining}

    # Heap con borrado perezoso: las entradas obsoletas se descartan al extraerlas
    # comparando con _valid_id. Acota el peor caso a O(D log D) si se eleva
    # PR_MAX_MOVES; con el tope actual un máximo lineal rinde igual.
    _counter = [0]
    _valid_id = {}

    def _heap_push(h, pair, score):
        _counter[0] += 1
        eid = _counter[0]
        _valid_id[pair] = eid
        heapq.heappush(h, (-score, eid, pair))

    heap = []
    for pair in remaining:
        _heap_push(heap, pair, cached_score[pair])

    while remaining:
        # Saca del heap el par de mayor beneficio, descartando entradas obsoletas (borrado perezoso)
        while heap:
            _, eid, pair = heapq.heappop(heap)
            if pair in remaining and _valid_id.get(pair) == eid:
                break
        else:
            break   # heap vacío de entradas válidas: no queda nada por procesar

        best_pair = pair
        remaining.remove(best_pair)
        _valid_id.pop(best_pair, None)

        # Adopta la ruta del guía para este par (el "movimiento" de PR)
        old_path = child.routing[best_pair]
        new_path = guide.routing[best_pair]

        child.update_pair(best_pair[0], best_pair[1], new_path)

        if child.max_load > parent_ceiling + tolerance:
            # El cambio superó el techo permitido: se revierte (ambos conjuntos de aristas cambiaron)
            child.update_pair(best_pair[0], best_pair[1], old_path)
            changed_edges = pair_edges[best_pair][0] | pair_edges[best_pair][1]
        else:
            # Cambio aceptado: solo las aristas que entraron o salieron (diferencia simétrica)
            changed_edges = pair_edges[best_pair][0] ^ pair_edges[best_pair][1]

        # Recalcula el beneficio de los pares afectados por las aristas que cambiaron
        to_rescore = set()
        for e in changed_edges:
            to_rescore.update(edge_to_remaining[e] & remaining)
        for pair in to_rescore:
            cached_score[pair] = compute_score(pair)
            _heap_push(heap, pair, cached_score[pair])

        for e in pair_edges[best_pair][0] | pair_edges[best_pair][1]:
            edge_to_remaining[e].discard(best_pair)

    return child


# =============================================================================
# BÚSQUEDA LOCAL (IMPROVE)
# =============================================================================

def _improve_one_pass(sol, graph, global_edge_load, max_moves=50):
    """
    Un paso de búsqueda local sobre la solución.

    Identifica las aristas más críticas (alta congestión actual + histórica),
    luego intenta reroutear los pares que las usan por caminos que eviten
    esas aristas. Acepta el cambio si mejora el índice real, el costo,
    o la firma de congestión de las top-10 aristas.

    Retorna el número de cambios aceptados.
    """
    # Puntaje combinado: congestión actual + contribución histórica
    score = {
        e: sol.edge_load[e] + IMPROVE_GLOBAL_WEIGHT * global_edge_load.get(e, 0)
        for e in sol.edge_load
    }

    critical = sorted(score, key=score.get, reverse=True)[:10]
    critical_set = set(critical)

    # Candidatos: pares que usan las aristas de congestión máxima
    max_edges = [e for e in sol.edge_load if sol.edge_load[e] == sol.max_load]
    candidate_pairs = set()
    for e in max_edges:
        candidate_pairs.update(sol.edge_to_pairs.get(e, set()))

    # Ampliar con pares de aristas críticas si hay pocos candidatos
    if len(candidate_pairs) < max_moves:
        for e in critical:
            candidate_pairs.update(sol.edge_to_pairs.get(e, set()))

    candidates = list(candidate_pairs)
    random.shuffle(candidates)
    candidates = candidates[:max_moves]

    accepted_changes = 0
    MAX_ACCEPTED = min(IMPROVE_MAX_ACCEPTED, max_moves)
    old_sig = congestion_signature(sol, limit=10)

    for (u, v) in candidates:
        # Prohíbe las aristas críticas que usa este par; si no cruza ninguna, se salta
        old_path = sol.routing[(u, v)]
        old_edges = path_to_edges(old_path)
        local_forbidden = old_edges & critical_set

        if not local_forbidden:
            continue

        old_real = sol.max_load
        old_cost = sol.cost

        # Aplica tentativamente el reencaminamiento evitando esas aristas críticas
        new_path = shortest_path_avoiding(graph, u, v, local_forbidden)

        if not new_path or new_path == old_path:
            continue

        sol.update_pair(u, v, new_path)

        new_real = sol.max_load

        # Criterio de aceptación en cascada: índice real > costo > firma de congestión
        if new_real < old_real:
            accept = True
            old_sig = congestion_signature(sol, limit=10)
        elif new_real == old_real and sol.cost < old_cost:
            accept = True
        elif new_real <= old_real + 1:   # tolera +1 en el máximo si mejora la firma top-10
            new_sig = congestion_signature(sol, limit=10)
            accept = improves_congestion(old_sig, new_sig)
            if accept:
                old_sig = new_sig
        else:
            accept = False

        if accept:
            accepted_changes += 1
            if accepted_changes >= MAX_ACCEPTED:
                break
        else:
            sol.update_pair(u, v, old_path)   # revierte el cambio no aceptado

    return accepted_changes


def improve(sol, graph, global_edge_load, max_moves=50, max_rounds=3):
    """
    Aplica múltiples pasadas de búsqueda local hasta que no haya mejoras
    o se alcance el límite de rondas.
    """
    for _ in range(max_rounds):
        changes = _improve_one_pass(sol, graph, global_edge_load, max_moves)
        if changes == 0:
            break
    return sol


# =============================================================================
# PERTURBACIÓN
# =============================================================================

def perturb_solution(solution, graph, num_changes=5):
    """
    Perturbación dirigida: reroutea num_changes pares que usan las aristas
    más congestionadas, prohibiendo esas aristas para forzar diversificación.
    Usada tanto en perturbación estratégica durante el loop como en
    post-procesamiento.
    """
    new_sol = solution.copy()

    # Las 10 aristas más congestionadas y los pares que las cruzan (candidatos a reencaminar)
    hot_edges = sorted(new_sol.edge_load, key=new_sol.edge_load.get, reverse=True)[:10]

    candidate_pairs = set()
    for e in hot_edges:
        candidate_pairs.update(new_sol.edge_to_pairs.get(e, set()))

    if not candidate_pairs:   # respaldo: si no hay aristas calientes, cualquier par sirve
        candidate_pairs = set(new_sol.routing.keys())

    chosen_pairs = random.sample(
        list(candidate_pairs),
        min(num_changes, len(candidate_pairs))
    )

    changes = 0
    for (u, v) in chosen_pairs:
        # Prohíbe las aristas calientes del camino actual y reencamina evitándolas
        old_path = new_sol.routing[(u, v)]
        forbidden = set(hot_edges) & path_to_edges(old_path)

        if not forbidden:
            continue

        new_path = shortest_path_avoiding(graph, u, v, forbidden)
        if new_path and new_path != old_path:
            new_sol.update_pair(u, v, new_path)
            changes += 1

    print(f"Perturbación: {changes} pares modificados (de {len(candidate_pairs)} conflictivos)")
    return new_sol


# =============================================================================
# SCATTER SEARCH PRINCIPAL
# =============================================================================

def scatter_search(graph, pop_size, refset_size, iterations):
    """
    Algoritmo principal de Scatter Search para minimizar el índice de transmisión.

    Flujo por iteración:
      1. Calcular distancias de población y RefSet (muestreadas) para thresholds adaptativos
      2. Mejorar en-place la mitad de calidad del RefSet
      3. Path Relinking entre pares del RefSet para generar hijos
      4. Insertar hijos en población (memoria extendida) si son diversos del RefSet actual
      5. Reconstruir RefSet desde población
      6. Actualizar best si hubo mejora
      7. Diversificar si hay estancamiento prolongado
    """

    # --- Precomputación e inicialización ---
    print("Precomputando caminos...")
    all_paths = precompute_all_pairs(graph)

    # Construcción base balanceada: primer elemento de la población; el resto se genera como variantes.
    print("Construyendo base balanceada genérica...")
    base_solution = build_balanced_base(graph, all_paths, K=BALANCED_BASE_K)
    print(f"Base inicial: indice={get_real_index(base_solution)}")

    population = [base_solution]

    print("Construyendo población...")
    for _ in range(pop_size - len(population)):
        r = random.random()
        if r < VARIANT_PROBS[0]:
            changes = VARIANT_CHANGES[0]
        elif r < VARIANT_PROBS[1]:
            changes = VARIANT_CHANGES[1]
        else:
            changes = VARIANT_CHANGES[2]
        population.append(generate_initial_variant(base_solution, graph, num_changes=changes))

    # Mejorar población inicial usando la congestión acumulada como guía global
    print("Mejorando población inicial...")
    seed_edge_load = defaultdict(int)
    for sol in population:
        for e, load in sol.edge_load.items():
            seed_edge_load[e] += load
    for sol in population:
        improve(sol, graph, seed_edge_load, max_moves=20, max_rounds=2)

    print("Construyendo RefSet...")
    refset = build_refset(population, refset_size)

    best = min(refset, key=lambda s: (get_real_index(s), s.cost)).copy()
    print(f"Iteración 1: real={get_real_index(best)} cost={best.cost:.4f}")

    # global_edge_load: memoria histórica de congestión mediante promedio móvil exponencial;
    # se actualiza al mejorar best, envejeciendo la información previa (MEMORY_DECAY).
    global_edge_load = {e: load for e, load in best.edge_load.items()}

    stagnation = 0
    diversifications_without_improvement = 0
    MAX_DIVERSIFICATIONS = max(3, min(6, int(math.log2(len(graph.nodes()))) - 3))
    # Disparadores de escape escalados al presupuesto de iteraciones (no a los nodos),
    # para que perturbación y diversificación tengan una ventana proporcional al correr
    # 15, 20 o más iteraciones.
    PERTURB_THRESHOLD = max(2, iterations // 5)   # 15->3, 20->4
    max_no_improve = max(3, iterations // 3)       # 15->5, 20->6
    MAX_PR_PAIRS = max(15, min(40, refset_size * 3))
    strategic_cooldown = 0

    # =========================================================================
    # LOOP PRINCIPAL
    # =========================================================================
    for it in range(iterations):

        print(f"\n--- Iteración {it + 1} ---")

        # --- Umbrales adaptativos ---

        # Distancia mediana de población (muestreada): base para POP_DISTANCE_THRESHOLD
        population_distances = sampled_population_distances(population, sample_size=120)
        median_population_dist = np.median(population_distances)
        POP_DISTANCE_THRESHOLD = max(POP_DIST_FLOOR, int(median_population_dist * POP_DIST_FACTOR))

        # Distancia mediana del RefSet (muestreada): base para MIN_PR_DISTANCE
        all_refset_pairs = [
            (i, j)
            for i in range(len(refset))
            for j in range(i + 1, len(refset))
        ]
        sample_size_refset = min(len(all_refset_pairs), max(30, len(all_refset_pairs) // 3))

        if len(all_refset_pairs) > sample_size_refset:
            sample_refset_pairs = random.sample(all_refset_pairs, sample_size_refset)
        else:
            sample_refset_pairs = all_refset_pairs

        distances_refset = [
            solution_distance(refset[i], refset[j])
            for i, j in sample_refset_pairs
        ]
        median_refset_dist = np.median(distances_refset)
        MIN_PR_DISTANCE = max(MIN_PR_DIST_FLOOR, int(median_refset_dist * MIN_PR_DIST_FACTOR))

        print(f"Median Population dist: {int(median_population_dist)} | POP_DISTANCE_THRESHOLD: {POP_DISTANCE_THRESHOLD}")
        print(f"Median RefSet dist: {int(median_refset_dist)} | MIN_PR_DISTANCE: {MIN_PR_DISTANCE}")

        # Registro del estado ANTES de mejorar la mitad de calidad: cualquier mejora posterior
        # (ya sea por esa mejora o por PR) debe superar esta línea base.
        prev_best_real = get_real_index(best)
        prev_best_cost = best.cost

        # --- Preparar padres para PR: copias mejoradas de la mitad de calidad ---
        # Se mejoran copias en lugar de los miembros originales del RefSet: así la mejora
        # de la mitad de calidad no altera soluciones que luego podrían ser el current_best.
        quality_half = refset_size // 2
        pr_parents = []
        for idx in range(len(refset)):
            if idx < quality_half:
                parent = refset[idx].copy()
                improve(parent, graph, global_edge_load, max_moves=10, max_rounds=1)
                pr_parents.append(parent)
            else:
                pr_parents.append(refset[idx])

        # --- Path Relinking ---
        new_solutions = []
        pr_pairs = [(i, j) for i in range(len(pr_parents)) for j in range(i + 1, len(pr_parents))]
        random.shuffle(pr_pairs)

        for (i, j) in pr_pairs[:min(MAX_PR_PAIRS, len(pr_pairs))]:
            parent_a = pr_parents[i]
            parent_b = pr_parents[j]

            if solution_distance(parent_a, parent_b, threshold=MIN_PR_DISTANCE) < MIN_PR_DISTANCE:
                continue

            child = path_relinking(parent_a, parent_b, global_edge_load)

            child_dist_a = solution_distance(child, parent_a, threshold=POP_DISTANCE_THRESHOLD)
            child_dist_b = solution_distance(child, parent_b, threshold=POP_DISTANCE_THRESHOLD)
            if child_dist_a < POP_DISTANCE_THRESHOLD and child_dist_b < POP_DISTANCE_THRESHOLD:
                continue

            parent_real = min(get_real_index(parent_a), get_real_index(parent_b))
            child_real_before_improve = get_real_index(child)
            child._pr_real_before = child_real_before_improve

            if child_real_before_improve > prev_best_real + 30:
                continue

            if child_real_before_improve <= prev_best_real:
                rounds = 3
            elif child_real_before_improve <= prev_best_real + 15:
                rounds = 2
            else:
                rounds = 1

            child = improve(
                child, graph, global_edge_load,
                max_moves=10 if child_real_before_improve <= parent_real + 8 else 6,
                max_rounds=rounds
            )
            child._pr_real_after = get_real_index(child)
            new_solutions.append(child)

        pr_count = len(new_solutions)

        # --- Perturbación estratégica si hay estancamiento ---
        if stagnation >= PERTURB_THRESHOLD and strategic_cooldown == 0:
            print("Estancamiento detectado, perturbación estratégica...")
            strategic_cooldown = STRATEGIC_COOLDOWN
            parent = min(refset[:quality_half], key=lambda s: (get_real_index(s), s.cost))
            perturbed = perturb_solution(parent, graph, num_changes=max(100, int(0.003 * len(parent.routing))))
            perturbed = improve(perturbed, graph, global_edge_load, max_moves=20)
            perturbed._origin = "perturbacion"
            new_solutions.insert(0, perturbed)

        new_solutions.sort(key=lambda s: (get_real_index(s), s.cost))
        print(f"Hijos PR: {pr_count} | Total (inc. perturbación): {len(new_solutions)}")

        # --- Insertar hijos en población (memoria extendida) ---
        # Descarte de duplicados contra el RefSet actual, no contra toda la población: como los hijos de PR
        # son cercanos a sus padres (que están en población), verificar contra toda la
        # población rechazaba casi todo y frenaba la evolución.
        inserted_pr = 0
        inserted_perturb = 0
        best_real = get_real_index(best)
        MAX_ACCEPT_GAP = max(60, best_real // 35)

        for child in new_solutions:
            if get_real_index(child) > best_real + MAX_ACCEPT_GAP:
                continue
            if min_distance_to_set(child, refset, threshold=POP_DEDUP_THRESHOLD) < POP_DEDUP_THRESHOLD:
                continue
            population.append(child)
            if getattr(child, "_origin", "pr") == "perturbacion":
                inserted_perturb += 1
            else:
                inserted_pr += 1

        print(f"Insertados en población: {inserted_pr + inserted_perturb} (PR={inserted_pr}, perturbación={inserted_perturb})")

        if strategic_cooldown > 0:
            strategic_cooldown -= 1

        # --- Trim población y reconstruir RefSet ---
        # El RefSet refleja siempre el estado actual de la población:
        # mitad de calidad = mejores diversos, mitad diversa = más distantes cerca del óptimo.
        population = sorted(population, key=lambda s: (get_real_index(s), s.cost))[:pop_size]
        refset = build_refset(population, refset_size)

        current_best = min(refset, key=lambda s: (get_real_index(s), s.cost))
        current_real = get_real_index(current_best)
        improved = (
            current_real < prev_best_real
            or (current_real == prev_best_real and current_best.cost < prev_best_cost)
        )

        if improved:
            best = current_best.copy()
            stagnation = 0
            diversifications_without_improvement = 0

            # Atribución del origen de la mejora: perturbación, PR puro, PR+Improve,
            # o diversificación (solución sin provenance, típicamente inyectada por diversificación).
            origin = getattr(current_best, "_origin", "pr")
            before = getattr(current_best, "_pr_real_before", None)
            after = getattr(current_best, "_pr_real_after", None)
            if origin == "perturbacion":
                attribution = f"perturbación (real={current_real})"
            elif before is not None:
                if before < prev_best_real:
                    attribution = f"PR puro (antes={before} -> después={after})"
                else:
                    attribution = f"PR+Improve (PR dejó={before}, improve bajó a {after})"
            else:
                attribution = f"diversificación/desconocido (real={current_real})"

            # Improve profundo del nuevo best
            real_before_deep = get_real_index(best)
            improve(best, graph, global_edge_load, max_moves=30, max_rounds=3)
            real_after_deep = get_real_index(best)
            deep_note = f" +deep={real_after_deep}" if real_after_deep < real_before_deep else ""

            print(f"Iter {it + 1}: Mejor real={real_after_deep} cost={best.cost:.4f} [{attribution}{deep_note}]")
            print(f"  Congestión min (10): {sorted(best.edge_load.values())[:10]}")
            print(f"  Congestión max (10): {sorted(best.edge_load.values(), reverse=True)[:10]}")

            # Actualizar memoria histórica de congestión con decay exponencial
            for e in global_edge_load:
                global_edge_load[e] *= MEMORY_DECAY
            for e, load in best.edge_load.items():
                global_edge_load[e] = global_edge_load.get(e, 0) + MEMORY_INCREMENT * load

            # Reinsertar best mejorado en población para la siguiente iteración
            population.append(best.copy())

        else:
            stagnation += 1
            print(f"Sin mejora. real={get_real_index(best)} cost={best.cost:.4f}")

        # --- Diversificación ante estancamiento prolongado ---
        if stagnation >= max_no_improve:
            print("Diversificando población...")
            diversifications_without_improvement += 1

            convergence_ratio = median_population_dist / max(1, len(best.routing))
            num_replace = pop_size // 2 if convergence_ratio < CONVERGENCE_RATIO_THRESHOLD else pop_size // 4

            if convergence_ratio < CONVERGENCE_RATIO_THRESHOLD:
                print(f"Población convergida (ratio={convergence_ratio:.3f}), reemplazando 50%")

            for i in range(num_replace):
                population[-(i + 1)] = build_initial_solution(graph, all_paths)

            refset = build_refset(population, refset_size)
            stagnation = 0

            print(f"Diversificaciones sin mejora: {diversifications_without_improvement}")

        if diversifications_without_improvement >= MAX_DIVERSIFICATIONS:
            print("Demasiadas diversificaciones sin mejora. Terminando.")
            break

    return best


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":

    start = time.time()

    input_file = sys.argv[1] if len(sys.argv) > 1 else "test16.txt"
    graph, pop, refset, iters = read_graph(input_file)

    n = len(graph.nodes())
    print("========================Parámetros========================")
    print(f"Archivo: {input_file}")
    print(f"Nodos: {n} | Población: {pop} | RefSet: {refset} | Iteraciones: {iters}")

    best = scatter_search(graph, pop, refset, iters)

    real_index = get_real_index(best)
    total_load = sum(best.edge_load.values())
    end = time.time()

    print("========================Resultados========================")
    print(f"Índice de Transmisión aproximado (real): {real_index}")
    print(f"Costo heurístico: {best.cost:.4f}")
    print(f"Congestión total: {total_load}")
    print(f"Numero de pares: {len(best.routing)}")
    print(f"Tiempo de ejecución: {end - start:.4f} segundos")