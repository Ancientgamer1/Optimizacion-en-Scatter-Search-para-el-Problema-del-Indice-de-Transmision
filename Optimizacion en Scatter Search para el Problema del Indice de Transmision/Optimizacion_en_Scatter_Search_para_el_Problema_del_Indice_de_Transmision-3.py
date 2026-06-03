import random
import itertools
import time
import copy
from collections import defaultdict, deque
from heapq import heappush, heappop

# =========================
# PARÁMETROS DE CONTROL
# =========================
ALPHA_DEFAULT = 0.5
BETA_DEFAULT = 0.3
CONGESTION_THRESHOLD = 10
IMPROVE_PROB = 0.3

# =========================
# UTILIDADES 
# =========================
def normalize_edge(u, v):
    return tuple(sorted((u, v)))

def is_valid_solution(solution):
    return solution is not None and solution.routing and len(solution.routing) > 0

# =========================
# GRAFO
# =========================
class Graph:
    def __init__(self):
        self.adj = defaultdict(list) # Lista de adyacencia: nodo -> lista de vecinos

     # Grafo no dirigido: Ejes se agregan en ambos sentidos
    def add_edge(self, u, v):
        if v not in self.adj[u]:
            self.adj[u].append(v)
        if u not in self.adj[v]:
            self.adj[v].append(u)

    def nodes(self):
        return list(self.adj.keys()) # Retorna lista de nodos

    def neighbors(self, u):
        return self.adj[u] # Retorna los vecinos de un nodo


# =========================
# SOLUCIÓN
# =========================
class Solution:
    def __init__(self, routing, alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT):  #Representacion de una solucíon
        self.routing = routing
        self.alpha = alpha
        self.beta = beta
        self.edge_load = self.compute_edge_load()          # Diccionario: (u, v) -> cantidad de rutas que pasan por esa arista
        self.cost = self.compute_cost()     # Costo de la solución (Índice de Transmisión) calculado a partir de la carga máxima en cualquier arista

    def copy(self):
        return copy.deepcopy(self)

    def compute_edge_load(self):        # Calcula la carga de cada arista contando cuántas rutas pasan por ella en la solución actual
        e_load = defaultdict(int)

        for path in self.routing.values():      
            for i in range(len(path) - 1):
                edge = normalize_edge(path[i], path[i + 1])     # Normaliza la arista para asegurar consistencia en el conteo de carga (u, v) es lo mismo que (v, u)
                e_load[edge] += 1

        return e_load

    def compute_cost(self):
        loads = list(self.edge_load.values())

        if not loads:
            return float('inf')

        if len(loads) == 0:
            return float('inf')

        max_load = max(loads)
        mean = sum(loads) / len(loads)

        std = (sum((x - mean)**2 for x in loads) / len(loads))**0.5     # Desviación estándar para penalizar soluciones con carga muy desigual (bottlenecks)

        return max_load + 0.3 * std         # Costo que combina la carga máxima con una penalización por desigualdad en la distribución de la carga (clave para evitar soluciones con bottlenecks extremos)

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
# INDICE DE TRANSMISIÓN REAL (COSTO REAL)
# =========================
def real_transmission_index(solution):
    return max(solution.edge_load.values()) if solution.edge_load else 0

# =========================
# MUESTREO INTELIGENTE
# =========================
def intelligent_sampling(graph, sample_size=800):
    nodes = list(graph.nodes())
    n = len(nodes)

    pairs = set()
    
    # Identificar hubs (nodos con alta conectividad) para asegurar que estén representados en el muestreo
    degree = {u: len(graph.neighbors(u)) for u in nodes}
    sorted_nodes = sorted(nodes, key=lambda x: degree[x], reverse=True)
    hubs = sorted_nodes[:max(2, int(0.15 * n))]

    # Deteccion simple de la estructura del grafo para ajustar las cuotas de muestreo
    max_deg = max(degree.values())
    avg_deg = sum(degree.values()) / n
    degree_values = list(degree.values())
    degree_variation = max(degree_values) - min(degree_values)

    # Parametros dinámicos basados en la estructura del grafo
    is_star_like = max_deg > 0.6 * n
    is_sparse = avg_deg < 3
    is_regular = degree_variation == 0

    if is_star_like:
        hub_ratio = 0.25
        far_ratio = 0.25
        cover_ratio = 0.20

    elif is_sparse:
        hub_ratio = 0.20
        far_ratio = 0.35
        cover_ratio = 0.25

    elif is_regular:
        hub_ratio = 0.15
        far_ratio = 0.30
        cover_ratio = 0.25

    else:
        hub_ratio = 0.35
        far_ratio = 0.25
        cover_ratio = 0.20

    hub_quota = int(sample_size * hub_ratio)
    far_quota = int(sample_size * far_ratio)
    cover_quota = int(sample_size * cover_ratio)

    random_quota = sample_size - (hub_quota + far_quota + cover_quota)

    # ---------------- HUBS ----------------
    while len(pairs) < hub_quota:
        u = random.choice(hubs)
        v = random.choice(nodes)
        if u != v:
            pairs.add((u, v))

    # ---------------- LEJANOS ----------------
    for _ in range(far_quota):
        start = random.choice(nodes)

        visited = {start}
        queue = deque([(start, 0)])
        farthest = (start, 0)

        while queue:
            u, d = queue.popleft()
            if d > farthest[1]:
                farthest = (u, d)

            for v in graph.neighbors(u):
                if v not in visited:
                    visited.add(v)
                    queue.append((v, d+1))

        if start != farthest[0]:
            pairs.add((start, farthest[0]))

    # ---------------- COBERTURA ----------------
    k = max(1, cover_quota // n)
    for u in nodes:
        for _ in range(k):
            v = random.choice(nodes)
            if u != v:
                pairs.add((u, v))

    # ---------------- RANDOM (USANDO random_quota) ----------------
    attempts = 0
    while len(pairs) < sample_size and attempts < random_quota * 5:
        u = random.choice(nodes)
        v = random.choice(nodes)
        if u != v:
            pairs.add((u, v))
        attempts += 1

    return list(pairs)

# =========================
# SAMPLING SESGADO (EVITANDO ARISTAS CON ALTA CARGA GLOBAL)
# =========================
def biased_sampling(graph, global_edge_load, sample_size):
    nodes = list(graph.nodes())

    # Score de nodo: inverso de la carga de sus aristas
    node_score = {}
    for u in nodes:
        total = 0
        for v in graph.neighbors(u):
            e = (u, v) if (u, v) in global_edge_load else (v, u)
            total += global_edge_load.get(e, 0)
        node_score[u] = 1 / (1 + total)

    # Normalizar probabilidades
    total_score = sum(node_score.values())
    probs = [node_score[u] / total_score for u in nodes]

    pairs = set()

    while len(pairs) < sample_size:
        u = random.choices(nodes, weights=probs, k=1)[0]
        v = random.choices(nodes, weights=probs, k=1)[0]

        if u != v:
            pairs.add((u, v))

    return list(pairs)

# =========================
# SAMPLING HÍBRIDO (INTELIGENTE + SESGADO)
# =========================
def hybrid_sampling(graph, global_edge_load, sample_size, bias_ratio=0.2):
    base_size = int(sample_size * (1 - bias_ratio))
    bias_size = sample_size - base_size

    base_pairs = intelligent_sampling(graph, base_size)
    biased_pairs = biased_sampling(graph, global_edge_load, bias_size)

    return base_pairs + biased_pairs

# =========================
# BFS SIMPLE 
# =========================
def bfs(graph, start, end):
    # Cola para BFS
    queue = deque([[start]])
    visited = set([start])

    # BFS clásico para encontrar un camino entre start y end
    while queue:
        path = queue.popleft()
        u = path[-1]

        if u == end:
            return path

        for v in graph.neighbors(u):
            if v not in visited:
                visited.add(v)
                queue.append(path + [v])

    return None

# =========================
# BFS CON CONGESTIÓN (INTENSIFICACIÓN)
# =========================
def bfs_congestion(graph, source, target, edge_load, global_edge_load, alpha=0.7, beta=BETA_DEFAULT):

    # Cola de prioridad para BFS con costo basado en la carga de las aristas
    pq = [(0, source, [source])]
    best_cost = {}

    # Mientras haya nodos por explorar, expandir el nodo con el menor costo acumulado (considerando la congestión local y global)
    while pq:
        cost, node, path = heappop(pq)

        if node == target:
            return path

        if node in best_cost and best_cost[node] <= cost:
            continue
        
        # Registrar el mejor costo para llegar a este nodo para evitar ciclos y expansiones innecesarias
        best_cost[node] = cost

        # Expandir vecinos considerando la carga de las aristas para guiar la búsqueda hacia rutas menos congestionadas
        for nei in graph.neighbors(node):
            edge = normalize_edge(node, nei)

            # Acceso directo (rápido)
            local = edge_load.get(edge, 0)
            global_l = global_edge_load.get(edge, 0)

            # Costo que combina la carga local de la arista con una penalización por congestión global para evitar saturar ciertas rutas
            new_cost = cost + alpha + beta * (local + global_l)

            # Solo expandir si el nuevo costo es mejor que cualquier costo previamente registrado para ese vecino
            heappush(pq, (new_cost, nei, path + [nei]))

    return None


# =========================
# GENERAR SOLUCIÓN
# =========================
def generate_solution(graph, pairs, alpha, beta, global_edge_load):
    # Genera una solución inicial para un conjunto de pares (u, v) utilizando una mezcla de BFS simple y BFS con congestión.
    routing = {}
    edge_load = defaultdict(int)

    for (u, v) in pairs:
        # Mezcla aleatoria para diversidad inicial (clave para evitar soluciones muy similares)
        if random.random() < IMPROVE_PROB:
            path = bfs(graph, u, v)
        else:
            path = bfs_congestion(graph, u, v, edge_load, global_edge_load, alpha, beta)
        
        # Solo agregar rutas válidas (u y v) con al menos un nodo
        if path and len(path) > 1:
            routing[(u, v)] = path

            # Actualizar carga de aristas
            for i in range(len(path) - 1):
                e = normalize_edge(path[i], path[i+1])
                edge_load[e] += 1

    return Solution(routing, alpha, beta)


# =========================
# DISTANCIA ENTRE SOLUCIONES (DIVERSIDAD)
# =========================
def solution_distance(s1, s2):
    e1 = s1.edge_load
    e2 = s2.edge_load

    dist = 0

    # Elegir el menor para iterar primero
    if len(e1) > len(e2):
        e1, e2 = e2, e1

    visited = set()

    # Sumar diferencias en cargas para aristas presentes en ambos
    for edge, load1 in e1.items():
        load2 = e2.get(edge, 0)
        dist += abs(load1 - load2)
        visited.add(edge)
    
    # Sumar cargas de aristas únicas en e2
    for edge, load2 in e2.items():
        if edge not in visited:
            dist += load2

    return dist


# =========================
# COMBINACIÓN
# =========================
def combine(parent1, parent2, graph, global_edge_load):

    new_routing = {}
    new_edge_load = defaultdict(int)

    # Combinar rutas de ambos padres con una mezcla aleatoria para fomentar diversidad
    for (u, v) in parent1.routing:

        # Selección base
        if random.random() < IMPROVE_PROB:
            path = parent1.routing[(u, v)]
        else:
            path = parent2.routing.get((u, v), parent1.routing[(u, v)])

        path = list(path)

        # Solo explorar si hay congestión real
        congested = False
        
        for i in range(len(path) - 1):
            e = normalize_edge(path[i], path[i+1])
            if global_edge_load.get(e, 0) > 10:
                congested = True
                break

        # Si la ruta heredada está congestionada, intentar encontrar una mejor alternativa con BFS congestionado
        if congested and random.random() < 0.3:
            alt = bfs_congestion(graph, u, v, new_edge_load, global_edge_load, parent1.alpha, parent1.beta)
            
            if alt:
                path = alt

        new_routing[(u, v)] = path

        # Actualizar carga de aristas para la nueva ruta combinada
        for i in range(len(path) - 1):
            e = normalize_edge(path[i], path[i+1])
            new_edge_load[e] += 1

    return Solution(new_routing, parent1.alpha, parent1.beta)

# =========================
# MEJORA
# =========================
def improve(solution, graph, global_edge_load, k=10):

    solution = solution.copy()
    routing = solution.routing
    edge_load = solution.edge_load

    # Top edges más cargados
    critical_edges = sorted(edge_load.items(), key=lambda x: x[1], reverse=True)[:k]

    # Solo considerar edges críticos para la mejora local para enfocar el esfuerzo en zonas problemáticas (bottlenecks)
    critical_edges = [e for e, _ in critical_edges]

    # Intentar mejorar rutas que pasen por edges críticos con una probabilidad controlada para evitar cambios excesivos que puedan destruir la estructura de la solución
    for (u, v), path in routing.items():

        # Solo si usa edges críticos
        if not any(normalize_edge(path[i], path[i+1]) in critical_edges
                   for i in range(len(path) - 1)):
            continue

        # Solo intentar mejorar con cierta probabilidad para evitar cambios innecesarias en rutas que ya son buenas o no están causando congestión
        if random.random() < IMPROVE_PROB:
            new_path = bfs_congestion(graph, u, v, edge_load, global_edge_load, solution.alpha, solution.beta)

            if new_path and (len(new_path) <= len(path) or random.random() < 0.1):
                routing[(u, v)] = new_path

    return solution


def kick_solution(solution, graph, global_edge_load, k=10, reroute_frac=0.4):

    routing = solution.routing
    edge_load = solution.edge_load

    # Identificar edges críticos (mezcla local + global)
    critical_edges = sorted(edge_load.items(), key=lambda x: x[1] + 0.5 * global_edge_load[x[0]], reverse=True)[:k]

    critical_edges = set(e for e, _ in critical_edges)

    # Seleccionar pares afectados
    affected_pairs = []

    for (u, v), path in routing.items():
        if any(normalize_edge(path[i], path[i+1]) in critical_edges
               for i in range(len(path) - 1)):
            affected_pairs.append((u, v))

    # Seleccionar subconjunto a reroutear
    num_reroute = max(1, int(len(affected_pairs) * reroute_frac))
    selected_pairs = random.sample(affected_pairs, min(num_reroute, len(affected_pairs)))

    # Crear copia
    new_routing = dict(routing)

    # Reroute forzado (evitando edges críticos)
    for (u, v) in selected_pairs:

        new_path = bfs_congestion(graph, u, v, edge_load, global_edge_load, 0.7, BETA_DEFAULT)

        if new_path:
            new_routing[(u, v)] = new_path

    # Reconstruir solución
    return Solution(new_routing, solution.alpha, solution.beta)

# =========================
# REFSET (DIVERSIDAD CONTROLADA)
# =========================
def build_refset(population, size):
    # Mejores soluciones
    sorted_pop = sorted(population, key=lambda s: s.cost)
    best = sorted_pop[: max(1, size // 2)]

    diverse = []

    # Candidatos (más robusto)
    candidates = random.sample(population, min(50, len(population)))

    # Selección diversa (max-min)
    while len(diverse) < max(1, size // 2):
        best_candidate = None
        best_dist = -1

        # Evaluar cada candidato contra el conjunto actual de soluciones seleccionadas (best + diverse)
        for c in candidates:
            if c in best or c in diverse:
                continue

            min_dist = float('inf')

            # Calcular la distancia mínima a cualquier solución en el conjunto seleccionado
            for s in best + diverse:
                d = solution_distance(c, s)

                # Actualizar la distancia mínima para este candidato
                if d < min_dist:
                    min_dist = d

                # Early stopping para eficiencia
                if min_dist <= best_dist:
                    break

            # Actualizar mejor candidato
            if min_dist > best_dist:
                best_dist = min_dist
                best_candidate = c

        # Evitar bucle infinito
        if best_candidate is not None:
            diverse.append(best_candidate)
        else:
            break

    return (best + diverse)[:size]

# =========================
# SCATTER SEARCH
# =========================
def scatter_search(graph, pop_size=None, refset_size=None, iterations=None):
    #Edge load global para guiar el muestreo y la mejora para evitar saturar ciertas aristas/fomentar soluciones más balanceadas
    global_edge_load = defaultdict(int)
    
    # Muestreo inteligente para generar pares de nodos (clave para calidad y diversidad de la población inicial)
    n = len(graph.nodes())
    sample_size = min(800, 5 * n)

    # Límite dinámico de diversificaciones
    max_diversifications = max(2, min(6, n // 50))

    pairs = intelligent_sampling(graph, sample_size)

    # Población inicial
    population = []
    
    for _ in range(pop_size):
        sol = generate_solution(graph, pairs, ALPHA_DEFAULT, BETA_DEFAULT, global_edge_load)
        population.append(sol)

    # Construir refset inicial
    refset = build_refset(population, refset_size)
    best = min(refset, key=lambda s: s.cost)

    print(f"Iteración 1: Mejor costo = {best.cost:.4f}")

    # Contadores de estancamiento/diversificaciones sin mejora para controlar que tanto diversificar y cuándo terminar
    stagnation = 0
    diversifications_without_improvement = 0

    # Iteraciones principales de Scatter Search
    for it in range(iterations):

        new_solutions = []

       # Separar refset en calidad y diversidad
        half = len(refset) // 2
        quality_set = refset[:half]
        diversity_set = refset[half:]

       # Parametros de control
        MAX_CHILDREN = 30

        # Presupuesto para cada tipo de combinación y controlar la cantidad de hijos generados
        BUDGET_QQ = MAX_CHILDREN // 3
        BUDGET_DD = MAX_CHILDREN // 3
        BUDGET_QD = MAX_CHILDREN - BUDGET_QQ - BUDGET_DD

        # Costo de la peor solución en el refset para decidir si vale la pena mejorar soluciones hijas
        worst_ref_cost = max(s.cost for s in refset)

        # Función para procesar pares con un límite de hijos generados
        def process_pairs_limited(pairs, limit):
            children = []

            for s1, s2 in pairs:

                # Genrar hijo combinando s1 y s2
                child = combine(s1, s2, graph, global_edge_load)

                # Solo intentar mejorar si el hijo tiene potencial de ser mejor que lo peor del refset
                if child.cost < worst_ref_cost and random.random() < IMPROVE_PROB:
                    child = improve(child, graph, global_edge_load,10)
                
                if not is_valid_solution(child):
                    continue

                # Agregar el hijo a la lista de nuevos candidatos
                children.append(child)

                # Early stopping para evitar generar demasiados hijos y saturar la población
                if len(children) >= limit:
                    break

            return children


        # -------- QUALITY × QUALITY --------
        pairs_qq = list(itertools.combinations(quality_set, 2))
        random.shuffle(pairs_qq)
        new_solutions += process_pairs_limited(pairs_qq, BUDGET_QQ)


        # -------- DIVERSITY × DIVERSITY --------
        pairs_dd = list(itertools.combinations(diversity_set, 2))
        random.shuffle(pairs_dd)
        new_solutions += process_pairs_limited(pairs_dd, BUDGET_DD)


        # -------- QUALITY × DIVERSITY --------
        pairs_qd = [(q, d) for q in quality_set for d in diversity_set]
        random.shuffle(pairs_qd)
        new_solutions += process_pairs_limited(pairs_qd, BUDGET_QD)

        # Agregar nuevas soluciones a la población
        population.extend(new_solutions)

        # Reconstruir refset
        refset = build_refset(population, refset_size)

        current_best = min(refset, key=lambda s: s.cost).copy()

        # Actualizar el mejor global y la memoria suave
        if current_best.cost < best.cost:
            best = current_best.copy()
            stagnation = 0
            diversifications_without_improvement = 0

            for e, load in best.edge_load.items():
                global_edge_load[e] = 0.8 * global_edge_load[e] + 0.2 * load

            print(f"Iter {it+1} mejor: {best.cost:.4f}")
            print(f"Real index: {real_transmission_index(best)}")
            print(sorted(best.edge_load.values())[:10])
            print(sorted(best.edge_load.values(), reverse=True)[:10])

        else:
            stagnation += 1
            print(f"Iteración {it+1}: sin mejora")

        # Diversificación selectiva para evitar estancamientos sin destruir completamente la estructura de la solución
        if stagnation >= max(10, it // 4):
            print("→ Diversificación selectiva")

            prev_best_cost = best.cost

            # Actualizar pares (Cambio controlado)
            pairs = hybrid_sampling(graph, global_edge_load, sample_size, bias_ratio=0.2)

            # Seleccionar peores soluciones
            k = pop_size // 4
            worst = sorted(population, key=lambda s: s.cost, reverse=True)[:k]

            new_solutions = []

            for s in worst:

                if random.random() < 0.5:
                    # Intensificación (mantener estructura)
                    kicked = kick_solution(s, graph, global_edge_load, 20, 0.5)
                    improved = improve(kicked, graph, global_edge_load, 10)
                    new_solutions.append(improved)

                else:
                    # Diversificación real (nuevos pares)
                    new_sol = generate_solution(graph, pairs, ALPHA_DEFAULT, BETA_DEFAULT, global_edge_load)
                    new_solutions.append(new_sol)

            # Reemplazo parcial
            population = sorted(population, key=lambda s: s.cost)
            population = population[:-k] + new_solutions

            refset = build_refset(population, refset_size)

            current_best = min(refset, key=lambda s: s.cost)

            if current_best.cost < prev_best_cost:
                best = current_best.copy()
                stagnation = 0
                diversifications_without_improvement = 0
            else:
                diversifications_without_improvement += 1

            stagnation = 0

        if diversifications_without_improvement >= max_diversifications:
            print("→ No hay mejoras tras múltiples diversificaciones, terminando")
            break

    return best


# =========================
# MAIN
# =========================

start = time.time()

graph, pop, refset, iters = read_graph("test7.txt")

print("========================Parámetros========================")
print(f"Nodos: {len(graph.nodes())} | Población: {pop} | RefSet: {refset} | Iteraciones: {iters}")

best = scatter_search(graph, pop, refset, iters)

real_index = real_transmission_index(best)
total_load = sum(best.edge_load.values())

end = time.time()

print("========================Resultados========================")
print(f"Índice de Transmisión aproximado (real): {real_index}")
print(f"Costo heurístico: {best.cost:.4f}")
print(f"Carga total: {total_load}")
print(f"Numero de pares: {len(best.routing)}")
print(f"Tiempo de ejecución: {end - start:.4f} segundos")