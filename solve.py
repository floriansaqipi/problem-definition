from __future__ import annotations

import argparse
import heapq
import math
import random
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple


CAPACITY = {"S": 10, "M": 20, "L": 30}


@dataclass(frozen=True)
class Street:
    id: int
    a: int
    b: int
    direction: int
    travel_time: int
    length: int
    category: str
    requirement: int

    def cleanable(self) -> bool:
        return self.category != "C"

    def mandatory(self) -> bool:
        return self.category == "M"

    def optional(self) -> bool:
        return self.category == "O"

    def directions(self) -> List[Tuple[int, int]]:
        if self.direction == 1:
            return [(self.a, self.b)]
        return [(self.a, self.b), (self.b, self.a)]


@dataclass(frozen=True)
class Instance:
    n: int
    m: int
    time_limit: int
    vehicle_count: int
    depot: int
    alpha: float
    streets: List[Street]
    vehicles: List[str]
    adjacency: List[List[Tuple[int, int, int]]]
    lmax: int
    wmax: float


@dataclass(frozen=True)
class CleanAction:
    street_id: int
    start: int
    end: int


@dataclass
class VehiclePlan:
    vehicle_type: str
    capacity: int
    actions: List[CleanAction] = field(default_factory=list)
    route_time: float = 0.0


@dataclass
class MaterializedSolution:
    routes: List[List[int]]
    cleaned_by_vehicle: List[List[int]]
    route_times: List[float]
    score: float
    valid: bool
    message: str


@dataclass(frozen=True)
class SubmissionStructureIssue:
    path: Path
    line_number: int
    message: str
    vehicle_index: Optional[int] = None
    declared_count: Optional[int] = None
    actual_count: Optional[int] = None

    def __str__(self) -> str:
        location = f"{self.path}:{self.line_number}"
        if self.vehicle_index is None:
            return f"{location}: {self.message}"
        return f"{location}: vehicle {self.vehicle_index}: {self.message}"


class SubmissionStructureError(ValueError):
    def __init__(self, issues: Sequence[SubmissionStructureIssue]) -> None:
        self.issues = list(issues)
        super().__init__("\n".join(str(issue) for issue in self.issues))


class PathCache:
    def __init__(self, instance: Instance) -> None:
        self.instance = instance
        self._cache: Dict[int, Tuple[List[float], List[Optional[int]], List[Optional[int]]]] = {}

    def _run_dijkstra(self, source: int) -> Tuple[List[float], List[Optional[int]], List[Optional[int]]]:
        n = self.instance.n
        dist = [math.inf] * n
        prev_node: List[Optional[int]] = [None] * n
        prev_edge: List[Optional[int]] = [None] * n
        dist[source] = 0
        heap: List[Tuple[float, int]] = [(0, source)]

        while heap:
            current_dist, node = heapq.heappop(heap)
            if current_dist != dist[node]:
                continue
            for nxt, travel_time, street_id in self.instance.adjacency[node]:
                new_dist = current_dist + travel_time
                if new_dist < dist[nxt]:
                    dist[nxt] = new_dist
                    prev_node[nxt] = node
                    prev_edge[nxt] = street_id
                    heapq.heappush(heap, (new_dist, nxt))

        return dist, prev_node, prev_edge

    def data(self, source: int) -> Tuple[List[float], List[Optional[int]], List[Optional[int]]]:
        if source not in self._cache:
            self._cache[source] = self._run_dijkstra(source)
        return self._cache[source]

    def distance(self, source: int, target: int) -> float:
        if source == target:
            return 0.0
        return self.data(source)[0][target]

    def path(self, source: int, target: int) -> Tuple[Optional[List[int]], Optional[List[int]]]:
        if source == target:
            return [source], []

        dist, prev_node, prev_edge = self.data(source)
        if math.isinf(dist[target]):
            return None, None

        nodes = [target]
        edges: List[int] = []
        current = target
        while current != source:
            previous = prev_node[current]
            edge_id = prev_edge[current]
            if previous is None or edge_id is None:
                return None, None
            nodes.append(previous)
            edges.append(edge_id)
            current = previous

        nodes.reverse()
        edges.reverse()
        return nodes, edges


def parse_instance(path: Path) -> Instance:
    lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not lines:
        raise ValueError(f"{path} is empty")

    header = lines[0].split()
    if len(header) != 6:
        raise ValueError("first line must contain N M T C S alpha")

    n = int(header[0])
    m = int(header[1])
    time_limit = int(header[2])
    vehicle_count = int(header[3])
    depot = int(header[4])
    alpha = float(header[5])

    street_start = 1
    has_coordinate_block = False
    if len(lines) >= 1 + n + 1:
        coordinate_block = True
        for line in lines[1 : 1 + n]:
            parts = line.split()
            if len(parts) != 2:
                coordinate_block = False
                break
            try:
                float(parts[0])
                float(parts[1])
            except ValueError:
                coordinate_block = False
                break
        if coordinate_block and len(lines[1 + n].split()) == 7:
            street_start = 1 + n
            has_coordinate_block = True

    available_street_rows = len(lines) - street_start - 1
    if available_street_rows < m and has_coordinate_block and available_street_rows > 0:
        m = available_street_rows

    if len(lines) < street_start + m + 1:
        raise ValueError("input file does not contain all streets and vehicle line")

    streets: List[Street] = []
    for street_id, line in enumerate(lines[street_start : street_start + m]):
        parts = line.split()
        if len(parts) != 7:
            raise ValueError(f"street line {street_id} must contain 7 fields")
        a, b, direction, travel_time, length = map(int, parts[:5])
        category = parts[5]
        requirement = int(parts[6])
        streets.append(Street(street_id, a, b, direction, travel_time, length, category, requirement))

    vehicles = lines[street_start + m].split()
    if len(vehicles) != vehicle_count:
        raise ValueError(f"expected {vehicle_count} vehicles, found {len(vehicles)}")

    adjacency: List[List[Tuple[int, int, int]]] = [[] for _ in range(n)]
    for street in streets:
        adjacency[street.a].append((street.b, street.travel_time, street.id))
        if street.direction == 2:
            adjacency[street.b].append((street.a, street.travel_time, street.id))

    cleanable = [street for street in streets if street.cleanable()]
    lmax = sum(street.length for street in cleanable)
    wmax = sum((30 - street.requirement) * street.length / 1000.0 for street in cleanable)

    return Instance(
        n=n,
        m=m,
        time_limit=time_limit,
        vehicle_count=vehicle_count,
        depot=depot,
        alpha=alpha,
        streets=streets,
        vehicles=vehicles,
        adjacency=adjacency,
        lmax=lmax,
        wmax=wmax,
    )


def street_gain(instance: Instance, street: Street, capacity: int) -> float:
    coverage_gain = street.length / instance.lmax if instance.lmax else 0.0
    waste = (capacity - street.requirement) * street.length / 1000.0
    waste_penalty = waste / instance.wmax if instance.wmax > 0 else 0.0
    return instance.alpha * coverage_gain - (1.0 - instance.alpha) * waste_penalty


def street_waste(instance: Instance, street: Street, capacity: int) -> float:
    if not street.cleanable():
        return 0.0
    return (capacity - street.requirement) * street.length / 1000.0


def route_time(instance: Instance, cache: PathCache, actions: Sequence[CleanAction]) -> float:
    current = instance.depot
    total = 0.0
    for action in actions:
        street = instance.streets[action.street_id]
        to_start = cache.distance(current, action.start)
        if math.isinf(to_start):
            return math.inf
        total += to_start + street.travel_time
        current = action.end
    to_depot = cache.distance(current, instance.depot)
    if math.isinf(to_depot):
        return math.inf
    return total + to_depot


def insertion_delta(
    instance: Instance,
    cache: PathCache,
    plan: VehiclePlan,
    action: CleanAction,
    position: int,
) -> float:
    previous_end = instance.depot if position == 0 else plan.actions[position - 1].end
    next_start = instance.depot if position == len(plan.actions) else plan.actions[position].start

    old_link = cache.distance(previous_end, next_start)
    first = cache.distance(previous_end, action.start)
    second = cache.distance(action.end, next_start)
    if math.isinf(first) or math.isinf(second) or math.isinf(old_link):
        return math.inf

    street = instance.streets[action.street_id]
    return first + street.travel_time + second - old_link


def removal_delta(instance: Instance, cache: PathCache, plan: VehiclePlan, position: int) -> float:
    action = plan.actions[position]
    street = instance.streets[action.street_id]
    previous_end = instance.depot if position == 0 else plan.actions[position - 1].end
    next_start = instance.depot if position + 1 == len(plan.actions) else plan.actions[position + 1].start

    old_link = cache.distance(previous_end, action.start) + street.travel_time + cache.distance(action.end, next_start)
    new_link = cache.distance(previous_end, next_start)
    if math.isinf(old_link) or math.isinf(new_link):
        return math.inf
    return new_link - old_link


def apply_insert(
    instance: Instance,
    cache: PathCache,
    plan: VehiclePlan,
    action: CleanAction,
    position: int,
) -> None:
    delta = insertion_delta(instance, cache, plan, action, position)
    plan.actions.insert(position, action)
    plan.route_time += delta


def can_vehicle_clean(plan: VehiclePlan, street: Street) -> bool:
    return street.cleanable() and plan.capacity >= street.requirement


def standalone_trip_time(instance: Instance, cache: PathCache, street: Street) -> float:
    best = math.inf
    for start, end in street.directions():
        start_dist = cache.distance(instance.depot, start)
        end_dist = cache.distance(end, instance.depot)
        if not math.isinf(start_dist) and not math.isinf(end_dist):
            best = min(best, start_dist + street.travel_time + end_dist)
    return best


def choose_ranked(items: List[Tuple[float, object]], rng: random.Random, top_k: int, minimize: bool) -> object:
    if minimize:
        items.sort(key=lambda item: item[0])
    else:
        items.sort(key=lambda item: item[0], reverse=True)

    if top_k <= 1 or len(items) == 1:
        return items[0][1]

    limit = min(top_k, len(items))
    weights = [1.0 / (rank + 1) for rank in range(limit)]
    total = sum(weights)
    pick = rng.random() * total
    running = 0.0
    for rank, weight in enumerate(weights):
        running += weight
        if pick <= running:
            return items[rank][1]
    return items[limit - 1][1]


def order_mandatory(
    instance: Instance,
    cache: PathCache,
    rng: random.Random,
    randomized: bool,
) -> List[Street]:
    mandatory = [street for street in instance.streets if street.mandatory()]

    def base_key(street: Street) -> Tuple[int, float, int]:
        return (street.requirement, standalone_trip_time(instance, cache, street), street.length)

    if not randomized:
        return sorted(mandatory, key=base_key, reverse=True)

    grouped: Dict[int, List[Street]] = {30: [], 20: [], 10: []}
    for street in mandatory:
        grouped.setdefault(street.requirement, []).append(street)

    ordered: List[Street] = []
    for requirement in (30, 20, 10):
        group = grouped.get(requirement, [])
        group.sort(
            key=lambda street: (
                standalone_trip_time(instance, cache, street) * rng.uniform(0.85, 1.15),
                street.length * rng.uniform(0.85, 1.15),
            ),
            reverse=True,
        )
        ordered.extend(group)
    return ordered


def best_mandatory_insertion(
    instance: Instance,
    cache: PathCache,
    plans: List[VehiclePlan],
    street: Street,
    rng: random.Random,
    top_k: int,
) -> Optional[Tuple[int, int, CleanAction]]:
    candidates: List[Tuple[float, Tuple[int, int, CleanAction]]] = []
    waste_weight = 0.0 if instance.alpha >= 0.999999 else (1.0 - instance.alpha) * 4.0

    for vehicle_index, plan in enumerate(plans):
        if not can_vehicle_clean(plan, street):
            continue
        for start, end in street.directions():
            action = CleanAction(street.id, start, end)
            for position in range(len(plan.actions) + 1):
                delta = insertion_delta(instance, cache, plan, action, position)
                if math.isinf(delta) or plan.route_time + delta > instance.time_limit:
                    continue

                normalized_time = delta / max(1, instance.time_limit)
                normalized_waste = street_waste(instance, street, plan.capacity) / instance.wmax if instance.wmax > 0 else 0.0
                oversize_penalty = (plan.capacity - street.requirement) / 30.0
                cost = normalized_time + waste_weight * normalized_waste + 0.15 * (1.0 - instance.alpha) * oversize_penalty
                cost *= rng.uniform(0.98, 1.02)
                candidates.append((cost, (vehicle_index, position, action)))

    if not candidates:
        return None
    return choose_ranked(candidates, rng, top_k, minimize=True)  # type: ignore[return-value]


def preferred_vehicle_phases(street: Street) -> List[set[str]]:
    if street.requirement == 10:
        return [{"S"}, {"M", "L"}]
    if street.requirement == 20:
        return [{"M"}, {"L"}]
    return [{"L"}]


def role_mandatory_insertion_candidates(
    instance: Instance,
    cache: PathCache,
    plans: List[VehiclePlan],
    street: Street,
    rng: random.Random,
    jitter: bool,
) -> List[Tuple[float, Tuple[int, int, CleanAction]]]:
    for phase_index, allowed_types in enumerate(preferred_vehicle_phases(street)):
        candidates: List[Tuple[float, Tuple[int, int, CleanAction]]] = []
        for vehicle_index, plan in enumerate(plans):
            if plan.vehicle_type not in allowed_types or not can_vehicle_clean(plan, street):
                continue
            for start, end in street.directions():
                action = CleanAction(street.id, start, end)
                for position in range(len(plan.actions) + 1):
                    delta = insertion_delta(instance, cache, plan, action, position)
                    if math.isinf(delta) or plan.route_time + delta > instance.time_limit:
                        continue

                    waste = street_waste(instance, street, plan.capacity)
                    phase_penalty = phase_index * instance.time_limit * 0.20
                    cost = delta + phase_penalty + waste * 5.0
                    if jitter:
                        cost *= rng.uniform(0.98, 1.02)
                    candidates.append((cost, (vehicle_index, position, action)))

        if candidates:
            candidates.sort(key=lambda item: item[0])
            return candidates

    return []


def construct_role_mandatory(
    instance: Instance,
    cache: PathCache,
    plans: List[VehiclePlan],
    rng: random.Random,
    top_k: int,
    jitter: bool,
) -> bool:
    remaining = [street for street in instance.streets if street.mandatory()]

    while remaining:
        ranked: List[Tuple[float, Tuple[Street, Tuple[int, int, CleanAction]]]] = []
        for street in remaining:
            candidates = role_mandatory_insertion_candidates(instance, cache, plans, street, rng, jitter)
            if not candidates:
                continue

            best_cost = candidates[0][0]
            second_cost = candidates[1][0] if len(candidates) > 1 else best_cost + instance.time_limit
            regret = second_cost - best_cost
            standalone = standalone_trip_time(instance, cache, street)
            hard_to_place_bonus = (standalone if not math.isinf(standalone) else instance.time_limit) * 0.05
            length_bonus = street.length * 0.01
            priority = regret + hard_to_place_bonus + length_bonus
            if jitter:
                priority *= rng.uniform(0.98, 1.02)
            ranked.append((priority, (street, candidates[0][1])))

        if not ranked:
            return False

        street, (vehicle_index, position, action) = choose_ranked(
            ranked,
            rng,
            top_k if jitter else 1,
            minimize=False,
        )  # type: ignore[misc]
        apply_insert(instance, cache, plans[vehicle_index], action, position)
        remaining = [candidate for candidate in remaining if candidate.id != street.id]

    return True


def missing_mandatory_streets(instance: Instance, solution: MaterializedSolution) -> List[Street]:
    cleaned = {street_id for vehicle in solution.cleaned_by_vehicle for street_id in vehicle}
    return [street for street in instance.streets if street.mandatory() and street.id not in cleaned]


def repair_missing_mandatory(
    instance: Instance,
    cache: PathCache,
    plans: List[VehiclePlan],
    rng: random.Random,
    top_k: int,
) -> bool:
    for _ in range(len([street for street in instance.streets if street.mandatory()])):
        solution = materialize_solution(instance, cache, plans)
        missing = missing_mandatory_streets(instance, solution)
        if not missing:
            return True

        repaired = False
        missing.sort(
            key=lambda street: (
                street.requirement,
                standalone_trip_time(instance, cache, street),
                street.length,
            ),
            reverse=True,
        )

        for street in missing:
            candidates = role_mandatory_insertion_candidates(instance, cache, plans, street, rng, jitter=False)
            if candidates:
                vehicle_index, position, action = choose_ranked(
                    [(cost, candidate) for cost, candidate in candidates],
                    rng,
                    top_k,
                    minimize=True,
                )  # type: ignore[misc]
                apply_insert(instance, cache, plans[vehicle_index], action, position)
                repaired = True
                break

            for vehicle_index, plan in enumerate(plans):
                if not can_vehicle_clean(plan, street):
                    continue
                for start, end in street.directions():
                    trial = VehiclePlan(plan.vehicle_type, plan.capacity, plan.actions + [CleanAction(street.id, start, end)])
                    trial.route_time = route_time(instance, cache, trial.actions)
                    if math.isinf(trial.route_time):
                        continue
                    rebuild_plan_nearest(instance, cache, trial, rng, randomized=False)
                    if trial.route_time <= instance.time_limit:
                        plan.actions = trial.actions
                        plan.route_time = trial.route_time
                        repaired = True
                        break
                if repaired:
                    break
            if repaired:
                break

        if not repaired:
            return False

    return not missing_mandatory_streets(instance, materialize_solution(instance, cache, plans))


def possible_optional_gain(instance: Instance, street: Street, capacities: Iterable[int]) -> float:
    best = -math.inf
    for capacity in capacities:
        if capacity >= street.requirement:
            best = max(best, street_gain(instance, street, capacity))
    return best


def select_optional_pool(
    instance: Instance,
    cache: PathCache,
    max_candidates: int,
) -> List[Street]:
    optionals = [street for street in instance.streets if street.optional()]
    capacities = [CAPACITY[vehicle] for vehicle in instance.vehicles]
    useful = []
    for street in optionals:
        gain = possible_optional_gain(instance, street, capacities)
        if instance.alpha >= 0.999999 or gain > 1e-12:
            useful.append(street)

    if len(useful) <= max_candidates:
        return useful

    def approximate_priority(street: Street) -> float:
        gain = max(0.0, possible_optional_gain(instance, street, capacities))
        trip_time = standalone_trip_time(instance, cache, street)
        if math.isinf(trip_time):
            return -math.inf
        return gain / max(1.0, trip_time)

    useful.sort(key=approximate_priority, reverse=True)
    return useful[:max_candidates]


def add_optional_streets(
    instance: Instance,
    cache: PathCache,
    plans: List[VehiclePlan],
    already_cleaned: set[int],
    rng: random.Random,
    top_k: int,
    max_optional_candidates: int,
) -> None:
    remaining = [street for street in select_optional_pool(instance, cache, max_optional_candidates) if street.id not in already_cleaned]
    remaining_ids = {street.id for street in remaining}
    streets_by_id = {street.id: street for street in remaining}

    while remaining_ids:
        candidates: List[Tuple[float, Tuple[int, int, CleanAction]]] = []
        for street_id in list(remaining_ids):
            street = streets_by_id[street_id]
            for vehicle_index, plan in enumerate(plans):
                if not can_vehicle_clean(plan, street):
                    continue
                gain = street_gain(instance, street, plan.capacity)
                if instance.alpha < 0.999999 and gain <= 1e-12:
                    continue

                for start, end in street.directions():
                    action = CleanAction(street.id, start, end)
                    for position in range(len(plan.actions) + 1):
                        delta = insertion_delta(instance, cache, plan, action, position)
                        if math.isinf(delta) or plan.route_time + delta > instance.time_limit:
                            continue

                        priority = gain / max(1.0, delta)
                        priority *= rng.uniform(0.98, 1.02)
                        candidates.append((priority, (vehicle_index, position, action)))

        if not candidates:
            break

        vehicle_index, position, action = choose_ranked(candidates, rng, top_k, minimize=False)  # type: ignore[misc]
        apply_insert(instance, cache, plans[vehicle_index], action, position)
        already_cleaned.add(action.street_id)
        remaining_ids.discard(action.street_id)


def rebuild_plan_nearest(
    instance: Instance,
    cache: PathCache,
    plan: VehiclePlan,
    rng: random.Random,
    randomized: bool,
) -> None:
    if len(plan.actions) <= 2:
        return

    remaining = plan.actions[:]
    rebuilt: List[CleanAction] = []
    current = instance.depot

    while remaining:
        ranked: List[Tuple[float, Tuple[int, CleanAction]]] = []
        for index, old_action in enumerate(remaining):
            street = instance.streets[old_action.street_id]
            for start, end in street.directions():
                action = CleanAction(street.id, start, end)
                travel = cache.distance(current, start)
                back = cache.distance(end, instance.depot)
                if math.isinf(travel) or math.isinf(back):
                    continue
                cost = travel + street.travel_time + 0.10 * back
                if randomized:
                    cost *= rng.uniform(0.95, 1.05)
                ranked.append((cost, (index, action)))
        if not ranked:
            return

        index, action = choose_ranked(ranked, rng, 2 if randomized else 1, minimize=True)  # type: ignore[misc]
        rebuilt.append(action)
        current = action.end
        remaining.pop(index)

    new_time = route_time(instance, cache, rebuilt)
    if new_time <= plan.route_time:
        plan.actions = rebuilt
        plan.route_time = new_time


def reassign_to_better_vehicle(
    instance: Instance,
    cache: PathCache,
    plans: List[VehiclePlan],
    max_moves: int = 200,
) -> None:
    for _ in range(max_moves):
        best_move = None
        best_improvement = 1e-12

        for source_index, source_plan in enumerate(plans):
            for action_index, action in enumerate(source_plan.actions):
                street = instance.streets[action.street_id]
                old_gain = street_gain(instance, street, source_plan.capacity)
                source_delta = removal_delta(instance, cache, source_plan, action_index)
                if math.isinf(source_delta):
                    continue
                source_new_time = source_plan.route_time + source_delta
                if source_new_time > instance.time_limit:
                    continue

                for target_index, target_plan in enumerate(plans):
                    if target_index == source_index or not can_vehicle_clean(target_plan, street):
                        continue
                    new_gain = street_gain(instance, street, target_plan.capacity)
                    improvement = new_gain - old_gain
                    if improvement <= best_improvement:
                        continue

                    for start, end in street.directions():
                        new_action = CleanAction(street.id, start, end)
                        for position in range(len(target_plan.actions) + 1):
                            target_delta = insertion_delta(instance, cache, target_plan, new_action, position)
                            if math.isinf(target_delta) or target_plan.route_time + target_delta > instance.time_limit:
                                continue
                            best_improvement = improvement
                            best_move = (source_index, action_index, source_delta, target_index, position, new_action, target_delta)

        if best_move is None:
            return

        source_index, action_index, source_delta, target_index, position, new_action, target_delta = best_move
        source_plan = plans[source_index]
        target_plan = plans[target_index]
        source_plan.actions.pop(action_index)
        source_plan.route_time += source_delta
        target_plan.actions.insert(position, new_action)
        target_plan.route_time += target_delta


def mark_traversed_cleanable(
    instance: Instance,
    cache: PathCache,
    plans: Sequence[VehiclePlan],
    cleaned: set[int],
    mandatory_only: bool,
) -> None:
    for plan in plans:
        _, traversed_edges = build_route_nodes_and_edges(instance, cache, plan.actions)
        if traversed_edges is None:
            continue

        for edge_id in traversed_edges:
            if edge_id in cleaned:
                continue
            street = instance.streets[edge_id]
            if not can_vehicle_clean(plan, street):
                continue
            if street.mandatory():
                cleaned.add(edge_id)
            elif not mandatory_only and (instance.alpha >= 0.999999 or street_gain(instance, street, plan.capacity) > 1e-12):
                cleaned.add(edge_id)


def build_route_nodes_and_edges(
    instance: Instance,
    cache: PathCache,
    actions: Sequence[CleanAction],
) -> Tuple[Optional[List[int]], Optional[List[int]]]:
    nodes = [instance.depot]
    edges: List[int] = []
    current = instance.depot

    for action in actions:
        path_nodes, path_edges = cache.path(current, action.start)
        if path_nodes is None or path_edges is None:
            return None, None
        nodes.extend(path_nodes[1:])
        edges.extend(path_edges)

        street = instance.streets[action.street_id]
        if nodes[-1] != action.start:
            return None, None
        nodes.append(action.end)
        edges.append(street.id)
        current = action.end

    path_nodes, path_edges = cache.path(current, instance.depot)
    if path_nodes is None or path_edges is None:
        return None, None
    nodes.extend(path_nodes[1:])
    edges.extend(path_edges)
    return nodes, edges


def materialize_solution(instance: Instance, cache: PathCache, plans: List[VehiclePlan]) -> MaterializedSolution:
    routes: List[List[int]] = []
    cleaned_by_vehicle: List[List[int]] = []
    route_times: List[float] = []
    globally_cleaned = {action.street_id for plan in plans for action in plan.actions}

    for plan in plans:
        nodes, traversed_edges = build_route_nodes_and_edges(instance, cache, plan.actions)
        if nodes is None or traversed_edges is None:
            return MaterializedSolution([], [], [], 0.0, False, "unreachable route segment")

        scheduled = {action.street_id for action in plan.actions}
        clean_line: List[int] = []
        already_on_this_vehicle: set[int] = set()

        for edge_id in traversed_edges:
            if edge_id in scheduled and edge_id not in already_on_this_vehicle:
                clean_line.append(edge_id)
                already_on_this_vehicle.add(edge_id)

        for edge_id in traversed_edges:
            if edge_id in globally_cleaned or edge_id in already_on_this_vehicle:
                continue
            street = instance.streets[edge_id]
            if not can_vehicle_clean(plan, street):
                continue
            if street.mandatory() or instance.alpha >= 0.999999 or street_gain(instance, street, plan.capacity) > 1e-12:
                clean_line.append(edge_id)
                already_on_this_vehicle.add(edge_id)
                globally_cleaned.add(edge_id)

        routes.append(nodes)
        cleaned_by_vehicle.append(clean_line)
        route_times.append(plan.route_time)

    score, valid, message = validate_and_score(instance, routes, cleaned_by_vehicle)
    return MaterializedSolution(routes, cleaned_by_vehicle, route_times, score, valid, message)


def build_pair_lookup(instance: Instance) -> Dict[Tuple[int, int], Street]:
    lookup: Dict[Tuple[int, int], Street] = {}
    for street in instance.streets:
        lookup[(street.a, street.b)] = street
        if street.direction == 2:
            lookup[(street.b, street.a)] = street
    return lookup


def validate_and_score(
    instance: Instance,
    routes: Sequence[Sequence[int]],
    cleaned_by_vehicle: Sequence[Sequence[int]],
) -> Tuple[float, bool, str]:
    if len(routes) != instance.vehicle_count or len(cleaned_by_vehicle) != instance.vehicle_count:
        return 0.0, False, "wrong number of vehicles"

    pair_lookup = build_pair_lookup(instance)
    total_waste = 0.0
    cleaned_once: set[int] = set()

    for vehicle_index, (route, cleaned_ids) in enumerate(zip(routes, cleaned_by_vehicle)):
        if not route or route[0] != instance.depot or route[-1] != instance.depot:
            return 0.0, False, f"vehicle {vehicle_index} does not start/end at depot"

        traversed: List[int] = []
        route_total = 0
        for a, b in zip(route, route[1:]):
            street = pair_lookup.get((a, b))
            if street is None:
                return 0.0, False, f"vehicle {vehicle_index} uses invalid edge {a}->{b}"
            traversed.append(street.id)
            route_total += street.travel_time

        if route_total > instance.time_limit:
            return 0.0, False, f"vehicle {vehicle_index} exceeds time limit"

        traversed_set = set(traversed)
        capacity = CAPACITY[instance.vehicles[vehicle_index]]
        for street_id in cleaned_ids:
            if street_id < 0 or street_id >= instance.m:
                return 0.0, False, f"invalid street id {street_id}"
            if street_id not in traversed_set:
                return 0.0, False, f"vehicle {vehicle_index} cleans street {street_id} without traversing it"
            street = instance.streets[street_id]
            if not street.cleanable():
                return 0.0, False, f"vehicle {vehicle_index} cleans connector {street_id}"
            if capacity < street.requirement:
                return 0.0, False, f"vehicle {vehicle_index} cannot clean street {street_id}"
            cleaned_once.add(street_id)
            total_waste += street_waste(instance, street, capacity)

    missing = [street.id for street in instance.streets if street.mandatory() and street.id not in cleaned_once]
    if missing:
        return 0.0, False, f"missing mandatory streets: {missing[:10]}"

    cleaned_length = sum(instance.streets[street_id].length for street_id in cleaned_once)
    coverage = cleaned_length / instance.lmax if instance.lmax else 1.0
    efficiency = 1.0 - total_waste / instance.wmax if instance.wmax > 0 else 1.0
    score = instance.alpha * coverage + (1.0 - instance.alpha) * efficiency
    return score, True, "valid"


def construct_solution(
    instance: Instance,
    cache: PathCache,
    seed: int,
    attempt: int,
    top_k: int,
    max_optional_candidates: int,
    mandatory_mode: str,
) -> Optional[MaterializedSolution]:
    rng = random.Random(seed)
    randomized = attempt > 0
    active_top_k = top_k if randomized else 1
    plans = [VehiclePlan(vehicle_type=vehicle, capacity=CAPACITY[vehicle]) for vehicle in instance.vehicles]
    cleaned: set[int] = set()
    used_opportunistic_mandatory = False

    if mandatory_mode == "role":
        mandatory_ok = construct_role_mandatory(
            instance=instance,
            cache=cache,
            plans=plans,
            rng=rng,
            top_k=max(1, top_k),
            jitter=True,
        )
        if not mandatory_ok:
            repair_missing_mandatory(instance, cache, plans, rng, max(1, top_k))
            return materialize_solution(instance, cache, plans)
        repair_missing_mandatory(instance, cache, plans, rng, max(1, top_k))
        role_solution = materialize_solution(instance, cache, plans)
        if missing_mandatory_streets(instance, role_solution):
            return role_solution
        cleaned = {action.street_id for plan in plans for action in plan.actions}
    else:
        for street in order_mandatory(instance, cache, rng, randomized):
            if street.id in cleaned:
                continue
            insertion = best_mandatory_insertion(instance, cache, plans, street, rng, active_top_k)
            if insertion is None:
                before = set(cleaned)
                mark_traversed_cleanable(instance, cache, plans, cleaned, mandatory_only=True)
                if street.id in cleaned:
                    used_opportunistic_mandatory = True
                    continue
                cleaned = before
                return materialize_solution(instance, cache, plans)
            vehicle_index, position, action = insertion
            apply_insert(instance, cache, plans[vehicle_index], action, position)
            cleaned.add(street.id)

        missing_mandatory = [street.id for street in instance.streets if street.mandatory() and street.id not in cleaned]
        if missing_mandatory:
            return materialize_solution(instance, cache, plans)

    if not used_opportunistic_mandatory:
        for plan in plans:
            rebuild_plan_nearest(instance, cache, plan, rng, randomized)

        reassign_to_better_vehicle(instance, cache, plans)

    if instance.alpha > 1e-12 and not used_opportunistic_mandatory:
        add_optional_streets(instance, cache, plans, cleaned, rng, active_top_k, max_optional_candidates)
        for plan in plans:
            rebuild_plan_nearest(instance, cache, plan, rng, randomized)
        reassign_to_better_vehicle(instance, cache, plans)
        add_optional_streets(instance, cache, plans, cleaned, rng, active_top_k, max_optional_candidates)

    return materialize_solution(instance, cache, plans)


def fallback_quality(instance: Instance, solution: MaterializedSolution) -> Tuple[int, float, int]:
    cleaned = {street_id for vehicle in solution.cleaned_by_vehicle for street_id in vehicle}
    mandatory_cleaned = sum(1 for street in instance.streets if street.mandatory() and street.id in cleaned)
    cleaned_length = sum(instance.streets[street_id].length for street_id in cleaned)
    return mandatory_cleaned, solution.score, cleaned_length


def write_solution(path: Optional[Path], solution: MaterializedSolution) -> None:
    lines: List[str] = [str(len(solution.routes))]
    for route, cleaned in zip(solution.routes, solution.cleaned_by_vehicle):
        lines.append(str(max(0, len(route) - 1)))
        lines.append(" ".join(str(node) for node in route))
        lines.append(" ".join(str(street_id) for street_id in cleaned))

    text = "\n".join(lines) + "\n"
    if path is None:
        sys.stdout.write(text)
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def inspect_submission_structure(path: Path) -> List[SubmissionStructureIssue]:
    lines = path.read_text().splitlines()
    issues: List[SubmissionStructureIssue] = []

    if not lines:
        return [SubmissionStructureIssue(path, 1, "submission file is empty")]

    try:
        vehicle_count = int(lines[0].strip())
    except ValueError:
        return [SubmissionStructureIssue(path, 1, "first line must be the vehicle count")]

    position = 1
    for vehicle_index in range(vehicle_count):
        if position + 2 >= len(lines):
            issues.append(
                SubmissionStructureIssue(
                    path=path,
                    line_number=position + 1,
                    vehicle_index=vehicle_index,
                    message="missing one or more of the 3 required vehicle lines",
                )
            )
            return issues

        count_line_number = position + 1
        route_line_number = position + 2
        try:
            declared_count = int(lines[position].strip())
        except ValueError:
            issues.append(
                SubmissionStructureIssue(
                    path=path,
                    line_number=count_line_number,
                    vehicle_index=vehicle_index,
                    message="route node count line must be an integer",
                )
            )
            position += 3
            continue

        actual_count = len(lines[position + 1].split())
        expected_nodes = declared_count + 1
        if expected_nodes != actual_count:
            issues.append(
                SubmissionStructureIssue(
                    path=path,
                    line_number=count_line_number,
                    vehicle_index=vehicle_index,
                    declared_count=declared_count,
                    actual_count=actual_count,
                    message=(
                        f"Route node count mismatch for vehicle {vehicle_index}: "
                        f"expected {expected_nodes}, got {actual_count}; "
                        f"route line is {route_line_number}"
                    ),
                )
            )

        position += 3

    if position != len(lines):
        issues.append(
            SubmissionStructureIssue(
                path=path,
                line_number=position + 1,
                message=f"extra line(s) after expected {vehicle_count} vehicle blocks",
            )
        )

    return issues


def check_submission_structure(path: Path) -> None:
    issues = inspect_submission_structure(path)
    if issues:
        raise SubmissionStructureError(issues)


def repair_submission_structure(path: Path) -> int:
    lines = path.read_text().splitlines()
    if not lines:
        raise SubmissionStructureError([SubmissionStructureIssue(path, 1, "submission file is empty")])

    try:
        vehicle_count = int(lines[0].strip())
    except ValueError as exc:
        raise SubmissionStructureError(
            [SubmissionStructureIssue(path, 1, "first line must be the vehicle count")]
        ) from exc

    repaired = 0
    position = 1
    for vehicle_index in range(vehicle_count):
        if position + 2 >= len(lines):
            raise SubmissionStructureError(
                [
                    SubmissionStructureIssue(
                        path=path,
                        line_number=position + 1,
                        vehicle_index=vehicle_index,
                        message="missing one or more of the 3 required vehicle lines; refusing repair",
                    )
                ]
            )

        try:
            declared_count = int(lines[position].strip())
        except ValueError as exc:
            raise SubmissionStructureError(
                [
                    SubmissionStructureIssue(
                        path=path,
                        line_number=position + 1,
                        vehicle_index=vehicle_index,
                        message="route node count line must be an integer; refusing repair",
                    )
                ]
            ) from exc

        actual_count = len(lines[position + 1].split())
        repaired_count = max(0, actual_count - 1)
        if declared_count != repaired_count:
            lines[position] = str(repaired_count)
            repaired += 1

        position += 3

    if position != len(lines):
        raise SubmissionStructureError(
            [
                SubmissionStructureIssue(
                    path=path,
                    line_number=position + 1,
                    message=f"extra line(s) after expected {vehicle_count} vehicle blocks; refusing repair",
                )
            ]
        )

    if repaired:
        path.write_text("\n".join(lines) + "\n")
    return repaired


def solve(
    instance: Instance,
    seconds: float,
    seed: int,
    restarts: int,
    top_k: int,
    max_optional_candidates: int,
    mandatory_mode: str = "standard",
) -> MaterializedSolution:
    cache = PathCache(instance)
    started = time.monotonic()
    best: Optional[MaterializedSolution] = None
    best_any: Optional[MaterializedSolution] = None
    attempt = 0

    while True:
        if restarts and attempt >= restarts:
            break
        if attempt > 0 and time.monotonic() - started >= seconds:
            break

        solution = construct_solution(
            instance=instance,
            cache=cache,
            seed=seed + attempt * 1_000_003,
            attempt=attempt,
            top_k=top_k,
            max_optional_candidates=max_optional_candidates,
            mandatory_mode=mandatory_mode,
        )

        if solution is not None:
            if best_any is None or fallback_quality(instance, solution) > fallback_quality(instance, best_any):
                best_any = solution
            if solution.valid and (best is None or solution.score > best.score):
                best = solution

        attempt += 1

    if best is not None:
        return best
    if best_any is not None:
        return best_any
    raise RuntimeError("could not construct any solution")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="GRASP-style solver for the Street Cleaning problem.")
    parser.add_argument("input", nargs="?", type=Path, help="input instance file")
    parser.add_argument("output", nargs="?", type=Path, help="output solution file; omit to write to stdout")
    output_tools = parser.add_mutually_exclusive_group()
    output_tools.add_argument("--check-output", type=Path, help="check submission file structure and exit")
    output_tools.add_argument("--repair-output", type=Path, help="repair route node count lines in a submission file and exit")
    parser.add_argument("--seconds", type=float, default=30.0, help="time budget per instance")
    parser.add_argument("--seed", type=int, default=1, help="base random seed")
    parser.add_argument("--restarts", type=int, default=0, help="fixed number of restarts; 0 means use time budget")
    parser.add_argument("--top-k", type=int, default=4, help="randomized choice width after the first deterministic run")
    parser.add_argument("--max-optional-candidates", type=int, default=20000, help="cap optional streets considered on large instances")
    parser.add_argument("--mandatory-mode", choices=["standard", "role"], default="standard", help="mandatory construction strategy")
    parser.add_argument("--quiet", action="store_true", help="suppress summary on stderr")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.check_output is not None:
        issues = inspect_submission_structure(args.check_output)
        if issues:
            for issue in issues:
                print(issue, file=sys.stderr)
            return 1
        if not args.quiet:
            print(f"structure_ok=True file={args.check_output}", file=sys.stderr)
        return 0

    if args.repair_output is not None:
        try:
            repaired = repair_submission_structure(args.repair_output)
            check_submission_structure(args.repair_output)
        except SubmissionStructureError as exc:
            print(exc, file=sys.stderr)
            return 1
        if not args.quiet:
            print(f"structure_ok=True repaired_counts={repaired} file={args.repair_output}", file=sys.stderr)
        return 0

    if args.input is None:
        raise SystemExit("input instance is required unless --check-output or --repair-output is used")

    instance = parse_instance(args.input)
    solution = solve(
        instance=instance,
        seconds=args.seconds,
        seed=args.seed,
        restarts=args.restarts,
        top_k=max(1, args.top_k),
        max_optional_candidates=max(0, args.max_optional_candidates),
        mandatory_mode=args.mandatory_mode,
    )
    write_solution(args.output, solution)

    if not args.quiet:
        cleaned = len({street_id for vehicle in solution.cleaned_by_vehicle for street_id in vehicle})
        print(
            f"valid={solution.valid} score={solution.score:.6f} cleaned={cleaned} "
            f"message={solution.message}",
            file=sys.stderr,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
