from __future__ import annotations

import copy
import heapq
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Iterable, Protocol, Sequence


UNLIMITED = (1 << 64) - 1

EXPECTED_REPAIR_FAILURES = {
    "cell timeout",
    "grammar repeated without progress",
    "RSR produced no unseen repair candidates",
    "RSR produced no queryable repair candidates",
    "maximum repair-iteration limit exhausted",
    "minimum-cost RSR candidate exceeds maximum candidate length",
    "maximum RSR queue size exceeded",
    "edit-distance cap reached",
}


@dataclass
class Config:
    n: int
    ngrams_batch_size: int
    max_candidate_length: int
    max_iterations: int
    max_states: int
    max_queue_size: int
    max_rsr_candidates: int
    positive_capacity: int
    negative_capacity: int
    seed: int = 0


@dataclass
class RepairCandidate:
    value: str
    edit_distance: int
    ngram_score: float = 0.0


@dataclass
class RsrIterationMeasurement:
    minimum_edit_cost: int = 0
    unique_candidates: int = 0
    candidates_after_ngrams: int = 0
    enumeration_complete: bool = False


@dataclass
class AlgorithmMeasurements:
    rsr_execution_time_ns: int = 0
    oracle_total_calls: int = 0
    oracle_candidates_submitted: int = 0
    oracle_execution_time_ns: int = 0
    rpni2_execution_time_ns: int = 0
    acceptance_prefix_ns: int = 0
    split_execution_time_ns: int = 0
    ngrams_execution_time_ns: int = 0
    total_iterations: int = 0
    rsr_iterations: list[RsrIterationMeasurement] = field(default_factory=list)


@dataclass
class DfaState:
    accepting: bool = False
    transitions: dict[int, int] = field(default_factory=dict)
    prefix: str | None = None


class Automaton:
    def __init__(self) -> None:
        self.states: list[DfaState] = []
        self.start = 0

    def add_state(self, accepting: bool = False, prefix: str | None = None) -> int:
        if len(self.states) > 0xFFFFFFFF:
            raise RuntimeError("DFA state identifier overflow")
        self.states.append(DfaState(accepting, prefix=prefix))
        return len(self.states) - 1

    def set_start_state(self, state: int) -> None:
        self.start = state

    def resolve(self, state: int) -> int:
        if not 0 <= state < len(self.states):
            raise RuntimeError("invalid DFA state")
        return state

    def transition(self, source: int, symbol: int) -> int | None:
        return self.states[self.resolve(source)].transitions.get(symbol)

    def accepts(self, value: str) -> bool:
        current = self.resolve(self.start)
        for symbol in _symbols(value):
            destination = self.transition(current, symbol)
            if destination is None:
                return False
            current = self.resolve(destination)
        return self.states[current].accepting

    def path(self, value: str) -> list[int]:
        current = self.resolve(self.start)
        result = [current]
        for symbol in _symbols(value):
            destination = self.transition(current, symbol)
            if destination is None:
                return []
            current = self.resolve(destination)
            result.append(current)
        return result

    def fingerprint(self) -> str:
        result = f"start={self.resolve(self.start)};"
        for state_id, state in enumerate(self.states):
            result += f"{state_id}:{int(state.accepting)}[{state_id},]{{"
            for symbol, destination in sorted(state.transitions.items()):
                result += f"{symbol}>{self.resolve(destination)},"
            result += "};"
        return result

    def clone(self) -> Automaton:
        return copy.deepcopy(self)


def _symbols(value: str) -> Iterable[int]:
    return map(ord, value)


def levenshtein_distance(left: str, right: str) -> int:
    """Return unit-cost insertion, deletion, and substitution distance."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(min(
                current[-1] + 1,
                previous[right_index] + 1,
                previous[right_index - 1] + (left_character != right_character),
            ))
        previous = current
    return previous[-1]


def build_pta(positives: Sequence[str], max_states: int) -> Automaton:
    automaton = Automaton()
    root = automaton.add_state(prefix="")
    automaton.set_start_state(root)
    for value in sorted(set(positives), key=lambda item: (len(item), item)):
        current = root
        if not value:
            automaton.states[current].accepting = True
            continue
        for position, symbol in enumerate(_symbols(value), 1):
            next_state = automaton.transition(current, symbol)
            if next_state is None:
                if len(automaton.states) >= max_states:
                    raise RuntimeError("maximum DFA state limit exceeded while building PTA")
                next_state = automaton.add_state(prefix=value[:position])
                automaton.states[current].transitions[symbol] = next_state
            current = next_state
        automaton.states[current].accepting = True
    return automaton


class NGramModel:
    START = -1
    END = 256

    def __init__(self, samples: Sequence[str], n: int, additional_alphabet: str = "") -> None:
        self.n = n
        self.vocabulary = {symbol for sample in samples for symbol in _symbols(sample)}
        self.vocabulary.update(_symbols(additional_alphabet))
        self.vocabulary.add(self.END)
        self.counts: dict[tuple[int, ...], dict[int, int]] = {}
        self.totals: dict[tuple[int, ...], int] = {}
        if not n:
            return
        for sample in samples:
            context = [self.START] * (n - 1)
            for symbol in _symbols(sample):
                self._observe(context, symbol)
                self._advance(context, symbol)
            self._observe(context, self.END)

    def _observe(self, context: list[int], token: int) -> None:
        key = tuple(context)
        token_counts = self.counts.setdefault(key, {})
        token_counts[token] = token_counts.get(token, 0) + 1
        self.totals[key] = self.totals.get(key, 0) + 1

    def _advance(self, context: list[int], token: int) -> None:
        if self.n > 1:
            del context[0]
            context.append(token)

    def score(self, value: str) -> float:
        if not self.n:
            return 0.0
        context = [self.START] * (self.n - 1)
        result = 0.0
        for symbol in _symbols(value):
            result += self._log_probability(context, symbol)
            self._advance(context, symbol)
        return result + self._log_probability(context, self.END)

    def _log_probability(self, context: list[int], token: int) -> float:
        key = tuple(context)
        count = self.counts.get(key, {}).get(token, 0)
        total = self.totals.get(key, 0)
        return math.log((count + 1.0) / (total + float(len(self.vocabulary))))


class PartitionedDfa:
    def __init__(self, base: Automaton) -> None:
        count = len(base.states)
        self.base = base
        self.parent = list(range(count))
        self.accepting = [state.accepting for state in base.states]
        self.overlays: list[dict[int, int] | None] = [None] * count

    def find(self, state: int) -> int:
        if not 0 <= state < len(self.parent):
            raise RuntimeError("invalid partition state")
        while self.parent[state] != state:
            state = self.parent[state]
        return state

    def roots(self) -> list[int]:
        return [state for state in range(len(self.parent)) if self.find(state) == state]

    def members(self, state: int) -> list[int]:
        root = self.find(state)
        return [original for original in range(len(self.parent)) if self.find(original) == root]

    def blocks(self) -> list[set[int]]:
        return [set(self.members(root)) for root in self.roots()]

    @classmethod
    def from_blocks(cls, base: Automaton,
                    blocks: Sequence[set[int]]) -> PartitionedDfa:
        result = cls(base)
        if set().union(*blocks) != set(range(len(base.states))) or sum(map(len, blocks)) != len(base.states):
            raise RuntimeError("partition blocks do not cover the PTA exactly once")
        for block in blocks:
            root = min(block)
            for state in block:
                result.parent[state] = root
            result.accepting[root] = any(base.states[state].accepting for state in block)
        for block in blocks:
            root = result.find(next(iter(block)))
            transitions: dict[int, int] = {}
            for state in block:
                for symbol, destination in base.states[state].transitions.items():
                    target = result.find(destination)
                    if symbol in transitions and transitions[symbol] != target:
                        raise RuntimeError("partition is not a deterministic PTA quotient")
                    transitions[symbol] = target
            result.overlays[root] = transitions
        return result

    def _transition_map(self, state: int) -> dict[int, int]:
        root = self.find(state)
        return self.overlays[root] if self.overlays[root] is not None else self.base.states[root].transitions

    def build_cache(self, deadline: float | None = None) -> tuple[list[dict[int, int]], list[bool], list[int]]:
        transitions = [{} for _ in self.parent]
        accepting = [False] * len(self.parent)
        roots = self.roots()
        for root in roots:
            _check_deadline(deadline)
            accepting[root] = self.accepting[root]
            transitions[root] = {
                symbol: self.find(destination)
                for symbol, destination in sorted(self._transition_map(root).items())
            }
        return transitions, accepting, roots

    def accepts(self, base: Automaton, value: str) -> bool:
        current = self.find(base.start)
        for symbol in _symbols(value):
            destination = self._transition_map(current).get(symbol)
            if destination is None:
                return False
            current = self.find(destination)
        return self.accepting[current]

    def materialize(self, base: Automaton) -> Automaton:
        transitions, accepting, roots = self.build_cache()
        result = Automaton()
        dense = {root: result.add_state(accepting[root]) for root in roots}
        result.set_start_state(dense[self.find(base.start)])
        for root in roots:
            for symbol, destination in transitions[root].items():
                result.states[dense[root]].transitions[symbol] = dense[self.find(destination)]
        return result


def rejects_all(base: Automaton, partition: PartitionedDfa, negatives: Iterable[str],
                deadline: float | None = None) -> bool:
    transitions, accepting, _ = partition.build_cache(deadline)
    start = partition.find(base.start)
    for value in negatives:
        _check_deadline(deadline)
        current = start
        for symbol in _symbols(value):
            _check_deadline(deadline)
            destination = transitions[current].get(symbol)
            if destination is None:
                break
            current = destination
        else:
            if accepting[current]:
                return False
    return True


def _elapsed(start: int) -> int:
    return time.perf_counter_ns() - start


def _check_deadline(deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise RuntimeError("cell timeout")


def _state_order(base: Automaton) -> list[int]:
    if any(state.prefix is None for state in base.states):
        raise RuntimeError("RPNI2 requires PTA prefix labels")
    return sorted(range(len(base.states)),
                  key=lambda state: (len(base.states[state].prefix or ""),
                                     base.states[state].prefix or ""))


def _split_state(blocks: list[set[int]], state: int) -> bool:
    block = next(block for block in blocks if state in block)
    if len(block) == 1:
        return False
    block.remove(state)
    blocks.append({state})
    return True


def _split_with_predecessor(base: Automaton, blocks: list[set[int]], state: int,
                            predecessors: dict[int, tuple[int, int]],
                            rank: dict[int, int], deadline: float | None = None) -> None:
    pending = {state}
    while pending:
        _check_deadline(deadline)
        current = max(pending, key=rank.__getitem__)
        pending.remove(current)
        if not _split_state(blocks, current):
            continue
        pending.update(base.states[current].transitions.values())
        if current not in predecessors:
            continue
        previous, symbol = predecessors[current]
        block_of = {member: index for index, block in enumerate(blocks) for member in block}
        previous_block = blocks[block_of[previous]]
        target = block_of[current]
        if any(symbol in base.states[other].transitions and
               block_of[base.states[other].transitions[symbol]] != target
               for other in previous_block if other != previous):
            pending.add(previous)


def _tail_accepts(base: Automaton, state: int, suffix: str,
                  deadline: float | None = None) -> bool:
    current = state
    for symbol in _symbols(suffix):
        _check_deadline(deadline)
        destination = base.transition(current, symbol)
        if destination is None:
            return False
        current = destination
    return base.states[current].accepting


def _acceptance_prefix_rank(base: Automaton, partition: PartitionedDfa,
                            negatives: set[str], rank: dict[int, int],
                            deadline: float | None = None) -> int:
    result = len(base.states)
    for value in negatives:
        _check_deadline(deadline)
        if not partition.accepts(base, value):
            continue
        prefix_state = base.start
        position = 0
        for position, symbol in enumerate(_symbols(value), 1):
            _check_deadline(deadline)
            destination = base.transition(prefix_state, symbol)
            if destination is None:
                position -= 1
                break
            prefix_state = destination
        witnesses = [
            state for state in partition.members(prefix_state)
            if _tail_accepts(base, state, value[position:], deadline)
        ]
        if witnesses:
            result = min(result, *(rank[state] for state in witnesses))
        else:
            # The paper's witness is only guaranteed for a characteristic sample.
            result = 1
    if result == len(base.states):
        raise RuntimeError("negative refinement requested for a compatible DFA")
    return result


def refine_for_negatives(base: Automaton, partition: PartitionedDfa,
                         negatives: set[str],
                         measurements: AlgorithmMeasurements | None = None,
                         deadline: float | None = None) -> PartitionedDfa:
    """RPNI2 acceptance-prefix refinement followed by splitting determinization."""
    if rejects_all(base, partition, negatives, deadline):
        return partition
    order = _state_order(base)
    rank = {state: position for position, state in enumerate(order)}
    started_prefix = time.perf_counter_ns()
    first_split = _acceptance_prefix_rank(base, partition, negatives, rank, deadline)
    if measurements:
        measurements.acceptance_prefix_ns += _elapsed(started_prefix)
    blocks = partition.blocks()
    predecessors = {
        destination: (source, symbol)
        for source, pta_state in enumerate(base.states)
        for symbol, destination in pta_state.transitions.items()
    }
    started_split = time.perf_counter_ns()
    for state in reversed(order[first_split:]):
        _check_deadline(deadline)
        _split_with_predecessor(base, blocks, state, predecessors, rank, deadline)
    if measurements:
        measurements.split_execution_time_ns += _elapsed(started_split)
    refined = PartitionedDfa.from_blocks(base, blocks)
    if not rejects_all(base, refined, negatives, deadline):
        # A non-characteristic sample can make a split revive an older negative.
        refined = PartitionedDfa(base)
        if not rejects_all(base, refined, negatives, deadline):
            conflict = next(value for value in negatives if base.accepts(value))
            raise RuntimeError(
                f"the same string is both a positive and negative example: {conflict!r}")
    return refined


def extend_positive(base: Automaton, partition: PartitionedDfa,
                    value: str, max_states: int) -> tuple[Automaton, PartitionedDfa]:
    """Apply RPNI2's shortest-suffix positive extension to the PTA quotient."""
    existing = base.start
    first_missing = len(value)
    for position, symbol in enumerate(_symbols(value)):
        destination = base.transition(existing, symbol)
        if destination is None:
            first_missing = position
            break
        existing = destination
    if len(base.states) + len(value) - first_missing > max_states:
        raise RuntimeError("maximum DFA state limit exceeded while extending PTA")

    blocks = partition.blocks()
    block_of = {state: index for index, block in enumerate(blocks) for state in block}
    quotient_transitions = {
        root: dict(partition._transition_map(root)) for root in partition.roots()
    }
    pta_state = base.start
    quotient_state = partition.find(base.start)
    extending = False
    for position, symbol in enumerate(_symbols(value), 1):
        destination = base.transition(pta_state, symbol)
        if destination is not None:
            pta_state = destination
            quotient_state = partition.find(destination)
            continue
        destination = base.add_state(prefix=value[:position])
        base.states[pta_state].transitions[symbol] = destination
        pta_state = destination
        if not extending:
            quotient_destination = quotient_transitions[quotient_state].get(symbol)
            if quotient_destination is not None:
                quotient_state = partition.find(quotient_destination)
                blocks[block_of[quotient_state]].add(destination)
                continue
            extending = True
        blocks.append({destination})
    base.states[pta_state].accepting = True
    return base, PartitionedDfa.from_blocks(base, blocks)


@dataclass(frozen=True)
class _Predecessor:
    previous_row: int
    previous_column: int
    operation: str
    symbol: int = 0


def _relax(candidate_cost: int, predecessor: _Predecessor, row: int, column: int,
           costs: list[list[int]], history: list[list[list[_Predecessor]]]) -> bool:
    if candidate_cost < costs[row][column]:
        costs[row][column] = candidate_cost
        history[row][column] = [predecessor]
        return True
    if candidate_cost == costs[row][column] and predecessor not in history[row][column]:
        history[row][column].append(predecessor)
    return False


def rsr_minimum_repairs(automaton: Automaton, input_value: str,
                        max_candidate_length: int = UNLIMITED,
                        max_rsr_candidates: int = UNLIMITED,
                        max_queue_size: int = UNLIMITED,
                        measurement: RsrIterationMeasurement | None = None,
                        deadline: float | None = None) -> list[RepairCandidate]:
    if not max_candidate_length or not max_rsr_candidates or not max_queue_size:
        raise ValueError("RSR candidate limits must be positive")
    edges = []
    for source, state in enumerate(automaton.states):
        _check_deadline(deadline)
        edges.extend((source, automaton.resolve(destination), symbol)
                     for symbol, destination in sorted(state.transitions.items()))
    edge_count, phi = len(edges), len(edges)
    incoming = [[] for _ in automaton.states]
    outgoing = [[] for _ in automaton.states]
    for edge_id, (source, destination, _) in enumerate(edges):
        _check_deadline(deadline)
        incoming[destination].append(edge_id)
        outgoing[source].append(edge_id)
    previous_edges: list[list[int]] = []
    next_edges: list[list[int]] = []
    start = automaton.resolve(automaton.start)
    for source, destination, _ in edges:
        previous = list(incoming[source])
        if source == start:
            previous.append(phi)
        previous_edges.append(previous)
        next_edges.append(list(outgoing[destination]))
    input_symbols = list(_symbols(input_value))
    infinity = UNLIMITED // 4
    costs = [[infinity] * (edge_count + 1) for _ in range(len(input_symbols) + 1)]
    history: list[list[list[_Predecessor]]] = [
        [[] for _ in range(edge_count + 1)] for _ in range(len(input_symbols) + 1)
    ]
    for row in range(len(input_symbols) + 1):
        costs[row][phi] = row
        if row:
            history[row][phi].append(_Predecessor(row - 1, phi, "deletion"))
    queue: list[tuple[int, int]] = []

    def push(item: tuple[int, int]) -> None:
        if max_queue_size != UNLIMITED and len(queue) >= max_queue_size:
            raise RuntimeError("maximum RSR queue size exceeded")
        heapq.heappush(queue, item)

    for edge_id, (source, _, symbol) in enumerate(edges):
        if source == start:
            _relax(1, _Predecessor(0, phi, "insertion", symbol), 0, edge_id, costs, history)
            push((1, edge_id))
    while queue:
        _check_deadline(deadline)
        cost, edge_id = heapq.heappop(queue)
        if cost != costs[0][edge_id]:
            continue
        for next_edge in next_edges[edge_id]:
            symbol = edges[next_edge][2]
            if _relax(cost + 1, _Predecessor(0, edge_id, "insertion", symbol),
                      0, next_edge, costs, history):
                push((cost + 1, next_edge))
    for row in range(1, len(input_symbols) + 1):
        _check_deadline(deadline)
        input_symbol = input_symbols[row - 1]
        for edge_id, (_, _, symbol) in enumerate(edges):
            _check_deadline(deadline)
            if costs[row - 1][edge_id] != infinity:
                _relax(costs[row - 1][edge_id] + 1,
                       _Predecessor(row - 1, edge_id, "deletion"),
                       row, edge_id, costs, history)
            for predecessor in previous_edges[edge_id]:
                if costs[row - 1][predecessor] == infinity:
                    continue
                operation = "match" if symbol == input_symbol else "substitution"
                _relax(costs[row - 1][predecessor] + (operation != "match"),
                       _Predecessor(row - 1, predecessor, operation, symbol),
                       row, edge_id, costs, history)
        queue = [(costs[row][edge_id], edge_id) for edge_id in range(edge_count)
                 if costs[row][edge_id] != infinity]
        if max_queue_size != UNLIMITED and len(queue) > max_queue_size:
            raise RuntimeError("maximum RSR queue size exceeded")
        heapq.heapify(queue)
        while queue:
            _check_deadline(deadline)
            cost, edge_id = heapq.heappop(queue)
            if cost != costs[row][edge_id]:
                continue
            for next_edge in next_edges[edge_id]:
                symbol = edges[next_edge][2]
                if _relax(cost + 1, _Predecessor(row, edge_id, "insertion", symbol),
                          row, next_edge, costs, history):
                    push((cost + 1, next_edge))
    minimum = costs[-1][phi] if automaton.states[start].accepting else infinity
    for edge_id, (_, destination, _) in enumerate(edges):
        if automaton.states[destination].accepting:
            minimum = min(minimum, costs[-1][edge_id])
    if minimum == infinity:
        return []
    final_columns = []
    if automaton.states[start].accepting and costs[-1][phi] == minimum:
        final_columns.append(phi)
    final_columns.extend(edge_id for edge_id, (_, destination, _) in enumerate(edges)
                         if automaton.states[destination].accepting and costs[-1][edge_id] == minimum)
    values: set[str] = set()
    reversed_symbols: list[int] = []

    def enumerate_values(row: int, column: int) -> None:
        stack = [(row, column, 0, False)]
        while stack and len(values) < max_rsr_candidates:
            _check_deadline(deadline)
            row, column, next_predecessor, emitted = stack[-1]
            if row == 0 and column == phi:
                values.add("".join(map(chr, reversed(reversed_symbols))))
                stack.pop()
                if emitted:
                    reversed_symbols.pop()
                continue
            predecessors = history[row][column]
            if next_predecessor == len(predecessors):
                stack.pop()
                if emitted:
                    reversed_symbols.pop()
                continue
            predecessor = predecessors[next_predecessor]
            stack[-1] = (row, column, next_predecessor + 1, emitted)
            child_emits = predecessor.operation in {"match", "insertion", "substitution"}
            if child_emits:
                if len(reversed_symbols) == max_candidate_length:
                    raise RuntimeError("minimum-cost RSR candidate exceeds maximum candidate length")
                reversed_symbols.append(predecessor.symbol)
            stack.append((predecessor.previous_row, predecessor.previous_column,
                          0, child_emits))

    for column in final_columns:
        enumerate_values(len(input_symbols), column)
        if len(values) >= max_rsr_candidates:
            break
    repairs = [RepairCandidate(value, minimum) for value in sorted(values)]
    if measurement:
        measurement.minimum_edit_cost = minimum
        measurement.unique_candidates = len(repairs)
        measurement.enumeration_complete = max_rsr_candidates == UNLIMITED or len(repairs) < max_rsr_candidates
    return repairs


def rsr_repair(automaton: Automaton, input_value: str) -> RepairCandidate | None:
    repairs = rsr_minimum_repairs(automaton, input_value, UNLIMITED, 1)
    return repairs[0] if repairs else None


def reject_exact_string_in_place(automaton: Automaton, value: str) -> bool:
    if not automaton.accepts(value):
        return False
    source_path = automaton.path(value)
    if len(source_path) != len(value) + 1:
        raise RuntimeError("accepted string has no complete DFA path")
    prefix_states = [
        automaton.add_state(position != len(value) and automaton.states[source].accepting)
        for position, source in enumerate(source_path)
    ]
    for position, source in enumerate(source_path):
        automaton.states[prefix_states[position]].transitions.update(automaton.states[source].transitions)
        if position < len(value):
            automaton.states[prefix_states[position]].transitions[ord(value[position])] = prefix_states[position + 1]
    automaton.set_start_state(prefix_states[0])
    return True


def reject_exact_string(source: Automaton, value: str) -> Automaton:
    result = source.clone()
    reject_exact_string_in_place(result, value)
    return result


def rsr_repairs(automaton: Automaton, input_value: str, model: NGramModel,
                ngrams_batch_size: int, max_candidate_length: int,
                max_rsr_candidates: int, max_queue_size: int,
                excluded: Iterable[str] = (), measurement: RsrIterationMeasurement | None = None,
                measurements: AlgorithmMeasurements | None = None,
                deadline: float | None = None) -> list[RepairCandidate]:
    if not all((ngrams_batch_size, max_candidate_length, max_rsr_candidates, max_queue_size)):
        raise ValueError("configured counts and resource limits must be positive")
    working = automaton.clone()
    for value in sorted(excluded):
        _check_deadline(deadline)
        reject_exact_string_in_place(working, value)
    repairs = rsr_minimum_repairs(working, input_value, max_candidate_length,
                                  max_rsr_candidates, max_queue_size,
                                  measurement, deadline)
    started = time.perf_counter_ns()
    for repair in repairs:
        _check_deadline(deadline)
        repair.ngram_score = model.score(repair.value)
    repairs.sort(key=lambda repair: (-repair.ngram_score, repair.edit_distance, repair.value))
    if measurements:
        measurements.ngrams_execution_time_ns += _elapsed(started)
    if ngrams_batch_size != UNLIMITED:
        del repairs[ngrams_batch_size:]
    if measurement:
        measurement.candidates_after_ngrams = len(repairs)
    return repairs


class Oracle(Protocol):
    def accepts_batch(self, values: Sequence[str],
                      deadline: float | None = None) -> list[bool]: ...


class ExampleFifo:
    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("negative capacity must be positive")
        self.capacity = capacity
        self.order: deque[str] = deque()
        self.members: set[str] = set()

    def add(self, value: str) -> bool:
        if value in self.members:
            return False
        if len(self.order) == self.capacity:
            self.members.remove(self.order.popleft())
        self.order.append(value)
        self.members.add(value)
        return True

    def values(self) -> set[str]:
        return set(self.order)

    def values_with(self, value: str) -> set[str]:
        values = self.values()
        if value not in values and len(values) == self.capacity:
            values.remove(self.order[0])
        values.add(value)
        return values

    def __len__(self) -> int:
        return len(self.order)

    def __contains__(self, value: str) -> bool:
        return value in self.members

    def __iter__(self):
        return iter(self.order)


def _oracle_accepts(oracle: Oracle, values: Sequence[str],
                    measurements: AlgorithmMeasurements,
                    deadline: float | None = None) -> list[bool]:
    if not values:
        return []
    measurements.oracle_total_calls += 1
    measurements.oracle_candidates_submitted += len(values)
    started = time.perf_counter_ns()
    result = oracle.accepts_batch(values, deadline)
    measurements.oracle_execution_time_ns += _elapsed(started)
    if len(result) != len(values):
        raise RuntimeError("oracle result count mismatch")
    if any(type(item) is not bool for item in result):
        raise RuntimeError("oracle result is not boolean")
    return result


class BatchSession:
    def __init__(self, positives: Sequence[str], config: Config,
                 measurements: AlgorithmMeasurements | None = None,
                 negatives: Sequence[str] = ()) -> None:
        self.config = config
        self.positives = ExampleFifo(config.positive_capacity)
        self.negatives = ExampleFifo(config.negative_capacity)
        for value in positives:
            self.positives.add(value)
        for value in negatives:
            self.negatives.add(value)
        self.all_positives = set(positives)
        self.pta = build_pta(list(self.all_positives), config.max_states)
        started = time.perf_counter_ns()
        self.model = NGramModel(list(self.all_positives), config.n)
        if measurements:
            measurements.ngrams_execution_time_ns += _elapsed(started)
        self.partition = PartitionedDfa.from_blocks(
            self.pta, [set(range(len(self.pta.states)))])
        started = time.perf_counter_ns()
        learned_negatives: set[str] = set()
        for value in self.negatives:
            learned_negatives.add(value)
            if self.partition.accepts(self.pta, value):
                self.partition = refine_for_negatives(
                    self.pta, self.partition, learned_negatives, measurements)
        if measurements:
            measurements.rpni2_execution_time_ns += _elapsed(started)

    @staticmethod
    def _check_deadline(deadline: float) -> None:
        _check_deadline(deadline)

    def _learn_negative(self, value: str, deadline: float,
                        measurements: AlgorithmMeasurements) -> None:
        if value in self.negatives:
            return
        if not self.partition.accepts(self.pta, value):
            self.negatives.add(value)
            return
        negatives = self.negatives.values_with(value)
        self._check_deadline(deadline)
        started = time.perf_counter_ns()
        self.partition = refine_for_negatives(
            self.pta, self.partition, negatives, measurements, deadline)
        measurements.rpni2_execution_time_ns += _elapsed(started)
        self._check_deadline(deadline)
        self.negatives.add(value)

    def _learn_positive(self, value: str, deadline: float,
                        measurements: AlgorithmMeasurements) -> None:
        if value in self.all_positives:
            self.positives.add(value)
            return
        self._check_deadline(deadline)
        started = time.perf_counter_ns()
        self.pta, self.partition = extend_positive(
            self.pta, self.partition, value, self.config.max_states)
        self.all_positives.add(value)
        self.positives.add(value)
        if not rejects_all(self.pta, self.partition, self.negatives.values(), deadline):
            self.partition = refine_for_negatives(
                self.pta, self.partition, self.negatives.values(), measurements, deadline)
        measurements.rpni2_execution_time_ns += _elapsed(started)
        started = time.perf_counter_ns()
        self.model = NGramModel(list(self.all_positives), self.config.n)
        measurements.ngrams_execution_time_ns += _elapsed(started)
        if not self.partition.accepts(self.pta, value):
            raise RuntimeError("positive example was not learned")
        self._check_deadline(deadline)

    def repair_candidates(self, dirty: str, oracle: Oracle, deadline: float,
                          measurements: AlgorithmMeasurements,
                          num_candidates: int,
                          edit_distance_cap: int) -> tuple[list[str], str]:
        """Collect distinct accepted repairs and retain partial results on failure."""
        if num_candidates <= 0:
            raise ValueError("num_candidates must be positive")
        if edit_distance_cap < 0:
            raise ValueError("edit_distance_cap must be non-negative")
        found: list[str] = []
        failure = ""
        try:
            self._learn_negative(dirty, deadline, measurements)
            fingerprints: set[tuple[str, tuple[str, ...]]] = set()
            for _ in range(self.config.max_iterations):
                self._check_deadline(deadline)
                measurements.total_iterations += 1
                working = self.partition.materialize(self.pta)
                fingerprint = (working.fingerprint(), tuple(found))
                if fingerprint in fingerprints:
                    raise RuntimeError("grammar repeated without progress")
                fingerprints.add(fingerprint)
                started = time.perf_counter_ns()
                ngrams_before = measurements.ngrams_execution_time_ns
                rsr_measurement = RsrIterationMeasurement()
                measurements.rsr_iterations.append(rsr_measurement)
                excluded = self.negatives.values() | set(found)
                repairs = rsr_repairs(
                    working, dirty, self.model, self.config.ngrams_batch_size,
                    self.config.max_candidate_length, self.config.max_rsr_candidates,
                    self.config.max_queue_size, excluded, rsr_measurement,
                    measurements, deadline,
                )
                combined = _elapsed(started)
                ngrams = measurements.ngrams_execution_time_ns - ngrams_before
                measurements.rsr_execution_time_ns += max(0, combined - ngrams)
                self._check_deadline(deadline)
                if not repairs:
                    raise RuntimeError("RSR produced no unseen repair candidates")
                candidates: list[str] = []
                cap_reached = False
                for repair in repairs:
                    if repair.value in excluded:
                        continue
                    if levenshtein_distance(dirty, repair.value) > edit_distance_cap:
                        cap_reached = True
                        break
                    candidates.append(repair.value)
                if not candidates:
                    if cap_reached:
                        raise RuntimeError("edit-distance cap reached")
                    raise RuntimeError("RSR produced no queryable repair candidates")
                position = 0
                while position < len(candidates) and len(found) < num_candidates:
                    values = candidates[position:]
                    accepted = _oracle_accepts(oracle, values, measurements, deadline)
                    answered = list(zip(values, accepted))
                    found.extend(candidate for candidate, result in answered if result)
                    for candidate, result in answered:
                        if not result:
                            self._learn_negative(candidate, deadline, measurements)
                    position += len(accepted)
                if len(found) >= num_candidates:
                    break
                if cap_reached:
                    raise RuntimeError("edit-distance cap reached")
            else:
                raise RuntimeError("maximum repair-iteration limit exhausted")
        except RuntimeError as error:
            if str(error) not in EXPECTED_REPAIR_FAILURES:
                raise
            failure = str(error)

        if failure != "cell timeout":
            try:
                for candidate in found:
                    self._learn_positive(candidate, deadline, measurements)
            except RuntimeError as error:
                if str(error) != "cell timeout":
                    raise
                failure = "cell timeout"
        return found, failure

    def repair(self, dirty: str, oracle: Oracle, deadline: float,
               measurements: AlgorithmMeasurements) -> str:
        candidates, failure = self.repair_candidates(
            dirty, oracle, deadline, measurements, 1, UNLIMITED)
        if candidates:
            return candidates[0]
        raise RuntimeError(failure)
