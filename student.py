import asyncio
import getpass
import json
import os
from collections import defaultdict, deque
from typing import List, Tuple, Optional, Set
import math

DIRECTIONS = {
    'w': (0, -1),
    's': (0, 1),
    'a': (-1, 0),
    'd': (1, 0),
}
LATE_GAME_MUSHROOM_THRESHOLD = 140
# Representa uma posição (x,y) no grid. Helpers básicos.
class Position:
    def __init__(self, x: int, y: int):
        self.x = x
        self.y = y
    def __eq__(self, other):
        return isinstance(other, Position) and self.x == other.x and self.y == other.y
    def __hash__(self):
        return hash((self.x, self.y))
    def __repr__(self):
        return f"Pos({self.x},{self.y})"
    def to_tuple(self):
        return (self.x, self.y)
    def manhattan_distance(self, other):
        return abs(self.x - other.x) + abs(self.y - other.y)
    def distance(self, other):
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
# Agente principal: guarda estado, avalia estratégias e decide ações.
class CentipedeAgent:
    def __init__(self):
        self.game_state = None
        self.map_size = None
        self.shot_cooldown = 0
        self.predicted_positions = {}
        self.danger_zones = set()
        self.centipede_position_history = defaultdict(lambda: deque(maxlen=15))
        self.stuck_centipedes = set()
        self.blaster_position_history = deque(maxlen=30)
        self.blaster_column_history = deque(maxlen=30)
        self.self_stuck_detected = False
        self.self_stuck_cooldown = 0
        self.last_target_name = None
        self.late_game = False
        self.camping_mode = False
        self.camping_target_name = None
        self.current_strategy = "aggressive"
        self.safe_zone_start = None
        self.frame_count = 0
        self.debug_info = {}
    # Atualiza estado do jogo e caches (tamanho mapa, cooldowns, perigos, estratégia)
    def update_state(self, state: dict):
        self.game_state = state
        self.frame_count += 1
        if self.map_size is None:
            self.map_size = state.get('size', (40, 24))
            self.safe_zone_start = self.map_size[1] - 5
        mushroom_count = len(state.get('mushrooms', []))
        was_late_game = self.late_game
        self.late_game = mushroom_count >= LATE_GAME_MUSHROOM_THRESHOLD
        if self.late_game and not was_late_game:
            pass
        elif not self.late_game and was_late_game:
            pass
            if self.camping_mode:
                pass
                self.camping_mode = False
                self.camping_target_name = None
        if self.shot_cooldown > 0:
            self.shot_cooldown -= 1
        self.update_centipede_predictions()
        self.update_danger_zones()
        self.evaluate_strategy()
    # Prediz movimentos das centipedes (até 5 passos) para evitar/antecipar colisões
    def update_centipede_predictions(self):
        self.predicted_positions = {}
        centipedes = self.game_state.get('centipedes', [])
        mushrooms = self.get_mushroom_positions()
        for centipede in centipedes:
            name = centipede['name']
            body = centipede['body']
            direction = centipede['direction']  # 0=N, 1=E, 2=S, 3=W
            if not body:
                continue
            head = body[-1]
            head_tuple = tuple(head)
            self.centipede_position_history[name].append(head_tuple)
            predictions = []
            sim_head = list(head)
            sim_dir = direction
            move_dir = 1  # 1 = down, -1 = up
            for step in range(5):
                if sim_dir == 1:  # EAST
                    next_x = sim_head[0] + 1
                elif sim_dir == 3:  # WEST
                    next_x = sim_head[0] - 1
                else:
                    next_x = sim_head[0]
                next_pos = [next_x, sim_head[1]]
                hit_obstacle = False
                if next_x < 0 or next_x >= self.map_size[0]:
                    hit_obstacle = True
                elif tuple(next_pos) in mushrooms:
                    hit_obstacle = True
                if hit_obstacle:
                    if sim_head[1] == 0:
                        move_dir = 1
                    elif sim_head[1] >= self.map_size[1] - 1:
                        move_dir = -1
                    next_pos = [sim_head[0], sim_head[1] + move_dir]
                    if sim_dir == 1:  # EAST -> WEST
                        sim_dir = 3
                    elif sim_dir == 3:  # WEST -> EAST
                        sim_dir = 1
                sim_head = next_pos
                predictions.append(tuple(sim_head))
            self.predicted_positions[name] = predictions
    # Marca centipedes como "presas" se quase não mexem nas últimas frames
    def detect_stuck_centipedes(self) -> List[str]:
        stuck = []
        for name, history in self.centipede_position_history.items():
            if len(history) >= 10:
                recent_positions = list(history)[-10:]
                unique_positions = set(recent_positions)
                if len(unique_positions) <= 2:
                    stuck.append(name)
                    if name not in self.stuck_centipedes:
                        pass
                        self.stuck_centipedes.add(name)
                else:
                    if name in self.stuck_centipedes:
                        pass
                        self.stuck_centipedes.discard(name)
        return stuck
    # Detecta se nós próprios estamos num ciclo horizontal chato (ADAD...) e sem progresso
    def detect_self_stuck(self) -> bool:
        if self.camping_mode:
            return False
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            return False
        my_pos = Position(*bug_blaster['pos'])
        self.blaster_position_history.append(my_pos.to_tuple())
        self.blaster_column_history.append(my_pos.x)
        if len(self.blaster_column_history) < 20:
            return False
        centipedes = self.game_state.get('centipedes', [])
        if not centipedes:
            return False
        recent_columns = list(self.blaster_column_history)[-20:]
        unique_columns = set(recent_columns)
        pattern_detected = False
        if len(unique_columns) == 2:
            col_a, col_b = sorted(unique_columns)
            alternations = 0
            for i in range(1, len(recent_columns)):
                if recent_columns[i] != recent_columns[i-1]:
                    alternations += 1
            if alternations > 12:
                pattern_detected = True
                if self.self_stuck_cooldown == 0:
                    if not self.self_stuck_detected:
                        pass
                    self.self_stuck_detected = True
                    self.self_stuck_cooldown = 30  # Check again in 30 frames
        if not pattern_detected and len(unique_columns) <= 3:
            num_stuck = sum(1 for c in centipedes if c['name'] in self.stuck_centipedes)
            if num_stuck >= len(centipedes) * 0.5:  # At least half are stuck
                pattern_detected = True
                if self.self_stuck_cooldown == 0:
                    if not self.self_stuck_detected:
                        col_range = max(unique_columns) - min(unique_columns) if len(unique_columns) > 1 else 0
                    self.self_stuck_detected = True
                    self.self_stuck_cooldown = 30
        all_stuck = len(centipedes) > 0 and all(c['name'] in self.stuck_centipedes for c in centipedes)
        if not pattern_detected and all_stuck and len(centipedes) >= 2:
            aligned_with_any = False
            for centipede in centipedes:
                if centipede['body']:
                    head_x = centipede['body'][-1][0]  # Head is last element
                    if abs(my_pos.x - head_x) <= 1:  # Within 1 column
                        aligned_with_any = True
                        break
            if not aligned_with_any:
                recent_positions = list(self.blaster_position_history)[-15:]
                frames_far_from_all = 0
                for pos in recent_positions:
                    far_from_all = True
                    for centipede in centipedes:
                        if centipede['body']:
                            head_x = centipede['body'][-1][0]  # Head is last element
                            if abs(pos[0] - head_x) <= 2:
                                far_from_all = False
                                break
                    if far_from_all:
                        frames_far_from_all += 1
                if frames_far_from_all > 12:
                    pattern_detected = True
                    if self.self_stuck_cooldown == 0:
                        if not self.self_stuck_detected:
                            pass
                        self.self_stuck_detected = True
                        self.self_stuck_cooldown = 30
        if self.self_stuck_cooldown > 0:
            self.self_stuck_cooldown -= 1
        if not pattern_detected and self.self_stuck_detected:
            pass
            self.self_stuck_detected = False
        return self.self_stuck_detected
    # Calcula zonas perigosas com base no corpo das centipedes e previsões imediatas
    def update_danger_zones(self):
        self.danger_zones = set()
        centipedes = self.game_state.get('centipedes', [])
        for centipede in centipedes:
            body = centipede['body']
            name = centipede['name']
            if name in self.stuck_centipedes:
                continue
            for segment in body:
                self.danger_zones.add(tuple(segment))
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0), (1,1), (1,-1), (-1,1), (-1,-1)]:
                    adj = (segment[0] + dx, segment[1] + dy)
                    if 0 <= adj[0] < self.map_size[0] and 0 <= adj[1] < self.map_size[1]:
                        self.danger_zones.add(adj)
            predictions = self.predicted_positions.get(name, [])
            for pred in predictions[:3]:
                self.danger_zones.add(pred)
                for dx, dy in [(0,1), (0,-1), (1,0), (-1,0)]:
                    adj = (pred[0] + dx, pred[1] + dy)
                    if 0 <= adj[0] < self.map_size[0] and 0 <= adj[1] < self.map_size[1]:
                        self.danger_zones.add(adj)
    # Verifica se uma posição será ocupada pela cabeça de alguma centipede em breve
    def will_be_hit_soon(self, pos: Position, horizon: int = 2) -> bool:
        pos_tuple = pos.to_tuple()
        for name, predictions in self.predicted_positions.items():
            if name in self.stuck_centipedes:
                continue
            for step_idx in range(min(horizon, len(predictions))):
                predicted_head = predictions[step_idx]
                if predicted_head == pos_tuple:
                    pass
                    return True
        centipedes = self.game_state.get('centipedes', [])
        for centipede in centipedes:
            if centipede['name'] in self.stuck_centipedes:
                continue
            body = centipede['body']
            if body:
                current_head = tuple(body[-1])  # Head is last element
                if current_head == pos_tuple:
                    pass
                    return True
        return False
    # Entre ações candidatas, escolhe a mais segura (considerando previsões e distâncias)
    def get_safest_action_with_prediction(self, candidate_actions: list, my_pos: Position) -> Optional[str]:
        mushrooms = self.get_mushroom_positions()
        centipedes = self.game_state.get('centipedes', [])
        safe_actions = []
        risky_actions = []  # Actions that might be hit but are better than certain death
        for action in candidate_actions:
            if action == '':
                new_pos = my_pos
            else:
                if action not in DIRECTIONS:
                    continue
                dx, dy = DIRECTIONS[action]
                new_pos = Position(my_pos.x + dx, my_pos.y + dy)
            if new_pos.x < 0 or new_pos.x >= self.map_size[0]:
                continue
            if new_pos.y < 0 or new_pos.y >= self.map_size[1]:
                continue
            if new_pos.to_tuple() in mushrooms:
                continue
            immediate_collision = False
            for centipede in centipedes:
                if new_pos.to_tuple() in [tuple(seg) for seg in centipede['body']]:
                    immediate_collision = True
                    break
            if immediate_collision:
                continue
            if self.will_be_hit_soon(new_pos, horizon=2):
                min_dist = float('inf')
                for centipede in centipedes:
                    for segment in centipede['body']:
                        dist = new_pos.manhattan_distance(Position(*segment))
                        min_dist = min(min_dist, dist)
                risky_actions.append((action, min_dist))
            else:
                min_dist = float('inf')
                for centipede in centipedes:
                    for segment in centipede['body']:
                        dist = new_pos.manhattan_distance(Position(*segment))
                        min_dist = min(min_dist, dist)
                safe_actions.append((action, min_dist))
        if safe_actions:
            safe_actions.sort(key=lambda x: x[1], reverse=True)
            return safe_actions[0][0]
        if risky_actions:
            risky_actions.sort(key=lambda x: x[1], reverse=True)
            return risky_actions[0][0]
        return None
    # Escolhe estratégia (aggressive/defensive/clearing) conforme ameaças e cogumelos
    def evaluate_strategy(self):
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            self.current_strategy = "aggressive"
            return
        my_pos = Position(*bug_blaster['pos'])
        centipedes = self.game_state.get('centipedes', [])
        mushroom_count = len(self.game_state.get('mushrooms', []))
        stuck_centipedes = self.detect_stuck_centipedes()
        old_strategy = self.current_strategy
        if stuck_centipedes:
            in_safe_zone = my_pos.y >= self.safe_zone_start
            closest_is_stuck = False
            min_distance = float('inf')
            closest_centipede_name = None
            for centipede in centipedes:
                for segment in centipede['body']:
                    seg_pos = Position(*segment)
                    dist = my_pos.manhattan_distance(seg_pos)
                    if dist < min_distance:
                        min_distance = dist
                        closest_centipede_name = centipede['name']
            if closest_centipede_name and closest_centipede_name in self.stuck_centipedes:
                closest_is_stuck = True
            if in_safe_zone and closest_is_stuck:
                self.current_strategy = "aggressive"
                if old_strategy != "aggressive":
                    pass
                return
            elif min_distance >= 3:
                self.current_strategy = "aggressive"
                if old_strategy != "aggressive":
                    pass
                return
            else:
                pass  # Fall through to defensive check below
        min_distance = float('inf')
        threat_below = False
        for centipede in centipedes:
            if centipede['name'] in self.stuck_centipedes:
                continue
            for segment in centipede['body']:
                seg_pos = Position(*segment)
                dist = my_pos.manhattan_distance(seg_pos)
                if dist < min_distance:
                    min_distance = dist
                if seg_pos.y >= my_pos.y - 2:  # Near or below us
                    threat_below = True
        if min_distance < 4 and threat_below:
            all_close_threats_stuck = True
            for centipede in centipedes:
                for segment in centipede['body']:
                    seg_pos = Position(*segment)
                    dist = my_pos.manhattan_distance(seg_pos)
                    if dist < 4 and centipede['name'] not in self.stuck_centipedes:
                        all_close_threats_stuck = False
                        break
                if not all_close_threats_stuck:
                    break
            if all_close_threats_stuck and stuck_centipedes:
                self.current_strategy = "aggressive"
                if old_strategy != "aggressive":
                    pass
                return
            self.current_strategy = "defensive"
            if old_strategy != "defensive":
                pass
            return
        if mushroom_count > 150 and not self.late_game:
            self.current_strategy = "clearing"
            if old_strategy != "clearing":
                pass
            return
        self.current_strategy = "aggressive"
        if old_strategy != "aggressive":
            if self.late_game:
                pass
            else:
                pass
    # Scora segmentos e escolhe melhor alvo (bónus para stuck, alinhamento, caminho limpo)
    def find_best_target(self) -> Optional[Tuple[str, Tuple[int, int], float]]:
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            return None
        my_pos = Position(*bug_blaster['pos'])
        centipedes = self.game_state.get('centipedes', [])
        mushrooms = self.get_mushroom_positions()
        self_stuck = self.detect_self_stuck()
        targets = []
        for centipede in centipedes:
            name = centipede['name']
            body = centipede['body']
            is_stuck = name in self.stuck_centipedes
            is_current_target = (name == self.last_target_name)
            for idx, segment in enumerate(body):
                seg_pos = Position(*segment)
                if seg_pos.y >= my_pos.y:
                    continue
                score = 0.0
                if is_stuck:
                    score += 200
                    if self.late_game:
                        score += 300
                distance = abs(seg_pos.x - my_pos.x) + abs(seg_pos.y - my_pos.y)
                score -= distance * 2  # Penalty: -2 per distance unit
                if self.late_game:
                    score += seg_pos.y * 0.5  # Small bonus per Y coordinate
                score += (self.map_size[1] - seg_pos.y) * 10
                if idx == len(body) - 1:  # Last element is head
                    score += 50
                if seg_pos.x == my_pos.x:
                    score += 100
                    path_clear = True
                    for y in range(seg_pos.y + 1, my_pos.y):
                        if (seg_pos.x, y) in mushrooms:
                            path_clear = False
                            break
                    if path_clear:
                        score += 150
                if self_stuck and is_current_target:
                    score -= 100
                if distance < 3:
                    score -= 100
                predictions = self.predicted_positions.get(name, [])
                if tuple(segment) in predictions[:2]:
                    score += 30
                targets.append((name, tuple(segment), score))
        if not targets:
            return None
        targets.sort(key=lambda x: x[2], reverse=True)
        best_target = targets[0]
        best_name, best_pos, best_score = best_target
        if best_name != self.last_target_name:
            if self.last_target_name:
                old_score = next((s for n, p, s in targets if n == self.last_target_name), None)
                if old_score is not None:
                    pass
                else:
                    pass
            else:
                pass
            self.last_target_name = best_name
        if best_name in self.stuck_centipedes:
            distance = abs(best_pos[0] - my_pos.x) + abs(best_pos[1] - my_pos.y)
        return best_target
    # Procura movimento mais seguro ponderando distância a inimigos, perigos e objetivo
    def find_safe_move(self, preferred_direction: Optional[str] = None, returning_to_safe_zone: bool = False) -> str:
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            return ''
        my_pos = Position(*bug_blaster['pos'])
        mushrooms = self.get_mushroom_positions()
        centipedes = self.game_state.get('centipedes', [])
        targeting_stuck = False
        target_stuck_pos = None
        if self.last_target_name and self.last_target_name in self.stuck_centipedes:
            targeting_stuck = True
            for c in centipedes:
                if c['name'] == self.last_target_name and c['body']:
                    target_stuck_pos = Position(*c['body'][-1])  # Head is last element
                    break
        best_action = ""
        best_score = -float('inf')
        for action in ['w', 'a', 's', 'd', '']:
            if action == '':
                new_pos = my_pos
            else:
                dx, dy = DIRECTIONS[action]
                new_pos = Position(my_pos.x + dx, my_pos.y + dy)
            score = 0
            if new_pos.x < 0 or new_pos.x >= self.map_size[0]:
                score -= 1000
                continue
            if new_pos.y < 0 or new_pos.y >= self.map_size[1]:
                score -= 1000
                continue
            if new_pos.to_tuple() in mushrooms:
                base_penalty = -500
                if self.self_stuck_detected and targeting_stuck and target_stuck_pos:
                    on_path_to_target = False
                    if action in ['a', 'd']:  # Horizontal movement
                        if (my_pos.x < new_pos.x <= target_stuck_pos.x) or \
                           (target_stuck_pos.x <= new_pos.x < my_pos.x):
                            on_path_to_target = True
                    if on_path_to_target:
                        score += base_penalty * 0.6  # -300 instead of -500
                    else:
                        score += base_penalty
                else:
                    score += base_penalty
                continue
            if new_pos.to_tuple() in self.danger_zones:
                if returning_to_safe_zone and action == 's':
                    immediate_collision = False
                    for centipede in centipedes:
                        if new_pos.to_tuple() in [tuple(seg) for seg in centipede['body']]:
                            immediate_collision = True
                            break
                    if immediate_collision:
                        score -= 1000  # Lethal, avoid completely
                        continue
                    else:
                        score -= 50
                else:
                    score -= 300
            if self.will_be_hit_soon(new_pos, horizon=2):
                score -= 5000  # Massive penalty - almost always avoid
            min_centipede_dist = float('inf')
            for centipede in centipedes:
                for segment in centipede['body']:
                    seg_pos = Position(*segment)
                    dist = new_pos.manhattan_distance(seg_pos)
                    min_centipede_dist = min(min_centipede_dist, dist)
            if returning_to_safe_zone and action == 's':
                if min_centipede_dist < 2:
                    score += min_centipede_dist * 5  # Reduced from *20
                else:
                    score += 20  # Small bonus, but don't let distance dominate
            else:
                score += min_centipede_dist * 20
            if new_pos.y >= self.safe_zone_start:
                score += 50
            if returning_to_safe_zone and action == 's':
                if new_pos.y > my_pos.y or new_pos.y >= self.safe_zone_start:
                    score += 500  # Dominant bonus to override distance penalties
            if preferred_direction and action == preferred_direction:
                score += 100
            if self.self_stuck_detected and action in ['w', 's']:
                if new_pos.to_tuple() not in self.danger_zones:
                    score += 80
            if self.self_stuck_detected and targeting_stuck and target_stuck_pos:
                current_dist = my_pos.manhattan_distance(target_stuck_pos)
                new_dist = new_pos.manhattan_distance(target_stuck_pos)
                if new_dist < current_dist:
                    score += 50
            if score > best_score:
                best_score = score
                best_action = action
        return best_action
    # Se estamos na última linha com ameaça lateral, tenta subir ou fugir lateralmente
    def evade_bottom_row_snake(self) -> Optional[str]:
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            return None
        my_pos = Position(*bug_blaster['pos'])
        if my_pos.y != self.map_size[1] - 1:
            return None
        centipedes = self.game_state.get('centipedes', [])
        mushrooms = self.get_mushroom_positions()
        threats = []
        for centipede in centipedes:
            for segment in centipede['body']:
                if segment[1] == my_pos.y:  # Same row
                    if abs(segment[0] - my_pos.x) <= 4:
                        threats.append(segment[0])
        if not threats:
            return None
        left_threat = any(x < my_pos.x for x in threats)
        right_threat = any(x > my_pos.x for x in threats)
        sandwiched = left_threat and right_threat
        candidate_actions = []
        up_pos = Position(my_pos.x, my_pos.y - 1)
        up_blocked = up_pos.to_tuple() in mushrooms or my_pos.y - 1 < 0
        if not up_blocked:
            if not self.will_be_hit_soon(up_pos, horizon=2):
                if sandwiched or up_pos.to_tuple() not in self.danger_zones:
                    pass
                    return 'w'
            else:
                pass
        if threats:
            closest_threat = min(threats, key=lambda x: abs(x - my_pos.x))
            if closest_threat < my_pos.x:
                candidate_actions = ['d', 'a', 'w', '']
            else:
                candidate_actions = ['a', 'd', 'w', '']
            safest = self.get_safest_action_with_prediction(candidate_actions, my_pos)
            if safest:
                pass
                return safest
            if closest_threat < my_pos.x:
                return self.find_safe_move('d')
            else:
                return self.find_safe_move('a')
        return None
    # Presos entre cogumelos nas laterais? tenta subir/descer ou disparar se possível
    def detect_horizontal_trap(self) -> Optional[str]:
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            return None
        my_pos = Position(*bug_blaster['pos'])
        centipedes = self.game_state.get('centipedes', [])
        for centipede in centipedes:
            for segment in centipede['body']:
                seg_pos = Position(*segment)
                if my_pos.manhattan_distance(seg_pos) <= 2:
                    return None  # Let emergency logic take over
        mushrooms = self.get_mushroom_positions()
        left_blocked = (my_pos.x - 1, my_pos.y) in mushrooms or my_pos.x - 1 < 0
        right_blocked = (my_pos.x + 1, my_pos.y) in mushrooms or my_pos.x + 1 >= self.map_size[0]
        if left_blocked and right_blocked:
            up_pos = (my_pos.x, my_pos.y - 1)
            down_pos = (my_pos.x, my_pos.y + 1)
            if up_pos not in mushrooms and my_pos.y > 0:
                return 'w'
            elif down_pos not in mushrooms and my_pos.y < self.map_size[1] - 1:
                return 's'
            if self.shot_cooldown == 0:
                return 'A'
        return None
    # Tiro seguro: evita trocar de lugar com a cabeça da centipede na próxima frame
    def is_safe_to_shoot(self) -> bool:
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            return False
        my_pos = Position(*bug_blaster['pos'])
        centipedes = self.game_state.get('centipedes', [])
        mushrooms = self.get_mushroom_positions()
        cobra_na_coluna = False
        for centipede in centipedes:
            body = centipede['body']
            direction = centipede['direction']  # 0=N, 1=E, 2=S, 3=W
            is_stuck = centipede['name'] in self.stuck_centipedes
            for idx, segment in enumerate(body):
                if segment[0] != my_pos.x:
                    continue
                cobra_na_coluna = True
                seg_y = segment[1]
                if is_stuck:
                    continue
                if seg_y >= my_pos.y:
                    return False
                if seg_y < my_pos.y - 1:
                    for y in range(seg_y + 1, my_pos.y):
                        if (my_pos.x, y) in mushrooms:
                            if seg_y > my_pos.y - 4:
                                return False
                    continue
                if seg_y == my_pos.y - 1:
                    if direction == 1:  # EAST -> WEST
                        cobra_next_x = my_pos.x - 1
                    elif direction == 3:  # WEST -> EAST
                        cobra_next_x = my_pos.x + 1
                    else:
                        return False
                    if direction == 1:  # Cobra vai para WEST, nós movemos EAST
                        safe_pos = (my_pos.x + 1, my_pos.y)
                    else:  # Cobra vai para EAST, nós movemos WEST
                        safe_pos = (my_pos.x - 1, my_pos.y)
                    if (safe_pos[0] < 0 or safe_pos[0] >= self.map_size[0] or
                        safe_pos in mushrooms or
                        safe_pos in self.danger_zones):
                        return False
                    safe_position = Position(safe_pos[0], safe_pos[1])
                    if self.will_be_hit_soon(safe_position, horizon=2):
                        pass
                        return False
        return True
    # Evasão de emergência quando há ameaça imediata (prioridade máxima)
    def emergency_evade(self) -> Optional[str]:
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            return None
        my_pos = Position(*bug_blaster['pos'])
        centipedes = self.game_state.get('centipedes', [])
        immediate_threat = False
        for centipede in centipedes:
            for segment in centipede['body']:
                seg_pos = Position(*segment)
                if my_pos.manhattan_distance(seg_pos) <= 1:
                    immediate_threat = True
                    break
            if immediate_threat:
                break
        if not immediate_threat:
            return None
        candidate_actions = ['w', 'a', 's', 'd', '']
        for centipede in centipedes:
            if not centipede['body']:
                continue
            head = centipede['body'][-1]  # Head is always last element
            head_x, head_y = head
            if head_x == my_pos.x and head_y == my_pos.y - 1:
                if 'w' in candidate_actions:
                    candidate_actions.remove('w')
            if head_x == my_pos.x and head_y == my_pos.y + 1:
                if 's' in candidate_actions:
                    candidate_actions.remove('s')
        safest = self.get_safest_action_with_prediction(candidate_actions, my_pos)
        if safest:
            pass
            return safest
        return self.find_safe_move()
    # No modo camping, sai se houver ameaça real ou se o alvo deixou de estar preso
    def check_camping_threats(self) -> Optional[str]:
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            return 'threat'
        my_pos = Position(*bug_blaster['pos'])
        centipedes = self.game_state.get('centipedes', [])
        if self.camping_target_name:
            target_exists = False
            target_still_stuck = False
            for centipede in centipedes:
                if centipede['name'] == self.camping_target_name:
                    target_exists = True
                    if self.camping_target_name in self.stuck_centipedes:
                        target_still_stuck = True
                    break
            if not target_exists:
                pass
                return 'unstuck'
            if not target_still_stuck:
                pass
                return 'unstuck'
        for centipede in centipedes:
            name = centipede['name']
            body = centipede['body']
            if name in self.stuck_centipedes:
                continue
            if not body:
                continue
            head = Position(*body[-1])
            column_distance = abs(head.x - my_pos.x)
            row_distance = my_pos.y - head.y  # Positive if centipede is above us
            if column_distance <= 1 and 0 < row_distance <= 5:
                pass
                return 'threat'
            if 0 < row_distance <= 3:
                direction = centipede.get('direction', 1)  # 0=N, 1=E, 2=S, 3=W
                if column_distance <= 3:
                    pass
                    return 'threat'
            if row_distance <= 0 and column_distance <= 4:
                pass
                return 'threat'
        return None
    # Entra em camping apenas se alvo estiver preso + alinhamento + sem ameaças próximas
    def should_enter_camping(self, target_name: str, is_aligned: bool, is_stuck: bool) -> bool:
        if not is_stuck or not is_aligned:
            return False
        threat = self.check_camping_threats()
        if threat:
            return False
        if self.late_game:
            return True
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            return False
        my_pos = Position(*bug_blaster['pos'])
        centipedes = self.game_state.get('centipedes', [])
        for centipede in centipedes:
            if centipede['name'] in self.stuck_centipedes:
                continue
            if not centipede['body']:
                continue
            head = Position(*centipede['body'][-1])
            distance = my_pos.manhattan_distance(head)
            if distance < 8:
                pass
                return False
        return True
    # No camping, mantém coluna e dispara quando seguro; sai só por ameaça/unstuck
    def decide_camping_action(self) -> Optional[str]:
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            self.camping_mode = False
            return None
        my_pos = Position(*bug_blaster['pos'])
        threat = self.check_camping_threats()
        if threat:
            if threat == 'threat':
                pass
                self.camping_mode = False
                self.camping_target_name = None
                return self.find_safe_move()
            elif threat == 'unstuck':
                pass
                self.camping_mode = False
                self.camping_target_name = None
                return None  # Let normal logic take over
        centipedes = self.game_state.get('centipedes', [])
        target_centipede = None
        for centipede in centipedes:
            if centipede['name'] == self.camping_target_name:
                target_centipede = centipede
                break
        if not target_centipede or not target_centipede['body']:
            pass
            self.camping_mode = False
            self.camping_target_name = None
            return None
        target_head = Position(*target_centipede['body'][-1])
        mushrooms = self.get_mushroom_positions()
        if my_pos.x != target_head.x:
            pass
            self.camping_mode = False
            self.camping_target_name = None
            return None
        mushrooms_in_path = 0
        for y in range(target_head.y + 1, my_pos.y):
            if (my_pos.x, y) in mushrooms:
                mushrooms_in_path += 1
        is_stuck = self.camping_target_name in self.stuck_centipedes
        if self.shot_cooldown == 0:
            if self.is_safe_to_shoot():
                if not is_stuck and mushrooms_in_path > 3:
                    self.camping_mode = False
                    self.camping_target_name = None
                    return None
                self.shot_cooldown = 10
                return 'A'
        self.debug_info['reason'] = 'camping_waiting'
        return ''
    # Loop principal de decisão: evasão > traps > emergência > camping > estratégia normal
    def decide_action(self) -> str:
        bug_blaster = self.game_state.get('bug_blaster', {})
        if not bug_blaster or 'pos' not in bug_blaster:
            return ''
        my_pos = Position(*bug_blaster['pos'])
        if action := self.evade_bottom_row_snake():
            self.debug_info['reason'] = 'bottom_row_escape'
            return action
        if action := self.detect_horizontal_trap():
            self.debug_info['reason'] = 'horizontal_trap'
            return action
        if action := self.emergency_evade():
            self.debug_info['reason'] = 'emergency_evade'
            if self.camping_mode:
                pass
                self.camping_mode = False
                self.camping_target_name = None
            return action
        if self.camping_mode:
            action = self.decide_camping_action()
            if action is not None:
                self.debug_info['reason'] = 'camping_mode'
                return action
        # Late game + auto-stuck: se alvo também está preso e estamos quase alinhados, disparar
        if self.self_stuck_detected and self.late_game:
            centipedes = self.game_state.get('centipedes', [])
            target = self.find_best_target()
            if target:
                target_name, target_pos, _ = target
                target_x, target_y = target_pos
                if target_name in self.stuck_centipedes:
                    # Dispara mesmo com muitos cogumelos: mantemos só o check de segurança
                    distance_x = abs(my_pos.x - target_x)
                    if distance_x <= 2 and self.shot_cooldown == 0 and self.is_safe_to_shoot():
                        self.debug_info['reason'] = 'adadada_loop_breaker'
                        self.shot_cooldown = 10
                        return 'A'
        if self.current_strategy == "defensive":
            self.debug_info['reason'] = 'defensive_mode'
            target = self.find_best_target()
            if target and self.shot_cooldown == 0:
                target_name, target_pos, target_score = target
                target_x, target_y = target_pos
                if my_pos.x == target_x:
                    if self.is_safe_to_shoot():
                        self.debug_info['reason'] = 'defensive_safe_shot'
                        self.shot_cooldown = 10
                        return 'A'
                else:
                    if target_y < my_pos.y - 2:
                        if my_pos.x < target_x:
                            self.debug_info['reason'] = 'defensive_align_right'
                            move = self.find_safe_move('d')
                            if move == 'd':  # Só move se for seguro
                                return move
                        elif my_pos.x > target_x:
                            self.debug_info['reason'] = 'defensive_align_left'
                            move = self.find_safe_move('a')
                            if move == 'a':  # Só move se for seguro
                                return move
            if my_pos.y < self.safe_zone_start:
                action = self.find_safe_move('s')
            else:
                action = self.find_safe_move()
            return action
        if self.current_strategy == "aggressive":
            target = self.find_best_target()
            if target:
                target_name, target_pos, target_score = target
                target_x, target_y = target_pos
                distance_x = abs(my_pos.x - target_x)
                is_stuck_target = target_name in self.stuck_centipedes
                is_aligned = (distance_x == 0)
                if is_aligned and is_stuck_target:
                    # Stuck + alinhado: tenta entrar/ficar em camping e limpar coluna
                    if self.should_enter_camping(target_name, True, True):
                        if not self.camping_mode:
                            pass
                            self.camping_mode = True
                            self.camping_target_name = target_name
                        action = self.decide_camping_action()
                        if action is not None:
                            self.debug_info['reason'] = 'midgame_camping'
                            return action
                mushrooms = self.get_mushroom_positions()
                current_path_clear = True
                if my_pos.x == target_x:
                    for y in range(target_y + 1, my_pos.y):
                        if (my_pos.x, y) in mushrooms:
                            current_path_clear = False
                            break
                reasonably_aligned_threshold = 1 if self.self_stuck_detected else 0
                is_reasonably_aligned = distance_x <= reasonably_aligned_threshold
                mushrooms_in_path = 0
                for y in range(target_y + 1, my_pos.y):
                    if (my_pos.x, y) in mushrooms:
                        mushrooms_in_path += 1
                if is_reasonably_aligned and self.shot_cooldown == 0:
                    path_clear_from_here = mushrooms_in_path == 0
                    if path_clear_from_here:
                        self.debug_info['reason'] = 'shooting_target'
                        self.shot_cooldown = 10
                        if self.self_stuck_detected and distance_x > 0:
                            pass
                        return 'A'
                    elif self.self_stuck_detected and mushrooms_in_path == 1 and target_name in self.stuck_centipedes:
                        # Quando estamos presos e só há 1 cogumelo, limpar rápido ajuda a destravar
                        centipedes = self.game_state.get('centipedes', [])
                        all_stuck = all(c['name'] in self.stuck_centipedes for c in centipedes)
                        if all_stuck and self.is_safe_to_shoot():
                            self.debug_info['reason'] = 'self_stuck_single_mushroom_clear'
                            self.shot_cooldown = 10
                            return 'A'
                    elif not self.late_game and my_pos.x == target_x:
                        # Midgame: usa tiros utilitários para abrir coluna até ao alvo
                        self.debug_info['reason'] = 'clearing_shot_path'
                        self.shot_cooldown = 10
                        return 'A'
                    elif self.late_game and my_pos.x == target_x:
                        # Late game: stuck dispara sempre; all_stuck + self_stuck também permite limpar
                        centipedes = self.game_state.get('centipedes', [])
                        all_stuck = all(c['name'] in self.stuck_centipedes for c in centipedes)
                        if target_name in self.stuck_centipedes:
                            if self.is_safe_to_shoot():
                                self.debug_info['reason'] = 'late_game_clearing_for_stuck_target'
                                self.shot_cooldown = 10
                                return 'A'
                        elif all_stuck and self.self_stuck_detected:
                            if self.is_safe_to_shoot():
                                self.debug_info['reason'] = 'late_game_unstuck_clearing_shot'
                                self.shot_cooldown = 10
                                return 'A'
                if my_pos.x < target_x:
                    self.debug_info['reason'] = 'aligning_right'
                    return self.find_safe_move('d')
                elif my_pos.x > target_x:
                    self.debug_info['reason'] = 'aligning_left'
                    return self.find_safe_move('a')
        if self.current_strategy == "clearing":
            self.debug_info['reason'] = 'clearing_mode'
            if self.shot_cooldown == 0:
                self.shot_cooldown = 10
                return 'A'
            return self.find_safe_move()
        if my_pos.y < self.safe_zone_start:
            self.debug_info['reason'] = 'return_to_safe_zone'
            return self.find_safe_move('s', returning_to_safe_zone=True)
        self.debug_info['reason'] = 'fallback_safe'
        return self.find_safe_move()
    def get_mushroom_positions(self) -> Set[Tuple[int, int]]:
        mushrooms = self.game_state.get('mushrooms', [])
        return {tuple(m['pos']) for m in mushrooms}
async def agent_loop(server_address="localhost:8000", agent_name="student"):
    import websockets
    agent = CentipedeAgent()
    async with websockets.connect(f"ws://{server_address}/player") as websocket:
        await websocket.send(json.dumps({"cmd": "join", "name": agent_name}))
        while True:
            try:
                state = json.loads(await websocket.recv())
                agent.update_state(state)
                if agent.frame_count % 50 == 0:
                    score = state.get('score', 0)
                    num_centipedes = len(state.get('centipedes', []))
                    num_mushrooms = len(state.get('mushrooms', []))
                    pass
                try:
                    key = agent.decide_action()
                    agent.last_action = key
                    if agent.self_stuck_detected and key in ['w', 's']:
                        pass
                        agent.self_stuck_detected = False
                except Exception as action_error:
                    pass
                    key = agent.find_safe_move()
                    if not key:
                        key = ''
                await websocket.send(json.dumps({"cmd": "key", "key": key}))
                if agent.frame_count % 100 == 0 and agent.debug_info.get('reason'):
                    pass
            except websockets.exceptions.ConnectionClosedOK:
                pass
                return
            except Exception as e:
                pass
                break
if __name__ == "__main__":
    SERVER = os.environ.get("SERVER", "localhost")
    PORT = os.environ.get("PORT", "8000")
    NAME = os.environ.get("NAME", getpass.getuser())
    try:
        asyncio.run(agent_loop(f"{SERVER}:{PORT}", NAME))
    except KeyboardInterrupt:
        pass
    except Exception as e:
        pass
