"""
IA Clássica para Centipede - Projeto Académico
Estratégia: Máquina de Estados Finitos (FSM) com Heurísticas Híbridas

PROIBIDO: Reinforcement Learning, Machine Learning, Redes Neuronais, Q-Learning, etc.
PERMITIDO: Apenas IA clássica/algoritmos determinísticos/heurísticas fixas

Autor: Student Agent
Data: 2025
"""

import asyncio
import json
import os
import getpass
import websockets
from enum import Enum
from typing import Optional
import math
from datetime import datetime
import time
from collections import deque


# ============================================================================
# CONSTANTES (ajustáveis para afinar comportamento)
# ============================================================================

# Thresholds de perigo
DANGER_DISTANCE_LINES = 3  # Se centipede está a menos de 3 linhas, EVADE
SAFE_DISTANCE_LINES = 4    # Após esta distância, voltar a ATTACK (sair mais cedo de EVADE)

# Preferências de posicionamento
PREFERRED_CENTER_X = 20    # Coluna central ideal
IDEAL_MIN_Y = 18           # Linha mínima ideal (zona de manobra)

# Pesos para scoring de alvos (priorizar segmentos mais altos = mais pontos)
TARGET_HEIGHT_WEIGHT = 2.0
TARGET_PROXIMITY_WEIGHT = 1.0

# Cooldown interno (frames esperados após disparar)
INTERNAL_COOLDOWN = 10

# Mapa
MAP_WIDTH = 40
MAP_HEIGHT = 24


# ============================================================================
# ENUMS
# ============================================================================

class AgentMode(Enum):
    """Estados da Máquina de Estados Finitos"""
    ATTACK = "ATTACK"          # Perseguir e disparar em centipedes
    EVADE = "EVADE"            # Fugir de centipedes próximas
    REPOSITION = "REPOSITION"  # Ajustar posição tática


# ============================================================================
# CLASSE PRINCIPAL DO AGENTE
# ============================================================================

class CentipedeAgent:
    """
    Agente baseado em FSM (Finite State Machine) para jogar Centipede.
    
    Estratégia:
    1. ATTACK: Procura cabeças de centipedes, alinha-se e dispara (maximizar score)
    2. EVADE: Foge quando centipedes estão perigosamente perto (sobrevivência)
    3. REPOSITION: Ajusta posição para melhor campo de tiro
    """
    
    def __init__(self):
        # Estado interno
        self.player_pos: Optional[tuple[int, int]] = None
        self.centipedes: list[dict] = []
        self.mushrooms: set[tuple[int, int]] = set()
        self.blasts: list[tuple[int, int]] = []
        self.score: int = 0
        self.step: int = 0
        
        # Controle de disparo (cooldown interno)
        self.frames_since_last_shot: int = INTERNAL_COOLDOWN
        self.can_shoot: bool = True
        
        # FSM
        self.current_mode: AgentMode = AgentMode.ATTACK
        self.last_action: Optional[str] = None

        # Logging estruturado (JSON Lines)
        self.logging_enabled: bool = True
        base_dir = os.path.dirname(__file__)
        self._log_dir = os.path.join(base_dir, "logs")
        os.makedirs(self._log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        user = getpass.getuser() or "user"
        self._log_path = os.path.join(self._log_dir, f"agent_{timestamp}_{user}.jsonl")
        self._session_id = f"{timestamp}-{os.getpid()}"
        self._last_reason: str = ""
        self._last_context: dict = {}
        self._last_transition_reason: Optional[str] = None

        # Históricos para detetar padrões indesejados (ex.: oscilação esquerda/direita)
        self._action_history: deque[str] = deque(maxlen=6)
        self._pos_history: deque[tuple[int, int]] = deque(maxlen=6)
        
    def update(self, state: dict) -> None:
        """
        Atualiza representação interna a partir do estado recebido do servidor.
        """
        # Jogador
        bug_blaster = state.get("bug_blaster", {})
        self.player_pos = tuple(bug_blaster.get("pos", (20, 23)))
        
        # Centipedes (apenas vivas)
        self.centipedes = state.get("centipedes", [])
        
        # Mushrooms (converter para set de posições para lookup O(1))
        self.mushrooms = {tuple(m["pos"]) for m in state.get("mushrooms", [])}
        
        # Blasts
        self.blasts = [tuple(b) for b in state.get("blasts", [])]
        
        # Metadata
        self.score = state.get("score", 0)
        self.step = state.get("step", 0)
        
        # Atualizar cooldown
        self.frames_since_last_shot += 1
        self.can_shoot = self.frames_since_last_shot >= INTERNAL_COOLDOWN
    
    # ------------------------------------------------------------------------
    # SELEÇÃO DE MODO (FSM TRANSITIONS)
    # ------------------------------------------------------------------------
    
    def select_mode(self) -> AgentMode:
        """
        Determina o modo atual da FSM com base no estado do jogo.
        
        Lógica de transição:
        - EVADE: Se há perigo iminente (centipede muito perto)
        - REPOSITION: Se não há alvos acessíveis ou posição é ruim
        - ATTACK: Caso contrário (condição default)
        """
        min_danger = self._calculate_min_danger_distance()
        
        # TRANSIÇÃO: QUALQUER → EVADE (prioridade máxima)
        if min_danger < DANGER_DISTANCE_LINES:
            next_mode = AgentMode.EVADE
            if next_mode != self.current_mode:
                self._last_transition_reason = "danger_below_threshold"
                self._log(
                    kind="mode_transition",
                    payload={
                        "from": self.current_mode.value,
                        "to": next_mode.value,
                        "danger": min_danger,
                        "reason": self._last_transition_reason,
                    },
                )
            return next_mode
        
        # TRANSIÇÃO: EVADE → REPOSITION/ATTACK
        if self.current_mode == AgentMode.EVADE and min_danger > SAFE_DISTANCE_LINES:
            # Sair de EVADE apenas se longe o suficiente
            next_mode = AgentMode.ATTACK if self._is_in_good_position() else AgentMode.REPOSITION
            if next_mode != self.current_mode:
                self._last_transition_reason = "safe_distance_reached"
                self._log(
                    kind="mode_transition",
                    payload={
                        "from": self.current_mode.value,
                        "to": next_mode.value,
                        "danger": min_danger,
                        "reason": self._last_transition_reason,
                    },
                )
            return next_mode
        
        # TRANSIÇÃO: REPOSITION → ATTACK
        if self.current_mode == AgentMode.REPOSITION and (self._is_in_good_position() or self._has_accessible_targets()):
            next_mode = AgentMode.ATTACK
            if next_mode != self.current_mode:
                self._last_transition_reason = "good_position_achieved"
                self._log(
                    kind="mode_transition",
                    payload={
                        "from": self.current_mode.value,
                        "to": next_mode.value,
                        "reason": self._last_transition_reason,
                    },
                )
            return next_mode
        
        # TRANSIÇÃO: ATTACK → REPOSITION
        if self.current_mode == AgentMode.ATTACK and not self._has_accessible_targets():
            next_mode = AgentMode.REPOSITION
            if next_mode != self.current_mode:
                self._last_transition_reason = "no_accessible_targets"
                self._log(
                    kind="mode_transition",
                    payload={
                        "from": self.current_mode.value,
                        "to": next_mode.value,
                        "reason": self._last_transition_reason,
                    },
                )
            return next_mode
        
        # Manter modo atual se nenhuma transição foi acionada
        return self.current_mode
    
    # ------------------------------------------------------------------------
    # DECISÃO DE AÇÃO (dentro de cada modo)
    # ------------------------------------------------------------------------
    
    def decide_action(self) -> Optional[str]:
        """
        Escolhe ação concreta (w/a/s/d/A) com base no modo atual.
        
        Retorna:
            str: "w", "a", "s", "d", "A", ou None (sem ação)
        """
        t0 = time.perf_counter()
        # Atualizar modo
        self.current_mode = self.select_mode()
        self._last_reason = ""
        self._last_context = {}
        action: Optional[str] = None
        
        # Decidir ação baseada no modo
        if self.current_mode == AgentMode.ATTACK:
            action = self._attack_logic()
        elif self.current_mode == AgentMode.EVADE:
            action = self._evade_logic()
        elif self.current_mode == AgentMode.REPOSITION:
            action = self._reposition_logic()
        
        dt_ms = (time.perf_counter() - t0) * 1000.0
        # Métricas para contexto da decisão
        danger = self._calculate_min_danger_distance()
        best_target = self._get_best_target()
        clear_shot = bool(best_target and self._has_clear_shot(best_target[0]))
        payload = {
            "action": action or "",
            "mode": self.current_mode.value,
            "reason": self._last_reason,
            "context": self._last_context,
            "step": self.step,
            "score": self.score,
            "player_pos": list(self.player_pos) if self.player_pos else None,
            "danger": danger,
            "best_target": list(best_target) if best_target else None,
            "clear_shot": clear_shot,
            "can_shoot": self.can_shoot,
            "frames_since_last_shot": self.frames_since_last_shot,
            "centipedes": len(self.centipedes),
            "decision_ms": dt_ms,
        }
        # Detetar oscilação e tentar quebrar com movimento vertical
        if action in ("a", "d") and self._is_oscillating():
            px, py = self.player_pos if self.player_pos else (None, None)
            break_action = None
            if px is not None:
                if self._is_safe_move(px, py + 1):
                    break_action = "s"
                    self._last_reason = "break_oscillation_down"
                    self._last_context = {"from_y": py, "to_y": py + 1}
                elif self._is_safe_move(px, py - 1):
                    break_action = "w"
                    self._last_reason = "break_oscillation_up"
                    self._last_context = {"from_y": py, "to_y": py - 1}
            if break_action:
                action = break_action
                payload["action"] = action
                payload["reason"] = self._last_reason
                payload["context"] = self._last_context

        # Atualizar históricos
        if self.player_pos:
            self._pos_history.append(self.player_pos)
        if action:
            self._action_history.append(action)

        self._log(kind="decision", payload=payload)
        return action
    
    # ------------------------------------------------------------------------
    # LÓGICA POR MODO
    # ------------------------------------------------------------------------
    
    def _attack_logic(self) -> Optional[str]:
        """
        Modo ATTACK: Procura melhor alvo, alinha-se e dispara.
        
        Estratégia:
        1. Identificar alvo prioritário (cabeça mais alta + mais próxima)
        2. Mover horizontalmente até alinhar com alvo
        3. Disparar quando alinhado e com cooldown pronto
        4. Subir ligeiramente para melhor campo de tiro (mas sem entrar na zona de perigo)
        """
        target = self._get_best_target()
        
        if target is None:
            # Sem alvos → trocar para REPOSITION
            self._last_reason = "no_target"
            self._last_context = {}
            return None
        
        target_x, target_y = target
        player_x, player_y = self.player_pos
        
        # 1. DISPARAR se alinhado e cooldown OK
        if self.can_shoot and player_x == target_x:
            clear = self._has_clear_shot(target_x)
            has_msh = bool(self._find_mushroom_in_column(player_x))
            has_cent = self._has_centipede_in_column(player_x)
            if clear or has_msh or has_cent:
                self.frames_since_last_shot = 0
                self.can_shoot = False
                self._last_reason = "aligned_shot"
                self._last_context = {
                    "target": [target_x, target_y],
                    "clear": clear,
                    "mushroom_in_front": has_msh,
                    "centipede_in_column": has_cent,
                }
                return "A"
        
        # 2. Manter-se na zona inferior ideal para poder disparar na cobra
        if player_y < IDEAL_MIN_Y:
            if self._is_safe_move(player_x, player_y + 1):
                self._last_reason = "return_to_bottom"
                self._last_context = {"from_y": player_y, "to_y": player_y + 1}
                return "s"

        # 3. ALINHAR HORIZONTALMENTE com alvo
        if player_x < target_x:
            # Verificar se pode mover para direita
            if self._is_safe_move(player_x + 1, player_y):
                self._last_reason = "move_right_to_align"
                self._last_context = {"from_x": player_x, "to_x": player_x + 1, "target_x": target_x}
                return "d"
        elif player_x > target_x:
            # Verificar se pode mover para esquerda
            if self._is_safe_move(player_x - 1, player_y):
                self._last_reason = "move_left_to_align"
                self._last_context = {"from_x": player_x, "to_x": player_x - 1, "target_x": target_x}
                return "a"
        else:
            # Já alinhado em X: se não puder disparar agora, tentar descer para manter pressão
            if not self.can_shoot and self._is_safe_move(player_x, player_y + 1):
                self._last_reason = "aligned_descend_waiting_cooldown"
                self._last_context = {"from_y": player_y, "to_y": player_y + 1}
                return "s"
        
        # 4. SUBIR ligeiramente se muito abaixo (para melhor ângulo)
        if player_y > IDEAL_MIN_Y and player_y > target_y + 5:
            if self._is_safe_move(player_x, player_y - 1):
                self._last_reason = "move_up_for_angle"
                self._last_context = {"from_y": player_y, "to_y": player_y - 1, "target_y": target_y}
                return "w"
        
        # 5. Se já alinhado mas em cooldown, esperar pelo disparo
        if player_x == target_x and not self.can_shoot:
            self._last_reason = "aligned_wait_cooldown"
            self._last_context = {"frames_since_last_shot": self.frames_since_last_shot}
        
        # 6. Fallback: tentar mover aleatoriamente mas com segurança
        action = self._random_safe_move(prefer_vertical=True)
        if action:
            self._last_reason = "random_safe_move"
            self._last_context = {"action": action}
        else:
            self._last_reason = "no_action_possible"
            self._last_context = {}
        return action
    
    def _evade_logic(self) -> Optional[str]:
        """
        Modo EVADE: Foge de centipedes próximas.
        
        Estratégia:
        1. Calcular coluna mais longe de todas as centipedes
        2. Mover horizontalmente para essa coluna
        3. NÃO disparar (prioridade = sobrevivência)
        4. Se encurralado por mushrooms, tentar destruir um
        """
        safe_column = self._find_safest_column()
        player_x, player_y = self.player_pos
        
        # Mover para coluna segura
        if player_x < safe_column:
            if self._is_safe_move(player_x + 1, player_y):
                self._last_reason = "move_right_to_safe_column"
                self._last_context = {"from_x": player_x, "to_x": player_x + 1, "safe_column": safe_column}
                return "d"
            else:
                # Bloqueado por mushroom, tentar destruir
                if self.can_shoot:
                    self.frames_since_last_shot = 0
                    self._last_reason = "blocked_by_mushroom_shoot"
                    self._last_context = {"dir": "right"}
                    return "A"
        elif player_x > safe_column:
            if self._is_safe_move(player_x - 1, player_y):
                self._last_reason = "move_left_to_safe_column"
                self._last_context = {"from_x": player_x, "to_x": player_x - 1, "safe_column": safe_column}
                return "a"
            else:
                # Bloqueado por mushroom, tentar destruir
                if self.can_shoot:
                    self.frames_since_last_shot = 0
                    self._last_reason = "blocked_by_mushroom_shoot"
                    self._last_context = {"dir": "left"}
                    return "A"
        
        # Se já na coluna segura, descer para ter mais espaço de manobra
        if player_y < MAP_HEIGHT - 2:
            if self._is_safe_move(player_x, player_y + 1):
                self._last_reason = "descend_for_space"
                self._last_context = {"from_y": player_y, "to_y": player_y + 1}
                return "s"
        
        self._last_reason = "no_evade_action"
        self._last_context = {}
        return None
    
    def _reposition_logic(self) -> Optional[str]:
        """
        Modo REPOSITION: Ajusta posição para melhor tática.
        
        Estratégia:
        1. Mover para coluna central (x=20) se não estiver lá
        2. Manter linha ideal (y=18-20)
        3. Limpar mushrooms no caminho
        """
        player_x, player_y = self.player_pos
        
        # 1. Ajustar X para centro
        if abs(player_x - PREFERRED_CENTER_X) > 2:
            if player_x < PREFERRED_CENTER_X:
                if self._is_safe_move(player_x + 1, player_y):
                    self._last_reason = "center_right"
                    self._last_context = {"from_x": player_x, "to_x": player_x + 1}
                    return "d"
            else:
                if self._is_safe_move(player_x - 1, player_y):
                    self._last_reason = "center_left"
                    self._last_context = {"from_x": player_x, "to_x": player_x - 1}
                    return "a"
        
        # 2. Ajustar Y para zona ideal
        if player_y < IDEAL_MIN_Y:
            if self._is_safe_move(player_x, player_y + 1):
                self._last_reason = "adjust_y_down"
                self._last_context = {"from_y": player_y, "to_y": player_y + 1}
                return "s"
        elif player_y > IDEAL_MIN_Y + 2:
            if self._is_safe_move(player_x, player_y - 1):
                self._last_reason = "adjust_y_up"
                self._last_context = {"from_y": player_y, "to_y": player_y - 1}
                return "w"
        
        # 3. Limpar mushroom à frente se houver
        if self.can_shoot:
            mushroom_ahead = self._find_mushroom_in_column(player_x)
            if mushroom_ahead:
                self.frames_since_last_shot = 0
                self._last_reason = "clear_mushroom_ahead"
                self._last_context = {"mushroom": list(mushroom_ahead)}
                return "A"
        
        self._last_reason = "hold_position"
        self._last_context = {}
        return None

    # ------------------------------------------------------------------------
    # LOGGING
    # ------------------------------------------------------------------------
    def _log(self, kind: str, payload: dict) -> None:
        if not self.logging_enabled:
            return
        try:
            record = {
                "ts": datetime.utcnow().isoformat() + "Z",
                "session": self._session_id,
                "kind": kind,
            }
            # Garantir serialização segura
            if isinstance(payload, dict):
                record.update(payload)
            else:
                record["payload"] = payload
            with open(self._log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception:
            # Não interromper o jogo por causa de logging
            pass
    
    # ------------------------------------------------------------------------
    # FUNÇÕES UTILITÁRIAS (HEURÍSTICAS)
    # ------------------------------------------------------------------------
    
    def _calculate_min_danger_distance(self) -> float:
        """
        Calcula distância (em linhas) até a centipede mais próxima.
        
        Retorna:
            float: Distância mínima ou infinito se não há centipedes
        """
        if not self.player_pos or not self.centipedes:
            return float('inf')
        
        player_x, player_y = self.player_pos
        min_dist = float('inf')
        
        for centipede in self.centipedes:
            body = centipede.get("body", [])
            for seg_x, seg_y in body:
                # Distância vertical (mais importante que horizontal)
                vertical_dist = abs(player_y - seg_y)
                horizontal_dist = abs(player_x - seg_x)
                
                # Combinação: priorizar distância vertical
                combined_dist = vertical_dist + horizontal_dist * 0.3
                min_dist = min(min_dist, combined_dist)
        
        return min_dist
    
    def _get_best_target(self) -> Optional[tuple[int, int]]:
        """
        Seleciona alvo prioritário: cabeça de centipede mais alta e mais próxima.
        
        Estratégia de scoring:
        - Prioridade 1: Altura (Y menor = mais pontos no jogo)
        - Prioridade 2: Proximidade horizontal ao jogador
        
        Retorna:
            tuple[int, int]: (x, y) do melhor alvo ou None
        """
        if not self.centipedes or not self.player_pos:
            return None
        
        player_x, player_y = self.player_pos
        best_target = None
        best_score = -float('inf')
        
        for centipede in self.centipedes:
            body = centipede.get("body", [])
            if not body:
                continue
            
            # Cabeça é o último elemento do body
            head_x, head_y = body[-1]
            
            # Score: quanto menor Y (mais alto), melhor
            # Score: quanto mais perto horizontalmente, melhor
            height_score = -head_y * TARGET_HEIGHT_WEIGHT
            proximity_score = -abs(player_x - head_x) * TARGET_PROXIMITY_WEIGHT
            
            total_score = height_score + proximity_score
            
            if total_score > best_score:
                best_score = total_score
                best_target = (head_x, head_y)
        
        return best_target
    
    def _has_clear_shot(self, column: int) -> bool:
        """
        Verifica se há caminho livre (sem mushrooms) entre jogador e topo da coluna.
        
        Args:
            column: Coluna X a verificar
            
        Retorna:
            bool: True se pode disparar livremente
        """
        if not self.player_pos:
            return False
        
        player_x, player_y = self.player_pos
        
        # Verificar se há mushrooms entre jogador e linha 0
        for y in range(0, player_y):
            if (column, y) in self.mushrooms:
                return False
        
        return True

    def _has_centipede_in_column(self, column: int) -> bool:
        """
        Verifica se existe alguma centipede na coluna acima do jogador.
        """
        if not self.player_pos:
            return False
        _, player_y = self.player_pos
        for centipede in self.centipedes:
            body = centipede.get("body", [])
            for seg_x, seg_y in body:
                if seg_x == column and seg_y < player_y:
                    return True
        return False
    
    def _find_safest_column(self) -> int:
        """
        Encontra coluna com maior distância média a todas as centipedes.
        
        Retorna:
            int: Coluna X mais segura
        """
        if not self.centipedes:
            return PREFERRED_CENTER_X
        
        # Calcular score de segurança para cada coluna
        column_scores = {}
        
        for x in range(MAP_WIDTH):
            total_dist = 0
            for centipede in self.centipedes:
                body = centipede.get("body", [])
                for seg_x, seg_y in body:
                    dist = abs(x - seg_x) + abs(self.player_pos[1] - seg_y) * 0.5
                    total_dist += dist
            
            column_scores[x] = total_dist
        
        # Retornar coluna com maior score (mais longe de centipedes)
        safest_column = max(column_scores, key=column_scores.get)
        return safest_column
    
    def _is_safe_move(self, new_x: int, new_y: int) -> bool:
        """
        Verifica se movimento para (new_x, new_y) é seguro.
        
        Critérios:
        - Dentro dos limites do mapa
        - Não há mushroom na posição
        - Não há centipede na posição
        
        Args:
            new_x, new_y: Posição destino
            
        Retorna:
            bool: True se movimento é seguro
        """
        # Limites do mapa
        if not (0 <= new_x < MAP_WIDTH and 0 <= new_y < MAP_HEIGHT):
            return False
        
        # Mushroom?
        if (new_x, new_y) in self.mushrooms:
            return False
        
        # Centipede?
        for centipede in self.centipedes:
            body = centipede.get("body", [])
            if (new_x, new_y) in body:
                return False
        
        return True
    
    def _is_in_good_position(self) -> bool:
        """
        Verifica se jogador está em posição tática favorável.
        
        Critérios:
        - Próximo do centro horizontal
        - Na zona ideal vertical
        - Sem mushrooms bloqueando caminho acima
        
        Retorna:
            bool: True se posição é boa
        """
        if not self.player_pos:
            return False
        
        player_x, player_y = self.player_pos
        
        # Critério 1: Proximidade ao centro
        if abs(player_x - PREFERRED_CENTER_X) > 5:
            return False
        
        # Critério 2: Linha ideal
        if not (IDEAL_MIN_Y <= player_y <= IDEAL_MIN_Y + 3):
            return False
        
        # Critério 3: Caminho livre acima (relaxado) — já não é obrigatório
        return True
    
    def _has_accessible_targets(self) -> bool:
        """
        Verifica se há alvos que podem ser atingidos.
        
        Retorna:
            bool: True se há pelo menos um alvo acessível
        """
        target = self._get_best_target()
        return target is not None
    
    def _find_mushroom_in_column(self, column: int) -> Optional[tuple[int, int]]:
        """
        Encontra mushroom mais próximo na coluna especificada.
        
        Args:
            column: Coluna X a procurar
            
        Retorna:
            tuple[int, int]: (x, y) do mushroom ou None
        """
        if not self.player_pos:
            return None
        
        player_x, player_y = self.player_pos
        closest_mushroom = None
        min_dist = float('inf')
        
        for mx, my in self.mushrooms:
            if mx == column and my < player_y:
                dist = player_y - my
                if dist < min_dist:
                    min_dist = dist
                    closest_mushroom = (mx, my)
        
        return closest_mushroom
    
    def _random_safe_move(self, prefer_vertical: bool = False) -> Optional[str]:
        """
        Fallback: tenta mover aleatoriamente mas com segurança.
        
        Retorna:
            str: Ação segura ou None
        """
        if not self.player_pos:
            return None
        
        player_x, player_y = self.player_pos
        # Preferir vertical para quebrar padrões de bounce em obstáculos
        if prefer_vertical:
            moves = [
                ("s", player_x, player_y + 1),
                ("w", player_x, player_y - 1),
                ("d", player_x + 1, player_y),
                ("a", player_x - 1, player_y),
            ]
        else:
            moves = [
                ("d", player_x + 1, player_y),
                ("a", player_x - 1, player_y),
                ("w", player_x, player_y - 1),
                ("s", player_x, player_y + 1),
            ]
        
        for action, new_x, new_y in moves:
            if self._is_safe_move(new_x, new_y):
                return action
        
        return None

    def _is_oscillating(self) -> bool:
        """
        Deteta padrão de oscilação horizontal (ex.: a,d,a,d) sem progresso de X.
        """
        if len(self._action_history) < 4 or len(self._pos_history) < 4:
            return False
        last_actions = list(self._action_history)[-4:]
        pattern1 = last_actions == ["a", "d", "a", "d"]
        pattern2 = last_actions == ["d", "a", "d", "a"]
        if not (pattern1 or pattern2):
            return False
        xs = [pos[0] for pos in self._pos_history][-4:]
        return max(xs) - min(xs) <= 1


# ============================================================================
# LOOP PRINCIPAL DO AGENTE
# ============================================================================

async def agent_loop(server_address="localhost:8000", agent_name="student"):
    """
    Loop principal de comunicação com o servidor.
    
    Baseado em client.py, mas sem Pygame (apenas lógica de agente).
    
    Args:
        server_address: Endereço do servidor (host:port)
        agent_name: Nome do agente (aparecerá nos highscores)
    """
    # Criar agente
    agent = CentipedeAgent()
    
    # Conectar ao servidor via WebSocket
    async with websockets.connect(f"ws://{server_address}/player") as websocket:
        # Enviar comando de join
        await websocket.send(json.dumps({"cmd": "join", "name": agent_name}))
        
        print(f"[{agent_name}] Conectado ao servidor {server_address}")
        print(f"[{agent_name}] Modo inicial: {agent.current_mode.value}")
        try:
            if getattr(agent, "_log_path", None):
                print(f"[{agent_name}] Logging: {agent._log_path}")
        except Exception:
            pass
        
        # Loop principal do jogo
        try:
            while True:
                # Receber estado do jogo
                state_json = await websocket.recv()
                state = json.loads(state_json)
                
                # Atualizar agente com novo estado
                agent.update(state)
                
                # Decidir ação
                action = agent.decide_action()
                
                # Log periódico (a cada 100 frames)
                if agent.step % 100 == 0:
                    print(f"[Step {agent.step}] Score: {agent.score} | "
                          f"Modo: {agent.current_mode.value} | "
                          f"Pos: {agent.player_pos} | "
                          f"Centipedes: {len(agent.centipedes)}")
                
                # Enviar ação ao servidor (se houver)
                if action:
                    await websocket.send(json.dumps({"cmd": "key", "key": action}))
                else:
                    # Enviar string vazia se sem ação
                    await websocket.send(json.dumps({"cmd": "key", "key": ""}))
        
        except websockets.exceptions.ConnectionClosedOK:
            print(f"[{agent_name}] Servidor fechou conexão (jogo terminou)")
            print(f"[{agent_name}] Score final: {agent.score} em {agent.step} steps")
        except Exception as e:
            print(f"[{agent_name}] Erro: {e}")


# ============================================================================
# ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    """
    Ponto de entrada do script.
    
    Pode-se customizar via variáveis de ambiente:
    - SERVER: endereço do servidor (default: localhost)
    - PORT: porta (default: 8000)
    - NAME: nome do agente (default: student)
    """
    # Ler configuração de environment variables
    SERVER = os.environ.get("SERVER", "localhost")
    PORT = os.environ.get("PORT", "8000")
    NAME = os.environ.get("NAME", "student")
    
    # Executar loop do agente
    loop = asyncio.get_event_loop()
    loop.run_until_complete(agent_loop(f"{SERVER}:{PORT}", NAME))


# ============================================================================
# SECÇÃO DE MELHORIAS FUTURAS (TODO/IDEIAS)
# ============================================================================

"""
TODO / IDEIAS DE EVOLUÇÃO:

1. MELHORIAS NAS HEURÍSTICAS:
   - Priorizar segmentos de centipede mais altos (linha Y < 5) ainda mais
   - Adicionar heurística para evitar ficar encurralado entre mushrooms
   - Implementar "pathfinding" simples (BFS) para encontrar caminho até coluna segura
   
2. NOVOS MODOS FSM:
   - CLEAR_PATH: Modo dedicado a limpar mushrooms estrategicamente
   - AGGRESSIVE: Modo ultra-agressivo quando score está baixo perto do timeout
   - DEFENSIVE: Modo ultra-defensivo quando há muitas centipedes simultâneas
   
3. OTIMIZAÇÕES:
   - Cache de cálculos pesados (ex: _find_safest_column)
   - Prever movimento das centipedes (1-2 frames à frente)
   - Usar A* para pathfinding em vez de movimentos greedy
   
4. TRATAMENTO DE CASOS ESPECIAIS:
   - Detectar padrões de "aprisionamento" e forçar escape
   - Priorizar destruir mushrooms que estão no caminho de centipedes (para forçá-las a descer)
   - Implementar "kiting" (ficar longe mas disparando constantemente)
   
5. TUNING DE PARÂMETROS:
   - Ajustar DANGER_DISTANCE_LINES baseado no número de centipedes
   - Ajustar TARGET_HEIGHT_WEIGHT dinamicamente (mais agressivo no início, defensivo no fim)
   - Adicionar randomização leve para evitar loops determinísticos
   
6. DEBUG E ANÁLISE:
   - Adicionar flag --verbose para logs detalhados
   - Gravar replays de jogos para análise offline
   - Estatísticas: % de tempo em cada modo, accuracy de disparos, etc.
   
7. MULTIAGENTE (se permitido):
   - Coordenar múltiplos agentes (cada um cobre uma zona do mapa)
   - Sistema de "comunicação" via estado compartilhado
"""
