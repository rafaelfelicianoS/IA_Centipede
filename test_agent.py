"""
Script de teste básico para validar lógica do agente (sem servidor).
Simula estados simples e verifica decisões.
"""

import sys
sys.path.append('.')

from student import CentipedeAgent, AgentMode

def test_initialization():
    """Testa inicialização do agente."""
    agent = CentipedeAgent()
    assert agent.current_mode == AgentMode.ATTACK
    assert agent.can_shoot == True
    assert agent.score == 0
    print("✓ Teste de inicialização passou")

def test_update_state():
    """Testa atualização do estado."""
    agent = CentipedeAgent()
    
    state = {
        "bug_blaster": {"pos": (20, 23), "alive": True},
        "centipedes": [
            {"name": "mother", "body": [(i, 0) for i in range(20)], "direction": 1}
        ],
        "mushrooms": [{"pos": (15, 10), "health": 4}],
        "blasts": [(20, 15)],
        "score": 100,
        "step": 50
    }
    
    agent.update(state)
    
    assert agent.player_pos == (20, 23)
    assert len(agent.centipedes) == 1
    assert (15, 10) in agent.mushrooms
    assert agent.score == 100
    assert agent.step == 50
    print("✓ Teste de atualização passou")

def test_danger_calculation():
    """Testa cálculo de perigo."""
    agent = CentipedeAgent()
    
    # Centipede longe
    agent.player_pos = (20, 23)
    agent.centipedes = [{"body": [(10, 0)]}]
    
    danger = agent._calculate_min_danger_distance()
    assert danger > 20  # Muito longe
    print(f"✓ Perigo com centipede longe: {danger:.1f}")
    
    # Centipede perto
    agent.centipedes = [{"body": [(20, 21)]}]
    danger = agent._calculate_min_danger_distance()
    assert danger < 5  # Muito perto
    print(f"✓ Perigo com centipede perto: {danger:.1f}")

def test_mode_transitions():
    """Testa transições de estados FSM."""
    agent = CentipedeAgent()
    agent.player_pos = (20, 23)
    
    # Cenário 1: Centipede longe → ATTACK
    agent.centipedes = [{"body": [(10, 0)]}]
    mode = agent.select_mode()
    assert mode == AgentMode.ATTACK or mode == AgentMode.REPOSITION
    print(f"✓ Centipede longe → {mode.value}")
    
    # Cenário 2: Centipede perto → EVADE
    agent.centipedes = [{"body": [(20, 21)]}]
    mode = agent.select_mode()
    assert mode == AgentMode.EVADE
    print(f"✓ Centipede perto → {mode.value}")

def test_target_selection():
    """Testa seleção de alvo."""
    agent = CentipedeAgent()
    agent.player_pos = (20, 23)
    
    # Múltiplas centipedes
    agent.centipedes = [
        {"body": [(10, 5)]},   # Mais alta, longe
        {"body": [(25, 10)]},  # Mais baixa, perto
        {"body": [(20, 3)]},   # Alta E perto → MELHOR
    ]
    
    target = agent._get_best_target()
    assert target is not None
    print(f"✓ Melhor alvo selecionado: {target} (esperado: próximo de (20, 3))")

def test_safe_move():
    """Testa verificação de movimento seguro."""
    agent = CentipedeAgent()
    agent.player_pos = (20, 23)
    agent.mushrooms = {(21, 23), (20, 22)}
    agent.centipedes = [{"body": [(19, 23)]}]
    
    # Movimento seguro
    assert agent._is_safe_move(20, 24) == False  # Fora do mapa (Y=24)
    assert agent._is_safe_move(21, 23) == False  # Mushroom
    assert agent._is_safe_move(19, 23) == False  # Centipede
    assert agent._is_safe_move(22, 23) == True   # OK
    print("✓ Verificação de movimento seguro funcionando")

def test_attack_logic():
    """Testa lógica do modo ATTACK."""
    agent = CentipedeAgent()
    agent.player_pos = (20, 20)
    agent.centipedes = [{"body": [(15, 5)]}]
    agent.can_shoot = True
    agent.current_mode = AgentMode.ATTACK
    
    action = agent._attack_logic()
    # Deve tentar mover para esquerda (alinhar com x=15)
    assert action in ["a", "d", "w", "A", None]
    print(f"✓ ATTACK decidiu ação: {action}")

def test_evade_logic():
    """Testa lógica do modo EVADE."""
    agent = CentipedeAgent()
    agent.player_pos = (20, 22)
    agent.centipedes = [{"body": [(20, 20)]}]  # Muito perto!
    agent.current_mode = AgentMode.EVADE
    
    action = agent._evade_logic()
    # Deve tentar mover para longe
    assert action in ["a", "d", "s", "A", None]
    print(f"✓ EVADE decidiu ação: {action}")

def run_all_tests():
    """Executa todos os testes."""
    print("=" * 50)
    print("TESTES UNITÁRIOS - student.py")
    print("=" * 50)
    
    try:
        test_initialization()
        test_update_state()
        test_danger_calculation()
        test_mode_transitions()
        test_target_selection()
        test_safe_move()
        test_attack_logic()
        test_evade_logic()
        
        print("=" * 50)
        print("✅ TODOS OS TESTES PASSARAM!")
        print("=" * 50)
        print("\nPróximo passo: Executar contra o servidor real:")
        print("  Terminal 1: python server.py")
        print("  Terminal 2: python student.py")
        
    except AssertionError as e:
        print(f"\n❌ TESTE FALHOU: {e}")
        import traceback
        traceback.print_exc()
    except Exception as e:
        print(f"\n❌ ERRO: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    run_all_tests()
