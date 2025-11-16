import os
import glob
import json
from collections import Counter, defaultdict
from statistics import mean

LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')

def load_latest_log():
    files = sorted(glob.glob(os.path.join(LOG_DIR, 'agent_*.jsonl')))
    if not files:
        print('Nenhum ficheiro de log encontrado em \'logs/\'.')
        return []
    latest = files[-1]
    print(f'➡️  A analisar: {os.path.basename(latest)}')
    events = []
    with open(latest, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                events.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return events


events = load_latest_log()
if not events:
    raise SystemExit(0)

# Filtrar só decisões
decisions = [e for e in events if e.get('kind') == 'decision']
transitions = [e for e in events if e.get('kind') == 'mode_transition']

print(f"Total de decisões: {len(decisions)} | Transições: {len(transitions)}")

# Contagens básicas
by_mode = Counter(d.get('mode') for d in decisions)
by_action = Counter(d.get('action') or '' for d in decisions)
by_reason = Counter(d.get('reason') or '' for d in decisions)

print('\n— Distribuição por modo —')
for m, c in by_mode.most_common():
    print(f"  {m:12s}: {c}")

print('\n— Distribuição de ações —')
for a, c in by_action.most_common():
    label = a if a else '(vazio)'
    print(f"  {label:8s}: {c}")

print('\n— Principais razões —')
for r, c in by_reason.most_common(10):
    label = r if r else '(sem razão)'
    print(f"  {label:28s}: {c}")

# Métricas por modo
danger_by_mode = defaultdict(list)
dt_by_mode = defaultdict(list)
shots = 0
for d in decisions:
    danger = d.get('danger')
    if isinstance(danger, (int, float)):
        danger_by_mode[d.get('mode')].append(danger)
    dt = d.get('decision_ms')
    if isinstance(dt, (int, float)):
        dt_by_mode[d.get('mode')].append(dt)
    if (d.get('action') or '') == 'A':
        shots += 1

print('\n— Métricas por modo —')
for m in sorted(danger_by_mode.keys() | dt_by_mode.keys()):
    dangers = danger_by_mode.get(m, [])
    dts = dt_by_mode.get(m, [])
    mdanger = f"{mean(dangers):.2f}" if dangers else 'n/a'
    mdt = f"{mean(dts):.2f} ms" if dts else 'n/a'
    print(f"  {m:12s}: perigo médio={mdanger} | decisão média={mdt}")

print('\n— Precisão de disparos (proxy) —')
total_dec = len(decisions)
rate = (shots/total_dec)*100 if total_dec else 0.0
print(f"  Disparos executados: {shots} de {total_dec} decisões ({rate:.1f}% taxa de disparo)")

print('\nSugestões iniciais (heurísticas):')
# Heurísticas simples com base nos dados
if total_dec == 0:
    print('  • Sem decisões registadas — execute o agente contra o servidor para gerar logs.')
    raise SystemExit(0)

attack_share = by_mode.get('ATTACK', 0) / total_dec
evade_share = by_mode.get('EVADE', 0) / total_dec
if attack_share < 0.4:
    print('  • Pouco tempo em ATTACK → considerar diminuir DANGER_DISTANCE_LINES (mais agressivo).')
if evade_share > 0.3:
    print('  • Muito tempo em EVADE → considerar aumentar SAFE_DISTANCE_LINES (sair mais cedo).')
if by_reason.get('no_accessible_targets', 0) > 10:
    print('  • Muitos \'no_accessible_targets\' → reforçar REPOSITION para limpar colunas bloqueadas.')
if by_reason.get('random_safe_move', 0) > 20:
    print('  • Muitos \"random_safe_move\" → implementar BFS simples para caminho até coluna alvo.')
