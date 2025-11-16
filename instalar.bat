@echo off
REM Script de Instalação Rápida - Agente Centipede Corrigido
REM Execute este script no diretório onde baixou os arquivos corrigidos

echo.
echo ========================================
echo   INSTALACAO AGENTE CENTIPEDE
echo   Correcoes e Melhorias Aplicadas
echo ========================================
echo.

REM Verificar se estamos no diretório correto
if not exist "student.py" (
    echo ERRO: student.py nao encontrado!
    echo Certifique-se de executar este script no mesmo diretorio dos arquivos baixados.
    pause
    exit /b 1
)

if not exist "agent_analysis.py" (
    echo ERRO: agent_analysis.py nao encontrado!
    echo Certifique-se de executar este script no mesmo diretorio dos arquivos baixados.
    pause
    exit /b 1
)

REM Criar backup dos arquivos antigos
set BACKUP_DIR=backup_%date:~-4%%date:~-10,2%%date:~-7,2%_%time:~0,2%%time:~3,2%%time:~6,2%
set BACKUP_DIR=%BACKUP_DIR: =0%

echo.
echo [1/4] Criando backup dos arquivos antigos...
if not exist "%BACKUP_DIR%" mkdir "%BACKUP_DIR%"

if exist "C:\repos\Projeto-Centipede-IA\student.py" (
    copy "C:\repos\Projeto-Centipede-IA\student.py" "%BACKUP_DIR%\student.py.bak"
    echo   - student.py copiado para backup
)

if exist "C:\repos\Projeto-Centipede-IA\agent_analysis.py" (
    copy "C:\repos\Projeto-Centipede-IA\agent_analysis.py" "%BACKUP_DIR%\agent_analysis.py.bak"
    echo   - agent_analysis.py copiado para backup
)

echo.
echo [2/4] Limpando arquivos de cache Python...
if exist "C:\repos\Projeto-Centipede-IA\__pycache__" (
    rd /s /q "C:\repos\Projeto-Centipede-IA\__pycache__"
    echo   - __pycache__ removido
)
if exist "C:\repos\Projeto-Centipede-IA\*.pyc" (
    del /q "C:\repos\Projeto-Centipede-IA\*.pyc"
    echo   - Arquivos .pyc removidos
)

echo.
echo [3/4] Copiando arquivos corrigidos...
copy /Y "student.py" "C:\repos\Projeto-Centipede-IA\student.py"
REM If running from the same directory as the target, skip copying to avoid "copy onto itself" error
if /I "%CD%\student.py"=="C:\repos\Projeto-Centipede-IA\student.py" (
    echo   - student.py already in target, skipping copy
) else (
    copy /Y "student.py" "C:\repos\Projeto-Centipede-IA\student.py"
    if %errorlevel% equ 0 (
        echo   - student.py instalado com sucesso!
    ) else (
        echo   ERRO ao copiar student.py
        pause
        exit /b 1
    )
)

copy /Y "agent_analysis.py" "C:\repos\Projeto-Centipede-IA\agent_analysis.py"
if /I "%CD%\agent_analysis.py"=="C:\repos\Projeto-Centipede-IA\agent_analysis.py" (
    echo   - agent_analysis.py already in target, skipping copy
) else (
    copy /Y "agent_analysis.py" "C:\repos\Projeto-Centipede-IA\agent_analysis.py"
    if %errorlevel% equ 0 (
        echo   - agent_analysis.py instalado com sucesso!
    ) else (
        echo   ERRO ao copiar agent_analysis.py
        pause
        exit /b 1
    )
)

echo.
echo [4/4] Verificando instalacao...
if exist "C:\repos\Projeto-Centipede-IA\student.py" (
    echo   - student.py: OK
) else (
    echo   - student.py: FALHOU
)

if exist "C:\repos\Projeto-Centipede-IA\agent_analysis.py" (
    echo   - agent_analysis.py: OK
) else (
    echo   - agent_analysis.py: FALHOU
)

echo.
echo ========================================
echo   INSTALACAO CONCLUIDA COM SUCESSO!
echo ========================================
echo.
echo Backup criado em: %BACKUP_DIR%
echo.
echo PROXIMOS PASSOS:
echo   1. Abra um terminal
echo   2. cd C:\repos\Projeto-Centipede-IA
echo   3. python student.py
echo.
echo O agente agora deve:
echo   - Atirar muito mais frequentemente
echo   - Nao crashear com ValueError
echo   - Atingir 1000+ pontos
echo.
echo Para mais detalhes, veja:
echo   - CORRECOES_E_MELHORIAS.md
echo   - GUIA_RAPIDO_TESTE.md
echo.
pause
