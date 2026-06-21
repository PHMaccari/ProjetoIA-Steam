# Steam Success Predictor

Sistema inteligente para **previsão de sucesso de jogos publicados na Steam**, desenvolvido como Projeto Final da disciplina de Inteligência Artificial Aplicada.

## Aviso

A pasta "steam_dataset" não esta presente no projeto conforme na imagem:
<img width="727" height="420" alt="{A0E21DDC-A96E-4B83-B00C-1221271990C6}" src="https://github.com/user-attachments/assets/251d5839-78e7-4fb7-9c7a-c55c3e9c0987" />  

Basta acessar o seguinte Link:  
<a href="https://drive.google.com/drive/folders/1ixBC-RpSDtkAoWpwYph4Ozjt0z0M7m6n?usp=drive_link" target="_blank">steam_dataset link</a>
  
Realizar o download do .ZIP, extrair e adicionar na pasta do projeto.

## Fluxo do sistema

```
Entrada (características do jogo) → Random Forest → Classificação → Explicação da decisão
```

## Estrutura do projeto

```
ProjetoIA/
├── app.py                          # Interface Streamlit
├── train.bat                       # Treina o modelo (Windows)
├── run_app.bat                     # Abre a interface (Windows)
├── install.bat                     # Instala dependências (Windows)
├── src/
│   ├── config.py                   # Constantes e caminhos
│   ├── data.py                     # ETL e feature engineering
│   ├── train.py                    # Treinamento do modelo
│   └── predict.py                  # Predição e explicabilidade
├── steam_dataset/                  # Dados brutos (steamds.csv)
├── artifacts/                      # Dataset processado e config do modelo
├── models/                         # Modelo treinado (.joblib)
└── requirements.txt
```

## Pré-requisitos

- Python 3.10+
- Arquivo `steamds.csv` em `steam_dataset/`

## Instalação e execução (mais fácil)

Os scripts `run_app.bat` e `install.bat` **se adaptam à pasta onde o projeto está** — não importa se você clonou em `Documents`, `Desktop` ou outro disco. Eles também tentam achar o Python automaticamente (`py`, `python` no PATH ou instalação padrão do Windows).

**Windows — clique duplo ou no CMD/PowerShell:**

```bat
cd CAMINHO\PARA\ProjetoIA2
install.bat      REM só na primeira vez
train.bat        REM treina o modelo (opcional; run_app.bat treina se faltar)
run_app.bat      REM abre a interface
```

**Linux / macOS / Git Bash:**

```bash
cd /caminho/para/ProjetoIA2
chmod +x run_app.sh
./run_app.sh
```

## Instalação manual

Na pasta do projeto (qualquer caminho):

```bash
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

## Treinar o modelo

**Windows — clique duplo ou:**

```bat
train.bat
```

**Linha de comando (qualquer SO):**

```bash
python -m src.train
```

Isso gera:

- `models/random_forest.joblib` — modelo treinado
- `artifacts/training_dataset.parquet` — dataset processado
- `artifacts/model_config.json` — configuração e métricas
- `artifacts/training_report.txt` — relatório completo

## Executar a interface

**PowerShell:**

```powershell
cd C:\Users\Pichau\Documents\ProjetoIA
& "$env:LOCALAPPDATA\Programs\Python\Python312\Scripts\streamlit.exe" run app.py
```

**Git Bash:**

```bash
cd /c/Users/Pichau/Documents/ProjetoIA
/c/Users/Pichau/AppData/Local/Programs/Python/Python312/Scripts/streamlit.exe run app.py
```

## Telas da interface

1. **Dashboard** — estatísticas gerais e gráficos do dataset
2. **Simulação** — formulário para inserir características do jogo
3. **Resultado** — classificação, probabilidades e explicação da IA

## Variável-alvo

O sucesso é classificado em três níveis com base em `recommendations_total`, dividido em **tercis balanceados** (33% dos jogos em cada classe):


| Classe          | Significado                     |
| --------------- | ------------------------------- |
| Baixo potencial | Terço inferior de recomendações |
| Médio potencial | Terço intermediário             |
| Alto potencial  | Terço superior                  |


## Features utilizadas

- Preço, free-to-play, idiomas, achievements
- Gêneros (Action, Indie, RPG, etc.)
- Multiplayer, single-player, co-op, early access
- Plataformas (Windows/Mac/Linux)
- Tier do publisher, suporte a controle, quantidade de tags

## Limitações

- A previsão reflete padrões históricos, não garante sucesso futuro
- Marketing, qualidade do gameplay e timing de lançamento não estão nos dados
- Jogos antigos tendem a acumular mais recomendações (viés temporal)


