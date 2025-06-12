
# 🌌 TRGI Simulator — Teoria da Realidade Geométrico-Informacional

Simulação computacional da Teoria da Realidade Geométrico-Informacional (TRGI), uma proposta teórica em que **informação quântica** é o substrato fundamental do universo, e **espaço-tempo, energia e matéria** são fenômenos emergentes.

Esta aplicação foi desenvolvida em **Python** com foco em visualização interativa, simulação de dinâmicas quânticas e análise métrica da geometria e energia informacional emergente.

---

## 📜 Sumário

- [🔭 Visão Geral da TRGI](#-visão-geral-da-trgi)
- [🧠 Estrutura da Simulação](#-estrutura-da-simulação)
- [📊 Métricas Computadas](#-métricas-computadas)
- [🛠️ Requisitos e Instalação](#️-requisitos-e-instalação)
- [🚀 Como Executar](#-como-executar)
- [📁 Estrutura do Projeto](#-estrutura-do-projeto)
- [📌 Exemplos de Resultados](#-exemplos-de-resultados)
- [📚 Referências e Inspirações](#-referências-e-inspirações)
- [🤝 Contribuições](#-contribuições)

---

## 🔭 Visão Geral da TRGI

A TRGI (Teoria da Realidade Geométrico-Informacional) propõe que:

1. **O substrato fundamental do universo é informação quântica** (`Ψ_I`), representada por qubits.
2. **A geometria do espaço-tempo emerge** da organização dessa informação (métrica emergente baseada na distância de estados quânticos).
3. **Energia e partículas são padrões estáveis de informação**, e o tensor energia-momento é reinterpretado como fluxo e densidade de informação.
4. **Um ciclo de feedback causal** regula o sistema:
   - A geometria influencia a dinâmica dos infons.
   - A dinâmica dos infons modifica a geometria.

---

## 🧠 Estrutura da Simulação

- A simulação ocorre em uma **grade 2D periódica** de qubits (`40x40` por padrão).
- Cada qubit evolui no tempo segundo um **Hamiltoniano local** (modelo de Ising Transverso com acoplamento variável):
  ```math
  H_{ij} = -J_{	ext{eff}} · Z_i ⊗ Z_j - h · (X_i ⊗ I + I ⊗ X_j)
  ```
- A **curvatura emergente** é calculada com base na variação de “distâncias informacionais” (ângulo de Bloch entre vizinhos).
- A energia local (**T₀₀**) é o valor esperado do Hamiltoniano local.
- O **acoplamento `J_eff`** depende da geometria: qubits mais alinhados interagem mais fortemente.

---

## 📊 Métricas Computadas

Durante a simulação, são coletadas e visualizadas:

- **Entropia de Shannon** (organização global da informação)
- **Curvatura Média** (estrutura geométrica emergente)
- **Energia Média** (T₀₀)
- **Correlação local** entre curvatura e energia (scatter plot + regressão linear)

---

## 🛠️ Requisitos e Instalação

```bash
git clone https://github.com/seunome/TRGI-simulator.git
cd TRGI-simulator
pip install -r requirements.txt
```

**Requisitos principais:**

- Python 3.10+
- numpy
- matplotlib
- scipy

---

## 🚀 Como Executar

Para abrir a interface gráfica interativa com visualização em tempo real:

```bash
python gui/interactive_sim_mpl.py
```

Ou, para rodar análises diretamente:

```python
from core.analysis import plot_history, plot_correlation
```

---

## 📁 Estrutura do Projeto

```
TRGI-simulator/
│
├── core/                    # Núcleo da simulação TRGI
│   ├── manifold.py          # Estrutura da grade e vizinhanças
│   ├── infon_qubit.py       # Definição dos qubits (infons)
│   ├── dynamics.py          # Regras de evolução (Ising quântico)
│   ├── geometry.py          # Cálculo da métrica e curvatura emergente
│   ├── t_tensor.py          # Tensor de energia informacional (T00)
│   ├── metrics.py           # Métricas globais (entropia, etc.)
│
├── gui/
│   └── interactive_sim_mpl.py   # Interface gráfica (matplotlib interativo)
│
├── config/
│   └── default_params.json      # Parâmetros da simulação
│
├── results/                 # Saídas gráficas e dados (opcional)
└── README.md
```

---

## 📌 Exemplos de Resultados

- **Fase desordenada (h = 0.8):**
  - Alta entropia e curvatura
  - Nenhuma correlação entre energia e curvatura

- **Fase ordenada (h = 0.2):**
  - Formação de domínios estruturados
  - Queda na entropia e energia
  - Correlação positiva entre curvatura e energia:
    ```math
    T_{00} ∝ Curvatura
    ```

---

## 📚 Referências e Inspirações

- John Wheeler – “It from Bit”
- Carlo Rovelli – Relational Quantum Mechanics
- Erik Verlinde – Gravidade emergente
- Computação Quântica e Modelo de Ising Transverso
- Autômatos celulares e sistemas auto-organizados

---

## 🤝 Contribuições

Ideias, críticas e sugestões são muito bem-vindas!  
Este é um projeto aberto, feito por curiosidade científica.

Se você é da física, ciência da computação, matemática, IA ou apenas curioso por universos simulados — vem junto!

---

**Autor:** Erik Mendes  
**Licença:** MIT (ou outra de sua escolha)
