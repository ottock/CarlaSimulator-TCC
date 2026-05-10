# CarlaSimulator-TCC

Simulador autônomo baseado no [CARLA 0.9.16](https://github.com/carla-simulator/carla/releases) desenvolvido como Trabalho de Conclusão de Curso. O sistema spawna um veículo ego com sensores LIDAR e câmera RGB, detecta obstáculos em tempo real e exibe uma janela de visualização multi-painel com a nuvem de pontos, feed de câmera e tabela de obstáculos.

---

## Sumário

- [Pré-requisitos](#pré-requisitos)
- [Instalação](#instalação)
- [Como executar](#como-executar)
- [Configuração do JSON](#configuração-do-json)
- [Estrutura do projeto](#estrutura-do-projeto)

---

## Pré-requisitos

| Requisito | Versão mínima |
|---|---|
| Python | 3.10+ |
| CARLA Simulator | 0.9.16 |
| Windows | 10/11 (64-bit) |

> **Download do CARLA:** https://github.com/carla-simulator/carla/releases

---

## Instalação

### 1. Clone o repositório

```bash
git clone <url-do-repositorio>
cd CarlaSimulator-TCC
```

### 2. Instale o CARLA

Extraia o CARLA 0.9.16 na raiz do projeto. A estrutura esperada é:

```
CarlaSimulator-TCC/
└── CARLA_0.9.16/
    └── CarlaUE4.exe
```

### 3. Crie o ambiente virtual e instale as dependências

Execute o script de setup na raiz do projeto:

```bash
scripts\createVenv.bat
```

Este script automaticamente:
1. Cria o ambiente virtual `.venv`
2. Ativa o ambiente
3. Instala todas as dependências do `requirements.txt`

> **Alternativa manual** (se o script não funcionar):
> ```bash
> python -m venv .venv
> .venv\Scripts\activate
> pip install -r requirements.txt
> ```

O arquivo `requirements.txt` instala:

| Pacote | Versão | Finalidade |
|---|---|---|
| `carla` | 0.9.16 | API Python do simulador |
| `numpy` | 2.4.4 | Processamento vetorizado da nuvem de pontos |
| `pygame` | 2.6.1 | Janela de visualização em tempo real |
| `logging` | 0.4.9.6 | Logging estruturado |

---

## Como executar

Com o ambiente virtual ativo, execute a partir da pasta `src/`:

```bash
python src/main.py settings/baseSettings.json
```

Ou use um arquivo de mapa específico:

```bash
python src/main.py settings/Town03.json
python src/main.py settings/Town05.json
```

O sistema:
1. Lança automaticamente o `CarlaUE4.exe`
2. Aguarda o boot do servidor (configurável via `server_boot_time`)
3. Conecta ao CARLA e carrega o mapa
4. Spawna o veículo ego com LIDAR e câmera
5. Spawna tráfego e pedestres conforme configurado
6. Abre a janela de visualização

### Controles da janela de visualização

| Tecla | Ação |
|---|---|
| `SPACE` | Pausar / retomar visualização |
| `ESC` | Fechar a janela |

---

## Configuração do JSON

Todos os arquivos de configuração ficam em `settings/`. A estrutura completa é:

```json
{
    "carla_client": { ... },
    "world": { ... }
}
```

### `carla_client` — Conexão com o servidor

```json
"carla_client": {
    "host": "localhost",
    "port": 2000,
    "timeout": 10.0,
    "max_retries": 3,
    "retry_delay": 2,
    "server_boot_time": 30
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `host` | string | Endereço IP do servidor CARLA |
| `port` | int | Porta do servidor (padrão 2000) |
| `timeout` | float | Timeout de conexão em segundos |
| `max_retries` | int | Tentativas de reconexão em caso de falha |
| `retry_delay` | int | Intervalo entre tentativas (segundos) |
| `server_boot_time` | int | Segundos para aguardar o boot do servidor |

---

### `world.map_name` — Mapa

```json
"map_name": "Town01"
```

Mapas disponíveis no CARLA 0.9.16: `Town01`, `Town02`, `Town03`, `Town04`, `Town05`, `Town06`, `Town07`, `Town10HD`.

---

### `world.actor_vehicle` — Veículo ego e sensores

```json
"actor_vehicle": {
    "enabled": true,
    "vehicle_filter": "vehicle.tesla.model3",
    "spawn_index": 0,
    "use_lidar": true,
    "use_camera": true,
    "lidar": {
        "channels": 32,
        "points_per_second": 100000,
        "rotation_frequency": 10,
        "range": 100,
        "upper_fov": 15,
        "lower_fov": -25
    }
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `enabled` | bool | Ativa ou desativa o veículo ego |
| `vehicle_filter` | string | Blueprint do veículo (ex: `vehicle.tesla.model3`) |
| `spawn_index` | int | Índice do ponto de spawn no mapa |
| `use_lidar` | bool | Monta sensor LIDAR no veículo |
| `use_camera` | bool | Monta câmera RGB no veículo |

#### Subcampo `lidar`

| Campo | Tipo | Descrição |
|---|---|---|
| `channels` | int | Número de feixes laser (resolução vertical) |
| `points_per_second` | int | Taxa de pontos por segundo |
| `rotation_frequency` | int | Frequência de rotação em Hz |
| `range` | float | Alcance máximo em metros |
| `upper_fov` | float | Ângulo vertical superior em graus |
| `lower_fov` | float | Ângulo vertical inferior em graus (negativo) |

---

### `world.settings` — Física da simulação

```json
"settings": {
    "synchronous_mode": true,
    "fixed_delta_seconds": 0.05,
    "no_rendering_mode": false,
    "substepping": true,
    "max_substep_delta_time": 0.01,
    "max_substeps": 10
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `synchronous_mode` | bool | Modo síncrono (recomendado para coleta de dados) |
| `fixed_delta_seconds` | float | Delta de tempo fixo por tick (0.05 = 20 FPS de simulação) |
| `no_rendering_mode` | bool | Desativa rendering gráfico (modo headless) |
| `substepping` | bool | Subdivide o passo de física para maior precisão |
| `max_substep_delta_time` | float | Delta máximo de cada sub-passo |
| `max_substeps` | int | Número máximo de sub-passos por tick |

---

### `world.weather` — Condições climáticas

```json
"weather": {
    "cloudiness": 80.0,
    "precipitation": 30.0,
    "precipitation_deposits": 50.0,
    "wind_intensity": 20.0,
    "sun_azimuth_angle": 45.0,
    "sun_altitude_angle": 30.0,
    "fog_density": 10.0,
    "fog_distance": 50.0,
    "wetness": 70.0,
    "dust_storm": 0.0
}
```

Todos os valores são `float` entre `0.0` e `100.0`, exceto ângulos solares (graus). Valores maiores intensificam o efeito.

---

### `world.traffic_manager` — Gerenciador de tráfego

```json
"traffic_manager": {
    "port": 8000,
    "synchronous_mode": true,
    "hybrid_physics_mode": true,
    "hybrid_physics_radius": 50.0,
    "global_distance": 3.0,
    "speed_difference": 0
}
```

| Campo | Tipo | Descrição |
|---|---|---|
| `port` | int | Porta do Traffic Manager |
| `hybrid_physics_mode` | bool | Física completa apenas dentro do raio, telepórtese fora |
| `hybrid_physics_radius` | float | Raio em metros para física completa |
| `global_distance` | float | Distância mínima entre veículos (metros) |
| `speed_difference` | int | Diferença de velocidade em relação ao limite (%) |

---

### `world.traffic` — Veículos de tráfego

```json
"traffic": {
    "vehicle_count": 30,
    "vehicle_filter": "vehicle.*",
    "auto_lane_change": false
}
```

---

### `world.pedestrians` — Pedestres

```json
"pedestrians": {
    "count": 20,
    "running_percentage": 10,
    "crossing_percentage": 20
}
```

---

### Arquivos de configuração disponíveis

| Arquivo | Mapa | Veículos | Pedestres | Uso |
|---|---|---|---|---|
| `baseSettings.json` | Town01 | 30 | 20 | Configuração padrão |
| `Town01.json` | Town01 | 30 | 20 | Town01 padrão |
| `Town03.json` | Town03 | 5 | 0 | Ambiente urbano leve |
| `Town05.json` | Town05 | 30 | 20 | Rodovias e cruzamentos |

---

## Estrutura do projeto

```
CarlaSimulator-TCC/
├── CARLA_0.9.16/           # Binários do simulador (não versionado)
├── log/                    # Logs gerados em execução
├── settings/               # Arquivos de configuração JSON
│   ├── baseSettings.json
│   ├── Town01.json
│   ├── Town03.json
│   └── Town05.json
├── src/                    # Código-fonte principal
│   ├── main.py             # Ponto de entrada da aplicação
│   ├── core/               # Lógica de simulação (domínio)
│   │   ├── carlaClient/    # Integração com a API do CARLA
│   │   │   ├── world_manager.py    # Ciclo de vida do mundo e cenário
│   │   │   ├── sensor_manager.py   # Coordenação e factory de sensores
│   │   │   ├── lidar_manager.py    # Sensor LIDAR + detecção de obstáculos
│   │   │   └── camera_manager.py   # Sensor câmera RGB
│   │   └── logger/         # Sistema de logging
│   │       ├── logger.py
│   │       └── configs/
│   │           └── base_config.json
│   ├── utils/              # Utilitários transversais
│   │   └── config.py       # Carregamento e validação do JSON
│   └── presentation/       # Camada de visualização
│       └── visualization.py # Janela pygame com nuvem de pontos
└── requirements.txt
```

### Camadas da arquitetura

```
┌─────────────────────────────────────┐
│           main.py                   │  Orquestração: boot, lifecycle, threads
├─────────────────────────────────────┤
│        presentation/                │  Visualização (pygame, 30 FPS)
│        visualization.py             │  Top view · Camera · Side · Front · Info
├─────────────────────────────────────┤
│          core/carlaClient/          │  Domínio da simulação
│  world_manager   sensor_manager     │  Mundo, cenário, tráfego, câmera espectadora
│  lidar_manager   camera_manager     │  Sensores, detecção de obstáculos (BFS grid)
├─────────────────────────────────────┤
│            utils/                   │  Infraestrutura transversal
│  config.py     core/logger/         │  Parsing/validação JSON · Logging rotativo
└─────────────────────────────────────┘
         ↑ configurado via settings/*.json
```

#### Fluxo de dados dos sensores

```
CARLA Server
    │
    ├─ LidarSensor._parse_lidar_data()
    │       │  detecção de obstáculos (BFS grid 0.5m)
    │       └─→ Queue → LidarVisualization.update_lidar_data()
    │
    └─ CameraRGBSensor._parse_camera_data()
            │  conversão para numpy array
            └─→ Queue → LidarVisualization.update_camera_data()

LidarVisualization.run()  [thread separada, 30 FPS]
    ├─ _draw_top_view()     — nuvem de pontos XY + obstáculos
    ├─ _draw_camera_view()  — feed RGB
    ├─ _draw_side_view()    — perfil de elevação XZ
    ├─ _draw_front_view()   — perspectiva frontal YZ
    └─ _draw_info_panel()   — tabela de obstáculos (distância, direção)
```
