# Fase 3 — Dual-input (câmera + LiDAR) + longitudinal — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Substituir o modelo só-câmera por um modelo dual-input (câmera + LiDAR) que também controla `throttle`/`brake`, para frear atrás de veículos sem perder a direção.

**Architecture:** Modelo único fundido (`DrivingNet`): braço de câmera PilotNet (idêntico ao `CameraSteeringNet`, warm-started do `cam_v2.pt`) + braço Conv1D de LiDAR (padding circular) → concat → FC → 3 saídas `[steer(tanh), throttle(sigmoid), brake(sigmoid)]`. Treino/validação nas Towns com tráfego dinâmico (só veículos). O pipeline de dados (writer/index) já é dual-input-ready.

**Tech Stack:** Python 3.x, PyTorch 2.6+cu124 (RTX 3070), NumPy, OpenCV, CARLA 0.9.16 (modo síncrono 20 Hz), pytest.

## Global Constraints

- **LiDAR:** 72 setores; `max_range = 12.0` m; normalização para a rede = `clip(metros/12.0, 0, 1)` (perto=0, livre=1).
- **Split sempre por episódio**, nunca por frame (usar `split_episodes`).
- **Não quebrar a Fase 2:** `CameraSteeringNet`, `SteeringDataset`, `ModelSteeringPolicy` permanecem intactos.
- **Warm-start:** o braço de câmera do `DrivingNet` deve ter a MESMA `nn.Sequential` de conv (mesmos nomes de submódulo `cnn.*`) que `CameraSteeringNet`, para cópia direta de `state_dict`.
- **`src/ai/shared/*` continua compatível com Python 3.6** (roda no Jetson depois). `model.py`/`dataset.py`/`train.py` são só-PC (podem usar sintaxe moderna).
- **Expert = autopilot do TM** (lateral + longitudinal), com `ignore_traffic_lights=True`. Os `throttle`/`brake` lidos de `vehicle.get_control()` são os rótulos.
- **Modelo pequeno e exportável** (dois inputs: `img`, `lidar`). < 2 M params.
- **Dados fora do OneDrive:** `D:\tcc_data`. `dataset_v1`/`v2` seguem válidos para direção; a coleta com tráfego gera **`dataset_v3`**. Checkpoints em `D:/tcc_data/runs/` (`driving_v1.pt`).
- **Rodar comandos do diretório raiz do repo.** Testes: `python -m pytest tests/ -q` (não precisam de servidor CARLA).

---

## File Structure

- `src/ai/shared/lidar_pipeline.py` **(modificar)** — + `normalize_sectors_m()` (fonte única metros→[0,1]).
- `src/ai/model.py` **(modificar)** — + `DrivingNet` (mantém `CameraSteeringNet`).
- `src/ai/dataset_index.py` **(modificar)** — + `sample_weights_dual()` (curva ∪ frenagem).
- `src/ai/dataset.py` **(modificar)** — + `DrivingDataset` (mantém `SteeringDataset`).
- `src/ai/warmstart.py` **(criar)** — `load_camera_backbone()` (copia `cnn.*` do checkpoint da câmera).
- `src/ai/train.py` **(modificar)** — + `train_dual()` + CLI `--dual`/`--init-from`.
- `src/ai/model_policy.py` **(modificar)** — + `DrivingPolicy` (mantém `ModelSteeringPolicy`).
- `src/ai/eval_openloop.py` **(modificar)** — + `evaluate_dual()` (MAE por eixo), auto-detecção de arch.
- `src/ai/collect.py` **(modificar)** — + `--traffic N` (spawn de veículos → frenagens).
- `src/ai/eval_closedloop.py` **(modificar)** — + `--traffic N`, `--ablate-lidar`, driver = `DrivingPolicy` quando arch=`DrivingNet`.
- `docs/ESTADO_IA.md` **(modificar)** — estado ao fim da fase.
- `tests/test_lidar_pipeline.py`, `tests/test_model.py`, `tests/test_dataset_index.py`, `tests/test_dataset.py`, `tests/test_model_policy.py` **(modificar/adicionar casos)**; `tests/test_warmstart.py`, `tests/test_train_dual.py`, `tests/test_eval_openloop_dual.py` **(criar)**.

---

## Task 1: Normalização do LiDAR (fonte única)

**Files:**
- Modify: `src/ai/shared/lidar_pipeline.py`
- Test: `tests/test_lidar_pipeline.py`

**Interfaces:**
- Produces: `normalize_sectors_m(sectors_m, max_range=MAX_RANGE_M) -> np.ndarray` (float32, shape igual à entrada, valores em [0,1]; perto=0, livre/além=1).

- [ ] **Step 1: Escrever o teste que falha**

Adicionar ao fim de `tests/test_lidar_pipeline.py`:

```python
import numpy as np
from ai.shared.lidar_pipeline import normalize_sectors_m


def test_normalize_maps_metres_to_unit_range():
    out = normalize_sectors_m(np.array([0.0, 6.0, 12.0], dtype=np.float32), max_range=12.0)
    assert np.allclose(out, [0.0, 0.5, 1.0])
    assert out.dtype == np.float32


def test_normalize_clips_beyond_range():
    out = normalize_sectors_m(np.array([-1.0, 24.0], dtype=np.float32), max_range=12.0)
    assert np.allclose(out, [0.0, 1.0])  # clipa abaixo de 0 e acima de max_range
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m pytest tests/test_lidar_pipeline.py -q`
Expected: FAIL (`ImportError: cannot import name 'normalize_sectors_m'`).

- [ ] **Step 3: Implementar**

Adicionar ao fim de `src/ai/shared/lidar_pipeline.py`:

```python
def normalize_sectors_m(sectors_m, max_range=MAX_RANGE_M):
    """Turn stored per-sector metres into model input in [0, 1] (near=0, free=1).

    Single source of truth used by both training (``DrivingDataset``) and
    inference (``DrivingPolicy``). Python 3.6-safe (runs on the Jetson too).
    """
    x = np.asarray(sectors_m, dtype=np.float32) / float(max_range)
    return np.clip(x, 0.0, 1.0).astype(np.float32)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python -m pytest tests/test_lidar_pipeline.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ai/shared/lidar_pipeline.py tests/test_lidar_pipeline.py
git commit -m "feat(ai): normalize_sectors_m — fonte unica metros->[0,1] do LiDAR"
```

---

## Task 2: `DrivingNet` (modelo fundido)

**Files:**
- Modify: `src/ai/model.py`
- Test: `tests/test_model.py`

**Interfaces:**
- Consumes: nada de tarefas anteriores.
- Produces: `DrivingNet(n_sectors=72, dropout=0.3)` com `forward(img (B,3,66,200), lidar (B,n_sectors)) -> (B,3)`, colunas `[steer∈[-1,1], throttle∈[0,1], brake∈[0,1]]`. O submódulo `self.cnn` é IDÊNTICO ao de `CameraSteeringNet` (para warm-start).

- [ ] **Step 1: Escrever os testes que falham**

Adicionar ao fim de `tests/test_model.py`:

```python
from ai.model import DrivingNet


def test_dual_forward_output_shape():
    net = DrivingNet()
    out = net(torch.zeros(4, 3, 66, 200), torch.zeros(4, 72))
    assert out.shape == (4, 3)


def test_dual_output_ranges():
    net = DrivingNet()
    out = net(torch.randn(8, 3, 66, 200), torch.rand(8, 72))
    steer, throttle, brake = out[:, 0], out[:, 1], out[:, 2]
    assert float(steer.min()) >= -1.0 and float(steer.max()) <= 1.0
    assert float(throttle.min()) >= 0.0 and float(throttle.max()) <= 1.0
    assert float(brake.min()) >= 0.0 and float(brake.max()) <= 1.0


def test_dual_lidar_changes_output():
    # Prova que o LiDAR está ligado à saída (relevante para a ablation).
    torch.manual_seed(0)
    net = DrivingNet().eval()
    img = torch.randn(2, 3, 66, 200)
    a = net(img, torch.zeros(2, 72))
    b = net(img, torch.ones(2, 72))
    assert not torch.allclose(a, b)


def test_dual_zero_lidar_forward_ok():
    # Caminho da ablation: LiDAR zerado não pode quebrar.
    net = DrivingNet()
    out = net(torch.zeros(1, 3, 66, 200), torch.zeros(1, 72))
    assert out.shape == (1, 3)


def test_dual_camera_backbone_matches_camera_net():
    # Mesmas chaves e shapes de conv que CameraSteeringNet (pré-requisito do warm-start).
    cam = CameraSteeringNet(); cam(torch.zeros(1, 3, 66, 200))
    dual = DrivingNet(); dual(torch.zeros(1, 3, 66, 200), torch.zeros(1, 72))
    cam_cnn = {k: v.shape for k, v in cam.state_dict().items() if k.startswith("cnn.")}
    dual_cnn = {k: v.shape for k, v in dual.state_dict().items() if k.startswith("cnn.")}
    assert cam_cnn == dual_cnn


def test_dual_model_is_small():
    net = DrivingNet()
    net(torch.zeros(1, 3, 66, 200), torch.zeros(1, 72))
    assert sum(p.numel() for p in net.parameters()) < 2_000_000
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m pytest tests/test_model.py -q`
Expected: FAIL (`ImportError: cannot import name 'DrivingNet'`).

- [ ] **Step 3: Implementar**

Adicionar ao fim de `src/ai/model.py` (o import de `torch` no topo do arquivo passa a ser necessário):

```python
import torch


class DrivingNet(nn.Module):
    """Dual-input BC net (Fase 3).

    Camera arm = the SAME PilotNet CNN as CameraSteeringNet (so its weights can be
    warm-started from cam_v2.pt). LiDAR arm = small Conv1D with circular padding
    (sector 71 neighbours sector 0). Outputs [steer, throttle, brake].

    forward(img (B,3,66,200) in [-1,1], lidar (B,n_sectors) in [0,1]) -> (B,3).
    """

    def __init__(self, n_sectors: int = 72, dropout: float = 0.3):
        super().__init__()
        self.cnn = nn.Sequential(                       # IDÊNTICO à CameraSteeringNet
            nn.Conv2d(3, 24, 5, stride=2), nn.ELU(),
            nn.Conv2d(24, 36, 5, stride=2), nn.ELU(),
            nn.Conv2d(36, 48, 5, stride=2), nn.ELU(),
            nn.Conv2d(48, 64, 3), nn.ELU(),
            nn.Conv2d(64, 64, 3), nn.ELU(),
            nn.Flatten(),
        )
        self.lidar = nn.Sequential(                     # padding circular = setores dão a volta
            nn.Conv1d(1, 16, 5, padding=2, padding_mode="circular"), nn.ELU(),
            nn.Conv1d(16, 32, 3, padding=1, padding_mode="circular"), nn.ELU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.LazyLinear(100), nn.ELU(), nn.Dropout(dropout),
            nn.Linear(100, 50), nn.ELU(),
        )
        self.steer_head = nn.Sequential(nn.Linear(50, 1), nn.Tanh())
        self.throttle_head = nn.Sequential(nn.Linear(50, 1), nn.Sigmoid())
        self.brake_head = nn.Sequential(nn.Linear(50, 1), nn.Sigmoid())

    def forward(self, img, lidar):
        cam = self.cnn(img)                             # (B, Fcam)
        lid = self.lidar(lidar.unsqueeze(1))            # (B, Flid)
        h = self.fc(torch.cat([cam, lid], dim=1))       # (B, 50)
        return torch.cat(
            [self.steer_head(h), self.throttle_head(h), self.brake_head(h)], dim=1
        )                                               # (B, 3)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python -m pytest tests/test_model.py -q`
Expected: PASS (todos, incluindo os da Fase 2).

- [ ] **Step 5: Commit**

```bash
git add src/ai/model.py tests/test_model.py
git commit -m "feat(ai): DrivingNet — dual-input camera+LiDAR (Conv1D circular) -> steer/throttle/brake"
```

---

## Task 3: `sample_weights_dual` (curva ∪ frenagem)

**Files:**
- Modify: `src/ai/dataset_index.py`
- Test: `tests/test_dataset_index.py`

**Interfaces:**
- Consumes: nada.
- Produces: `sample_weights_dual(steers, brakes, straight_thr=0.05, brake_thr=0.05, straight_w=1.0, curve_w=3.0, brake_w=3.0) -> np.ndarray` (float64). Peso do frame = máximo entre o peso de curva e o de frenagem.

- [ ] **Step 1: Escrever o teste que falha**

Adicionar ao fim de `tests/test_dataset_index.py`:

```python
from ai.dataset_index import sample_weights_dual


def test_sample_weights_dual_upweights_curve_and_brake():
    steers = [0.0, 0.5, 0.0, 0.5]
    brakes = [0.0, 0.0, 0.8, 0.8]
    w = sample_weights_dual(steers, brakes, curve_w=3.0, brake_w=4.0)
    assert w[0] == 1.0          # reto, sem freio
    assert w[1] == 3.0          # curva
    assert w[2] == 4.0          # freio
    assert w[3] == 4.0          # curva + freio -> máximo dos dois
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m pytest tests/test_dataset_index.py -q`
Expected: FAIL (`ImportError: cannot import name 'sample_weights_dual'`).

- [ ] **Step 3: Implementar**

Adicionar ao fim de `src/ai/dataset_index.py`:

```python
def sample_weights_dual(steers, brakes, straight_thr=0.05, brake_thr=0.05,
                        straight_w=1.0, curve_w=3.0, brake_w=3.0):
    """Per-sample weights combining two imbalances: curves and braking.

    Both curves (|steer| high) and braking frames (brake high) are rare; a frame
    that is either gets upweighted. Weight = max(curve weight, brake weight).
    """
    s = np.abs(np.asarray(steers, dtype=np.float64))
    b = np.asarray(brakes, dtype=np.float64)
    curve = np.where(s < straight_thr, straight_w, curve_w)
    brake = np.where(b < brake_thr, straight_w, brake_w)
    return np.maximum(curve, brake).astype(np.float64)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python -m pytest tests/test_dataset_index.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ai/dataset_index.py tests/test_dataset_index.py
git commit -m "feat(ai): sample_weights_dual — sobre-amostra curvas e frenagem"
```

---

## Task 4: `DrivingDataset`

**Files:**
- Modify: `src/ai/dataset.py`
- Test: `tests/test_dataset.py`

**Interfaces:**
- Consumes: `normalize_sectors_m` (Task 1); `build_index` (existente, já entrega `image`/`lidar`/`row`/`steer`/`throttle`/`brake`).
- Produces: `DrivingDataset(index, max_range=12.0)`; `__getitem__` → `(img tensor (3,66,200) f32, lidar tensor (72,) f32 em [0,1], target tensor (3,) f32 = [steer,throttle,brake])`.

- [ ] **Step 1: Escrever os testes que falham**

Adicionar ao fim de `tests/test_dataset.py`:

```python
from ai.dataset import DrivingDataset


def _episode_dual(path, rows):
    """rows: lista de (steer, throttle, brake). LiDAR do frame i = np.full(72, i)."""
    w = EpisodeWriter(str(path))
    for i, (s, th, br) in enumerate(rows):
        w.add(np.zeros((360, 640, 3), dtype=np.uint8),
              np.full(72, float(i), dtype=np.float32),
              {"steer": s, "throttle": th, "brake": br, "v": 1.0,
               "x": 0, "y": 0, "yaw": 0, "noise_active": False})
    w.close()


def test_driving_item_shapes_and_target(tmp_path):
    _episode_dual(tmp_path / "ep_0001", [(0.1, 0.5, 0.0), (-0.2, 0.0, 0.9)])
    ds = DrivingDataset(build_index([str(tmp_path / "ep_0001")]))
    img, lidar, target = ds[1]
    assert tuple(img.shape) == (3, 66, 200) and img.dtype == torch.float32
    assert tuple(lidar.shape) == (72,) and lidar.dtype == torch.float32
    assert tuple(target.shape) == (3,)
    assert float(target[0]) == pytest.approx(-0.2)
    assert float(target[1]) == pytest.approx(0.0)
    assert float(target[2]) == pytest.approx(0.9)


def test_driving_lidar_row_alignment_and_norm(tmp_path):
    # frame 1 -> lidar.npy linha 1 = full(1.0) -> normalizado 1/12.
    _episode_dual(tmp_path / "ep_0001", [(0.0, 0.5, 0.0), (0.0, 0.5, 0.0)])
    ds = DrivingDataset(build_index([str(tmp_path / "ep_0001")]), max_range=12.0)
    _, lidar, _ = ds[1]
    assert float(lidar[0]) == pytest.approx(1.0 / 12.0)
    assert float(lidar.min()) >= 0.0 and float(lidar.max()) <= 1.0
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m pytest tests/test_dataset.py -q`
Expected: FAIL (`ImportError: cannot import name 'DrivingDataset'`).

- [ ] **Step 3: Implementar**

Adicionar ao fim de `src/ai/dataset.py` (usa `np`, `torch`, `cv2`, `preprocess` já importados no arquivo):

```python
from ai.shared.lidar_pipeline import normalize_sectors_m


class DrivingDataset(Dataset):
    """Dual-input: yields ``(img (3,66,200), lidar (72,) in [0,1], target (3,))``.

    Target = [steer, throttle, brake]. LiDAR is loaded per-episode via mmap and
    normalized with the SAME function the car will use at inference.
    """

    def __init__(self, index, max_range=12.0):
        self.index = index
        self.max_range = max_range
        self._lidar_cache = {}

    def __len__(self):
        return len(self.index)

    def _lidar_array(self, path):
        arr = self._lidar_cache.get(path)
        if arr is None:
            arr = np.load(path, mmap_mode="r")
            self._lidar_cache[path] = arr
        return arr

    def __getitem__(self, i):
        rec = self.index[i]
        img_bgr = cv2.imread(rec["image"], cv2.IMREAD_COLOR)
        if img_bgr is None:
            raise FileNotFoundError(rec["image"])
        x = preprocess(img_bgr)
        sectors_m = np.asarray(self._lidar_array(rec["lidar"])[rec["row"]], dtype=np.float32)
        lidar = normalize_sectors_m(sectors_m, self.max_range)
        target = np.array([rec["steer"], rec["throttle"], rec["brake"]], dtype=np.float32)
        return torch.from_numpy(x), torch.from_numpy(lidar), torch.from_numpy(target)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python -m pytest tests/test_dataset.py -q`
Expected: PASS (Fase 2 + novos).

- [ ] **Step 5: Commit**

```bash
git add src/ai/dataset.py tests/test_dataset.py
git commit -m "feat(ai): DrivingDataset — img + LiDAR normalizado + alvo [steer,throttle,brake]"
```

---

## Task 5: Warm-start do braço de câmera

**Files:**
- Create: `src/ai/warmstart.py`
- Test: `tests/test_warmstart.py`

**Interfaces:**
- Consumes: `DrivingNet` (Task 2); `CameraSteeringNet` (existente).
- Produces: `load_camera_backbone(model, ckpt_path, device="cpu") -> int` — copia as chaves `cnn.*` do checkpoint (dict com `model_state_dict`) para `model`; retorna quantos tensores foram copiados.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_warmstart.py`:

```python
"""Warm-start: copiar o backbone de câmera do checkpoint da Fase 2 para o DrivingNet."""
import torch

from ai.model import CameraSteeringNet, DrivingNet
from ai.warmstart import load_camera_backbone


def _save_camera_ckpt(path):
    cam = CameraSteeringNet(); cam(torch.zeros(1, 3, 66, 200))
    torch.save({"model_state_dict": cam.state_dict(), "arch": "CameraSteeringNet"}, path)
    return cam


def test_load_camera_backbone_copies_conv_weights(tmp_path):
    ckpt = tmp_path / "cam.pt"
    cam = _save_camera_ckpt(ckpt)
    dual = DrivingNet(); dual(torch.zeros(1, 3, 66, 200), torch.zeros(1, 72))
    n = load_camera_backbone(dual, str(ckpt), device="cpu")
    assert n > 0
    for k, v in cam.state_dict().items():
        if k.startswith("cnn."):
            assert torch.allclose(dual.state_dict()[k], v)


def test_load_camera_backbone_leaves_lidar_untouched(tmp_path):
    ckpt = tmp_path / "cam.pt"
    _save_camera_ckpt(ckpt)
    dual = DrivingNet(); dual(torch.zeros(1, 3, 66, 200), torch.zeros(1, 72))
    before = {k: v.clone() for k, v in dual.state_dict().items() if k.startswith("lidar.")}
    load_camera_backbone(dual, str(ckpt), device="cpu")
    for k, v in before.items():
        assert torch.allclose(dual.state_dict()[k], v)  # braço de LiDAR intacto
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m pytest tests/test_warmstart.py -q`
Expected: FAIL (`ModuleNotFoundError: No module named 'ai.warmstart'`).

- [ ] **Step 3: Implementar**

Criar `src/ai/warmstart.py`:

```python
"""Warm-start the DrivingNet camera arm from a CameraSteeringNet checkpoint (Fase 3).

The two nets share the exact same ``cnn.*`` submodule, so transferring the proven
PilotNet features is a filtered state_dict copy. The LiDAR arm and head keep their
fresh initialization.
"""
import torch


def load_camera_backbone(model, ckpt_path, device="cpu"):
    """Copy ``cnn.*`` tensors from a camera checkpoint into ``model`` in place.

    Args:
        model: an initialized ``DrivingNet`` (run one forward first so lazy layers exist).
        ckpt_path: path to a checkpoint dict containing ``model_state_dict``.
        device: map_location for loading.

    Returns:
        Number of tensors copied.
    """
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    src = state["model_state_dict"]
    cnn = {k: v for k, v in src.items() if k.startswith("cnn.")}
    missing, unexpected = model.load_state_dict(cnn, strict=False)
    # strict=False: 'missing' lists the lidar/fc/head keys we intentionally skip.
    return len(cnn)
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python -m pytest tests/test_warmstart.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ai/warmstart.py tests/test_warmstart.py
git commit -m "feat(ai): load_camera_backbone — warm-start do braco de camera (cnn.*) no DrivingNet"
```

---

## Task 6: Treino dual (`train_dual` + CLI)

**Files:**
- Modify: `src/ai/train.py`
- Test: `tests/test_train_dual.py`

**Interfaces:**
- Consumes: `DrivingDataset` (Task 4), `sample_weights_dual` (Task 3), `DrivingNet` (Task 2), `load_camera_backbone` (Task 5), `build_index`/`list_episodes`/`split_episodes`/`mae` (existentes).
- Produces: `train_dual(data_dir, out_path, init_from=None, epochs=40, batch=128, lr=1e-4, weight_decay=1e-5, dropout=0.3, val_frac=0.2, seed=0, workers=0, limit=0, patience=6, w_steer=1.0, w_throttle=0.5, w_brake=1.0, device=None)` — treina e salva checkpoint `{"model_state_dict","val_loss","mae_steer","mae_throttle","mae_brake","var_ratio","epoch","arch":"DrivingNet"}`.

- [ ] **Step 1: Escrever o teste que falha (smoke, sem CARLA)**

Criar `tests/test_train_dual.py`:

```python
"""Smoke test do treino dual: 2 episódios sintéticos, 1 época, CPU -> gera checkpoint."""
import numpy as np
import torch

from ai.dataset_writer import EpisodeWriter
from ai.train import train_dual


def _episode(path, n):
    w = EpisodeWriter(str(path))
    for i in range(n):
        s = 0.4 if i % 2 else -0.4
        br = 0.8 if i % 3 == 0 else 0.0
        w.add(np.zeros((360, 640, 3), dtype=np.uint8),
              np.full(72, float(i % 12), dtype=np.float32),
              {"steer": s, "throttle": 0.5, "brake": br, "v": 1.0,
               "x": 0, "y": 0, "yaw": 0, "noise_active": False})
    w.close()


def test_train_dual_writes_checkpoint(tmp_path):
    _episode(tmp_path / "ep_0001", 8)
    _episode(tmp_path / "ep_0002", 8)
    out = tmp_path / "driving_smoke.pt"
    train_dual(str(tmp_path), str(out), epochs=1, batch=4, workers=0, device="cpu")
    assert out.exists()
    state = torch.load(str(out), map_location="cpu", weights_only=False)
    assert state["arch"] == "DrivingNet"
    assert {"mae_steer", "mae_throttle", "mae_brake"} <= set(state)
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m pytest tests/test_train_dual.py -q`
Expected: FAIL (`ImportError: cannot import name 'train_dual'`).

- [ ] **Step 3: Implementar**

Adicionar em `src/ai/train.py` — novos imports no topo (junto aos existentes) e a função. Imports a adicionar:

```python
import torch.nn.functional as F

from ai.dataset import DrivingDataset
from ai.dataset_index import sample_weights_dual
from ai.model import DrivingNet
from ai.warmstart import load_camera_backbone
```

Função (adicionar após `train`):

```python
def _evaluate_dual(model, loader, device):
    """Return per-axis MAE (steer, throttle, brake) and steer var_ratio."""
    model.eval()
    preds, tgts = [], []
    with torch.no_grad():
        for img, lidar, y in loader:
            out = model(img.to(device), lidar.to(device))
            preds.append(out.cpu().numpy())
            tgts.append(y.numpy())
    p, t = np.concatenate(preds), np.concatenate(tgts)
    maes = [mae(p[:, i], t[:, i]) for i in range(3)]
    return maes[0], maes[1], maes[2], variance_ratio(p[:, 0], t[:, 0])


def train_dual(data_dir, out_path, init_from=None, epochs=40, batch=128, lr=1e-4,
               weight_decay=1e-5, dropout=0.3, val_frac=0.2, seed=0, workers=0,
               limit=0, patience=6, w_steer=1.0, w_throttle=0.5, w_brake=1.0, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    episodes = list_episodes(data_dir)
    if not episodes:
        raise SystemExit("No ep_* episodes found in %s" % data_dir)
    train_eps, val_eps = split_episodes(episodes, val_frac, seed)
    train_index = build_index(train_eps)
    val_index = build_index(val_eps)
    if limit:
        train_index = train_index[:limit]
        val_index = val_index[:max(1, limit // 5)]
    print("device=%s | episodes: %d train / %d val | frames: %d train / %d val"
          % (device, len(train_eps), len(val_eps), len(train_index), len(val_index)))

    w = sample_weights_dual([r["steer"] for r in train_index],
                            [r["brake"] for r in train_index])
    sampler = WeightedRandomSampler(torch.as_tensor(w, dtype=torch.double),
                                    num_samples=len(w), replacement=True)
    train_loader = DataLoader(DrivingDataset(train_index), batch_size=batch, sampler=sampler,
                              num_workers=workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(DrivingDataset(val_index), batch_size=batch, shuffle=False,
                            num_workers=workers, pin_memory=True)

    model = DrivingNet(dropout=dropout).to(device)
    model(torch.zeros(1, 3, 66, 200, device=device), torch.zeros(1, 72, device=device))  # init lazy
    if init_from:
        n = load_camera_backbone(model, init_from, device=device)
        print("warm-start: copied %d camera tensors from %s" % (n, init_from))

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    axis_w = torch.tensor([w_steer, w_throttle, w_brake], device=device)

    best, best_epoch, since_best = float("inf"), 0, 0
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    for epoch in range(1, epochs + 1):
        model.train()
        t0, running, nb = time.time(), 0.0, 0
        for img, lidar, y in train_loader:
            img, lidar, y = img.to(device), lidar.to(device), y.to(device)
            opt.zero_grad()
            per = F.smooth_l1_loss(model(img, lidar), y, reduction="none")  # (B,3)
            loss = (per * axis_w).mean()
            loss.backward()
            opt.step()
            running += loss.item(); nb += 1
        sched.step()
        m_s, m_t, m_b, vr = _evaluate_dual(model, val_loader, device)
        val_loss = w_steer * m_s + w_throttle * m_t + w_brake * m_b
        flag = ""
        if val_loss < best:
            best, best_epoch, since_best = val_loss, epoch, 0
            torch.save({"model_state_dict": model.state_dict(), "val_loss": val_loss,
                        "mae_steer": m_s, "mae_throttle": m_t, "mae_brake": m_b,
                        "var_ratio": vr, "epoch": epoch, "arch": "DrivingNet"}, out_path)
            flag = " *"
        else:
            since_best += 1
        print("epoch %2d/%d  train=%.4f  MAE s/t/b=%.4f/%.4f/%.4f  var=%.2f  (%.1fs)%s"
              % (epoch, epochs, running / max(1, nb), m_s, m_t, m_b, vr, time.time() - t0, flag))
        if since_best >= patience:
            print("early stop (no val improvement for %d epochs)" % patience); break

    print("best val_loss=%.4f at epoch %d  ->  %s" % (best, best_epoch, out_path))
```

Estender o CLI em `main()` — adicionar flags e despachar para `train_dual` quando `--dual`:

```python
    p.add_argument("--dual", action="store_true", help="Train the dual-input DrivingNet (Fase 3)")
    p.add_argument("--init-from", default=None, help="Camera checkpoint to warm-start from (cam_v2.pt)")
    p.add_argument("--w-steer", type=float, default=1.0)
    p.add_argument("--w-throttle", type=float, default=0.5)
    p.add_argument("--w-brake", type=float, default=1.0)
```

E, no fim de `main()`, antes da chamada existente a `train(...)`, ramificar:

```python
    if a.dual:
        train_dual(a.data, a.out, init_from=a.init_from, epochs=a.epochs, batch=a.batch,
                   lr=a.lr, weight_decay=a.weight_decay, dropout=a.dropout, val_frac=a.val_frac,
                   seed=a.seed, workers=a.workers, limit=a.limit, patience=a.patience,
                   w_steer=a.w_steer, w_throttle=a.w_throttle, w_brake=a.w_brake, device=a.device)
        return
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python -m pytest tests/test_train_dual.py -q`
Expected: PASS (gera o checkpoint smoke).

- [ ] **Step 5: Commit**

```bash
git add src/ai/train.py tests/test_train_dual.py
git commit -m "feat(ai): train_dual — treino dual-input com warm-start, loss multi-eixo, sampler curva+freio"
```

---

## Task 7: `DrivingPolicy` (inferência)

**Files:**
- Modify: `src/ai/model_policy.py`
- Test: `tests/test_model_policy.py`

**Interfaces:**
- Consumes: `DrivingNet` (Task 2), `normalize_sectors_m` (Task 1), `points_to_sectors_m` (existente em `ai.sim_lidar`), `preprocess` (existente).
- Produces: `DrivingPolicy(ckpt_path, device=None, throttle_floor=0.0, brake_deadzone=0.1, n_sectors=72, max_range=12.0, ablate_lidar=False)`; `__call__(obs) -> (steer, throttle, brake)`. `obs` tem `image` (BGR HxWx3 uint8 ou None) e `lidar` (dict com `points` (N,3) ou None). Com `image=None` retorna `(0.0, 0.0, 0.0)`. Com `ablate_lidar=True` alimenta o LiDAR **livre (ones = pista limpa)** (para a ablation; na convenção near=0/free=1, zeros seria "obstáculo em tudo"). **Deadzone de freio:** se `brake < brake_deadzone` zera o freio (o brake residual do modelo segurava o carro parado); se `brake ≥ deadzone`, throttle vai a 0 (mutuamente exclusivos).

- [ ] **Step 1: Escrever os testes que falham**

Adicionar ao fim de `tests/test_model_policy.py`:

```python
import numpy as np

from ai.model import DrivingNet
from ai.model_policy import DrivingPolicy


def _save_driving_ckpt(path):
    import torch
    net = DrivingNet(); net(torch.zeros(1, 3, 66, 200), torch.zeros(1, 72))
    torch.save({"model_state_dict": net.state_dict(), "arch": "DrivingNet"}, path)


def _fake_obs():
    return {"image": np.zeros((360, 640, 3), dtype=np.uint8),
            "lidar": {"points": np.array([[5.0, 0.0, 0.0], [3.0, 1.0, 0.0]], dtype=np.float32)}}


def test_driving_policy_returns_three_controls_in_range(tmp_path):
    ckpt = tmp_path / "driving.pt"; _save_driving_ckpt(ckpt)
    pol = DrivingPolicy(str(ckpt), device="cpu")
    steer, throttle, brake = pol(_fake_obs())
    assert -1.0 <= steer <= 1.0
    assert 0.0 <= throttle <= 1.0
    assert 0.0 <= brake <= 1.0


def test_driving_policy_no_image_is_safe(tmp_path):
    ckpt = tmp_path / "driving.pt"; _save_driving_ckpt(ckpt)
    pol = DrivingPolicy(str(ckpt), device="cpu")
    assert pol({"image": None, "lidar": None}) == (0.0, 0.0, 0.0)


def test_driving_policy_throttle_floor(tmp_path):
    ckpt = tmp_path / "driving.pt"; _save_driving_ckpt(ckpt)
    pol = DrivingPolicy(str(ckpt), device="cpu", throttle_floor=0.4)
    steer, throttle, brake = pol(_fake_obs())
    if brake < 0.05:
        assert throttle >= 0.4  # piso aplicado quando não está freando
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m pytest tests/test_model_policy.py -q`
Expected: FAIL (`ImportError: cannot import name 'DrivingPolicy'`).

- [ ] **Step 3: Implementar**

Adicionar ao fim de `src/ai/model_policy.py` (novos imports no topo do arquivo):

```python
from ai.model import DrivingNet
from ai.sim_lidar import points_to_sectors_m
from ai.shared.lidar_pipeline import normalize_sectors_m


class DrivingPolicy:
    """Dual-input trained model as a closed-loop policy (Fase 3).

    Consumes camera + LiDAR from ``obs`` and outputs (steer, throttle, brake).
    LiDAR points are converted with the SAME sim adapter used at collection time,
    then normalized with the SAME function used in training.
    """

    def __init__(self, ckpt_path, device=None, throttle_floor=0.0, brake_deadzone=0.1,
                 n_sectors=72, max_range=12.0, ablate_lidar=False):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.throttle_floor = throttle_floor
        self.brake_deadzone = brake_deadzone
        self.n_sectors = n_sectors
        self.max_range = max_range
        self.ablate_lidar = ablate_lidar
        self.model = DrivingNet().to(self.device)
        self.model(torch.zeros(1, 3, 66, 200, device=self.device),
                   torch.zeros(1, n_sectors, device=self.device))  # init lazy
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model_state_dict"])
        self.model.eval()

    def _lidar_vector(self, obs):
        data = obs.get("lidar") if obs else None
        if self.ablate_lidar or not data or data.get("points") is None:
            # near=0/free=1: ones = "pista limpa" é o sinal neutro sem-LiDAR
            # (ablation + fallback sem nuvem); zeros significaria "obstáculo em tudo".
            return np.ones(self.n_sectors, dtype=np.float32)
        sectors_m = points_to_sectors_m(data["points"], n_sectors=self.n_sectors,
                                        max_range=self.max_range)
        return normalize_sectors_m(sectors_m, self.max_range)

    def __call__(self, obs):
        img = obs.get("image") if obs else None
        if img is None:
            return 0.0, 0.0, 0.0
        x = preprocess(img)
        lidar = self._lidar_vector(obs)
        with torch.no_grad():
            xt = torch.from_numpy(x).unsqueeze(0).to(self.device)
            lt = torch.from_numpy(lidar).unsqueeze(0).to(self.device)
            out = self.model(xt, lt).cpu().numpy().ravel()
        steer, throttle, brake = float(out[0]), float(out[1]), float(out[2])
        # brake e throttle mutuamente exclusivos; brake residual (~0.02) junto com
        # throttle segura o carro parado. Abaixo do deadzone zera o freio (e aplica
        # o piso de throttle); no/acima do deadzone freia de verdade e corta o throttle.
        if brake < self.brake_deadzone:
            brake = 0.0
            throttle = max(throttle, self.throttle_floor)
        else:
            throttle = 0.0
        return steer, throttle, brake
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python -m pytest tests/test_model_policy.py -q`
Expected: PASS (Fase 2 + novos).

- [ ] **Step 5: Commit**

```bash
git add src/ai/model_policy.py tests/test_model_policy.py
git commit -m "feat(ai): DrivingPolicy — inferencia dual-input (camera+LiDAR) -> steer/throttle/brake"
```

---

## Task 8: Avaliação open-loop dual (MAE por eixo)

**Files:**
- Modify: `src/ai/eval_openloop.py`
- Test: `tests/test_eval_openloop_dual.py`

**Interfaces:**
- Consumes: `DrivingDataset` (Task 4), `DrivingNet` (Task 2), `mae`/`variance_ratio`/`build_index`/`list_episodes`/`split_episodes`.
- Produces: `evaluate_dual(ckpt, data_dir, split="val", val_frac=0.2, seed=0, batch=128, workers=0, device=None) -> dict` com chaves `mae_steer`, `mae_throttle`, `mae_brake`, `var_ratio`, `frames`. O `main()` chama `evaluate_dual` quando o checkpoint tem `arch == "DrivingNet"`.

- [ ] **Step 1: Escrever o teste que falha**

Criar `tests/test_eval_openloop_dual.py`:

```python
"""Open-loop dual: roda o DrivingNet sobre um dataset sintético e devolve MAE por eixo."""
import numpy as np
import torch

from ai.dataset_writer import EpisodeWriter
from ai.model import DrivingNet
from ai.eval_openloop import evaluate_dual


def _episode(path, n):
    w = EpisodeWriter(str(path))
    for i in range(n):
        w.add(np.zeros((360, 640, 3), dtype=np.uint8),
              np.full(72, float(i % 12), dtype=np.float32),
              {"steer": 0.1, "throttle": 0.5, "brake": 0.0, "v": 1.0,
               "x": 0, "y": 0, "yaw": 0, "noise_active": False})
    w.close()


def test_evaluate_dual_returns_axis_maes(tmp_path):
    _episode(tmp_path / "ep_0001", 6)
    _episode(tmp_path / "ep_0002", 6)
    net = DrivingNet(); net(torch.zeros(1, 3, 66, 200), torch.zeros(1, 72))
    ckpt = tmp_path / "driving.pt"
    torch.save({"model_state_dict": net.state_dict(), "arch": "DrivingNet"}, ckpt)
    out = evaluate_dual(str(ckpt), str(tmp_path), split="all", workers=0, device="cpu")
    assert {"mae_steer", "mae_throttle", "mae_brake", "var_ratio", "frames"} <= set(out)
    assert out["frames"] == 12
```

- [ ] **Step 2: Rodar e ver falhar**

Run: `python -m pytest tests/test_eval_openloop_dual.py -q`
Expected: FAIL (`ImportError: cannot import name 'evaluate_dual'`).

- [ ] **Step 3: Implementar**

Adicionar imports no topo de `src/ai/eval_openloop.py`:

```python
from ai.dataset import DrivingDataset
from ai.model import DrivingNet
```

Adicionar a função (após `evaluate`):

```python
def evaluate_dual(ckpt, data_dir, split="val", val_frac=0.2, seed=0, batch=128,
                  workers=4, device=None):
    """Per-axis open-loop metrics for a DrivingNet checkpoint."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    episodes = list_episodes(data_dir)
    train_eps, val_eps = split_episodes(episodes, val_frac, seed)
    chosen = {"val": val_eps, "train": train_eps}.get(split, episodes)
    index = build_index(chosen)
    loader = DataLoader(DrivingDataset(index), batch_size=batch, shuffle=False, num_workers=workers)

    model = DrivingNet().to(device)
    model(torch.zeros(1, 3, 66, 200, device=device), torch.zeros(1, 72, device=device))
    state = torch.load(ckpt, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    preds, tgts = [], []
    with torch.no_grad():
        for img, lidar, y in loader:
            preds.append(model(img.to(device), lidar.to(device)).cpu().numpy())
            tgts.append(y.numpy())
    p, t = np.concatenate(preds), np.concatenate(tgts)
    out = {"frames": len(t),
           "mae_steer": mae(p[:, 0], t[:, 0]),
           "mae_throttle": mae(p[:, 1], t[:, 1]),
           "mae_brake": mae(p[:, 2], t[:, 2]),
           "var_ratio": variance_ratio(p[:, 0], t[:, 0])}
    print("split=%s  frames=%d" % (split, out["frames"]))
    print("  MAE steer=%.4f  throttle=%.4f  brake=%.4f  |  var_ratio=%.2f"
          % (out["mae_steer"], out["mae_throttle"], out["mae_brake"], out["var_ratio"]))
    return out
```

Fazer o `main()` despachar por arch — logo após ler `a = p.parse_args()`:

```python
    _st = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    if _st.get("arch") == "DrivingNet":
        evaluate_dual(a.ckpt, a.data, a.split, a.val_frac, a.seed, a.batch, a.workers, a.device)
        return
```

- [ ] **Step 4: Rodar e ver passar**

Run: `python -m pytest tests/test_eval_openloop_dual.py -q`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add src/ai/eval_openloop.py tests/test_eval_openloop_dual.py
git commit -m "feat(ai): evaluate_dual — MAE por eixo (steer/throttle/brake) + auto-deteccao de arch"
```

---

## Task 9: Coleta com tráfego (`collect.py --traffic`)

**Files:**
- Modify: `src/ai/collect.py`

**Interfaces:**
- Consumes: `spawn_random_vehicles` (existente em `core.carlaClient.world_manager`).
- Produces: parâmetro `traffic=0` em `collect(...)` e flag CLI `--traffic N`; registra `"traffic": N` no `meta.json`.

> **Nota:** tarefa de integração (exige servidor CARLA). Sem teste pytest; verificação é uma coleta curta real + contagem de frames com `brake>0`.

- [ ] **Step 1: Adicionar o import**

Em `src/ai/collect.py`, no bloco que importa de `core.carlaClient.world_manager`, incluir `spawn_random_vehicles`:

```python
from core.carlaClient.world_manager import (
    connect_to_carla,
    simulation_context,
    spawn_actor_vehicle,
    spawn_random_vehicles,
    setup_spectator_follow_vehicle,
    update_spectator_position,
)
```

- [ ] **Step 2: Assinatura + meta**

Adicionar `traffic=0` à assinatura de `collect(...)` (por ex. após `quality="Low"`). No dict de `write_meta`, adicionar a chave:

```python
        "traffic": traffic,
```

- [ ] **Step 3: Spawnar o tráfego após o ego assentar**

Em `collect(...)`, logo após o laço `for _ in range(10): world.tick()` (que deixa o spawn assentar) e ANTES do `setup_spectator_follow_vehicle`, inserir:

```python
            if traffic:
                spawned = spawn_random_vehicles(world, actor_list, traffic, traffic_manager)
                for _ in range(10):
                    world.tick()  # deixa o tráfego assentar e o autopilot assumir
                logger.info("Traffic: %d vehicles spawned", len(spawned))
```

- [ ] **Step 4: Flag CLI + repasse**

Adicionar em `main()`:

```python
    parser.add_argument("--traffic", type=int, default=0,
                        help="Spawn N autopilot vehicles to create braking events")
```

E passar `traffic=args.traffic` na chamada a `collect(...)`.

- [ ] **Step 5: Verificação manual (CARLA) + commit**

Coleta curta com tráfego (2 episódios, 60 s):

```bash
python src/ai/collect.py --out D:/tcc_data/dataset_v3 --episodes 2 --seconds 60 --slow 20 --traffic 30
```

Confirmar que há frenagem no dataset (fração de frames com `brake>0` deve ser claramente > 0):

```bash
python -c "import glob,csv;
rows=[r for f in glob.glob('D:/tcc_data/dataset_v3/ep_*/labels.csv') for r in csv.DictReader(open(f))];
b=sum(float(r['brake'])>0.05 for r in rows);
print('frames=%d  brake>0=%d  (%.1f%%)'%(len(rows),b,100*b/max(1,len(rows))))"
```
Expected: `brake>0` com percentual > 0 (idealmente ≥ 5%). Se ~0%, aumentar `--traffic` e/ou `--slow`.

```bash
git add src/ai/collect.py
git commit -m "feat(ai): collect --traffic N — spawna veiculos p/ gerar eventos de frenagem (dataset_v3)"
```

---

## Task 10: Closed-loop com tráfego + ablation

**Files:**
- Modify: `src/ai/eval_closedloop.py`

**Interfaces:**
- Consumes: `spawn_random_vehicles` (existente), `DrivingPolicy` (Task 7).
- Produces: flags CLI `--traffic N` e `--ablate-lidar`; quando o checkpoint tem `arch == "DrivingNet"`, o driver passa a ser `DrivingPolicy` (senão segue `ModelSteeringPolicy`). Assinatura de `run_closed_loop` ganha `traffic=0, ablate_lidar=False`.

> **Nota:** tarefa de integração (exige CARLA). Sem teste pytest; verificação = rodar com tráfego e comparar com/sem LiDAR (ablation).

- [ ] **Step 1: Import + assinatura**

Em `src/ai/eval_closedloop.py`, adicionar `spawn_random_vehicles` ao import de `core.carlaClient.world_manager`. Adicionar `traffic=0, ablate_lidar=False` à assinatura de `run_closed_loop(...)`.

- [ ] **Step 2: Selecionar o driver por arch**

No bloco `if not autopilot:` que hoje cria `ModelSteeringPolicy` quando `model_ckpt`, trocar por seleção de arch:

```python
                if model_ckpt:
                    import torch
                    arch = torch.load(model_ckpt, map_location="cpu", weights_only=False).get("arch")
                    if arch == "DrivingNet":
                        from ai.model_policy import DrivingPolicy
                        policy = DrivingPolicy(model_ckpt, ablate_lidar=ablate_lidar)
                        logger.info("Driving with DrivingNet%s: %s",
                                    " (LiDAR ABLATED)" if ablate_lidar else "", model_ckpt)
                    else:
                        from ai.model_policy import ModelSteeringPolicy
                        policy = ModelSteeringPolicy(model_ckpt, throttle=model_throttle)
                        logger.info("Driving with camera model: %s", model_ckpt)
                else:
                    policy = ExpertPolicy(world, vehicle, target_speed_kmh=target_speed_kmh, seed=seed)
```

- [ ] **Step 3: Spawnar tráfego**

Logo após o laço `for _ in range(10): world.tick()` (assentar spawn) e antes de `_attach_collision_sensor`, inserir:

```python
            if traffic:
                tm_traffic = client.get_trafficmanager(tm_port)
                tm_traffic.set_synchronous_mode(True)
                spawn_random_vehicles(world, actor_list, traffic, tm_traffic)
                for _ in range(10):
                    world.tick()
                logger.info("Traffic: %d vehicles spawned", traffic)
```

- [ ] **Step 4: Flags CLI + repasse**

Adicionar em `main()`:

```python
    parser.add_argument("--traffic", type=int, default=0, help="Spawn N autopilot vehicles")
    parser.add_argument("--ablate-lidar", action="store_true",
                        help="Feed the LiDAR as free/clear (ones) — ablation: prove the model uses it")
```

E repassar `traffic=args.traffic, ablate_lidar=args.ablate_lidar` na chamada a `run_closed_loop(...)`.

- [ ] **Step 5: Verificação manual (CARLA) — critério de aprovação**

Rodar com tráfego, com e sem LiDAR, várias seeds. O modelo COM LiDAR deve ter `collisions=0` em ≥8/10; SEM LiDAR deve colidir mais.

```bash
# COM LiDAR
python src/ai/eval_closedloop.py --model D:/tcc_data/runs/driving_v1.pt --routes 10 --seconds 60 --traffic 30 --seed 1
# ABLATION (sem LiDAR) — mesmas condições
python src/ai/eval_closedloop.py --model D:/tcc_data/runs/driving_v1.pt --routes 10 --seconds 60 --traffic 30 --seed 1 --ablate-lidar
```
Expected: com LiDAR, SUMMARY reporta ≥8/10 rotas limpas (sem colisão); com `--ablate-lidar`, colisões aumentam claramente.

```bash
git add src/ai/eval_closedloop.py
git commit -m "feat(ai): eval_closedloop --traffic/--ablate-lidar + driver DrivingPolicy p/ arch DrivingNet"
```

---

## Task 11: Coleta, treino e avaliação de verdade + atualização do estado

**Files:**
- Modify: `docs/ESTADO_IA.md`

> Tarefa operacional (roda o pipeline completo em CARLA/GPU) e fecha a fase. Sem pytest.

- [ ] **Step 1: Coletar `dataset_v3` (com tráfego + recuperação lateral)**

```bash
python src/ai/collect.py --out D:/tcc_data/dataset_v3 --episodes 20 --seconds 120 --slow 20 --traffic 30 --recovery
```
Conferir o relatório final (histograma de steer + fração de `brake>0`).

- [ ] **Step 2: Treinar com warm-start**

```bash
python -u src/ai/train.py --dual --data D:/tcc_data/dataset_v3 --out D:/tcc_data/runs/driving_v1.pt --init-from D:/tcc_data/runs/cam_v2.pt --epochs 40
```
Expected: log de warm-start ("copied N camera tensors") + MAE por eixo caindo; salva `driving_v1.pt`.

- [ ] **Step 3: Avaliação open-loop**

```bash
python src/ai/eval_openloop.py --ckpt D:/tcc_data/runs/driving_v1.pt --data D:/tcc_data/dataset_v3 --split val
```
Expected (gate): `MAE steer` no patamar da Fase 2; `var_ratio ≥ 0.6`.

- [ ] **Step 4: Avaliação closed-loop + ablation (critério da fase)**

Rodar os dois comandos da Task 10, Step 5. Registrar a tabela com/sem LiDAR.
Expected: ≥8/10 rotas sem colisão COM LiDAR; pior SEM LiDAR.

- [ ] **Step 5: Atualizar `docs/ESTADO_IA.md` e commitar**

Marcar a Fase 3 como concluída, apontar `driving_v1.pt`, registrar os números (MAE por eixo, rotas limpas com/sem LiDAR) e listar as novas peças (`DrivingNet`, `DrivingDataset`, `DrivingPolicy`, `train_dual`, `collect --traffic`, `eval_closedloop --traffic/--ablate-lidar`).

```bash
git add docs/ESTADO_IA.md
git commit -m "docs(ai): Fase 3 concluida — dual-input dirige e freia (com resultados + ablation)"
```

---

## Self-Review (feito na escrita do plano)

- **Cobertura do spec:** modelo fundido (T2), warm-start (T5), Conv1D circular (T2), LiDAR normalizado fonte única (T1), coleta com tráfego/só-veículos (T9), dataset dual (T4), sampler curva+freio (T3), treino multi-eixo (T6), política dual (T7), open-loop por eixo (T8), closed-loop com tráfego + ablation (T10), operação/estado (T11). ✔ Todos os itens do spec têm tarefa.
- **Sem placeholders:** todo passo de código traz o código completo; passos de CARLA trazem comando exato + resultado esperado.
- **Consistência de tipos:** `DrivingNet.forward(img, lidar)->(B,3)`, `DrivingDataset`→`(img,lidar,target)`, `sample_weights_dual(steers,brakes)`, `load_camera_backbone(model,ckpt,device)`, `DrivingPolicy(...).__call__(obs)->(steer,throttle,brake)`, `evaluate_dual(...)->dict` — usados de forma idêntica em T6/T8/T10.
