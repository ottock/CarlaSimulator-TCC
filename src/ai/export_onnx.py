"""Exporta o DrivingNet treinado para ONNX (Fase 6a) e prova a paridade.

O Jetson Nano nao roda PyTorch: o caminho e' checkpoint -> ONNX -> TensorRT. Este
script faz a primeira metade e, no mesmo passo, MEDE se o ONNX responde o mesmo
que o checkpoint. Sem essa prova o carro poderia dirigir com uma rede diferente da
que passou no `eval_track`.

Opset 11: validado contra o TensorRT 8.x do JetPack 4.6 (que cobre ate 13-14). O
padding circular do braco de LiDAR nao vira um `Pad(wrap)` exotico -- o exportador
o decompoe em `Slice`+`Concat`, operadores nativos do TRT.

O `fov_deg` do treino vai gravado nos `metadata_props` do .onnx para o runtime da
Fase 6b mascarar o scan real com o MESMO campo de visao, sem depender de repetir
uma flag na linha de comando.

Uso (da raiz do repo):
    # paridade medida no split de VALIDACAO (o jeito certo)
    python src/ai/export_onnx.py --model D:/tcc_data/runs/driving_track_180.pt \
        --out D:/tcc_data/runs/driving_track_180.onnx --data D:/tcc_data/dataset_track_v1

    # sem dataset a mao: paridade em entradas sinteticas (fraca, so um smoke)
    python src/ai/export_onnx.py --model ... --out ...
"""
import os
import sys

_SRC_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

import argparse

import numpy as np
import onnx
import onnxruntime as ort
import torch

from ai.model import DrivingNet

N_SECTORS = 72
MAX_RANGE_M = 12.0
DEFAULT_OPSET = 11
PARITY_TOL = 1e-4


def _load_model(ckpt_path, n_sectors=N_SECTORS, device="cpu"):
    """Return ``(model in eval mode, checkpoint dict)``."""
    model = DrivingNet().to(device)
    model(torch.zeros(1, 3, 66, 200, device=device),
          torch.zeros(1, n_sectors, device=device))  # inicializa a LazyLinear
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, state


def export_onnx(ckpt_path, out_path, opset=DEFAULT_OPSET, n_sectors=N_SECTORS,
                max_range=MAX_RANGE_M):
    """Export ``ckpt_path`` to ``out_path`` and return the metadata written.

    Batch is a dynamic axis, so the same file serves batched validation and the
    single-frame loop on the car.
    """
    model, state = _load_model(ckpt_path, n_sectors=n_sectors)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    torch.onnx.export(
        model,
        (torch.zeros(1, 3, 66, 200), torch.zeros(1, n_sectors)),
        out_path,
        opset_version=opset,
        input_names=["image", "lidar"],
        output_names=["control"],
        dynamic_axes={"image": {0: "batch"}, "lidar": {0: "batch"},
                      "control": {0: "batch"}},
    )

    # fov_deg ausente = checkpoint anterior a Fase 6a = visao completa.
    fov = state.get("fov_deg")
    meta = {
        "arch": state.get("arch", "DrivingNet"),
        "fov_deg": str(float(fov if fov is not None else 360.0)),
        "n_sectors": str(int(n_sectors)),
        "max_range_m": str(float(max_range)),
        "opset": str(int(opset)),
        "source_checkpoint": os.path.basename(ckpt_path),
    }
    model_onnx = onnx.load(out_path)
    for k, v in meta.items():
        entry = model_onnx.metadata_props.add()
        entry.key, entry.value = k, v
    onnx.checker.check_model(model_onnx)
    onnx.save(model_onnx, out_path)
    return meta


def synthetic_samples(n, n_sectors=N_SECTORS, seed=0):
    """Random inputs in the model's domain: image in [-1, 1], LiDAR in [0, 1].

    A weak stand-in for real frames -- good enough to catch a broken graph, not
    to certify the export. Prefer :func:`validation_samples`.
    """
    rng = np.random.default_rng(seed)
    imgs = rng.uniform(-1.0, 1.0, (n, 3, 66, 200)).astype(np.float32)
    lidars = rng.uniform(0.0, 1.0, (n, n_sectors)).astype(np.float32)
    return imgs, lidars


def validation_samples(data_dir, n=64, fov_deg=None, max_range=MAX_RANGE_M,
                       val_frac=0.2, seed=0):
    """Real frames from the VALIDATION split, preprocessed exactly as in training.

    Uses the same ``val_frac``/``seed`` split as ``train.py``, so these are frames
    the model was never fitted on, and applies the checkpoint's ``fov_deg`` -- the
    parity number then describes the input the car will actually feed the net.
    """
    from ai.dataset import DrivingDataset
    from ai.dataset_index import build_index, list_episodes, split_episodes

    episodes = list_episodes(data_dir)
    if not episodes:
        raise SystemExit("No ep_* episodes found in %s" % data_dir)
    _, val_eps = split_episodes(episodes, val_frac, seed)
    index = build_index(val_eps)
    if not index:
        raise SystemExit("Validation split is empty in %s" % data_dir)

    ds = DrivingDataset(index, max_range=max_range, fov_deg=fov_deg)
    # Espalhado pelo split inteiro em vez dos n primeiros frames (que seriam um
    # unico trecho de um unico episodio).
    step = max(1, len(ds) // n)
    rows = [ds[i] for i in range(0, len(ds), step)][:n]
    imgs = np.stack([r[0].numpy() for r in rows]).astype(np.float32)
    lidars = np.stack([r[1].numpy() for r in rows]).astype(np.float32)
    return imgs, lidars


def check_parity(ckpt_path, onnx_path, samples, n_sectors=N_SECTORS):
    """Max absolute difference between the PyTorch and the ONNX outputs."""
    imgs, lidars = samples
    model, _ = _load_model(ckpt_path, n_sectors=n_sectors)
    with torch.no_grad():
        ref = model(torch.from_numpy(imgs), torch.from_numpy(lidars)).numpy()
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    got = sess.run(["control"], {"image": imgs, "lidar": lidars})[0]
    return float(np.max(np.abs(ref - got)))


def main():
    p = argparse.ArgumentParser(description="Exporta o DrivingNet para ONNX (Fase 6a)")
    p.add_argument("--model", required=True, help="Checkpoint .pt do DrivingNet")
    p.add_argument("--out", default=None, help="Arquivo .onnx (padrao: ao lado do .pt)")
    p.add_argument("--opset", type=int, default=DEFAULT_OPSET)
    p.add_argument("--data", default=None,
                   help="Dataset para medir a paridade no split de validacao (recomendado)")
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--val-frac", type=float, default=0.2, help="Igual ao usado no treino")
    p.add_argument("--seed", type=int, default=0, help="Igual ao usado no treino")
    p.add_argument("--tol", type=float, default=PARITY_TOL)
    a = p.parse_args()

    out_path = a.out or (os.path.splitext(a.model)[0] + ".onnx")
    meta = export_onnx(a.model, out_path, opset=a.opset)
    size_mb = os.path.getsize(out_path) / (1024.0 * 1024.0)
    print("exportado: %s (%.1f MB, opset %s)" % (out_path, size_mb, meta["opset"]))
    print("metadados: fov_deg=%s  n_sectors=%s  max_range_m=%s"
          % (meta["fov_deg"], meta["n_sectors"], meta["max_range_m"]))
    print("onnx.checker: OK")

    fov = float(meta["fov_deg"])
    if a.data:
        samples = validation_samples(a.data, n=a.samples,
                                     fov_deg=None if fov >= 360.0 else fov,
                                     val_frac=a.val_frac, seed=a.seed)
        origem = "split de validacao de %s" % a.data
    else:
        samples = synthetic_samples(a.samples)
        origem = "entradas SINTETICAS (rode com --data para a medida que vale)"
    max_diff = check_parity(a.model, out_path, samples)
    print("paridade PyTorch<->ONNX em %d amostras (%s): max|diff| = %.3e"
          % (len(samples[0]), origem, max_diff))

    if max_diff > a.tol:
        raise SystemExit("FALHOU: paridade %.3e acima da tolerancia %.1e" % (max_diff, a.tol))
    print("paridade OK (< %.1e)" % a.tol)


if __name__ == "__main__":
    main()
