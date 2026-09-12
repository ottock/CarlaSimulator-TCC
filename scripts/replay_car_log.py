"""Re-executa uma corrida real no PC e atribui o esterco a camera ou ao LiDAR.

Roda sobre o diretorio gravado no Jetson com `--log-inputs`, que guarda a imagem
JA preprocessada de cada quadro -- exatamente o que a rede viu. Com isso da para
alimentar o modelo de novo e neutralizar um sensor de cada vez:

    completo      : imagem real  + LiDAR real   (tem de bater com o log)
    sem LiDAR     : imagem real  + vetor "tudo livre"
    sem camera    : imagem neutra + LiDAR real

O sensor cuja neutralizacao MAIS muda o esterco e o que estava mandando. Isso
substitui a inferencia por correlacao que usamos antes.

A fidelidade (diferenca entre o replay completo e o steer gravado) e reportada
primeiro de proposito: se ela for alta, o replay nao esta reproduzindo a corrida
e a atribuicao nao significa nada.

Uso:
    python scripts/replay_car_log.py runs/pista_XXXX --ckpt D:/tcc_data/runs/driving_track_180.pt
"""
import argparse
import io
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))


def attribution(full, sem_lidar, sem_camera, gravado=None, limiar=1.3):
    """Quantify how much each sensor moved the steering.

    ``limiar`` is how many times larger one delta must be to be called dominant;
    below it the model is using both and calling a winner would be noise.
    """
    full = np.asarray(full, dtype=np.float64)
    dl = float(np.abs(full - np.asarray(sem_lidar, dtype=np.float64)).mean())
    dc = float(np.abs(full - np.asarray(sem_camera, dtype=np.float64)).mean())
    if dl > dc * limiar:
        dom = "lidar"
    elif dc > dl * limiar:
        dom = "camera"
    else:
        dom = "equilibrado"
    out = {"delta_lidar": dl, "delta_camera": dc, "dominante": dom}
    if gravado is not None:
        out["fidelidade"] = float(
            np.abs(full - np.asarray(gravado, dtype=np.float64)).mean())
    return out


def _load(run_dir):
    with io.open(os.path.join(run_dir, "meta.json"), encoding="utf-8") as fh:
        meta = json.load(fh)
    imgs = np.load(os.path.join(run_dir, "model_input.npy"))
    sec = np.load(os.path.join(run_dir, "sectors.npy"))
    with io.open(os.path.join(run_dir, "frames.jsonl"), encoding="utf-8") as fh:
        frames = [json.loads(l) for l in fh if l.strip()]
    return meta, imgs, sec, frames


def main():
    p = argparse.ArgumentParser(description="Re-executa a corrida e atribui o esterco")
    p.add_argument("run_dir")
    p.add_argument("--ckpt", required=True, help="Checkpoint .pt do DrivingNet")
    a = p.parse_args()

    import torch
    from ai.model import DrivingNet

    meta, imgs, sec, frames = _load(a.run_dir)
    if imgs.shape[0] == 0:
        raise SystemExit("model_input.npy vazio -- rode o carro com --log-inputs")
    n = min(len(imgs), len(sec), len(frames))
    print("corrida: %s  (%d quadros)" % (a.run_dir, n))
    print("meta: fov=%s  span=%s  crop=%s" % (meta.get("fov_deg"),
                                              meta.get("steer_span_us"),
                                              meta.get("crop_frac")))

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = DrivingNet().to(dev)
    model(torch.zeros(1, 3, 66, 200, device=dev), torch.zeros(1, 72, device=dev))
    state = torch.load(a.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    # imagem -> [-1,1] CHW, desfazendo o uint8 do log
    X = (imgs[:n].astype(np.float32) / 127.5 - 1.0).transpose(0, 3, 1, 2)
    L = sec[:n].astype(np.float32)
    livre = np.ones_like(L)                       # LiDAR neutro = tudo livre
    cinza = np.zeros_like(X)                      # camera neutra = cinza medio
    rodou = np.array([abs(f["steer"]) > 0 or not f.get("blind", False) for f in frames[:n]])

    def infer(xs, ls):
        out = []
        with torch.no_grad():
            for i in range(0, len(xs), 256):
                xb = torch.from_numpy(xs[i:i + 256]).to(dev)
                lb = torch.from_numpy(ls[i:i + 256]).to(dev)
                out.append(model(xb, lb).cpu().numpy())
        return np.concatenate(out)[:, 0]

    full = infer(X, L)
    sem_l = infer(X, livre)
    sem_c = infer(cinza, L)
    gravado = np.array([f["steer"] for f in frames[:n]], dtype=np.float64)

    # so os quadros em que o modelo realmente rodou no carro
    v = rodou & (gravado != 0.0)
    if v.sum() < 10:
        print("AVISO: poucos quadros com inferencia real; usando todos")
        v = np.ones(n, dtype=bool)

    r = attribution(full[v], sem_l[v], sem_c[v], gravado=gravado[v])
    print("")
    print("--- FIDELIDADE DO REPLAY (menor = melhor) ---")
    print("  |replay - gravado| medio: %.4f" % r["fidelidade"])
    if r["fidelidade"] > 0.05:
        print("  ALTA: o replay nao esta reproduzindo a corrida. Pode ser diferenca")
        print("  entre a engine TensorRT e este checkpoint, ou checkpoint errado.")
        print("  A atribuicao abaixo NAO e confiavel enquanto isso nao fechar.")
    print("")
    print("--- ATRIBUICAO (%d quadros) ---" % int(v.sum()))
    print("  neutralizar o LiDAR  muda o esterco em %.4f" % r["delta_lidar"])
    print("  neutralizar a camera muda o esterco em %.4f" % r["delta_camera"])
    print("  -> sensor dominante: %s" % r["dominante"].upper())
    print("")
    print("  Leitura: se a camera domina e o carro dirige mal, o problema esta na")
    print("  imagem (luz/textura). Se o LiDAR domina, esta no vetor de setores.")


if __name__ == "__main__":
    main()
