"""
Treino YOLOv8-seg — Viaturas Blindadas do Exército Brasileiro
=============================================================
Detecta hardware automaticamente e ajusta todos os parâmetros
(batch size, workers, imgsz, device) para o melhor desempenho
disponível: CPU fraca, CPU forte, GPU com pouca ou muita VRAM.

INSTALAÇÃO:
    pip install ultralytics pycocotools Pillow tqdm pyyaml

USO:
    # Prepara dataset e treina (fluxo completo)
    python treino_yolov8seg.py --coco anotacoes_coco --imagens repositorio_viaturas

    # Só prepara o dataset (sem treinar)
    python treino_yolov8seg.py --coco anotacoes_coco --imagens repositorio_viaturas --so-preparar

    # Retoma treino interrompido
    python treino_yolov8seg.py --retomar runs/segment/viaturas_eb/weights/last.pt

    # Avalia modelo treinado
    python treino_yolov8seg.py --avaliar runs/segment/viaturas_eb/weights/best.pt

    # Força perfil específico (ignora detecção automática)
    python treino_yolov8seg.py --coco anotacoes_coco --imagens repositorio_viaturas --perfil gpu_alta
"""

import os, sys, json, shutil, argparse, math, time
from pathlib import Path

# ══════════════════════════════════════════════════════════════════
# PERFIS DE HARDWARE
# ══════════════════════════════════════════════════════════════════

PERFIS_HW = {
    "cpu_fraca": {
        "descricao":  "CPU sem GPU (PC doméstico)",
        "device":     "cpu",
        "batch":      4,
        "workers":    2,
        "imgsz":      512,
        "modelo":     "yolov8n-seg.pt",   # nano — mais rápido
        "epochs_rec": 50,
        "amp":        False,
    },
    "cpu_forte": {
        "descricao":  "CPU multi-core sem GPU (servidor/workstation)",
        "device":     "cpu",
        "batch":      8,
        "workers":    8,
        "imgsz":      640,
        "modelo":     "yolov8s-seg.pt",   # small
        "epochs_rec": 250,
        "amp":        False,
    },
    "gpu_baixa": {
        "descricao":  "GPU com < 6 GB VRAM",
        "device":     "0",
        "batch":      8,
        "workers":    4,
        "imgsz":      640,
        "modelo":     "yolov8s-seg.pt",
        "epochs_rec": 100,
        "amp":        True,
    },
    "gpu_media": {
        "descricao":  "GPU com 6–12 GB VRAM",
        "device":     "0",
        "batch":      16,
        "workers":    8,
        "imgsz":      640,
        "modelo":     "yolov8m-seg.pt",   # medium
        "epochs_rec": 150,
        "amp":        True,
    },
    "gpu_alta": {
        "descricao":  "GPU com 12+ GB VRAM",
        "device":     "0",
        "batch":      32,
        "workers":    8,
        "imgsz":      1024,
        "modelo":     "yolov8l-seg.pt",   # large
        "epochs_rec": 200,
        "amp":        True,
    },
}

CLASSES = [
    "VBTP_Guarani",
    "Cascavel_EE9",
    "VBC_Leopard",
    "Guepard_M113",
    "Lince_LMV",
]

SPLIT_TREINO = 0.80   # 80% treino, 20% validação


# ══════════════════════════════════════════════════════════════════
# DETECÇÃO DE HARDWARE
# ══════════════════════════════════════════════════════════════════

def detectar_hardware() -> dict:
    info = {
        "tem_gpu":  False,
        "nome_gpu": "Nenhuma",
        "vram_gb":  0.0,
        "n_cpu":    os.cpu_count() or 1,
        "device":   "cpu",
    }
    try:
        import torch
        if torch.cuda.is_available():
            idx  = torch.cuda.current_device()
            prop = torch.cuda.get_device_properties(idx)
            info.update({
                "tem_gpu":  True,
                "nome_gpu": prop.name,
                "vram_gb":  round(prop.total_memory / 1024**3, 1),
                "device":   "0",
            })
    except Exception:
        pass
    return info


def recomendar_perfil(hw: dict) -> str:
    if not hw["tem_gpu"]:
        return "cpu_forte" if hw["n_cpu"] >= 8 else "cpu_fraca"
    vram = hw["vram_gb"]
    if vram >= 12:
        return "gpu_alta"
    if vram >= 6:
        return "gpu_media"
    return "gpu_baixa"


def exibir_menu_perfil(hw: dict) -> str:
    recomendado = recomendar_perfil(hw)

    print("\n┌─────────────────────────────────────────────────────────────┐")
    print("│                   Configuração de Hardware                    │")
    print("├─────────────────────────────────────────────────────────────┤")
    if hw["tem_gpu"]:
        print(f"│  GPU   : {hw['nome_gpu'][:50]:<50} │")
        print(f"│  VRAM  : {hw['vram_gb']:.1f} GB{'':<48} │")
    else:
        print(f"│  GPU   : Nenhuma detectada{'':<36} │")
    print(f"│  CPUs  : {hw['n_cpu']} núcleos{'':<48} │")
    print("├─────────────────────────────────────────────────────────────┤")
    print("│  Perfil       Modelo          Batch  Resolução  Descrição    │")
    print("├─────────────────────────────────────────────────────────────┤")

    for i, (chave, p) in enumerate(PERFIS_HW.items(), 1):
        rec = " ←" if chave == recomendado else "  "
        print(f"│  [{i}] {chave:<12} {p['modelo']:<15} {p['batch']:<6} "
              f"{p['imgsz']}px{rec:<10} │")

    print("└─────────────────────────────────────────────────────────────┘")

    chaves = list(PERFIS_HW.keys())
    mapa = {str(i+1): k for i, k in enumerate(chaves)}
    mapa.update({k: k for k in chaves})

    while True:
        entrada = input(
            f"\n  Escolha o perfil [1-{len(chaves)}] (Enter = {recomendado}): "
        ).strip().lower()
        if entrada == "":
            print(f"  ✅ Perfil: {recomendado} — {PERFIS_HW[recomendado]['descricao']}")
            return recomendado
        if entrada in mapa:
            escolha = mapa[entrada]
            print(f"  ✅ Perfil: {escolha} — {PERFIS_HW[escolha]['descricao']}")
            return escolha
        print(f"  ⚠  Opção inválida. Digite um número de 1 a {len(chaves)}.")


# ══════════════════════════════════════════════════════════════════
# CONVERSÃO COCO → YOLO (segmentação)
# ══════════════════════════════════════════════════════════════════

def coco_para_yolo_seg(
    coco: dict,
    pasta_imgs_src: Path,
    pasta_imgs_dst: Path,
    pasta_labels_dst: Path,
    cat_para_idx: dict,
) -> int:
    """
    Converte anotações COCO para formato YOLO-seg.
    Cada linha do .txt: class_id x1 y1 x2 y2 ... (coords normalizadas 0-1)
    Retorna número de imagens processadas.
    """
    pasta_imgs_dst.mkdir(parents=True, exist_ok=True)
    pasta_labels_dst.mkdir(parents=True, exist_ok=True)

    # índice: image_id → lista de anotações
    anns_por_img: dict[int, list] = {}
    for ann in coco["annotations"]:
        anns_por_img.setdefault(ann["image_id"], []).append(ann)

    processadas = 0
    for img_info in coco["images"]:
        iid = img_info["id"]
        if iid not in anns_por_img:
            continue   # sem anotação → pula

        W, H = img_info["width"], img_info["height"]
        nome  = img_info["file_name"]
        src   = pasta_imgs_src / nome

        if not src.exists():
            continue

        linhas = []
        for ann in anns_por_img[iid]:
            cat_id  = ann["category_id"]
            cls_idx = cat_para_idx.get(cat_id)
            if cls_idx is None:
                continue

            for poli in ann.get("segmentation", []):
                if len(poli) < 6:
                    continue
                # normaliza coordenadas
                coords = []
                for j in range(0, len(poli), 2):
                    x = max(0.0, min(1.0, poli[j]   / W))
                    y = max(0.0, min(1.0, poli[j+1] / H))
                    coords.extend([f"{x:.6f}", f"{y:.6f}"])
                linhas.append(f"{cls_idx} " + " ".join(coords))

        if not linhas:
            continue

        # copia imagem
        shutil.copy2(src, pasta_imgs_dst / nome)
        # salva label
        (pasta_labels_dst / (Path(nome).stem + ".txt")).write_text(
            "\n".join(linhas)
        )
        processadas += 1

    return processadas


def preparar_dataset(
    pasta_coco: Path,
    pasta_repositorio: Path,
    pasta_dataset: Path,
) -> Path:
    """
    Lê os JSONs COCO, converte para YOLO-seg e cria a estrutura:
        dataset/
          images/train/   images/val/
          labels/train/   labels/val/
          dataset.yaml
    Retorna o caminho do dataset.yaml.
    """
    import random
    from tqdm import tqdm

    print("\n📁 Preparando dataset YOLO…")

    # mapeamento category_id → índice YOLO (0-based)
    cat_para_idx = {}
    for i, nome_classe in enumerate(CLASSES):
        # procura nos JSONs qual category_id corresponde a esse nome
        for json_path in pasta_coco.glob("*_coco.json"):
            try:
                coco = json.loads(json_path.read_bytes().decode("latin-1"))
                for cat in coco.get("categories", []):
                    if cat["name"] == nome_classe:
                        cat_para_idx[cat["id"]] = i
            except Exception:
                pass

    total_treino = total_val = 0
    random.seed(42)

    for nome_classe in CLASSES:
        json_path = pasta_coco / f"{nome_classe}_coco.json"
        if not json_path.exists():
            print(f"  ⚠  {json_path.name} não encontrado, pulando.")
            continue

        coco = json.loads(json_path.read_bytes().decode("latin-1"))
        imgs = coco["images"]
        random.shuffle(imgs)

        n_treino  = max(1, int(len(imgs) * SPLIT_TREINO))
        imgs_tr   = imgs[:n_treino]
        imgs_val  = imgs[n_treino:]

        pasta_imgs_src = (
            pasta_repositorio / nome_classe / "imagens"
        )

        # garante que os dicts de images só têm as imgs do split
        for split_nome, split_imgs in [("train", imgs_tr), ("val", imgs_val)]:
            ids_split = {i["id"] for i in split_imgs}
            coco_split = dict(coco)
            coco_split["images"] = split_imgs
            coco_split["annotations"] = [
                a for a in coco["annotations"]
                if a["image_id"] in ids_split
            ]

            n = coco_para_yolo_seg(
                coco_split,
                pasta_imgs_src,
                pasta_dataset / "images" / split_nome,
                pasta_dataset / "labels" / split_nome,
                cat_para_idx,
            )
            if split_nome == "train":
                total_treino += n
            else:
                total_val    += n

        print(f"  ✅ {nome_classe:<22} treino={n_treino}  val={len(imgs_val)}")

    # dataset.yaml
    yaml_path = pasta_dataset / "dataset.yaml"
    yaml_content = (
        f"path: {pasta_dataset.resolve()}\n"
        f"train: images/train\n"
        f"val:   images/val\n"
        f"nc: {len(CLASSES)}\n"
        f"names: {CLASSES}\n"
    )
    yaml_path.write_text(yaml_content)

    print(f"\n  Total treino : {total_treino} imagens")
    print(f"  Total val    : {total_val} imagens")
    print(f"  dataset.yaml : {yaml_path}")

    return yaml_path


# ══════════════════════════════════════════════════════════════════
# DATA AUGMENTATION (configuração inline para YOLO)
# ══════════════════════════════════════════════════════════════════

def augmentation_args(perfil: str) -> dict:
    """
    Retorna hiperparâmetros de augmentation adequados ao perfil.
    Mais augmentation em datasets pequenos para compensar.
    """
    base = {
        "hsv_h":    0.015,   # variação de matiz
        "hsv_s":    0.7,     # variação de saturação
        "hsv_v":    0.4,     # variação de brilho
        "degrees":  10.0,    # rotação
        "translate":0.1,     # translação
        "scale":    0.5,     # escala
        "shear":    2.0,     # cisalhamento
        "flipud":   0.0,     # flip vertical (veículos não ficam de cabeça p/ baixo)
        "fliplr":   0.5,     # flip horizontal
        "mosaic":   1.0,     # mosaico de 4 imagens
        "mixup":    0.1,     # mixup entre imagens
        "copy_paste": 0.1,   # copy-paste de segmentos
    }
    # Em CPU fraca, reduz augmentation pesado para acelerar
    if "cpu_fraca" in perfil:
        base.update({"mosaic": 0.5, "mixup": 0.0, "copy_paste": 0.0})
    return base


# ══════════════════════════════════════════════════════════════════
# TREINO
# ══════════════════════════════════════════════════════════════════

def treinar(
    yaml_path: Path,
    perfil_nome: str,
    perfil: dict,
    epochs: int,
    nome_run: str,
) -> Path:
    try:
        from ultralytics import YOLO
    except ImportError:
        print("\n❌ ultralytics não instalado.")
        print("   Execute: pip install ultralytics\n")
        sys.exit(1)

    print(f"\n{'═'*60}")
    print("🚀  INICIANDO TREINO")
    print(f"{'═'*60}")
    print(f"  Modelo     : {perfil['modelo']}")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch      : {perfil['batch']}")
    print(f"  Resolução  : {perfil['imgsz']}px")
    print(f"  Device     : {perfil['device'].upper() if perfil['device']=='cpu' else 'GPU ' + perfil['device']}")
    print(f"  AMP        : {'Sim (mixed precision)' if perfil['amp'] else 'Não'}")
    print(f"  Workers    : {perfil['workers']}")
    print(f"{'─'*60}\n")

    model = YOLO(perfil["modelo"])
    aug   = augmentation_args(perfil_nome)

    results = model.train(
        data       = str(yaml_path),
        epochs     = epochs,
        batch      = perfil["batch"],
        imgsz      = perfil["imgsz"],
        device     = perfil["device"],
        workers    = perfil["workers"],
        amp        = perfil["amp"],
        name       = nome_run,
        project    = "runs/segment",
        patience   = 30,          # early stopping se não melhorar em 30 epochs
        save       = True,
        save_period= 10,          # salva checkpoint a cada 10 epochs
        plots      = True,        # gera gráficos de treino
        val        = True,
        verbose    = True,
        # augmentation
        **aug,
    )

    best = Path("runs/segment") / nome_run / "weights" / "best.pt"
    print(f"\n✅ Treino concluído!")
    print(f"   Melhor modelo : {best}")
    return best


# ══════════════════════════════════════════════════════════════════
# AVALIAÇÃO
# ══════════════════════════════════════════════════════════════════

def avaliar(modelo_path: str, yaml_path: Path = None):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics não instalado: pip install ultralytics")
        sys.exit(1)

    print(f"\n📊 Avaliando: {modelo_path}")
    model = YOLO(modelo_path)

    # tenta encontrar o yaml se não foi passado
    if yaml_path is None:
        candidatos = list(Path("dataset_yolo").glob("*.yaml"))
        if candidatos:
            yaml_path = candidatos[0]
        else:
            print("❌ dataset.yaml não encontrado. Passe --coco e --imagens para preparar o dataset.")
            sys.exit(1)

    metrics = model.val(data=str(yaml_path), plots=True)

    print(f"\n{'═'*50}")
    print("📊  MÉTRICAS DE AVALIAÇÃO")
    print(f"{'═'*50}")
    try:
        print(f"  mAP50 (box)  : {metrics.box.map50:.3f}")
        print(f"  mAP50-95 (box): {metrics.box.map:.3f}")
        print(f"  mAP50 (mask) : {metrics.seg.map50:.3f}")
        print(f"  mAP50-95 (mask): {metrics.seg.map:.3f}")
        print(f"\n  Por classe:")
        for i, nome in enumerate(CLASSES):
            try:
                print(f"    {nome:<22} mAP50={metrics.seg.ap50[i]:.3f}")
            except Exception:
                pass
    except Exception as e:
        print(f"  (métricas detalhadas: {e})")
    print(f"{'═'*50}")


# ══════════════════════════════════════════════════════════════════
# INFERÊNCIA RÁPIDA (teste visual)
# ══════════════════════════════════════════════════════════════════

def inferir(modelo_path: str, imagem_path: str):
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ pip install ultralytics")
        sys.exit(1)

    print(f"\n🔍 Inferindo: {imagem_path}")
    model   = YOLO(modelo_path)
    results = model.predict(
        source     = imagem_path,
        save       = True,
        conf       = 0.25,
        iou        = 0.45,
        show_labels= True,
        show_conf  = True,
    )
    print(f"  Detecções: {len(results[0].boxes)}")
    print(f"  Resultado salvo em: runs/segment/predict/")


# ══════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Treino YOLOv8-seg para segmentação de viaturas blindadas do EB."
    )
    ap.add_argument("--coco",        type=str, default=None,
                    help="Pasta com os JSONs COCO (ex: anotacoes_coco)")
    ap.add_argument("--imagens",     type=str, default=None,
                    help="Pasta raiz do repositório de imagens (ex: repositorio_viaturas)")
    ap.add_argument("--dataset",     type=str, default="dataset_yolo",
                    help="Pasta de saída do dataset YOLO (padrão: dataset_yolo)")
    ap.add_argument("--epochs",      type=int, default=0,
                    help="Número de epochs (0 = usa recomendação do perfil)")
    ap.add_argument("--perfil",      type=str, default=None,
                    choices=list(PERFIS_HW.keys()),
                    help="Força um perfil de hardware específico")
    ap.add_argument("--nome-run",    type=str, default="viaturas_eb",
                    help="Nome do experimento em runs/segment/ (padrão: viaturas_eb)")
    ap.add_argument("--so-preparar", action="store_true",
                    help="Apenas converte o dataset, não treina")
    ap.add_argument("--retomar",     type=str, default=None,
                    help="Retoma treino a partir de um checkpoint (last.pt)")
    ap.add_argument("--avaliar",     type=str, default=None,
                    help="Avalia um modelo treinado (best.pt)")
    ap.add_argument("--inferir",     type=str, default=None,
                    help="Roda inferência em uma imagem com o modelo --avaliar")
    args = ap.parse_args()

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Treino YOLOv8-seg — Viaturas Blindadas do Exército Br  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # ── Modo avaliação ────────────────────────────────────────────
    if args.avaliar:
        yaml_path = Path(args.dataset) / "dataset.yaml" if Path(args.dataset).exists() else None
        if args.inferir:
            inferir(args.avaliar, args.inferir)
        else:
            avaliar(args.avaliar, yaml_path)
        return

    # ── Modo retomar treino ───────────────────────────────────────
    if args.retomar:
        try:
            from ultralytics import YOLO
        except ImportError:
            print("❌ pip install ultralytics")
            sys.exit(1)
        print(f"\n▶ Retomando treino: {args.retomar}")
        model = YOLO(args.retomar)
        model.train(resume=True)
        return

    # ── Modo normal: prepara + treina ─────────────────────────────
    if not args.coco or not args.imagens:
        print("\n❌ Informe --coco e --imagens para preparar o dataset.")
        print("   Use --help para ver todas as opções.\n")
        ap.print_help()
        sys.exit(1)

    # Detecta hardware e escolhe perfil
    hw = detectar_hardware()

    if args.perfil:
        perfil_nome = args.perfil
        print(f"\n  Perfil forçado: {perfil_nome}")
    else:
        perfil_nome = exibir_menu_perfil(hw)

    perfil = PERFIS_HW[perfil_nome]
    epochs = args.epochs if args.epochs > 0 else perfil["epochs_rec"]

    # Prepara dataset
    pasta_dataset = Path(args.dataset)
    yaml_path = preparar_dataset(
        Path(args.coco),
        Path(args.imagens),
        pasta_dataset,
    )

    if args.so_preparar:
        print("\n✅ Dataset preparado. Pronto para treinar com:")
        print(f"   python treino_yolov8seg.py --coco {args.coco} "
              f"--imagens {args.imagens} --perfil {perfil_nome}\n")
        return

    # Treina
    best_pt = treinar(yaml_path, perfil_nome, perfil, epochs, args.nome_run)

    # Avalia automaticamente após treino
    print("\n📊 Avaliando modelo final…")
    avaliar(str(best_pt), yaml_path)

    print(f"\n{'═'*60}")
    print("🎯  PRÓXIMOS PASSOS")
    print(f"{'═'*60}")
    print(f"  Testar em uma imagem:")
    print(f"    python treino_yolov8seg.py --avaliar {best_pt} "
          f"--inferir caminho/para/imagem.jpg")
    print(f"\n  Exportar para ONNX (deploy):")
    print(f"    yolo export model={best_pt} format=onnx")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
