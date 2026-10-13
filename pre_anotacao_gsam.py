"""
Pré-anotação automática com Grounded-SAM
=========================================
Usa GroundingDINO (detecção por texto) + SAM (segmentação)
para gerar máscaras automáticas das viaturas blindadas.

Exporta no formato COCO JSON — compatível com CVAT e Label Studio.

INSTALAÇÃO (execute uma vez antes de rodar):
    python pre_anotacao_gsam.py --instalar

USO:
    # Anota todas as viaturas do repositório
    python pre_anotacao_gsam.py --repo repositorio_viaturas

    # Anota uma pasta específica
    python pre_anotacao_gsam.py --pasta repositorio_viaturas/VBTP_Guarani/imagens

    # Ajusta a confiança mínima de detecção (0.0 a 1.0)
    python pre_anotacao_gsam.py --repo repositorio_viaturas --confianca 0.35
"""

import os, sys, json, shutil, argparse, subprocess, time, math
from pathlib import Path
from datetime import datetime

# ══════════════════════════════════════════════════════════════════
# PERFIS DE MODELO
# ══════════════════════════════════════════════════════════════════

PERFIS = {
    "leve": {
        "sam_tipo":  "vit_b",
        "sam_nome":  "sam_vit_b.pth",
        "sam_url":   "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
        "sam_desc":  "SAM ViT-B (375 MB)",
        "descricao": "PC fraco / sem GPU  — rápido, precisão boa",
        "vram_min":  0,
    },
    "medio": {
        "sam_tipo":  "vit_l",
        "sam_nome":  "sam_vit_l.pth",
        "sam_url":   "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
        "sam_desc":  "SAM ViT-L (1.2 GB)",
        "descricao": "GPU com 6–8 GB VRAM — equilíbrio velocidade/precisão",
        "vram_min":  6,
    },
    "pesado": {
        "sam_tipo":  "vit_h",
        "sam_nome":  "sam_vit_h.pth",
        "sam_url":   "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
        "sam_desc":  "SAM ViT-H (2.4 GB)",
        "descricao": "GPU com 10+ GB VRAM — máxima precisão",
        "vram_min":  10,
    },
}


def detectar_hardware() -> dict:
    """Detecta GPU disponível e retorna informações de hardware."""
    info = {"tem_gpu": False, "nome_gpu": "CPU only", "vram_gb": 0, "device": "cpu"}
    try:
        import torch
        if torch.cuda.is_available():
            idx  = torch.cuda.current_device()
            nome = torch.cuda.get_device_name(idx)
            vram = torch.cuda.get_device_properties(idx).total_memory / 1024**3
            info.update({"tem_gpu": True, "nome_gpu": nome,
                         "vram_gb": round(vram, 1), "device": "cuda"})
    except Exception:
        pass
    return info


def recomendar_perfil(hw: dict) -> str:
    """Retorna o perfil recomendado com base no hardware detectado."""
    if not hw["tem_gpu"]:
        return "leve"
    if hw["vram_gb"] >= PERFIS["pesado"]["vram_min"]:
        return "pesado"
    if hw["vram_gb"] >= PERFIS["medio"]["vram_min"]:
        return "medio"
    return "leve"


def escolher_perfil_interativo(hw: dict) -> str:
    """Mostra menu interativo e retorna o perfil escolhido pelo usuário."""
    recomendado = recomendar_perfil(hw)

    print("\n┌─────────────────────────────────────────────────────┐")
    print("│              Configuração de Hardware                 │")
    print("├─────────────────────────────────────────────────────┤")
    if hw["tem_gpu"]:
        print(f"│  GPU  : {hw['nome_gpu'][:43]:<43} │")
        print(f"│  VRAM : {hw['vram_gb']:.1f} GB{'':<39} │")
    else:
        print(f"│  GPU  : Nenhuma detectada — modo CPU               │")
    print("├─────────────────────────────────────────────────────┤")
    print("│  Perfis disponíveis:                                 │")
    for chave, p in PERFIS.items():
        rec = " ← recomendado" if chave == recomendado else ""
        print(f"│  [{list(PERFIS).index(chave)+1}] {chave:<8}  {p['sam_desc']:<18}  {p['descricao'][:16]:<16}{rec:<15}│")
    print("└─────────────────────────────────────────────────────┘")

    mapa = {"1": "leve", "2": "medio", "3": "pesado",
            "leve": "leve", "medio": "medio", "pesado": "pesado",
            "l": "leve", "m": "medio", "p": "pesado"}

    while True:
        entrada = input(
            f"\n  Escolha o perfil [1/2/3] (Enter = {recomendado}): "
        ).strip().lower()
        if entrada == "":
            return recomendado
        if entrada in mapa:
            escolha = mapa[entrada]
            print(f"  ✅ Perfil selecionado: {escolha} — {PERFIS[escolha]['descricao']}")
            return escolha
        print("  ⚠  Opção inválida. Digite 1, 2 ou 3.")


# ── Mapeamento viatura → prompts GroundingDINO ─────────────────────
VIATURA_PROMPTS = {
    "VBTP_Guarani":  "military armored personnel carrier 6x6 wheeled",
    "Cascavel_EE9":  "military wheeled armored car cannon turret",
    "VBC_Leopard":   "military main battle tank tracks turret",
    "Guepard_M113":  "military tracked armored personnel carrier",
    "Lince_LMV":     "military light protected patrol vehicle 4x4",
}
PROMPT_GENERICO = "military armored vehicle"

CLASSES_COCO = [
    {"id": 1, "name": "VBTP_Guarani",  "supercategory": "viatura_blindada"},
    {"id": 2, "name": "Cascavel_EE9",  "supercategory": "viatura_blindada"},
    {"id": 3, "name": "VBC_Leopard",   "supercategory": "viatura_blindada"},
    {"id": 4, "name": "Guepard_M113",  "supercategory": "viatura_blindada"},
    {"id": 5, "name": "Lince_LMV",     "supercategory": "viatura_blindada"},
]
NOME_PARA_ID = {c["name"]: c["id"] for c in CLASSES_COCO}


# ══════════════════════════════════════════════════════════════════
# INSTALAÇÃO
# ══════════════════════════════════════════════════════════════════

def _rodar_cmd(cmd: list, descricao: str, cwd=None) -> bool:
    """Executa um comando e imprime resultado."""
    print(f"▶ {descricao}…")
    result = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    if result.returncode != 0:
        print(f"  ⚠  {result.stderr[-400:]}")
        return False
    print("  ✅ OK")
    return True


def _instalar_groundingdino(base_dir: Path) -> bool:
    """
    Instala GroundingDINO no Windows sem precisar de compilador C++.
    Estratégia: clona o repo e adiciona ao sys.path diretamente,
    evitando a etapa de build que exige Visual Studio Build Tools.
    """
    gdino_dir = base_dir / "GroundingDINO"

    # 1. Clona se não existir
    if not gdino_dir.exists():
        ok = _rodar_cmd(
            ["git", "clone", "--depth", "1",
             "https://github.com/IDEA-Research/GroundingDINO.git",
             str(gdino_dir)],
            "Clonando GroundingDINO"
        )
        if not ok:
            return False
    else:
        print("  ✅ Repositório GroundingDINO já existe")

    # 2. Tenta instalar dependências sem compilar extensões nativas
    _rodar_cmd(
        [sys.executable, "-m", "pip", "install",
         "transformers==4.35.2", "timm==0.9.12",
         "addict", "yapf", "pycocotools", "supervision==0.19.0",
         "huggingface_hub"],
        "Instalando dependências Python do GroundingDINO"
    )

    # 3. Registra o caminho no arquivo .pth do ambiente Python
    #    (equivalente a pip install -e . mas sem compilação)
    try:
        import site
        site_packages = Path(site.getsitepackages()[0])
        pth_file = site_packages / "groundingdino_local.pth"
        pth_file.write_text(str(gdino_dir.resolve()))
        print(f"  ✅ GroundingDINO registrado via .pth: {pth_file}")
    except Exception as e:
        print(f"  ⚠  Não foi possível criar .pth: {e}")
        # fallback: adiciona ao PYTHONPATH do sistema
        pth_alt = gdino_dir.parent / "gdino_path.py"
        pth_alt.write_text(
            f"import sys\nsys.path.insert(0, r'{gdino_dir.resolve()}')"
        )
        print(f"  ✅ Fallback: use 'import gdino_path' antes de rodar")

    # 4. Verifica se o módulo ficou acessível
    result = subprocess.run(
        [sys.executable, "-c", "import groundingdino; print('OK')"],
        capture_output=True, text=True
    )
    if "OK" in result.stdout:
        print("  ✅ groundingdino importável com sucesso")
        return True
    else:
        # Última tentativa: injeta o path na sessão atual
        if str(gdino_dir.resolve()) not in sys.path:
            sys.path.insert(0, str(gdino_dir.resolve()))
        result2 = subprocess.run(
            [sys.executable, "-c",
             f"import sys; sys.path.insert(0,r'{gdino_dir.resolve()}'); "
             "import groundingdino; print('OK')"],
            capture_output=True, text=True
        )
        if "OK" in result2.stdout:
            print("  ✅ groundingdino acessível via sys.path")
            return True
        print(f"  ⚠  Verificação falhou: {result2.stderr[-200:]}")
        return False


def corrigir_dependencias():
    """Corrige incompatibilidade transformers/GroundingDINO e problemas de módulo."""
    print("╔══════════════════════════════════════════════════════╗")
    print("║         Corrigindo dependências incompatíveis        ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    base_dir = Path("modelos_gsam")
    base_dir.mkdir(exist_ok=True)

    fixes = [
        ([sys.executable, "-m", "pip", "install", "transformers==4.35.2"],
         "Fixando transformers==4.35.2"),
        ([sys.executable, "-m", "pip", "install", "timm==0.9.12"],
         "Fixando timm==0.9.12"),
        ([sys.executable, "-m", "pip", "install",
          "addict", "yapf", "pycocotools", "supervision==0.19.0"],
         "Instalando dependências do GroundingDINO"),
    ]

    for cmd, descricao in fixes:
        _rodar_cmd(cmd, descricao)

    # GroundingDINO via clone (mais confiável que pip install git+)
    ok = _instalar_groundingdino(base_dir)
    if not ok:
        print("\n❌ Falha ao instalar GroundingDINO.")
        print("   Verifique se o Git está instalado: https://git-scm.com/download/win")
        return

    print("\n✅ Correção concluída! Tente rodar novamente:\n")
    print("   python pre_anotacao_gsam.py --repo repositorio_viaturas\n")


def instalar_dependencias():
    print("╔══════════════════════════════════════════════════════╗")
    print("║         Instalando dependências – Grounded-SAM       ║")
    print("╚══════════════════════════════════════════════════════╝\n")

    cmds = [
        # PyTorch com CUDA (se não tiver GPU, usa CPU automaticamente)
        [sys.executable, "-m", "pip", "install", "torch", "torchvision",
         "--index-url", "https://download.pytorch.org/whl/cu118"],
        # transformers em versão compatível com GroundingDINO
        # (versões >= 4.36 quebraram a API BertModel usada internamente)
        [sys.executable, "-m", "pip", "install", "transformers==4.35.2"],
        # Dependências gerais
        [sys.executable, "-m", "pip", "install",
         "opencv-python", "Pillow", "numpy", "supervision==0.19.0",
         "pycocotools", "tqdm", "matplotlib",
         "timm==0.9.12", "addict", "yapf"],
        # Segment Anything Model
        [sys.executable, "-m", "pip", "install",
         "git+https://github.com/facebookresearch/segment-anything.git"],
    ]

    for cmd in cmds:
        print(f"▶ {' '.join(cmd[-3:])}")
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"  ⚠  Aviso: {result.stderr[-300:]}")
        else:
            print("  ✅ OK")

    # Download dos pesos dos modelos
    pesos_dir = Path("modelos_gsam")
    pesos_dir.mkdir(exist_ok=True)

    # GroundingDINO via clone — mais confiável no Windows que pip install git+
    _instalar_groundingdino(pesos_dir)

    # Detecta hardware e pergunta qual perfil SAM baixar
    hw         = detectar_hardware()
    recomendado = recomendar_perfil(hw)
    perfil_inst = escolher_perfil_interativo(hw)
    p_sam       = PERFIS[perfil_inst]

    modelos = {
        p_sam["sam_nome"]: (
            p_sam["sam_url"],
            f"{p_sam['sam_desc']} — {p_sam['descricao']}",
        ),
        "groundingdino_swint_ogc.pth": (
            "https://github.com/IDEA-Research/GroundingDINO/releases/download/"
            "v0.1.0-alpha/groundingdino_swint_ogc.pth",
            "GroundingDINO SwinT (662 MB)",
        ),
        "GroundingDINO_SwinT_OGC.py": (
            "https://raw.githubusercontent.com/IDEA-Research/GroundingDINO/main/"
            "groundingdino/config/GroundingDINO_SwinT_OGC.py",
            "Config GroundingDINO",
        ),
    }

    print("\n📥 Baixando pesos dos modelos…")
    import requests as _req

    def _baixar(url: str, destino: Path, descricao: str):
        print(f"  ⬇  {descricao}")
        tmp = destino.with_suffix(".tmp")
        try:
            with _req.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                total    = int(r.headers.get("Content-Length", 0))
                baixado  = 0
                inicio   = time.time()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                            baixado += len(chunk)
                            if total:
                                pct = baixado / total * 100
                                mb  = baixado / 1024 / 1024
                                tmb = total   / 1024 / 1024
                                vel = mb / max(time.time() - inicio, 0.1)
                                print(f"\r    {pct:5.1f}%  {mb:.0f}/{tmb:.0f} MB"
                                      f"  {vel:.1f} MB/s    ", end="", flush=True)
            tmp.rename(destino)
            print(f"\r    ✅ {destino.name} ({baixado/1024/1024:.0f} MB)          ")
        except Exception as e:
            if tmp.exists():
                tmp.unlink()
            print(f"\n    ❌ Erro ao baixar {descricao}: {e}")

    for nome, (url, descricao) in modelos.items():
        destino = pesos_dir / nome
        if destino.exists():
            print(f"  ✅ {nome} já existe")
            continue
        _baixar(url, destino, descricao)

    print("\n✅ Instalação concluída!")
    print(f"   Pesos salvos em: {pesos_dir.resolve()}")
    print("\nAgora execute:")
    print("   python pre_anotacao_gsam.py --repo repositorio_viaturas\n")


# ══════════════════════════════════════════════════════════════════
# CARREGAMENTO DOS MODELOS
# ══════════════════════════════════════════════════════════════════

def carregar_modelos(pesos_dir: Path, device: str):
    import torch
    from segment_anything import sam_model_registry, SamPredictor

    # Garante que o GroundingDINO clonado está no path (Windows sem compilador)
    gdino_clone = pesos_dir / "GroundingDINO"
    if gdino_clone.exists() and str(gdino_clone.resolve()) not in sys.path:
        sys.path.insert(0, str(gdino_clone.resolve()))

    try:
        import groundingdino.datasets.transforms as T
        from groundingdino.models import build_model
        from groundingdino.util.slconfig import SLConfig
        from groundingdino.util.utils import clean_state_dict
    except ModuleNotFoundError:
        print("\n❌ Módulo groundingdino não encontrado.")
        print("   Execute: python pre_anotacao_gsam.py --corrigir\n")
        sys.exit(1)

    print(f"  🔧 Dispositivo: {device.upper()}")

    # SAM — escolhe automaticamente o melhor modelo disponível
    # Ordem de preferência: pesado > medio > leve (usa o melhor que encontrar)
    sam_tipo, sam_nome, sam_path_found = None, None, None
    preferencia = (
        ["vit_h","vit_l","vit_b"] if device == "cuda"
        else ["vit_b","vit_l","vit_h"]
    )
    mapa_nomes = {"vit_h": "sam_vit_h.pth", "vit_l": "sam_vit_l.pth",
                  "vit_b": "sam_vit_b.pth"}
    for tipo in preferencia:
        candidato = pesos_dir / mapa_nomes[tipo]
        if candidato.exists():
            sam_tipo, sam_nome, sam_path_found = tipo, mapa_nomes[tipo], candidato
            break
    if sam_path_found is None:
        print("\n❌ Nenhum peso SAM encontrado em", pesos_dir)
        print("   Execute: python pre_anotacao_gsam.py --instalar")
        sys.exit(1)
    print(f"  📦 Carregando SAM ({sam_nome})…")
    sam = sam_model_registry[sam_tipo](checkpoint=str(sam_path_found))
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    # GroundingDINO
    cfg_path = pesos_dir / "GroundingDINO_SwinT_OGC.py"
    dino_path = pesos_dir / "groundingdino_swint_ogc.pth"
    print(f"  📦 Carregando GroundingDINO…")
    args = SLConfig.fromfile(str(cfg_path))
    args.device = device
    dino_model = build_model(args)
    checkpoint = torch.load(str(dino_path), map_location="cpu")
    dino_model.load_state_dict(
        clean_state_dict(checkpoint["model"]), strict=False
    )
    dino_model.eval()
    dino_model = dino_model.to(device)

    return sam_predictor, dino_model


# ══════════════════════════════════════════════════════════════════
# DETECÇÃO + SEGMENTAÇÃO
# ══════════════════════════════════════════════════════════════════

def detectar_e_segmentar(
    imagem_bgr,
    prompt: str,
    sam_predictor,
    dino_model,
    device: str,
    box_threshold: float = 0.35,
    text_threshold: float = 0.25,
):
    import torch
    import numpy as np
    import cv2
    # Garante path do clone no Windows
    gdino_clone = Path(__file__).parent / "modelos_gsam" / "GroundingDINO"
    if not gdino_clone.exists():
        # tenta localização alternativa
        for p in sys.path:
            c = Path(p).parent / "modelos_gsam" / "GroundingDINO"
            if c.exists():
                gdino_clone = c
                break
    if gdino_clone.exists() and str(gdino_clone.resolve()) not in sys.path:
        sys.path.insert(0, str(gdino_clone.resolve()))

    from groundingdino.util.inference import predict
    from groundingdino.datasets import transforms as T
    from PIL import Image

    # Converte BGR → RGB para PIL
    imagem_rgb = cv2.cvtColor(imagem_bgr, cv2.COLOR_BGR2RGB)
    pil_img    = Image.fromarray(imagem_rgb)

    # Transforma para GroundingDINO
    transform = T.Compose([
        T.RandomResize([800], max_size=1333),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    img_tensor, _ = transform(pil_img, None)

    # Detecção com GroundingDINO
    boxes, logits, phrases = predict(
        model       = dino_model,
        image       = img_tensor,
        caption     = prompt,
        box_threshold  = box_threshold,
        text_threshold = text_threshold,
        device      = device,
    )

    if len(boxes) == 0:
        return [], [], []

    # Converte boxes para coordenadas de pixel (formato xyxy)
    H, W = imagem_bgr.shape[:2]
    boxes_pixel = boxes.clone()
    boxes_pixel[:, [0, 2]] *= W
    boxes_pixel[:, [1, 3]] *= H
    # de cx,cy,w,h → x1,y1,x2,y2
    boxes_xyxy = torch.zeros_like(boxes_pixel)
    boxes_xyxy[:, 0] = boxes_pixel[:, 0] - boxes_pixel[:, 2] / 2
    boxes_xyxy[:, 1] = boxes_pixel[:, 1] - boxes_pixel[:, 3] / 2
    boxes_xyxy[:, 2] = boxes_pixel[:, 0] + boxes_pixel[:, 2] / 2
    boxes_xyxy[:, 3] = boxes_pixel[:, 1] + boxes_pixel[:, 3] / 2
    boxes_xyxy = boxes_xyxy.cpu().numpy()

    # Segmentação com SAM
    sam_predictor.set_image(imagem_rgb)
    masks_list, scores_list = [], []

    for box in boxes_xyxy:
        masks, scores, _ = sam_predictor.predict(
            point_coords = None,
            point_labels = None,
            box          = box[None, :],
            multimask_output = False,
        )
        masks_list.append(masks[0])    # máscara booleana HxW
        scores_list.append(float(scores[0]))

    return boxes_xyxy, masks_list, scores_list


# ══════════════════════════════════════════════════════════════════
# CONVERSÃO MÁSCARA → SEGMENTAÇÃO COCO (polígono RLE)
# ══════════════════════════════════════════════════════════════════

def mascara_para_poligono(mascara_bool, tolerancia: int = 2):
    """Converte máscara binária em lista de polígonos [x,y,x,y,...]."""
    import cv2
    import numpy as np

    mascara_uint8 = mascara_bool.astype("uint8") * 255
    contornos, _ = cv2.findContours(
        mascara_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    poligonos = []
    for c in contornos:
        # simplifica o contorno para reduzir pontos
        epsilon = tolerancia * cv2.arcLength(c, True) / 100
        approx  = cv2.approxPolyDP(c, epsilon, True)
        if len(approx) < 4:   # ignora contornos muito pequenos
            continue
        pts = approx.flatten().tolist()
        if len(pts) >= 6:     # mínimo 3 pontos (6 coordenadas)
            poligonos.append(pts)

    return poligonos


def bbox_de_mascara(mascara_bool):
    """Retorna [x, y, largura, altura] da bounding box da máscara."""
    import numpy as np
    rows = np.any(mascara_bool, axis=1)
    cols = np.any(mascara_bool, axis=0)
    if not rows.any():
        return [0, 0, 0, 0]
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return [int(cmin), int(rmin), int(cmax - cmin), int(rmax - rmin)]


def area_mascara(mascara_bool) -> int:
    import numpy as np
    return int(np.sum(mascara_bool))


# ══════════════════════════════════════════════════════════════════
# PROCESSAMENTO DE UMA PASTA
# ══════════════════════════════════════════════════════════════════

def processar_pasta(
    pasta: Path,
    nome_viatura: str,
    sam_predictor,
    dino_model,
    device: str,
    conf: float,
    saida_dir: Path,
):
    import cv2
    from tqdm import tqdm

    imagens = sorted([
        p for p in pasta.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    ])

    if not imagens:
        print(f"  ⚠  Nenhuma imagem encontrada em {pasta}")
        return None

    cat_id   = NOME_PARA_ID.get(nome_viatura, 0)
    prompt   = VIATURA_PROMPTS.get(nome_viatura, PROMPT_GENERICO)

    coco = {
        "info": {
            "description": f"Pré-anotação automática Grounded-SAM — {nome_viatura}",
            "date_created": datetime.now().isoformat(),
            "version": "1.0",
        },
        "licenses": [],
        "categories": CLASSES_COCO,
        "images": [],
        "annotations": [],
    }

    img_id  = 1
    ann_id  = 1
    ok_count = 0
    sem_det  = 0

    print(f"\n  🔍 Prompt: «{prompt}»")
    print(f"  📁 {len(imagens)} imagens a processar")

    for caminho_img in tqdm(imagens, desc=f"  {nome_viatura[:18]}", unit="img"):
        bgr = cv2.imread(str(caminho_img))
        if bgr is None:
            continue

        H, W = bgr.shape[:2]

        # Registra a imagem no COCO
        coco["images"].append({
            "id":        img_id,
            "file_name": caminho_img.name,
            "width":     W,
            "height":    H,
        })

        try:
            boxes, masks, scores = detectar_e_segmentar(
                bgr, prompt, sam_predictor, dino_model, device,
                box_threshold=conf, text_threshold=conf * 0.7,
            )
        except Exception as e:
            print(f"    ⚠  Erro em {caminho_img.name}: {e}")
            img_id += 1
            continue

        if not masks:
            sem_det += 1
            img_id  += 1
            continue

        for mask, score, box in zip(masks, scores, boxes):
            poligonos = mascara_para_poligono(mask)
            if not poligonos:
                continue

            coco["annotations"].append({
                "id":            ann_id,
                "image_id":      img_id,
                "category_id":   cat_id,
                "segmentation":  poligonos,
                "bbox":          bbox_de_mascara(mask),
                "area":          area_mascara(mask),
                "iscrowd":       0,
                "score":         round(score, 4),   # confiança do SAM
                "auto_labeled":  True,               # flag para revisão
            })
            ann_id += 1

        ok_count += 1
        img_id   += 1

    # Salva o JSON
    saida_dir.mkdir(parents=True, exist_ok=True)
    json_path = saida_dir / f"{nome_viatura}_coco.json"
    json_path.write_text(json.dumps(coco, indent=2, ensure_ascii=False))

    print(f"  ✅ {ok_count}/{len(imagens)} imagens anotadas")
    if sem_det:
        print(f"  ⚠  {sem_det} imagens sem detecção (revise manualmente)")
    print(f"  💾 {json_path}")

    return {
        "viatura":      nome_viatura,
        "imagens":      len(imagens),
        "anotadas":     ok_count,
        "sem_deteccao": sem_det,
        "json":         str(json_path),
    }


# ══════════════════════════════════════════════════════════════════
# VISUALIZAÇÃO DE RESULTADO (DEBUG)
# ══════════════════════════════════════════════════════════════════

def visualizar_amostra(coco_json: Path, pasta_imagens: Path, n: int = 6):
    """Gera grade com amostra de imagens anotadas para conferência visual."""
    import cv2
    import numpy as np

    coco = json.loads(coco_json.read_text())
    imgs_com_ann = {a["image_id"] for a in coco["annotations"]}
    imgs = [i for i in coco["images"] if i["id"] in imgs_com_ann][:n]

    if not imgs:
        print("  Nenhuma imagem com anotação para visualizar.")
        return

    cores = [
        (52, 199, 89), (255, 149, 0), (0, 122, 255),
        (255, 59, 48), (175, 82, 222),
    ]

    grade = []
    thumb_size = (320, 240)

    for img_info in imgs:
        caminho = pasta_imagens / img_info["file_name"]
        bgr = cv2.imread(str(caminho))
        if bgr is None:
            continue

        overlay = bgr.copy()
        anns = [a for a in coco["annotations"] if a["image_id"] == img_info["id"]]

        for ann in anns:
            cor = cores[(ann["category_id"] - 1) % len(cores)]
            for poli in ann["segmentation"]:
                pts = np.array(poli, dtype=np.int32).reshape(-1, 2)
                cv2.fillPoly(overlay, [pts], cor)
                cv2.polylines(bgr, [pts], True, cor, 2)

        bgr = cv2.addWeighted(bgr, 0.55, overlay, 0.45, 0)

        # label
        cat = next((c for c in coco["categories"] if c["id"] == anns[0]["category_id"]), None)
        if cat:
            cv2.putText(bgr, cat["name"].replace("_", " "), (10, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        bgr = cv2.resize(bgr, thumb_size)
        grade.append(bgr)

    # monta grade 3×2
    while len(grade) < 6:
        grade.append(np.zeros((*thumb_size[::-1], 3), dtype=np.uint8))

    linha1 = np.hstack(grade[:3])
    linha2 = np.hstack(grade[3:6])
    final  = np.vstack([linha1, linha2])

    out_path = coco_json.parent / f"{coco_json.stem}_preview.jpg"
    cv2.imwrite(str(out_path), final)
    print(f"  🖼  Preview salvo: {out_path}")


# ══════════════════════════════════════════════════════════════════
# ENTRADA PRINCIPAL
# ══════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser(
        description="Pré-anotação automática com Grounded-SAM → formato COCO"
    )
    ap.add_argument("--instalar",    action="store_true",
                    help="Instala todas as dependências e baixa os pesos")
    ap.add_argument("--corrigir",    action="store_true",
                    help="Corrige incompatibilidade transformers/GroundingDINO sem reinstalar tudo")
    ap.add_argument("--repo",        type=str, default=None,
                    help="Pasta raiz do repositório de viaturas")
    ap.add_argument("--pasta",       type=str, default=None,
                    help="Pasta de imagens de uma viatura específica")
    ap.add_argument("--viatura",     type=str, default=None,
                    help="Nome da viatura (ex: VBTP_Guarani) ao usar --pasta")
    ap.add_argument("--confianca",   type=float, default=0.35,
                    help="Limiar de confiança do GroundingDINO (padrão: 0.35)")
    ap.add_argument("--pesos",       type=str, default="modelos_gsam",
                    help="Pasta com os pesos dos modelos (padrão: modelos_gsam)")
    ap.add_argument("--saida",       type=str, default="anotacoes_coco",
                    help="Pasta de saída dos JSONs COCO (padrão: anotacoes_coco)")
    ap.add_argument("--preview",     action="store_true",
                    help="Gera imagem de preview com as máscaras sobrepostas")
    ap.add_argument("--cpu",         action="store_true",
                    help="Força uso de CPU mesmo com GPU disponível")
    args = ap.parse_args()

    if args.corrigir:
        corrigir_dependencias()
        return

    if args.instalar:
        instalar_dependencias()
        return

    # Importações pesadas só quando for anotar
    try:
        import torch
    except ImportError:
        print("❌ PyTorch não instalado. Execute primeiro:")
        print("   python pre_anotacao_gsam.py --instalar\n")
        sys.exit(1)

    device = "cpu" if args.cpu or not torch.cuda.is_available() else "cuda"
    pesos_dir = Path(args.pesos)
    saida_dir = Path(args.saida)

    # Verifica pesos — pelo menos um modelo SAM deve existir
    algum_sam = any(
        (pesos_dir / nome).exists()
        for nome in ["sam_vit_b.pth", "sam_vit_l.pth", "sam_vit_h.pth"]
    )
    for nome in ["groundingdino_swint_ogc.pth", "GroundingDINO_SwinT_OGC.py"]:
        if not (pesos_dir / nome).exists():
            print(f"❌ Peso não encontrado: {pesos_dir / nome}")
            print("   Execute: python pre_anotacao_gsam.py --instalar\n")
            sys.exit(1)
    if not algum_sam:
        print("❌ Nenhum peso SAM encontrado em", pesos_dir)
        print("   Execute: python pre_anotacao_gsam.py --instalar\n")
        sys.exit(1)

    # Detecta hardware e mostra perfil que será usado
    hw = detectar_hardware()
    perfil_auto = recomendar_perfil(hw)
    sam_disponivel = next(
        (PERFIS[p]["sam_nome"] for p in ["pesado","medio","leve"]
         if (pesos_dir / PERFIS[p]["sam_nome"]).exists()),
        "nenhum"
    )

    print("╔══════════════════════════════════════════════════════╗")
    print("║      Pré-anotação Automática – Grounded-SAM          ║")
    print("╚══════════════════════════════════════════════════════╝")
    if hw["tem_gpu"]:
        print(f"  GPU         : {hw['nome_gpu'][:40]}")
        print(f"  VRAM        : {hw['vram_gb']:.1f} GB")
    else:
        print("  GPU         : Nenhuma — rodando em CPU")
    print(f"  SAM carregado: {sam_disponivel}")
    print(f"  Confiança   : {args.confianca}")
    print(f"  Saída       : {saida_dir.resolve()}\n")

    print("🔧 Carregando modelos…")
    sam_predictor, dino_model = carregar_modelos(pesos_dir, device)
    print("✅ Modelos prontos\n")

    resultados = []

    # Modo repositório completo
    if args.repo:
        repo = Path(args.repo)
        for nome_viatura in VIATURA_PROMPTS:
            pasta_imgs = repo / nome_viatura / "imagens"
            if not pasta_imgs.exists():
                print(f"⚠  {pasta_imgs} não existe, pulando.")
                continue
            print(f"\n{'─'*54}")
            print(f"🚛  {nome_viatura.replace('_', ' ')}")
            print(f"{'─'*54}")
            r = processar_pasta(
                pasta_imgs, nome_viatura,
                sam_predictor, dino_model, device,
                args.confianca, saida_dir,
            )
            if r:
                resultados.append(r)
                if args.preview:
                    visualizar_amostra(
                        Path(r["json"]), pasta_imgs
                    )

    # Modo pasta única
    elif args.pasta:
        pasta = Path(args.pasta)
        nome  = args.viatura or pasta.parent.name
        r = processar_pasta(
            pasta, nome,
            sam_predictor, dino_model, device,
            args.confianca, saida_dir,
        )
        if r:
            resultados.append(r)
            if args.preview:
                visualizar_amostra(Path(r["json"]), pasta)

    else:
        print("❌ Informe --repo ou --pasta. Use --help para ajuda.")
        sys.exit(1)

    # Relatório final
    rel_path = saida_dir / "relatorio_anotacao.json"
    rel = {
        "gerado_em":  datetime.now().isoformat(),
        "dispositivo": device,
        "confianca":  args.confianca,
        "viaturas":   resultados,
        "total_imagens":  sum(r["imagens"]  for r in resultados),
        "total_anotadas": sum(r["anotadas"] for r in resultados),
        "total_sem_det":  sum(r["sem_deteccao"] for r in resultados),
    }
    rel_path.write_text(json.dumps(rel, indent=2, ensure_ascii=False))

    print(f"\n{'═'*54}")
    print("📊  RESUMO FINAL")
    print(f"{'═'*54}")
    for r in resultados:
        pct = 100 * r["anotadas"] / r["imagens"] if r["imagens"] else 0
        print(f"  {r['viatura']:<22} {r['anotadas']:>3}/{r['imagens']:<3} "
              f"({pct:5.1f}%)  sem detecção: {r['sem_deteccao']}")
    print(f"{'─'*54}")
    tot_i = rel["total_imagens"]
    tot_a = rel["total_anotadas"]
    pct_t = 100 * tot_a / tot_i if tot_i else 0
    print(f"  {'TOTAL':<22} {tot_a:>3}/{tot_i:<3} ({pct_t:5.1f}%)")
    print(f"{'═'*54}")
    print(f"\n📁 JSONs COCO : {saida_dir.resolve()}")
    print(f"📄 Relatório  : {rel_path.resolve()}")
    print("\n💡 Próximo passo:")
    print("   Importe os JSONs no CVAT ou Label Studio para revisão.")
    print("   Campos 'auto_labeled: true' indicam anotações para conferir.\n")


if __name__ == "__main__":
    main()
