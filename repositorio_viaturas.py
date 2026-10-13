"""
Repositório de Imagens de Viaturas Blindadas do Exército Brasileiro
====================================================================
Usa Selenium (browser real) para contornar bloqueios anti-bot do Bing/Google.
O browser abre minimizado e fecha automaticamente ao terminar.

INSTALAÇÃO:
    pip install selenium webdriver-manager Pillow imagehash tqdm requests

USO:
    python repositorio_viaturas.py                         # 30 imgs, threshold 8
    python repositorio_viaturas.py --images 50             # mais imagens
    python repositorio_viaturas.py --threshold 5           # filtro mais agressivo
    python repositorio_viaturas.py --output C:/meu_dataset # pasta personalizada
    python repositorio_viaturas.py --visible               # browser visível (debug)
"""

import os, re, sys, time, json, shutil, argparse
from pathlib import Path
from io import BytesIO

# ── Verifica dependências ──────────────────────────────────────────
faltando = []
try:
    import requests
except ImportError:
    faltando.append("requests")
try:
    from PIL import Image
except ImportError:
    faltando.append("Pillow")
try:
    import imagehash
except ImportError:
    faltando.append("imagehash")
try:
    from tqdm import tqdm
except ImportError:
    faltando.append("tqdm")
try:
    from selenium import webdriver
    from selenium.webdriver.common.by import By
    from selenium.webdriver.chrome.service import Service
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC
except ImportError:
    faltando.append("selenium")

if faltando:
    print("\n❌ Dependências faltando. Execute:")
    print(f"   pip install {' '.join(faltando)}\n")
    sys.exit(1)

# ── Viaturas e queries ─────────────────────────────────────────────

VIATURAS = {
    "VBTP_Guarani": [
        "VBTP Guarani blindado exército brasileiro",
        "Guarani VBTP-MR armored vehicle Brazil",
        "viatura blindada Guarani 6x6",
    ],
    "Cascavel_EE9": [
        "Cascavel EE-9 blindado exército brasileiro",
        "EE-9 Cascavel armored car Brazil army",
        "Cascavel wheeled armored vehicle",
    ],
    "VBC_Leopard": [
        "Leopard 1A5 BR exército brasileiro tanque",
        "VBC Leopard 1 Brazil main battle tank",
        "Leopard 1A5 carro de combate brasileiro",
    ],
    "Guepard_M113": [
        "Guepard M113 exército brasileiro blindado",
        "M113 Gepard Brazil army APC",
        "Guepard blindado transporte pessoal EB",
    ],
    "Lince_LMV": [
        "Lince LMV exército brasileiro veículo blindado leve",
        "Iveco LMV Lynx Brazil army light armored",
        "Lince LMV 4x4 blindado leve exército",
    ],
}

MIN_W, MIN_H = 200, 150
HASH_SIZE    = 16
IMG_HEADERS  = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )
}


# ── Selenium: configura Chrome ─────────────────────────────────────

def criar_driver(visivel: bool = False) -> webdriver.Chrome:
    opts = Options()
    if not visivel:
        opts.add_argument("--headless=new")
    opts.add_argument("--no-sandbox")
    opts.add_argument("--disable-dev-shm-usage")
    opts.add_argument("--disable-blink-features=AutomationControlled")
    opts.add_argument("--window-size=1280,900")
    opts.add_argument("--lang=pt-BR")
    opts.add_experimental_option("excludeSwitches", ["enable-automation"])
    opts.add_experimental_option("useAutomationExtension", False)
    opts.add_argument(
        "user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    )

    erros = []

    # Opção 1: Selenium Manager embutido (Selenium >= 4.10, sem dependência extra)
    try:
        driver = webdriver.Chrome(options=opts)
        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        return driver
    except Exception as e:
        erros.append(f"Selenium Manager: {e}")

    # Opção 2: webdriver-manager (corrige bug de timeout forçando int)
    try:
        from webdriver_manager.chrome import ChromeDriverManager
        from webdriver_manager.core.os_manager import ChromeType
        import os
        os.environ.setdefault("WDM_SSL_VERIFY", "0")
        mgr = ChromeDriverManager()
        mgr._http_client.timeout = 30   # força timeout como int
        path = mgr.install()
        driver = webdriver.Chrome(service=Service(path), options=opts)
        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        return driver
    except Exception as e:
        erros.append(f"webdriver-manager: {e}")

    # Opção 3: chromedriver já no PATH
    try:
        driver = webdriver.Chrome(service=Service("chromedriver"), options=opts)
        driver.execute_script(
            "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
        )
        return driver
    except Exception as e:
        erros.append(f"chromedriver no PATH: {e}")

    print("\n❌ Não foi possível iniciar o ChromeDriver:")
    for err in erros:
        print(f"   • {err}")
    print("\n💡 Solução:")
    print("   pip install --upgrade selenium webdriver-manager")
    sys.exit(1)


# ── Busca URLs via Bing Images ─────────────────────────────────────

def buscar_urls_bing(driver: webdriver.Chrome, query: str, n: int) -> list[str]:
    """Abre Bing Images no browser real e extrai URLs das imagens."""
    url_busca = (
        "https://www.bing.com/images/search"
        f"?q={requests.utils.quote(query)}&form=HDRSC2&first=1"
    )
    driver.get(url_busca)
    time.sleep(2)

    urls: list[str] = []
    tentativas_scroll = 0
    max_scroll = 8

    while len(urls) < n and tentativas_scroll < max_scroll:
        # extrai URLs das thumbnails e imagens originais
        elementos = driver.find_elements(By.CSS_SELECTOR, "a.iusc")
        for el in elementos:
            try:
                m = el.get_attribute("m")
                if not m:
                    continue
                data = json.loads(m)
                murl = data.get("murl", "")
                if murl and murl not in urls and murl.startswith("http"):
                    urls.append(murl)
            except Exception:
                continue

        if len(urls) >= n:
            break

        # rola a página para carregar mais imagens
        driver.execute_script("window.scrollBy(0, 2000)")
        time.sleep(1.5)

        # clica em "Ver mais imagens" se disponível
        try:
            btn = driver.find_element(By.CSS_SELECTOR, ".btn_seemore, #bop_container button")
            btn.click()
            time.sleep(2)
        except Exception:
            pass

        tentativas_scroll += 1

    return list(dict.fromkeys(urls))[:n * 2]   # dedup + margem


def buscar_urls_google(driver: webdriver.Chrome, query: str, n: int) -> list[str]:
    """Fallback: Google Images."""
    url_busca = (
        "https://www.google.com/search"
        f"?q={requests.utils.quote(query)}&tbm=isch&hl=pt-BR"
    )
    driver.get(url_busca)
    time.sleep(2)

    # fecha popup de cookies se aparecer
    for seletor in ["button[id*='accept']", "button[aria-label*='Aceitar']",
                    "button[aria-label*='Accept']"]:
        try:
            driver.find_element(By.CSS_SELECTOR, seletor).click()
            time.sleep(1)
            break
        except Exception:
            pass

    urls: list[str] = []
    tentativas_scroll = 0

    while len(urls) < n and tentativas_scroll < 6:
        # clica nas thumbnails para obter URL original
        thumbs = driver.find_elements(By.CSS_SELECTOR, "img.Q4LuWd, img.YQ4gaf")
        for thumb in thumbs:
            if len(urls) >= n:
                break
            try:
                src = thumb.get_attribute("src") or thumb.get_attribute("data-src") or ""
                # pega a URL da imagem completa via data-ou ou src com tamanho original
                parent = thumb.find_element(By.XPATH, "./ancestor::a[1]")
                href = parent.get_attribute("href") or ""
                # extrai imgurl do href do Google
                m = re.search(r"imgurl=([^&]+)", href)
                if m:
                    img_url = requests.utils.unquote(m.group(1))
                    if img_url not in urls and img_url.startswith("http"):
                        urls.append(img_url)
                        continue
                # usa src como último recurso (será thumbnail)
                if src and src.startswith("http") and src not in urls:
                    urls.append(src)
            except Exception:
                continue

        driver.execute_script("window.scrollBy(0, 3000)")
        time.sleep(1.5)
        tentativas_scroll += 1

    return list(dict.fromkeys(urls))[:n * 2]


# ── Download de imagem individual ─────────────────────────────────

def baixar_imagem(url: str, caminho: Path, timeout: int = 14) -> bool:
    try:
        r = requests.get(url, headers=IMG_HEADERS, timeout=timeout, stream=True)
        r.raise_for_status()
        img = Image.open(BytesIO(r.content))
        if img.width < MIN_W or img.height < MIN_H:
            return False
        if img.mode not in ("RGB", "L"):
            img = img.convert("RGB")
        img.save(caminho.with_suffix(".jpg"), "JPEG", quality=90)
        return True
    except Exception:
        return False


# ── Pipeline por viatura ───────────────────────────────────────────

def baixar_viatura(
    driver: webdriver.Chrome,
    nome: str,
    queries: list[str],
    pasta: Path,
    n: int,
) -> list[Path]:
    pasta_raw = pasta / "_raw"
    pasta_raw.mkdir(parents=True, exist_ok=True)

    urls_vistas: set[str] = set()
    arquivos:    list[Path] = []
    idx = 0

    for query in queries:
        if len(arquivos) >= n:
            break
        print(f"  🔍 {query}")

        urls = buscar_urls_bing(driver, query, n)
        if len(urls) < 5:
            print(f"    ⚠  Bing retornou {len(urls)}, tentando Google…")
            urls += buscar_urls_google(driver, query, n)

        with tqdm(
            total=min(len(urls), n - len(arquivos)),
            desc=f"    ↓ {nome[:18]}",
            unit="img",
            leave=False,
        ) as bar:
            for url in urls:
                if len(arquivos) >= n:
                    break
                if url in urls_vistas:
                    continue
                urls_vistas.add(url)

                dest = pasta_raw / f"{nome}_{idx:04d}"
                if baixar_imagem(url, dest):
                    arquivos.append(dest.with_suffix(".jpg"))
                    idx += 1
                    bar.update(1)
                time.sleep(0.2)

    return arquivos


# ── Filtro pHash ───────────────────────────────────────────────────

def remover_similares(pasta: Path, threshold: int) -> tuple[int, int]:
    pasta_raw   = pasta / "_raw"
    pasta_final = pasta / "imagens"
    pasta_final.mkdir(exist_ok=True)

    arquivos = sorted(pasta_raw.glob("*.jpg"))
    total = len(arquivos)
    if total == 0:
        return 0, 0

    hashes: list[imagehash.ImageHash] = []
    mantidos = 0

    print(f"  🔎 Filtrando {total} imagens (threshold={threshold})…")
    for arq in tqdm(arquivos, desc="    pHash", unit="img", leave=False):
        try:
            ph = imagehash.phash(Image.open(arq), hash_size=HASH_SIZE)
        except Exception:
            arq.unlink(missing_ok=True)
            continue
        if not any(abs(ph - h) <= threshold for h in hashes):
            hashes.append(ph)
            shutil.copy2(arq, pasta_final / f"{arq.stem}_ok.jpg")
            mantidos += 1

    shutil.rmtree(pasta_raw, ignore_errors=True)
    return total, mantidos


# ── Relatório ──────────────────────────────────────────────────────

def gerar_relatorio(raiz: Path, resultados: dict, threshold: int) -> None:
    rel = {
        "threshold_similaridade": threshold,
        "viaturas": {},
        "total_baixadas": 0,
        "total_mantidas": 0,
    }
    for nome, (b, m) in resultados.items():
        rel["viaturas"][nome] = {"baixadas": b, "mantidas": m, "removidas": b - m}
        rel["total_baixadas"] += b
        rel["total_mantidas"] += m
    rel["total_removidas"] = rel["total_baixadas"] - rel["total_mantidas"]

    (raiz / "relatorio.json").write_text(
        json.dumps(rel, indent=2, ensure_ascii=False)
    )

    sep = "═" * 58
    print(f"\n{sep}")
    print("📊  RELATÓRIO FINAL")
    print(sep)
    for nome, d in rel["viaturas"].items():
        print(
            f"  {nome:<22} baixadas={d['baixadas']:>3}  "
            f"mantidas={d['mantidas']:>3}  removidas={d['removidas']:>3}"
        )
    print("─" * 58)
    print(
        f"  {'TOTAL':<22} baixadas={rel['total_baixadas']:>3}  "
        f"mantidas={rel['total_mantidas']:>3}  "
        f"removidas={rel['total_removidas']:>3}"
    )
    print(sep)
    print(f"\n📁 Repositório : {raiz.resolve()}")
    print(f"📄 Relatório   : {(raiz / 'relatorio.json').resolve()}")


# ── Entry point ────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description="Download + filtragem de imagens de viaturas blindadas do EB."
    )
    ap.add_argument("--images",    type=int,  default=30,
                    help="Imagens por viatura (padrão: 30)")
    ap.add_argument("--threshold", type=int,  default=8,
                    help="Limiar pHash 0-64 (padrão: 8)")
    ap.add_argument("--output",    type=str,  default="repositorio_viaturas",
                    help="Pasta raiz do repositório")
    ap.add_argument("--visible",   action="store_true",
                    help="Abre o Chrome visível (útil para debug)")
    args = ap.parse_args()

    raiz = Path(args.output)
    raiz.mkdir(parents=True, exist_ok=True)

    print("╔══════════════════════════════════════════════════════════╗")
    print("║   Repositório de Viaturas Blindadas – Exército Brasileiro ║")
    print("╚══════════════════════════════════════════════════════════╝")
    print(f"  Imagens/viatura : {args.images}")
    print(f"  Threshold pHash : {args.threshold}")
    print(f"  Destino         : {raiz.resolve()}")
    print(f"  Modo browser    : {'visível' if args.visible else 'headless'}\n")

    print("🌐 Iniciando Chrome…")
    driver = criar_driver(visivel=args.visible)

    try:
        resultados = {}

        for nome, queries in VIATURAS.items():
            print(f"\n{'─'*58}")
            print(f"🚛  {nome.replace('_', ' ')}")
            print(f"{'─'*58}")

            arquivos = baixar_viatura(
                driver, nome, queries, raiz / nome, args.images
            )
            print(f"  ✅ {len(arquivos)} imagens baixadas")

            total, mantidas = remover_similares(raiz / nome, args.threshold)
            print(
                f"  🗑  {total - mantidas} similares removidas → {mantidas} únicas"
            )

            resultados[nome] = (len(arquivos), mantidas)

    finally:
        driver.quit()
        print("\n🔒 Browser encerrado.")

    gerar_relatorio(raiz, resultados, args.threshold)


if __name__ == "__main__":
    main()
