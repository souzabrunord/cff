import os
import time
import re
import json
import platform
import numpy as np
import cv2
import pytesseract
import requests
import unicodedata
from math import radians, cos, sin, asin, sqrt 
from PIL import Image, ImageOps
from bs4 import BeautifulSoup
import streamlit as st
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning) 

# --- AJUSTE INTELIGENTE DO TESSERACT ---
if platform.system() == "Windows":
    pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

headers = {
    'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    "Referer": "https://site.cff.org.br/farmaceutico/pesquisa"
}

UF_CODES = {
    'AC': 12, 'AL': 27, 'AP': 16, 'AM': 13, 'BA': 29, 'CE': 23, 'DF': 53,
    'ES': 32, 'GO': 52, 'MA': 21, 'MT': 51, 'MS': 50, 'MG': 31, 'PA': 15,
    'PB': 25, 'PR': 41, 'PE': 26, 'PI': 22, 'RJ': 33, 'RN': 24, 'RS': 43,
    'RO': 11, 'RR': 14, 'SC': 42, 'SP': 35, 'SE': 28, 'TO': 17
}

# ---------------- CACHE DO MAPA (BAIXA SÓ 1 VEZ) ----------------
@st.cache_data(show_spinner=False)
def carregar_mapa_ibge():
    try:
        resp = requests.get("https://raw.githubusercontent.com/kelvins/Municipios-Brasileiros/main/json/municipios.json", timeout=15)
        if resp.status_code == 200:
            return resp.json()
        return None
    except:
        return None

# ---------------- FUNÇÕES DE DISTÂNCIA E ORDENAÇÃO ----------------
def remover_acentos(txt):
    return unicodedata.normalize('NFKD', str(txt)).encode('ASCII', 'ignore').decode('ASCII').upper().strip()

def calcular_distancia_km(lat1, lon1, lat2, lon2):
    R = 6371
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    a = sin(dLat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dLon/2)**2
    return R * (2 * asin(sqrt(a)))

def ordenar_cidades_por_distancia(lista_cidades, estado_sigla, cidade_referencia, ui_status):
    if not cidade_referencia:
        return lista_cidades, []
        
    cidade_ref_norm = remover_acentos(cidade_referencia)
    codigo_alvo = UF_CODES.get(estado_sigla.upper())
    
    # Busca os dados cacheados em vez de fazer download novamente
    dados_municipios = carregar_mapa_ibge()
    
    if not dados_municipios:
        ui_status.error("❌ Erro ao baixar o mapa de distâncias do IBGE (Bloqueio de Servidor). Indo por Ordem Alfabética.")
        time.sleep(3)
        return lista_cidades, []

    mapa_coordenadas = {}
    lat_alvo, lon_alvo = None, None
    cidade_encontrada_mapa = ""

    for mun in dados_municipios:
        if mun["codigo_uf"] == codigo_alvo:
            nome_norm = remover_acentos(mun["nome"])
            mapa_coordenadas[nome_norm] = (mun["latitude"], mun["longitude"])
            
            if nome_norm == cidade_ref_norm or cidade_ref_norm in nome_norm:
                lat_alvo, lon_alvo = mun["latitude"], mun["longitude"]
                cidade_encontrada_mapa = nome_norm

    if lat_alvo is not None and lon_alvo is not None:
        cidades_com_distancia = []
        cidades_sem_coordenada = []

        for c in lista_cidades:
            c_norm = remover_acentos(c)
            if c_norm in mapa_coordenadas:
                lat_c, lon_c = mapa_coordenadas[c_norm]
                dist = calcular_distancia_km(lat_alvo, lon_alvo, lat_c, lon_c)
                cidades_com_distancia.append((c, dist))
            else:
                cidades_sem_coordenada.append(c)

        cidades_com_distancia.sort(key=lambda x: x[1])
        nova_lista = [item[0] for item in cidades_com_distancia] + cidades_sem_coordenada
        top_5 = [f"{item[0]} ({round(item[1], 1)} km)" for item in cidades_com_distancia[:5]]
        
        ui_status.success(f"📍 **Rota Traçada (Base: {cidade_encontrada_mapa})!**\n\n*Próximas paradas:* {', '.join(top_5)}...")
        return nova_lista, top_5
    else:
        ui_status.warning(f"⚠️ A cidade '{cidade_referencia}' não foi encontrada no mapa do IBGE. Verifique a digitação. Iniciando varredura por Ordem Alfabética.")
        time.sleep(3)
        return lista_cidades, []

# ---------------- FUNÇÕES DE APOIO ----------------
def resolver_captcha(caminho_imagem):
    try:
        img_captcha = Image.open(caminho_imagem)
        img_np = np.array(img_captcha.convert("RGB"))
        clean_color = cv2.medianBlur(img_np, 3)
        mask_bg_color = (clean_color[:,:,0] > 180) & (clean_color[:,:,1] > 180) & (clean_color[:,:,2] > 180)
        final_bin = np.zeros_like(clean_color)
        final_bin[mask_bg_color] = [255, 255, 255] 
        final_bin[~mask_bg_color] = [0, 0, 0]      
        
        img_captcha = Image.fromarray(final_bin).convert("L")
        inverted_bg = ImageOps.invert(img_captcha)
        bbox = inverted_bg.getbbox()
        if bbox: img_captcha = img_captcha.crop(bbox)
        
        img_captcha = ImageOps.expand(img_captcha, border=10, fill=255)
        img_np_resized = cv2.resize(np.array(img_captcha.convert("RGB"))[:,:,::-1], None, fx=3, fy=3, interpolation=cv2.INTER_LANCZOS4)
        
        config_tess = r'--psm 7 -c tessedit_char_whitelist=0123456789+-'
        texto_limpo = pytesseract.image_to_string(img_np_resized, config=config_tess).strip().replace(" ", "").replace("+-", "+").replace("-+", "-").replace("++", "+").replace("--", "-")
        
        match = re.search(r'(\d+)([+\-])(\d+)', texto_limpo)
        if match:
            num1, operador, num2 = int(match.group(1)), match.group(2), int(match.group(3))
            if num1 <= 50 and num2 <= 50:
                resultado = int(eval(f"{num1}{operador}{num2}"))
                if 0 <= resultado <= 50: return str(resultado)
        return None
    except: return None

# ---------------- MOTOR PRINCIPAL ----------------
def buscar_profissional_por_raio(nome_pesquisa, estado_sigla, cidade_referencia, ui_status, ui_alerts):
    caminho_json = os.path.join("ufs_json", f"{estado_sigla.upper()}.json")
    if not os.path.exists(caminho_json):
        return None, f"Erro: Arquivo {estado_sigla.upper()}.json não encontrado na pasta ufs_json."

    with open(caminho_json, "r", encoding="utf-8") as f:
        cidades_json = json.load(f)
    
    lista_cidades = [c["NomeCidade"].replace("'", '') for c in cidades_json]
    
    lista_cidades, top_5 = ordenar_cidades_por_distancia(lista_cidades, estado_sigla, cidade_referencia, ui_status)
    
    session = requests.Session()
    session.verify = False
    session.headers.update(headers)
    caminho_img_temp = "captcha_raio_temp.png"

    progress_bar = st.progress(0)
    total_cidades = len(lista_cidades)

    for i, cidade in enumerate(lista_cidades):
        progress_bar.progress((i + 1) / total_cidades)
        ui_status.info(f"🔎 Varrendo cidade: **{cidade.upper()}** ({i+1}/{total_cidades})...")
        
        tentativa_atual = 0
        encontrou_profissional = False
        soup_resultado = None

        while tentativa_atual < 5:
            try:
                resp = session.get("https://site.cff.org.br/farmaceutico/pesquisa", timeout=15)
                soup = BeautifulSoup(resp.text, 'html.parser')
                
                token_tag = soup.find("input", {"name": "_token"})
                if not token_tag: break
                token = token_tag.get("value")
                
                img_tag = soup.select_one("#Imgcaptcha img") or soup.find("img", src=re.compile(r'captcha', re.I))
                if not img_tag: break
                    
                img_url = img_tag.get('src')
                if not img_url.startswith('http'):
                    img_url = "https://site.cff.org.br" + (img_url if img_url.startswith('/') else '/' + img_url)
                    
                resp_img = session.get(img_url, timeout=15)
                with open(caminho_img_temp, "wb") as f: f.write(resp_img.content)
                
                valor_captcha = resolver_captcha(caminho_img_temp)
                if not valor_captcha:
                    tentativa_atual += 1
                    continue

                payload = {'_token': token, 'uf': estado_sigla, 'cidade': cidade, 'categoria': 'farmaceutico', 'nome': nome_pesquisa, 'captcha': valor_captcha, 'search': '1'}
                resp_post = session.post("https://site.cff.org.br/farmaceutico/pesquisar", data=payload, timeout=20)
                texto_pagina = resp_post.text.lower()
                soup_post = BeautifulSoup(resp_post.text, 'html.parser')
                
                if "captcha está incorreto" in texto_pagina or "inválido" in texto_pagina:
                    tentativa_atual += 1
                    continue
                
                if "favor informar os campos obrigatórios" in texto_pagina:
                    tentativa_atual += 1
                    continue

                if "não foi possível encontrar um profissional" in texto_pagina:
                    break 
                
                registros = soup_post.find_all("div", class_="team-info")
                if registros:
                    encontrou_profissional = True
                    soup_resultado = soup_post
                    break
                    
                tentativa_atual += 1
                
            except:
                tentativa_atual += 1
                time.sleep(1)
            
        if tentativa_atual >= 5:
            ui_alerts.warning(f"⚠️ Pulei a cidade de {cidade.upper()} porque o site não respondeu ou o captcha falhou 5 vezes.")
            continue

        if encontrou_profissional and soup_resultado:
            profissionais_encontrados = []
            for r in soup_resultado.find_all("div", class_="team-info"):
                linhas = [l.strip() for l in r.text.split("\n") if l.strip()]
                if len(linhas) >= 2:
                    raw_nome = linhas[0].strip().upper()
                    raw_crf = next((l.replace("CRF", "").replace(":", "").strip() for l in linhas if "CRF" in l.upper()), "")
                    raw_situacao = next((l.replace("SITUAÇÃO:", "").strip() for l in linhas if "SITUAÇÃO:" in l.upper()), "")
                    raw_data = next((l.replace("DATA DA INSCRIÇÃO:", "").strip() for l in linhas if "DATA DA INSCRIÇÃO:" in l.upper()), "")
                    
                    profissionais_encontrados.append({
                        "Nome": raw_nome,
                        "CRF": raw_crf,
                        "Cidade de Registro": cidade.upper(),
                        "Estado": estado_sigla.upper(),
                        "Situação": raw_situacao,
                        "Inscrição": raw_data
                    })

            if os.path.exists(caminho_img_temp): os.remove(caminho_img_temp)
            return profissionais_encontrados, "Sucesso"

    if os.path.exists(caminho_img_temp): os.remove(caminho_img_temp)
    return None, f"A busca varreu as {total_cidades} cidades, mas '{nome_pesquisa}' não foi encontrado(a)."


# =======================================================
# INTERFACE GRÁFICA WEB (STREAMLIT)
# =======================================================
st.set_page_config(page_title="Radar CFF", page_icon="📡")

st.title("Busca por Região")
st.write("Insira o nome exato e a cidade base. O robô varrerá a região em formato de raio (do mais próximo ao mais distante).")

col1, col2, col3 = st.columns([1, 2, 2])
with col1:
    estado_input = st.text_input("UF (ex: SP):").upper()
with col2:
    cidade_input = st.text_input("Cidade Base (ex: Campinas):").upper()
with col3:
    nome_input = st.text_input("Nome Exato do Farmacêutico:").upper()

if st.button("Buscar"):
    if not estado_input or not cidade_input or not nome_input:
        st.warning("⚠️ Preencha todos os campos para prosseguir.")
    else:
        status_container = st.empty()
        alerts_container = st.container()
        
        resultados, msg = buscar_profissional_por_raio(nome_input, estado_input, cidade_input, status_container, alerts_container)
        
        status_container.empty()
        
        if resultados:
            st.success("✅ Alvo detectado e extraído com sucesso!")
            st.dataframe(resultados, use_container_width=True)
        else:
            st.error(msg)
