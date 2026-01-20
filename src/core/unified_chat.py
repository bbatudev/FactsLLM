# -*- coding: utf-8 -*-
"""
Unified Chat System (Birlesik Sohbet Sistemi)
=============================================
Bu dosya, projenin "Beyni" olarak calisir. RAG (Bilgi Getirme), CoT (Zincirleme Dusunce)
ve Web Search (Internet Aramasi) ozelliklerini tek bir cati altinda toplar.

Basitce anlatmak gerekirse:
1. Kullanici bir soru sorar.
2. "Router" (Yonlendirici) bu sorunun ne tur bir soru olduguna karar verir (Sohbet mi? Bilgi mi? Guncel haber mi?).
3. Eger bilgi soruysa "RAG" devreye girer, kutuphaneden (Vector Database) bilgi toplar.
4. Toplanan bilgiler "CoT" ile analiz edilir ve son kullaniciya sunulur.

Ozellikler:
- Model yukleme (32GB RAM yerine 4-bit ile daha az RAM kullanimi)
- Dual RAG (Hem Teyit.org hem Wikipedia verisi kullanilir)
- Akilli Yonlendirme (Sohbet vs Bilgi sorusu ayrimi)
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from sentence_transformers import SentenceTransformer
import faiss
import json
import os
import sys
import gc
import time
import mmap
import re
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
from contextlib import contextmanager
import numpy as np

# ============================================================================
# BÖLÜM 1: AYARLAR
# ============================================================================

# başlamadan kısa bir özet ,  aslında 3 tane veri kaynağı var,  1 tane intent sağlayıcı var ,,

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model Yolları
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "base_model")  # lama 3 8parametrelik
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "adapters", "checkpoint_cot_lora_ultra") # son yaptığımız fine tune dosyası 

# Wikipedia RAG - DEVRE DIŞI (Veri kalitesi düşük: tab kirliliği, bağlam eksikliği, etiket yok)
USE_WIKI_RAG = False  # Şimdilik sadece Teyit.org + FEVER Gold kullan

WIKI_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "indices", "faiss_index_fast.bin")
WIKI_DOCS_PATH = os.path.join(PROJECT_ROOT, "data", "indices", "documents.jsonl")  
WIKI_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
 

 
# Teyit.org RAG (Türkçe) - TEMİZLENMİŞ VERSİYON
TEYIT_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "indices", "teyit_index.bin")
TEYIT_DOCS_PATH = os.path.join(PROJECT_ROOT, "data", "indices", "teyit_documents_clean.json")
TEYIT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# FEVER Gold (EN) - FACTS
FEVER_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "indices", "fever_gold.index")
FEVER_DOCS_PATH = os.path.join(PROJECT_ROOT, "data", "indices", "fever_gold_docs.json")
FEVER_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Sistem Ayarları
MAX_NEW_TOKENS = 3000   #512'de gayet stabil çalışıyor ancak ben 1024 çektim olmazsa dönmeyi unutma+++
USE_LORA = True  # LoRA aktif

# Threshold Değerleri
TEYIT_THRESHOLD = 0.45
WIKI_THRESHOLD = 1.0
MIN_DOCS_FOR_VERDICT = 1

# ============================================================================
# BÖLÜM 2: VERİ YAPILARI
# ============================================================================

class Verdict(Enum):
    """Doğrulama sonucu"""
    TRUE = "DOĞRUDUR"
    FALSE = "YANLIŞTIR"
    UNVERIFIABLE = "DOĞRULAMAZ"
    UNKNOWN = "BİLİNMİYOR"


@dataclass
class VerdictResult:
    """CoT analiz sonucu"""
    verdict: Verdict
    confidence: str  # YÜKSEK/ORTA/DÜŞÜK
    supporting_points: List[str]
    refuting_points: List[str]
    explanation: str
    sources: List[str]
    raw_response: str


@dataclass
class ChatResponse:
    """Kullanıcıya dönecek son cevap"""
    verdict: str
    explanation: str
    sources: List[str]
    source_type: str  # RAG/LLM/WEB/CHAT
    processing_time: float
    confidence: str


# ============================================================================
# BÖLÜM 3: WEB SEARCH
# ============================================================================

try:
    from duckduckgo_search import DDGS # duck duck go motorunu kullanmaktayız
    HAS_DDGS = True
except ImportError:
    HAS_DDGS = False


def search_web(query: str, max_results: int = 3) -> List[Dict[str, str]]:
    """
    DuckDuckGo ile web araması yapar.
    
    Args:
        query: Arama sorgusu
        max_results: Maksimum sonuç sayısı
        
    Returns:
        Liste of dict: [{"title": ..., "body": ..., "href": ...}, ...]
    """
    if not HAS_DDGS:
        print("⚠️ Web search kullanılamıyor (duckduckgo_search paketi yok)")
        return []

    try:
        print(f"\n🔍 [DEBUG] Web Araması Başlatılıyor: '{query}'")
        with DDGS() as ddgs:
            # Önce varsayılan backend, olmazsa html backend dene
            try:
                results = list(ddgs.text(query, max_results=max_results))
            except Exception as e_inner:
                print(f"⚠️ Varsayılan backend hatası: {e_inner}, 'html' backend deneniyor...")
                results = list(ddgs.text(query, max_results=max_results, backend='html'))
            except Exception as e_inner:
                print(f"⚠️ Varsayılan backend hatası: {e_inner}, 'html' backend deneniyor...")
                results = list(ddgs.text(query, max_results=max_results, backend='html'))
            
        print(f"🔍 [DEBUG] Bulunan sonuç sayısı: {len(results)}")
        
        formatted = []
        for r in results:
            formatted.append({
                "title": r.get("title", ""),
                "body": r.get("body", ""),
                "href": r.get("href", r.get("link", ""))
            })
        return formatted
        
    except Exception as e:
        print(f"❌ Web search Kritik Hata: {e}")
        import traceback
        traceback.print_exc()
        return []


# ============================================================================
# BÖLÜM 4: ROUTER (Niyet Sınıflandırma - Yönlendirici)
# ============================================================================
# Burası ofisin "Resepsiyonu" gibidir.
# Gelen sorunun "Geyik muhabbeti mi?", "Bilgi sorusu mu?" yoksa "Güncel haber mi?"
# olduğuna karar verir ve ilgili departmana yönlendirir.

def classify_intent(query: str, tokenizer, model) -> str:
    """
    Kullanıcı sorgusunun niyetini belirler: CHAT, RAG veya WEB.
    
    Args:
        query: Kullanıcı sorusu
        tokenizer, model: Yüklü Llama modeli
        
    Returns:
        str: 'CHAT', 'RAG' veya 'WEB'
    """
    # 1. Manuel Override (Hız ve Kesinlik için)
    q_lower = query.lower()
    web_keywords = ["web search", "internet", "google", "arama yap", "araştır", "son dakika", "bugün", "webden", "internetten"]  # bilerek hardcode ekledik,
    # zararı yok bu kadarın 
    if any(k in q_lower for k in web_keywords):
        return "WEB"
        
    # 2. LLM ile Sınıflandırma


    # aşağıdaki prmpotun 1. kısmı düzgün açlışmıyor sadece evet hayır ile doğrulanabilen ya da teyit gerektiren sorularda stabil çalışıyor bunun da tek sebebi 
    # 3. finetune formatının sadece doğruluğa yönelik olması olabilir o yüzden kim sorularında belki ek olarak basit bir modelin websearch ile ya da base modelin
    # devreye girip cevap vermesi gibi bir ek özellik olabilir 
    system_prompt = """You are a routing assistant. Classify the user question into one of these three categories:

1. RAG: For questions about facts, claims, verification, companies, policies, history, science, biography, geography, or general knowledge. Examples: 
   - "Who is Einstein?"
   - "Capital of France"
   - "Did Meta close fact-checking program?"
   - "Is this claim true?"
   
2. WEB: For questions about RIGHT NOW, TODAY, or VERY RECENT events (last 24 hours). Examples:
   - "Dollar price today"
   - "Weather right now"
   - "Match score today"
   
3. CHAT: For greetings, philosophy, coding, translation, or casual conversation. Examples:
   - "Hello"
   - "Write a Python script"
   - "How are you?"

Return ONLY the category name: RAG, WEB, or CHAT. No explanation."""

    user_prompt = f"Question: {query}\nCategory:"
    
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]
    
    input_ids = tokenizer.apply_chat_template(
        messages, 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(model.device)
    
    terminators = [
        tokenizer.eos_token_id,
        tokenizer.convert_tokens_to_ids("<|eot_id|>")
    ]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=10,
            eos_token_id=terminators,
            do_sample=False,
            temperature=0.0
        )
    
    response = outputs[0][input_ids.shape[-1]:]
    category = tokenizer.decode(response, skip_special_tokens=True).strip().upper()
    
    if "RAG" in category: return "RAG"
    if "WEB" in category: return "WEB"
    if "CHAT" in category: return "CHAT"
    
    return "CHAT"  # Fallback

# websearchten adaptörü çıakrıp da areama yapmak 
# ============================================================================
# BÖLÜM 5: BELLEK YÖNETİMİ
# ============================================================================

_encoder_cache = {}


@contextmanager
def memory_efficient_context():
    """Context manager for memory-efficient operations"""
    try:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        yield
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def get_cached_encoder(model_name: str) -> SentenceTransformer:
    """Get cached sentence transformer encoder"""
    if model_name not in _encoder_cache:
        print(f"Loading encoder: {model_name}")
        encoder = SentenceTransformer(model_name, device='cpu')
        if torch.cuda.is_available():
            try:
                encoder = encoder.to('cuda')
                print(f"Encoder {model_name} moved to GPU")
            except Exception as e:
                print(f"Warning: Could not move encoder to GPU: {e}")
        _encoder_cache[model_name] = encoder
    return _encoder_cache[model_name]


def cleanup_resources():
    """Clean up global resources and free memory"""
    global _encoder_cache
    
    print("Temizlik yapılıyor...")
    for model_name, encoder in _encoder_cache.items():
        del encoder
    _encoder_cache.clear()
    
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("Kaynaklar temizlendi.")


# ============================================================================
# BÖLÜM 6: RAG SİSTEMİ YÜKLEME (Başlatma Tuşu)
# ============================================================================
# Bu bölüm sanki bilgisayarı açar gibi tüm sistemi hazır hale getirir.
# Burada iki farklı "Zeka" (Transformer) işe alınır:
# 1. YAZAR/DEDEKTİF (LLM - Llama): Zekidir, konuşur, karar verir ama yavaştır. (Patron)
# 2. KÜTÜPHANECİ (Embedding - MiniLM): Konuşamaz ama çok hızlı okur ve aradığını bulur. (Eleman)
# 
# Neden 2 tane? Çünkü Patronu (LLM) kütüphane rafları arasında koşturmak istemeyiz, o masasında oturup karar vermeli.
# Kitapları getir-götür işini Kütüphaneci (Embedding) yapar.

def load_rag_system() -> Dict:
    """
    RAG sistemini yükler: Model + Encoders + Indexes
    
    Returns:
        Dict: {"tokenizer", "model", "wiki": {...}, "teyit": {...}}
    """
    print("=" * 60)
    print("RAG Sistemi ve Modeller Baslatiliyor...")
    print("=" * 60)
    
    with memory_efficient_context():
        # 1. LLM (Base Model + LoRA)
        print(f"\n[LLM] Yukleniyor: {BASE_MODEL_PATH}")
        
        tokenizer = AutoTokenizer.from_pretrained(
            BASE_MODEL_PATH,
            use_fast=True,
            padding_side='left',
            truncation_side='left'
        )
        
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_storage=torch.uint8
        )
        
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,
            max_memory={0: "10GB", "cpu": "20GB"},
            torch_dtype=torch.float16,
            trust_remote_code=False,
            use_cache=True,
            attn_implementation="sdpa"
        )
        
        if hasattr(base_model, 'gradient_checkpointing_enable'):
            base_model.gradient_checkpointing_enable()
        
        if USE_LORA:
            from peft import PeftModel
            print(f"[LoRA] Adaptör: {ADAPTER_PATH}")
            model = PeftModel.from_pretrained(
                base_model,
                ADAPTER_PATH,
                torch_dtype=torch.float16
            )
            print("[LoRA] Adaptor basariyla yuklendi!")
        else:
            model = base_model
            print("[UYARI] LoRA DEVRE DISI - Base model kullaniliyor")
        
        model.eval()

        # 2. Wikipedia RAG (EN) - OPSİYONEL
        wiki_encoder = None
        wiki_index = None
        
        if USE_WIKI_RAG:
            print("\n[Wiki-RAG] Yukleniyor...")
            wiki_encoder = get_cached_encoder(WIKI_EMBED_MODEL)
            
            if os.path.exists(WIKI_INDEX_PATH):
                try:
                    wiki_index = faiss.read_index(WIKI_INDEX_PATH, faiss.IO_FLAG_MMAP)
                    size_mb = os.path.getsize(WIKI_INDEX_PATH) / 1024 / 1024
                    print(f"   Wiki indeksi yuklendi ({size_mb:.1f} MB)")
                except Exception as e:
                    print(f"   Warning: {e}")
                    wiki_index = faiss.read_index(WIKI_INDEX_PATH)
            else:
                print(f"   [UYARI] Wiki indeksi bulunamadi: {WIKI_INDEX_PATH}")
        else:
            print("\n[Wiki-RAG] DEVRE DISI (USE_WIKI_RAG = False)")

        # 3. Teyit RAG (TR)
        print("\n[Teyit-RAG] Yukleniyor...")
        teyit_encoder = get_cached_encoder(TEYIT_EMBED_MODEL)
        
        teyit_index = None
        teyit_docs_data = []
        if os.path.exists(TEYIT_INDEX_PATH):
            try:
                teyit_index = faiss.read_index(TEYIT_INDEX_PATH, faiss.IO_FLAG_MMAP)
                size_mb = os.path.getsize(TEYIT_INDEX_PATH) / 1024 / 1024
                print(f"   Teyit indeksi yuklendi ({size_mb:.1f} MB)")
            except Exception as e:
                print(f"   Warning: {e}")
                teyit_index = faiss.read_index(TEYIT_INDEX_PATH)
            
            # Teyit dokümanlarını yükle
            try:
                with open(TEYIT_DOCS_PATH, "r", encoding="utf-8") as f:
                    teyit_docs_data = json.load(f)
                print(f"   Teyit dokümanlari yuklendi ({len(teyit_docs_data)} adet)")
            except Exception as e:
                print(f"   Warning: {e}")
        else:
            print(f"   [UYARI] Teyit indeksi bulunamadi: {TEYIT_INDEX_PATH}")

        # 4. FEVER Gold RAG (Yeni - Unified veya Legacy)
        print("\n[FEVER-Gold] RAG Yukleniyor...")
        
        # YENİ: Multilingual Unified Index Kontrolü
        unified_fever_index_path = os.path.join(PROJECT_ROOT, "data", "indices", "fever_gold_multilingual.index")
        unified_fever_docs_path = os.path.join(PROJECT_ROOT, "data", "indices", "fever_docs_multilingual.json")
        
        fever_index = None
        fever_docs_data = []
        # Encoder başlangıçta None, duruma göre atanacak
        fever_encoder = None 
        
        if os.path.exists(unified_fever_index_path):
            print("   [INFO] Unified Multilingual Index bulundu. Teyit Encoder kullanilacak (RAM Tasarrufu).")
            # Teyit encoder'ını tekrar kullan (Aynı model: paraphrase-multilingual-MiniLM-L12-v2)
            fever_encoder = teyit_encoder 
            
            try:
                fever_index = faiss.read_index(unified_fever_index_path, faiss.IO_FLAG_MMAP)
                size_mb = os.path.getsize(unified_fever_index_path) / 1024 / 1024
                print(f"   FEVER Unified indeksi yuklendi ({size_mb:.1f} MB)")
                
                with open(unified_fever_docs_path, "r", encoding="utf-8") as f:
                    fever_docs_data = json.load(f)
                print(f"   FEVER dokümanlari yuklendi ({len(fever_docs_data)} adet)")
            except Exception as e:
                print(f"   [HATA] Unified index yuklenemedi: {e}")
                
        # Eğer Unified yoksa veya yüklenemediyse ESKİ (Legacy) sisteme dön
        if fever_index is None:
            if os.path.exists(FEVER_INDEX_PATH):
                print("   [INFO] Legacy English Index kullaniliyor (Ekstra RAM kullanacak).")
                fever_encoder = get_cached_encoder(FEVER_EMBED_MODEL)
                try:
                    fever_index = faiss.read_index(FEVER_INDEX_PATH, faiss.IO_FLAG_MMAP)
                    size_mb = os.path.getsize(FEVER_INDEX_PATH) / 1024 / 1024
                    print(f"   FEVER Legacy indeksi yuklendi ({size_mb:.1f} MB)")
                    
                    with open(FEVER_DOCS_PATH, "r", encoding="utf-8") as f:
                        fever_docs_data = json.load(f)
                    print(f"   FEVER dokümanlari yuklendi ({len(fever_docs_data)} adet)")
                except Exception as e:
                     print(f"   Warning: {e}")
            else:
                print(f"   [UYARI] Hicbir FEVER indeksi bulunamadi.")

    print("\n" + "=" * 60)
    print("[TAMAM] TUM SISTEMLER HAZIR!")
    print("=" * 60 + "\n")
    
    return {
        "tokenizer": tokenizer,
        "model": model,
        "wiki": {"encoder": wiki_encoder, "index": wiki_index},
        "teyit": {"encoder": teyit_encoder, "index": teyit_index, "docs": teyit_docs_data},
        "fever": {"encoder": fever_encoder, "index": fever_index, "docs": fever_docs_data}
    }


# ============================================================================
# BÖLÜM 7: RETRIEVAL FONKSİYONLARI
# ============================================================================

def search_rag(query: str, resources: Dict, k: int = 3, movies: List[Dict] = None) -> List[str]:
    """Tüm kaynaklardan bilgi getirir (Wiki + Teyit + FEVER)
    
    🎓 EĞİTİM NOTU - KÜTÜPHANE TARAMASI:
    Burada "Kütüphaneci" (Transformer 1) devreye girer.
    Sizin sorunuzu alır, raflar arasında koşarak en alakalı kitapları (belgeleri)
    seçip Patron'un (LLM) masasına bırakır.
    """
    all_results = []
    
    # Kaynakları al
    wiki_index = resources["wiki"]["index"]
    wiki_encoder = resources["wiki"]["encoder"]
    
    teyit_index = resources["teyit"]["index"]
    teyit_encoder = resources["teyit"]["encoder"]
    teyit_docs_data = resources["teyit"]["docs"]
    
    fever_index = resources["fever"]["index"]
    fever_encoder = resources["fever"]["encoder"]
    fever_docs_data = resources["fever"]["docs"]
    
    # Kaynak sayısına göre k değerini paylaştır
    active_sources = sum([1 for x in [wiki_index, teyit_index, fever_index] if x is not None])
    k_source = k // active_sources if active_sources > 0 else k
    
    # 1. Wiki Araması
    # 1. Wiki Araması
    if wiki_index:
        try:
            with memory_efficient_context():
                query_vector = wiki_encoder.encode(
                    [query], 
                    normalize_embeddings=True,
                    batch_size=1
                )
                
                distances, indices = wiki_index.search(
                    query_vector.astype('float32'), 
                    min(k_source * 2, wiki_index.ntotal)
                )
                
                valid_results = []
                # WIKI_THRESHOLD kullanılmalı
                for dist, idx in zip(distances[0], indices[0]):
                    if dist < WIKI_THRESHOLD and idx >= 0:
                        valid_results.append((dist, idx))
                        if len(valid_results) >= k_source:
                            break
                            
                if valid_results:
                    target_ids = {idx for _, idx in valid_results}
                    
                    with open(WIKI_DOCS_PATH, 'r', encoding='utf-8') as f:
                        for line_idx, line in enumerate(f):
                            if line_idx in target_ids:
                                doc = json.loads(line)
                                score = next(dist for dist, idx in valid_results if idx == line_idx)
                                relevance_pct = max(0, int((1 - score/2) * 100))
                                all_results.append(f"[Alaka: %{relevance_pct}] {doc['text']}")
                                target_ids.remove(line_idx)
                                if not target_ids:
                                    break
        except Exception as e:
            print(f"Wiki hatasi: {e}")
        
    # 2. Teyit Araması
    if teyit_index and teyit_docs_data:
        try:
            with torch.no_grad():
                q_emb = teyit_encoder.encode([query]).astype("float32")
            
            D, I = teyit_index.search(q_emb, k_source)
            
            for score, idx in zip(D[0], I[0]):
                if idx < len(teyit_docs_data):
                    doc = teyit_docs_data[idx]
                    
                    # PERFORMANCE SAFETY CAP (4000 Chars)
                    # Modelin donmasını engellemek için 4000 karakter (yaklaşık 1-2 sayfa) yeterli.
                    # Bu, makalenin %95'ini kapsar ama "Sınırsız" modun kilitlenmesini önler.
                    full_text = doc.get('text', '')
                    safe_text = full_text[:4000] 
                    if len(full_text) > 4000:
                        safe_text += "... [Metin Performans İçin Kısaltıldı]"
                        
                    formatted = f"[Alaka: {int(score*100)}] [KAYNAK: Teyit.org] {doc.get('title', '')}\n{safe_text}"
                    all_results.append(formatted)
        except Exception as e:
            print(f"Teyit hatasi: {e}")

    # 3. FEVER Gold Araması
    if fever_index and fever_docs_data:
        try:
            with torch.no_grad():
                q_emb = fever_encoder.encode([query]).astype("float32")
            
            D, I = fever_index.search(q_emb, k_source)
            
            for score, idx in zip(D[0], I[0]):
                if idx < len(fever_docs_data):
                    doc = fever_docs_data[idx]
                    # FEVER verisinde 'claim_ref' ve 'text' (evidence) var
                    evidence_text = doc.get('text', '')
                    claim_context = doc.get('claim_ref', '')
                    label = doc.get('label', '')
                    
                    formatted = f"[Alaka: {int(score*100)}] [KAYNAK: FEVER Gold] (Doğrulayan İddia: {claim_context})\nKANIT: {evidence_text}\n(Etiket: {label})"
                    all_results.append(formatted)
        except Exception as e:
            print(f"FEVER hatasi: {e}")
    
    # SONUÇLARI SKORA GÖRE SIRALA (YÜKSEKTEN DÜŞÜĞE)
    # Format: "[Alaka: SCORE] ..."
    def extract_score(res_str):
        try:
            if "[Alaka: %" in res_str:
                return int(res_str.split("[Alaka: %")[1].split("]")[0])
            elif "[Alaka: " in res_str:
                return int(res_str.split("[Alaka: ")[1].split("]")[0])
        except:
            return 0
        return 0

    # Skora göre tersten sırala (En yüksek puan en üstte)
    all_results.sort(key=extract_score, reverse=True)

    return all_results
    



def retrieve_documents_teyit(query: str, resources: Dict, k: int = 3,
                             threshold: float = TEYIT_THRESHOLD) -> List[str]:
    """
    Teyit.org'dan (TR) doküman çeker.
    
    Args:
        query: Arama sorgusu
        resources: RAG resources dict
        k: Çekilecek doküman sayısı
        threshold: Cosine similarity threshold (yüksek = iyi)
    """
    encoder = resources["teyit"]["encoder"]
    index = resources["teyit"]["index"]
    docs_data = resources["teyit"]["docs"]
    
    if not index or not docs_data: 
        return []

    with memory_efficient_context():
        query_vector = encoder.encode(
            [query], 
            normalize_embeddings=True,
            batch_size=1
        )
        
        distances, indices = index.search(
            query_vector.astype('float32'), 
            min(k * 2, len(docs_data))
        )
        
        results = []
        for score, idx in zip(distances[0], indices[0]):
            if score >= threshold and 0 <= idx < len(docs_data):
                doc = docs_data[idx]
                relevance_pct = int(score * 100)
                # ÖNEMLİ: Teyit makaleleri çok uzun, 2000 karakter sınırı
                text_content = doc['text'][:2000]
                formatted = f"[Alaka: %{relevance_pct}] [BAŞLIK: {doc['title']}]\n{text_content}\n(Link: {doc['url']})"
                results.append(formatted)
                
                if len(results) >= k:
                    break
    
    return results


# ============================================================================
# BÖLÜM 8: CHAIN-OF-THOUGHT ANALİZCİ
# ============================================================================

class ChainOfThoughtAnalyzer:
    """RAG kanıtlarını analiz eden sınıf - Kapsamlı CoT
    
     EĞİTİM NOTU - ZİNCİRLEME DÜŞÜNCE (CHAIN OF THOUGHT):
    Normalde yapay zeka hemen "Cevap 5" der. Ama hata yapabilir.
    "Zincirleme Düşünce" (CoT) yönteminde ona "Sesli Düşün" diyoruz.
    
    Örnek:
    - Normal Model: "Cevap 5" (Belki attı, belki bildi)
    - CoT Modeli: "Önce 3 ile 2'yi topladım, 5 buldum. Sonra sağlamasını yaptım. Sonuç 5."
    
    Bu sayede model "halüsinasyon" görmez (uydurmaz), mantıklı adım atar.
    """
    
    # String concatenation for precise control over newlines (Training formatına birebir uyum)
    COT_PROMPT_TEMPLATE = (
        "### Instruction:\n"
        "Sen titiz bir fact-checker'sın. Aşağıdaki İDDİA ve KANITLARI analiz et.\n"
        "Kurallar:\n"
        "1. Eğer kanıt 'Yanlış Çeviri', 'Montaj' veya 'İddia edildiği gibi değil' diyorsa karar: YANLIŞTIR.\n"
        "2. İddia sadece kelimesi kelimesine ve bağlamıyla doğruysa: DOĞRUDUR de.\n"
        "3. Eğer kanıtlar iddia konusuyla (Kişi, Takım, Olay) ALAKASIZSA: 'DOĞRULANAMAZ' de ve uydurma.\n"
        "4. Cevabını '## ADIM 1 - Bulgular:', '## ADIM 2 - Bağlam:', '## ADIM 3 - SON KARAR:' formatında ver.\n\n"
        "### Input:\n"
        "İDDİA: {claim}\n"
        "KANITLAR: {evidence}\n\n"
        "### Response:\n"
    )

    def __init__(self, tokenizer, model):
        self.tokenizer = tokenizer
        self.model = model
        
    def analyze_with_evidence(self, claim: str, evidence_docs: List[str]) -> VerdictResult:
        """
        Kanıtları CoT ile analiz edip verdict döndür - DEBUG EKLENDİ
        """
        # Debug: Kaynak seçim sürecini göster
        self.debug_source_selection(claim, evidence_docs)
        
        evidence_text = self._format_evidence(evidence_docs)
        
        prompt = self.COT_PROMPT_TEMPLATE.format(
            claim=claim,
            evidence=evidence_text
        )
        
        response = self._generate_response(prompt)
        return self._parse_cot_response(response, evidence_docs)
    
    def _format_evidence(self, docs: List[str]) -> str:
        """
        Kanıtları model için formatla - YENİ YAKLAŞIM:
        - Sadece en alakalı 1. kaynağın tam içeriğini modele sun
        - Diğer kaynakların sadece başlıklarını debug için göster
        
        Bu yaklaşımın avantajları:
        1. Model daha az bilgi ile daha odaklı karar verir
        2. Farklı kaynaklardan gelen çelişkili bilgiler karışmaz
        3. Daha hızlı işlem ve daha az token kullanımı
        
        Dezavantajları:
        1. Diğer kaynaklardaki potansiyel olarak değerli bilgiler kaybolabilir
        2. Sadece ilk kaynağın alaka skoruna güvenilir
        3. İlk kaynak yanlış veya yetersiz ise doğruluk düşebilir
        """
        if not docs:
            return "Kanıt bulunamadı."
        
        formatted = []
        
        # İlk kaynağı tam olarak ekle (en alakalı olan)
        first_doc = docs[0]
        formatted.append(f"Kanıt 1 (En Alakalı):\n{first_doc}")
        
        # Diğer kaynakların sadece başlıklarını göster (debug için)
        if len(docs) > 1:
            formatted.append("\n--- DİĞER KAYNAKLAR (SADECE BAŞLIKLAR) ---")
            for i, doc in enumerate(docs[1:], 2):
                # Başlığı çıkar - [BAŞLIK: ...] formatını kullan
                if "[BAŞLIK:" in doc:
                    title_match = doc.split("[BAŞLIK:")[1].split("]")[0]
                    formatted.append(f"Kanıt {i}: {title_match}")
                elif "[KAYNAK:" in doc:
                    # FEVER veya diğer formatlar için
                    parts = doc.split("[KAYNAK:")
                    source_match = parts[1].split("]")[0]
                    
                    title = "Başlık mevcut değil"
                    if len(parts) > 1 and "]" in parts[1]:
                        after_source = parts[1].split("]", 1)[1].strip()
                        if "(Doğrulayan İddia:" in after_source:
                            try:
                                title = after_source.split("(Doğrulayan İddia:")[1].split(")")[0].strip()
                            except:
                                pass
                        else:
                             # Try first line
                            lines = [l.strip() for l in after_source.split('\n') if l.strip()]
                            if lines:
                                title = lines[0]

                    formatted.append(f"Kanıt {i}: [{source_match}] {title}")
                else:
                    # İlk 100 karakteri başlık olarak kullan
                    title_preview = doc[:100].replace('\n', ' ')
                    formatted.append(f"Kanıt {i}: {title_preview}...")
        
        return "\n\n".join(formatted)
    
    def _generate_response(self, prompt: str) -> str:
        """LLM'den cevap al (Raw Format)"""
        
        # DEBUG SADELEŞTİRME: Ekrana basılan prompt'u kısalt
        # Kullanıcı prompt'un tamamını görmek istemiyor, sadece başını görmek istiyor
        # Ancak model'e TAM prompt gidecek.
        debug_prompt_view = prompt
        
        # Eğer prompt çok uzunsa (özellikle Teyit.org metinleri) kısaltarak göster
        if len(prompt) > 2000:
             # Sadece debug baskısı için kısaltma
             # Bu işlem modele giden veriyi ETKİLEMEZ
             lines = prompt.split('\n')
             truncated_lines = []
             for line in lines:
                 if len(line) > 300: # 300 karakterden uzun satırları kırp
                     truncated_lines.append(line[:300] + "... [DEVAMI GİZLENDİ]")
                 else:
                     truncated_lines.append(line)
             debug_prompt_view = '\n'.join(truncated_lines)

        print(f"\n[DEBUG] PROMPT:\n{repr(debug_prompt_view)}\n" + "-"*20)
        
        inputs = self.tokenizer(prompt, return_tensors="pt")
        input_ids = inputs.input_ids.to(self.model.device)
        attention_mask = inputs.attention_mask.to(self.model.device)
        
        # Prompt uzunluğu kontrolü
        prompt_length = input_ids.shape[1]
        if prompt_length > 6000:
            print(f"[UYARI] Prompt cok uzun ({prompt_length} token)! Model yanit vermeyebilir.")
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=self.tokenizer.eos_token_id,
                do_sample=True,      
                temperature=0.3,
                top_p=0.9,
                repetition_penalty=1.2
            )
            
        response = self.tokenizer.decode(outputs[0][input_ids.shape[1]:], skip_special_tokens=True)
        
        # Halüsinasyon Temizliği (assistant: vb. başlarsa kes)
        stop_words = [
            "assistant:", "Human:", "User:", "### Instruction", 
            "### Response:", "### Input:", "Counter-Analysis:", 
            "## Kaynak:", "## Tarih:", "```"
        ]
        
        for stop_word in stop_words:
            if stop_word in response:
                response = response.split(stop_word)[0]
                
        print(f"\n[DEBUG] HAM CEVAP:\n{response.strip()}\n" + "-"*40)
        return response.strip()
  
    
    def _parse_cot_response(self, response: str, sources: List[str]) -> VerdictResult:
        """
        Yeni CoT formatındaki cevabı parse et - KAYNAK GÖSTERİMİ GÜNCELLEMESİ:
        - Sadece 1. kaynağın tam içeriğini, diğerlerinin sadece başlıklarını göster
        - Debug bilgilerini daha bilgilendirici hale getir
        """
        verdict = self._extract_verdict(response)
        confidence = self._extract_confidence(response)
        
        # Bulgular (ADIM 1)
        supporting = self._extract_section(response, "ADIM 1", "ADIM 2")
        
        # Bağlam Kontrolü (ADIM 2)
        refuting = self._extract_section(response, "ADIM 2", "ADIM 3")
        
        # Açıklama
        explanation = self._extract_section(response, "Açıklama:", None)
        if not explanation:
            explanation = self._extract_section(response, "## Açıklama", None)
        
        # FALLBACK: Açıklama bulunamazsa kaynaktan özet çıkar
        if not explanation or len(explanation) < 10:
            explanation = self._generate_fallback_explanation(sources, verdict)
        
        # Kaynakları formatla - sadece 1. kaynağın tam içeriği
        formatted_sources = []
        if sources:
            # İlk kaynağı tam olarak ekle
            formatted_sources.append(sources[0])
            
            # Diğer kaynakların sadece başlıklarını ekle
            for i, source in enumerate(sources[1:], 2):
                title = self._extract_title_from_source(source)
                formatted_sources.append(f"[KAYNAK {i}] {title}")
        
        return VerdictResult(
            verdict=verdict,
            confidence=confidence,
            supporting_points=[s.strip() for s in supporting.split('\n') if s.strip()] if supporting else [],
            refuting_points=[r.strip() for r in refuting.split('\n') if r.strip()] if refuting else [],
            explanation=explanation.strip() if explanation else "Yeterli bilgi bulunamadı.",
            sources=formatted_sources,
            raw_response=response
        )
    
    def _extract_title_from_source(self, source: str) -> str:
        """
        Kaynak metninden başlık bilgisini çıkarır
        """
        # [BAŞLIK: ...] formatını kontrol et
        if "[BAŞLIK:" in source:
            title_match = source.split("[BAŞLIK:")[1].split("]")[0]
            return title_match
        
        # [KAYNAK: ...] formatını kontrol et
        elif "[KAYNAK:" in source:
            # Extract content after [KAYNAK: Source]
            parts = source.split("[KAYNAK:")
            if len(parts) > 1 and "]" in parts[1]:
                # parts[1] -> " Teyit.org] Title\nText..."
                after_source = parts[1].split("]", 1)[1].strip()
                
                # Check for FEVER format explicitly
                if "(Doğrulayan İddia:" in after_source:
                    try:
                        claim_part = after_source.split("(Doğrulayan İddia:")[1].split(")")[0]
                        return claim_part.strip()
                    except:
                        pass

                # Try to get the first non-empty line as title
                lines = [l.strip() for l in after_source.split('\n') if l.strip()]
                if lines:
                    return lines[0]
            
            # Fallback if no text found
            source_match = parts[1].split("]")[0]
            return f"[{source_match}] - Başlık bulunamadı"
        
        # [Alaka: %...] formatını kontrol et
        elif "[Alaka:" in source:
            # İlk satırı başlık olarak al
            lines = source.split('\n')
            if lines:
                return lines[0].strip()
        
        # Varsayılan: ilk 100 karakteri başlık olarak kullan
        return source[:100].replace('\n', ' ') + "..."
    
    def _generate_fallback_explanation(self, sources: List[str], verdict: Verdict) -> str:
        """Açıklama bulunamazsa kaynaktan özet oluştur"""
        if not sources:
            return "Kaynak bulunamadı."
        
        # En yüksek alaka skorlu kaynağı bul
        best_source = sources[0]  # İlk kaynak genellikle en alakalı
        
        # Kaynaktan metin çıkar
        text = best_source
        
        # [Alaka: %XX] ve [BAŞLIK: ...] kısımlarını atla, asıl içeriğe ulaş
        if "[BAŞLIK:" in text:
            parts = text.split("]", 2)
            if len(parts) > 2:
                text = parts[2]
        elif "[Alaka:" in text:
            parts = text.split("]", 1)
            if len(parts) > 1:
                text = parts[1]
        
        # Tam metni al
        text = text.strip()
        
        # Verdict'e göre prefix ekle
        verdict_prefix = {
            Verdict.TRUE: "✅ Doğru: ",
            Verdict.FALSE: "❌ Yanlış: ",
            Verdict.UNVERIFIABLE: "⚠️ Doğrulanamaz: ",
            Verdict.UNKNOWN: "❓ Belirsiz: "
        }
        
        prefix = verdict_prefix.get(verdict, "")
        
        return f"{prefix}{text}"
    
    def _extract_verdict(self, response: str) -> Verdict:
        """Cevaptan verdict çıkar"""
        response_upper = response.upper()
        
        if "SON KARAR" in response_upper:
            karar_section = response_upper.split("SON KARAR")[1][:100]
            
            if "YANLIŞTIR" in karar_section or "YANLISTIR" in karar_section:
                return Verdict.FALSE
            elif "DOĞRUDUR" in karar_section or "DOGRUDUR" in karar_section:
                return Verdict.TRUE
            elif "DOĞRULAMAZ" in karar_section or "DOGRULAMAZ" in karar_section:
                return Verdict.UNVERIFIABLE
        
        if "YANLIŞTIR" in response_upper or "YANLISTIR" in response_upper:
            return Verdict.FALSE
        elif "DOĞRUDUR" in response_upper or "DOGRUDUR" in response_upper:
            return Verdict.TRUE
        elif "DOĞRULAMAZ" in response_upper or "DOGRULAMAZ" in response_upper:
            return Verdict.UNVERIFIABLE
        
        return Verdict.UNKNOWN
    
    def _extract_confidence(self, response: str) -> str:
        """Cevaptan güven seviyesi çıkar"""
        response_upper = response.upper()
        
        if "YÜKSEK" in response_upper or "YUKSEK" in response_upper:
            return "YÜKSEK"
        elif "DÜŞÜK" in response_upper or "DUSUK" in response_upper:
            return "DÜŞÜK"
        else:
            return "ORTA"
    
    def _extract_section(self, response: str, start_marker: str, 
                        end_marker: Optional[str]) -> str:
        """Belirli bir bölümü çıkar (Başlangıç -> Bitiş/Son)"""
        if start_marker not in response:
            return ""
            
        start_idx = response.find(start_marker) + len(start_marker)
        content = response[start_idx:]
        
        if end_marker and end_marker in content:
            end_idx = content.find(end_marker)
            return content[:end_idx].strip()
            
        return content.strip()
    
    def debug_source_selection(self, query: str, sources: List[str]) -> None:
        """
        Debug fonksiyonu - Tüm kaynakların başlıklarını ve alaka skorlarını göster
        Hangi kaynağın seçildiğini belirt
        
        Bu fonksiyon, kaynak seçim sürecini şeffaf hale getirir ve
        modelin hangi bilgiye dayanarak karar verdiğini gösterir.
        """
        print("\n" + "="*60)
        print("KAYNAK SECIM DEBUG BILGILERI")
        print("="*60)
        print(f"Sorgu: {query}")
        print(f"Toplam Kaynak Sayisi: {len(sources)}")
        print("-"*60)
        
        if not sources:
            print("[UYARI] Hic kaynak bulunamadi!")
            return
        
        # Tüm kaynakları analiz et
        for i, source in enumerate(sources):
            print(f"\nKaynak {i+1}:")
            
            # Alaka skorunu çıkar
            relevance_score = "Bilinmiyor"
            if "[Alaka: %" in source:
                try:
                    score_part = source.split("[Alaka: %")[1].split("]")[0]
                    relevance_score = f"%{score_part}"
                except:
                    relevance_score = "Parse edilemedi"
            elif "[Alaka: " in source:
                try:
                    score_part = source.split("[Alaka: ")[1].split("]")[0]
                    relevance_score = f"{score_part}"
                except:
                    relevance_score = "Parse edilemedi"
            
            print(f"   Alaka Skoru: {relevance_score}")
            
            # Kaynak tipini belirle
            source_type = "Bilinmiyor"
            if "[KAYNAK: Teyit.org]" in source:
                source_type = "Teyit.org"
            elif "[KAYNAK: FEVER Gold]" in source:
                source_type = "FEVER Gold"
            elif "Wikipedia" in source or "wiki" in source.lower():
                source_type = "Wikipedia"
            
            print(f"   Kaynak Turu: {source_type}")
            
            # Başlığı çıkar
            title = self._extract_title_from_source(source)
            print(f"   Baslik: {title}")
        
        # Seçilen kaynağı belirt
        print("\n" + "-"*60)
        print("KARAR:")
        if sources:
            selected_title = self._extract_title_from_source(sources[0])
            print(f"Modelin kullandigi birincil kaynak: {selected_title}")
            print("Neden: En yuksek alaka skoruna sahip kaynak")
        else:
            print("Modelin kullandigi kaynak: YOK")
        
        print("="*60 + "\n")


# ============================================================================
# BÖLÜM 9: ANA CHAT SİSTEMİ
# ============================================================================

class UnifiedChatSystem:
    """
    Birleşik sohbet sistemi
    
    Akış (Workflow):
    1. Router (Yönlendirici): Soru ne hakkında? (Hava durumu mu? Tarih mi? Merhaba mı?)
    2. RAG (Araştırmacı): Eğer bilgi sorusuysa, kütüphaneden (Teyit, Wiki, FEVER) belge bul.
    3. CoT (Analizci): Bulunan belgeleri oku, "Doğru mu Yanlış mı" karar ver.
    4. LLM (Yazar): Sonucu güzelce yaz.
    5. Web Search (Muhabir): Eğer kütüphanede yoksa, internetten bak.
    """
    ## ADIM 3 - SON KARAR (Kesin Hüküm):
    # Analizine göre aşağıdaki etiketlerden SADECE BİRİNİ seç. DUYGUSAL DAVRANMA, SADECE MANTIĞA BAK.
    # 
    # *   **DOĞRUDUR:** İddiadaki olay GERÇEKTEN OLMUŞSA ve detaylar DOĞRUYSA. (Önemli: Eğer haber "Yanlış çeviri" veya "Yalanlandı" diyorsa bu ASLA doğru olamaz).
    # *   **YANLIŞTIR:** Kanıt metni iddiayı REDDEDİYORSA (Örn: "Yanlış çevrilmiş", "İddia edildiği gibi değil", "Montaj", "Yalan"). EĞER KANIT "YANLIŞ ANLAŞILMA" DİYORSA KARAR "YANLIŞTIR" OLMALIDIR.
    # *   **BAĞLAMDAN KOPARILMIŞ:** Olay gerçek ama anlatıldığı gibi değilse.
    # *   **DOĞRULANAMAZ:** Kanıt yetersizse.
    # 
    # ## ADIM 4 - Açıklama Yazımı (Kullanıcıya Sunum):
    UNCERTAINTY_PHRASES = [
        "bilmiyorum", "emin değilim", "emin olamıyorum", 
        "belirsiz", "doğrulayamıyorum", "yeterli bilgi yok",
        "i don't know", "not sure", "cannot verify", "uncertain"
    ]
    
    def __init__(self, rag_resources: Dict):
        self.resources = rag_resources
        self.tokenizer = rag_resources["tokenizer"]
        self.model = rag_resources["model"]
        self.cot_analyzer = ChainOfThoughtAnalyzer(self.tokenizer, self.model)
    
    def chat(self, query: str) -> ChatResponse:
        """
        Ana sohbet fonksiyonu
        
        Args:
            query: Kullanıcı sorusu/iddiası
            
        Returns:
            ChatResponse objesi
        """
        start_time = time.time()
        
        # 1. Router - sorgu tipini belirle
        intent = classify_intent(query, self.tokenizer, self.model)
        print(f"[Intent]: {intent}")
        
        if intent == "CHAT":
            response = self._generate_chat_response(query)
            return ChatResponse(
                verdict="N/A",
                explanation=response,
                sources=[],
                source_type="CHAT",
                processing_time=time.time() - start_time,
                confidence="N/A"
            )
        
        if intent == "WEB":
            return self._handle_web_search(query, start_time)
        
        # 2. RAG modu - Kanıt ara
        print("[RAG] Arama yapiliyor...")
        
        # Tek arama fonksiyonu tüm kaynaklardan getirir
        # Wiki + Teyit + FEVER
        all_docs = search_rag(
            query, self.resources, k=5  # Toplamda 5 iyi kanıt yeterli
        )
        print(f"   Bulunan Toplam Kanit: {len(all_docs)}")
        
        # 3. Kanıt yeterli mi?
        if len(all_docs) >= MIN_DOCS_FOR_VERDICT:
            print("[CoT] Analiz yapiliyor...")
            result = self.cot_analyzer.analyze_with_evidence(query, all_docs)
            
            return ChatResponse(
                verdict=result.verdict.value,
                explanation=result.explanation,
                sources=result.sources,
                source_type="RAG",
                processing_time=time.time() - start_time,
                confidence=result.confidence
            )
        
        # 4. RAG yetersiz - LLM kendi bilgisi
        print("[UYARI] RAG sonucu yetersiz, LLM bilgisi deneniyor...")
        llm_response = self._generate_knowledge_response(query)
        
        if self._is_llm_uncertain(llm_response):
            print("[Web] LLM emin degil, Web Search deneniyor...")
            return self._handle_web_search(query, start_time)
        
        return ChatResponse(
            verdict="N/A",
            explanation=llm_response,
            sources=["LLM kendi bilgisi"],
            source_type="LLM",
            processing_time=time.time() - start_time,
            confidence="ORTA"
        )
    
    def _generate_chat_response(self, query: str) -> str:
        """
        Sohbet modu cevabı (LoRA DEVRE DIŞI BIRAKILIR)
        Burada modelin 'Dedektif' şapkasını çıkarıp, normal 'Arkadaş' moduna geçmesini sağlıyoruz.
        Böylece 'Nasılsın?' dendiğinde 'İddia doğrulanabilir değil' demez. :)
        """
        messages = [
            {"role": "system", "content": "Sen yardımsever bir AI asistanısın. Türkçe cevap ver."},
            {"role": "user", "content": query}
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        
        # SİHİRLİ DOKUNUŞ: LoRA Adaptörünü geçici olarak kapatıyoruz
        # Bu context manager (with bloğu) bittiğinde adaptör otomatik geri açılır.
        # Eğer model normal bir modelse (LoRA yoksa) hata vermemesi için kontrol ediyoruz.
        
        if hasattr(self.model, "disable_adapter"):
            context_manager = self.model.disable_adapter()
        else:
            # Boş bir context manager (işlevsiz)
            from contextlib import nullcontext
            context_manager = nullcontext()

        with context_manager:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids,
                    max_new_tokens=1000,
                    do_sample=True,
                    temperature=0.7,
                    pad_token_id=self.tokenizer.eos_token_id
                )
        
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)
    
    def _generate_knowledge_response(self, query: str) -> str:
        """LLM kendi bilgisiyle cevap"""
        system_prompt = """Sen bir bilgi asistanısın. Kullanıcının sorusunu kendi bilginle yanıtla.
        
KURALLAR:
1. Emin değilsen "Bu konuda emin değilim" de
2. Bilmiyorsan "Bu konuda yeterli bilgim yok" de
3. Kısa ve öz cevap ver"""
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query}
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=1500,
                do_sample=True,
                temperature=0.5,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)
    
    def _is_llm_uncertain(self, response: str) -> bool:
        """LLM cevabında belirsizlik var mı?"""
        response_lower = response.lower()
        return any(phrase in response_lower for phrase in self.UNCERTAINTY_PHRASES)
    
    def _handle_web_search(self, query: str, start_time: float) -> ChatResponse:
        """Web search ile cevap"""
        try:
            results = search_web(query, max_results=3)
            
            if not results:
                return ChatResponse(
                    verdict="DOĞRULAMAZ",
                    explanation="Bu konuda yeterli bilgi bulunamadı.",
                    sources=[],
                    source_type="ERROR",
                    processing_time=time.time() - start_time,
                    confidence="DÜŞÜK"
                )
            
            web_docs = [f"[{r['title']}]\n{r['body']}" for r in results]
            result = self.cot_analyzer.analyze_with_evidence(query, web_docs)
            
            return ChatResponse(
                verdict=result.verdict.value,
                explanation=result.explanation,
                sources=[r['href'] for r in results],
                source_type="WEB",
                processing_time=time.time() - start_time,
                confidence=result.confidence
            )
        except Exception as e:
            print(f"Web search hatası: {e}")
            return ChatResponse(
                verdict="DOĞRULAMAZ",
                explanation=f"Web araması sırasında hata oluştu: {str(e)}",
                sources=[],
                source_type="ERROR",
                processing_time=time.time() - start_time,
                confidence="DÜŞÜK"
            )


# ============================================================================
# BÖLÜM 10: YARDIMCI FONKSİYONLAR
# ============================================================================

def load_system() -> UnifiedChatSystem:
    """
    Sistemi yükle ve döndür (convenience function)
    
    Returns:
        UnifiedChatSystem instance
    """
    resources = load_rag_system()
    return UnifiedChatSystem(resources)


def print_response(result: ChatResponse):
    """Cevabı güzel formatta yazdır"""
    print(f"\n[SONUC] ({result.source_type})")
    print("-" * 40)
    
    if result.verdict != "N/A":
        print(f"[Karar]: {result.verdict}")
        print(f"[Güven]: {result.confidence}")
    
    print(f"\n[Açıklama]:\n{result.explanation}")
    
    if result.sources:
        print(f"\n[Kaynaklar]:")
        for i, src in enumerate(result.sources[:3]):
            print(f"   {i+1}. {src[:80]}...")
    
    print(f"\n[Sure]: {result.processing_time:.2f}s")
    print("-" * 40)


# ============================================================================
# BÖLÜM 11: MAIN (CLI)
# ============================================================================

def main():
    """Ana CLI fonksiyonu"""
    print("=" * 60)
    print("Unified Chat System")
    print("   RAG + CoT + Web Search")
    print("=" * 60)
    
    try:
        print("\nSistem yukleniyor...")
        system = load_system()
        
        print("\nSistem hazir!")
        print("Cikis icin 'q' yazin.\n")
        
        while True:
            query = input("\nSoru: ").strip()
            
            if query.lower() in ['q', 'quit', 'exit', 'cikis']:
                print("Gorusuruz!")
                break
                
            if not query:
                continue
            
            print("\n" + "-" * 40)
            result = system.chat(query)
            print_response(result)
            
    except KeyboardInterrupt:
        print("\n\nProgramdan cikiliyor...")
    except Exception as e:
        print(f"Hata: {e}")
    finally:
        cleanup_resources()


if __name__ == "__main__":
    main()
