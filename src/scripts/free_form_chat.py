import os
import sys
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
from transformers import BitsAndBytesConfig
import time
import json
import logging
from typing import Dict, List, Tuple, Optional, Any

# Import RAG components
from sentence_transformers import SentenceTransformer
import faiss
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
try:
    from core.web_search import search_web
except ImportError:
    # Fallback if import fails
    def search_web(query, max_results=3):
        """Fallback web search function"""
        print(f"⚠️ Web search not available: {query}")
        return []

# Windows için UTF-8 encoding ayarı
if sys.platform == "win32":
    import locale
    # Sadece locale ayarını değiştir, stdout/stderr'i değiştirme
    try:
        locale.setlocale(locale.LC_ALL, 'Turkish_Turkey.1254')
    except:
        try:
            locale.setlocale(locale.LC_ALL, 'tr_TR.UTF-8')
        except:
            pass  # Varsayılan ayarlar devam etsin

# --- AYARLAR ---
# Proje Kök Dizini (src/scripts/free_form_chat.py -> ../../)
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Model Yolları
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "base_model")
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "checkpoint_lora_1k")

# RAG 1: Wikipedia (İngilizce - Mevcut)
WIKI_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "indices", "faiss_index_fast.bin")
WIKI_DOCS_PATH = os.path.join(PROJECT_ROOT, "data", "indices", "documents.jsonl")
WIKI_EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# RAG 2: Teyit.org (Türkçe - Yeni)
TEYIT_INDEX_PATH = os.path.join(PROJECT_ROOT, "data", "indices", "teyit_index.bin")
TEYIT_DOCS_PATH = os.path.join(PROJECT_ROOT, "data", "indices", "teyit_documents_clean.json")
TEYIT_EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# Sistem Ayarları
MAX_NEW_TOKENS = 1024  # Daha uzun yanıtlar için artırıldı
CHAT_HISTORY_LIMIT = 5  # Son 5 konuşmayı hafızada tut

# Hybrid System Settings
RAG_QUALITY_THRESHOLD = 0.4  # Minimum RAG quality score to use RAG response
ENABLE_RAG = True  # RAG sistemini aktif et
ENABLE_WEB_SEARCH = True  # Web search fallback'ini aktif et

def load_model():
    """Base model ve LoRA adaptörünü yükler"""
    print("Model ve LoRA adaptörü yükleniyor...")
    
    try:
        # Tokenizer yükle
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
        
        # BitsAndBytes quantization config (4-bit ile optimize)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,  # Çift quantization - daha az VRAM
        )
        
        # Base model yükle
        base_model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_PATH,
            quantization_config=bnb_config,
            device_map="auto",
            low_cpu_mem_usage=True,  # CPU RAM kullanımını azalt
            max_memory={0: "10GB", "cpu": "20GB"}  # VRAM ve RAM limitleri
        )
        
        # LoRA adaptörünü yükle
        model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
        print("Model ve LoRA adaptörü başarıyla yüklendi!")
        
        return model, tokenizer
        
    except Exception as e:
        print(f"Model yükleme hatası: {e}")
        sys.exit(1)

def load_rag_system():
    """RAG sistemini yükler"""
    print("RAG sistemleri yükleniyor...")
    
    rag_resources = {}
    
    # Wikipedia RAG (EN)
    print("Wiki-RAG Yükleniyor...")
    try:
        wiki_encoder = SentenceTransformer(WIKI_EMBED_MODEL)
        if os.path.exists(WIKI_INDEX_PATH):
            wiki_index = faiss.read_index(WIKI_INDEX_PATH)
            rag_resources["wiki"] = {"encoder": wiki_encoder, "index": wiki_index}
            print("[OK] Wikipedia RAG yuklendi")
        else:
            print(f"[WARNING] Wiki Indeksi bulunamadi ({WIKI_INDEX_PATH})")
            rag_resources["wiki"] = {"encoder": wiki_encoder, "index": None}
    except Exception as e:
        print(f"[ERROR] Wikipedia RAG yukleme hatasi: {e}")
        rag_resources["wiki"] = {"encoder": None, "index": None}
    
    # Teyit RAG (TR)
    print("Teyit-RAG Yükleniyor...")
    try:
        teyit_encoder = SentenceTransformer(TEYIT_EMBED_MODEL)
        if os.path.exists(TEYIT_INDEX_PATH):
            teyit_index = faiss.read_index(TEYIT_INDEX_PATH)
            # Teyit dokümanlarını hafızaya al
            with open(TEYIT_DOCS_PATH, "r", encoding="utf-8") as f:
                teyit_docs_data = json.load(f)
            rag_resources["teyit"] = {"encoder": teyit_encoder, "index": teyit_index, "docs": teyit_docs_data}
            print("[OK] Teyit.org RAG yuklendi")
        else:
            print(f"[WARNING] Teyit Indeksi bulunamadi ({TEYIT_INDEX_PATH})")
            rag_resources["teyit"] = {"encoder": teyit_encoder, "index": None, "docs": []}
    except Exception as e:
        print(f"[ERROR] Teyit.org RAG yukleme hatasi: {e}")
        rag_resources["teyit"] = {"encoder": None, "index": None, "docs": []}
    
    return rag_resources

def retrieve_documents_wiki(query: str, rag_resources: Dict, k: int = 3) -> Tuple[List[str], List[float]]:
    """Wikipedia (EN) indeksinde arama yapar"""
    if not rag_resources["wiki"]["encoder"] or not rag_resources["wiki"]["index"]:
        return [], []
    
    encoder = rag_resources["wiki"]["encoder"]
    index = rag_resources["wiki"]["index"]
    
    query_vector = encoder.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_vector.astype('float32'), k)
    
    # Convert L2 distances to similarity scores (0-1 range)
    max_distance = 2.0
    scores = [max(0, 1 - (dist / max_distance)) for dist in distances[0]]
    
    results = []
    filtered_scores = []
    threshold = 0.3  # Minimum relevance threshold
    
    for i, (score, idx) in enumerate(zip(scores, indices[0])):
        if score >= threshold:
            target_ids = {idx}
            found_count = 0
            
            with open(WIKI_DOCS_PATH, 'r', encoding='utf-8') as f:
                for line_idx, line in enumerate(f):
                    if line_idx in target_ids:
                        doc = json.loads(line)
                        results.append(doc['text'])
                        filtered_scores.append(score)
                        found_count += 1
                        if found_count == len(target_ids):
                            break
    
    return results, filtered_scores

def retrieve_documents_teyit(query: str, rag_resources: Dict, k: int = 3) -> Tuple[List[str], List[float]]:
    """Teyit.org (TR) indeksinde arama yapar"""
    if not rag_resources["teyit"]["encoder"] or not rag_resources["teyit"]["index"]:
        return [], []
    
    encoder = rag_resources["teyit"]["encoder"]
    index = rag_resources["teyit"]["index"]
    docs_data = rag_resources["teyit"]["docs"]
    
    query_vector = encoder.encode([query], normalize_embeddings=True)
    distances, indices = index.search(query_vector.astype('float32'), k)
    
    results = []
    scores = []
    threshold = 0.3  # Minimum relevance threshold
    
    for i, (score, idx) in enumerate(zip(distances[0], indices[0])):
        if score >= threshold and idx < len(docs_data):
            doc = docs_data[idx]
            results.append(doc['text'][:1000])  # İlk 1000 karakter
            scores.append(score)
            
    return results, scores

def calculate_rag_quality(documents: List[str], scores: List[float]) -> float:
    """RAG sonuçlarının kalitesini hesaplar"""
    if not documents or not scores:
        return 0.0
    
    # Ortalama relevance skoru
    avg_score = sum(scores) / len(scores)
    
    # Doküman sayısı faktörü
    doc_count_factor = min(len(documents) / 3.0, 1.0)  # Max 1.0 için 3 doküman
    
    # İçerik uzunluğu faktörü
    total_length = sum(len(doc) for doc in documents)
    length_factor = min(total_length / 1000.0, 1.0)  # Max 1.0 için 1000 karakter
    
    # Ağırlıklı ortalama
    quality = (avg_score * 0.6) + (doc_count_factor * 0.2) + (length_factor * 0.2)
    
    return min(max(quality, 0.0), 1.0)

def generate_response_with_rag(model, tokenizer, rag_resources, query: str, messages: List[Dict]) -> str:
    """Hibrit sistem ile yanıt üretir: RAG → Model → WebSearch"""
    
    # Adım 1: RAG dene (önce Teyit.org, sonra Wikipedia)
    rag_response = None
    rag_quality = 0.0
    
    if ENABLE_RAG:
        try:
            # Önce Türkçe Teyit.org'u dene
            teyit_docs, teyit_scores = retrieve_documents_teyit(query, rag_resources)
            if teyit_docs:
                rag_quality = calculate_rag_quality(teyit_docs, teyit_scores)
                if rag_quality >= RAG_QUALITY_THRESHOLD:
                    context = "\n\n".join([f"Kaynak {i+1}: {doc}" for i, doc in enumerate(teyit_docs)])
                    rag_messages = messages.copy()
                    rag_messages[-1] = {
                        "role": "user",
                        "content": f"Soru: {query}\n\nKaynaklar:\n{context}\n\nBu kaynaklara göre soruyu yanıtla:"
                    }
                    rag_response = generate_response(model, tokenizer, rag_messages)
            
            # Eğer Teyit.org yeterli değilse, Wikipedia'ı dene
            if not rag_response:
                wiki_docs, wiki_scores = retrieve_documents_wiki(query, rag_resources)
                if wiki_docs:
                    wiki_quality = calculate_rag_quality(wiki_docs, wiki_scores)
                    if wiki_quality >= RAG_QUALITY_THRESHOLD:
                        context = "\n\n".join([f"Kaynak {i+1}: {doc}" for i, doc in enumerate(wiki_docs)])
                        rag_messages = messages.copy()
                        rag_messages[-1] = {
                            "role": "user",
                            "content": f"Soru: {query}\n\nKaynaklar:\n{context}\n\nBu kaynaklara göre soruyu yanıtla:"
                        }
                        rag_response = generate_response(model, tokenizer, rag_messages)
                        rag_quality = wiki_quality
        except Exception as e:
            print(f"[RAG ERROR] {e}")
    
    # Eğer RAG yüksek kalitede sonuç bulduysa kullan
    if rag_response and rag_quality >= RAG_QUALITY_THRESHOLD:
        return rag_response
    
    # Adım 2: Model kendi bilgisini kullan
    try:
        model_response = generate_response(model, tokenizer, messages)
        if model_response and len(model_response.strip()) > 20:
            return model_response
    except Exception as e:
        print(f"[MODEL ERROR] {e}")
    
    # Adım 3: Web search fallback
    if ENABLE_WEB_SEARCH:
        try:
            search_results = search_web(query, max_results=3)
            if search_results:
                context = "\n\n".join([f"Kaynak {i+1}: {result['title']}\n{result['body']}" for i, result in enumerate(search_results)])
                web_messages = messages.copy()
                web_messages[-1] = {
                    "role": "user",
                    "content": f"Soru: {query}\n\nWeb Arama Sonuçları:\n{context}\n\nBu arama sonuçlarına göre soruyu yanıtla:"
                }
                web_response = generate_response(model, tokenizer, web_messages)
                if web_response and len(web_response.strip()) > 20:
                    return web_response
        except Exception as e:
            print(f"[WEB SEARCH ERROR] {e}")
    
    # Son çare: Basit model yanıtı
    return generate_response(model, tokenizer, messages)

def generate_response(model, tokenizer, messages):
    """Modelden yanıt üretir"""
    try:
        # Input tensor'ını oluştur
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            return_tensors="pt",
            return_dict=True
        )
        
        input_ids = inputs["input_ids"].to(model.device)
        
        # Attention mask oluştur
        attention_mask = inputs.get("attention_mask", None)
        if attention_mask is None:
            # Eğer attention mask yoksa, padding token'ları için oluştur
            attention_mask = (input_ids != tokenizer.pad_token_id).long()
        attention_mask = attention_mask.to(model.device)
        
        # Geliştirilmiş generation parametreleri
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=MAX_NEW_TOKENS,
            min_new_tokens=130,  # Minimum yanıt uzunluğu
            pad_token_id=tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,  # Daha düşük temperature tutarlılık için
            top_p=0.85,      # Daha konservatif top_p
            top_k=40,        # Top-k sampling ekle
            repetition_penalty=1.15,  # Tekrarları azalt
            length_penalty=1.0,       # Uzunluk cezası
            early_stopping=True,      # Erken durdurma
            no_repeat_ngram_size=3,   # 3-gram tekrarlarını engelle
            use_cache=True,           # Cache kullanımını aktif et
        )
        
        # Yanıtı decode et
        response = outputs[0][input_ids.shape[-1]:]
        decoded_response = tokenizer.decode(response, skip_special_tokens=True)
        
        # Boş veya çok kısa yanıtları kontrol et
        if len(decoded_response.strip()) < 10:
            return "Üzgünüm, anlamlı bir yanıt üretemedim. Lütfen sorunuzu farklı şekilde ifade edin."
            
        return decoded_response
        
    except Exception as e:
        print(f"Yanıt üretme hatası: {e}")
        import traceback
        traceback.print_exc()
        return "Üzgünüm, bir hata oluştu. Lütfen tekrar deneyin."

def print_welcome():
    """Hoş geldin mesajını yazdırır"""
    print("\n" + "="*60)
    print("[ROBOT] HIBRIT SERBEST CHAT SISTEMI")
    print("="*60)
    print("Bu sistem arka planda RAG -> Model -> WebSearch akisi calistirir.")
    print("Arayuz basit kalir, ancak yanitlar coklu kaynaktan dogrulanir.")
    print("\nKullanim:")
    print("- Turkce sorularinizi yazin")
    print("- 'temizle' yazarak sohbet gecmisini temizleyin")
    print("- 'cikis' veya 'q' yazarak cikis yapin")
    print("="*60 + "\n")

def main():
    """Ana chat döngüsü"""
    print_welcome()
    
    # Modeli yükle
    model, tokenizer = load_model()
    
    # RAG sistemini yükle
    rag_resources = load_rag_system()
    
    # Sohbet geçmişi
    chat_history = []
    
    # Sistem mesajı
    system_message = {
        "role": "system",
        "content": "Sen yardımsever bir AI asistanısın. Kullanıcının sorularını Türkçe olarak yanıtla. Dürüst ol ve bildiklerini paylaş. Kaynaklardan elde ettiğin bilgileri kullanırken doğru ve faydalı olmaya özen göster."
    }
    
    print("\n[ROBOT] Chat hazir! Sorularinizi yazabilirsiniz.\n")
    
    while True:
        try:
            # Kullanıcı girdisini al
            user_input = input("[KULLANICI] Siz: ").strip()
            
            # Çıkış komutlarını kontrol et
            if user_input.lower() in ['çıkış', 'q', 'quit', 'exit']:
                print("\n[BYE] Gorusuruz!")
                break
                
            # Temizle komutunu kontrol et
            if user_input.lower() in ['temizle', 'clear']:
                chat_history = []
                print("\n[CLEAR] Sohbet gecmisi temizlendi.\n")
                continue
            
            # Boş girdiyi kontrol et
            if not user_input:
                continue
            
            # Kullanıcı mesajını ekle
            user_message = {"role": "user", "content": user_input}
            
            # Mesajları hazırla
            messages = [system_message] + chat_history + [user_message]
            
            print("\n[ROBOT] Dusunuyorum...")
            start_time = time.time()
            
            # Hibrit yanıt üret
            response = generate_response_with_rag(model, tokenizer, rag_resources, user_input, messages)
            
            response_time = time.time() - start_time
            
            # Yanıtı yazdır
            print(f"\n[ASISTAN] Asistan: {response}")
            print(f"[TIME] Yanit suresi: {response_time:.2f} saniye\n")
            
            # Asistan yanıtını sohbet geçmişine ekle
            assistant_message = {"role": "assistant", "content": response}
            chat_history.append(user_message)
            chat_history.append(assistant_message)
            
            # Sohbet geçmişini sınırla
            if len(chat_history) > CHAT_HISTORY_LIMIT * 2:  # Her konuşma 2 mesaj (kullanıcı + asistan)
                chat_history = chat_history[-CHAT_HISTORY_LIMIT * 2:]
                
        except KeyboardInterrupt:
            print("\n\n[EXIT] Programdan cikiliyor...")
            break
        except Exception as e:
            print(f"\n[ERROR] Hata: {e}\n")
            continue

if __name__ == "__main__":
    main()