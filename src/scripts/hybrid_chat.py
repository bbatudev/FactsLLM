import os
import sys
import time
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import required modules
from core.enhanced_rag_chat import (
    load_enhanced_rag_system, 
    retrieve_documents_wiki, 
    retrieve_documents_teyit,
    ContentAnalysis,
    EnhancedResponse
)
from core.web_search import search_web
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
import torch

# Project paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BASE_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "base_model")
ADAPTER_PATH = os.path.join(PROJECT_ROOT, "models", "checkpoint_lora_1k")

@dataclass
class ChatResponse:
    """Simple response structure for hybrid chat"""
    answer: str
    source: str  # "RAG", "Model", or "WebSearch"
    confidence: str
    details: str

class HybridChatSystem:
    """Hybrid chat system with RAG, Model Knowledge, and Web Search fallbacks"""
    
    def __init__(self):
        self.rag_resources = None
        self.tokenizer = None
        self.model = None
        self.chat_history = []
        self.rag_threshold = 0.6  # Threshold for RAG quality
        
    def load_system(self):
        """Load all required components"""
        print("[YUKLENIYOR] Hibrit Chat Sistemi Yükleniyor...")
        
        # Load RAG system
        try:
            self.rag_resources = load_enhanced_rag_system()
            print("[TAMAMLANDI] RAG Sistemi yüklendi")
        except Exception as e:
            print(f"[UYARI] RAG Sistemi yüklenemedi: {e}")
            self.rag_resources = None
        
        # Load model separately for direct use
        try:
            print("[MODEL] Model yükleniyor...")
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
            
            # Quantization config
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            
            # Load base model
            base_model = AutoModelForCausalLM.from_pretrained(
                BASE_MODEL_PATH,
                quantization_config=bnb_config,
                device_map="auto",
                low_cpu_mem_usage=True,
                max_memory={0: "10GB", "cpu": "20GB"}
            )
            
            # Load LoRA adapter
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
            print("[TAMAMLANDI] Model ve LoRA adaptörü yüklendi")
            
        except Exception as e:
            print(f"[HATA] Model yüklenemedi: {e}")
            sys.exit(1)
        
        print("[HAZIR] Hibrit Chat Sistemi hazır!\n")
    
    def try_rag_response(self, query: str) -> Optional[ChatResponse]:
        """Try to get response from RAG system"""
        if not self.rag_resources:
            return None
        
        try:
            # Try Teyit first (Turkish content)
            docs, scores = retrieve_documents_teyit(query, self.rag_resources, k=3)
            
            if not docs or not scores:
                # Try Wikipedia as fallback
                docs, scores = retrieve_documents_wiki(query, self.rag_resources, k=3)
            
            if docs and scores:
                # Check if we have high-quality results
                max_score = max(scores)
                if max_score >= self.rag_threshold:
                    # Generate response using RAG content
                    context = "\n\n".join(docs[:2])
                    prompt = f"""Aşağıdaki kaynaklara dayanarak soruyu Türkçe olarak yanıtla:
                    
Soru: {query}

Kaynaklar:
{context}

Yanıt:"""
                    
                    response = self._generate_model_response(prompt)
                    
                    return ChatResponse(
                        answer=response,
                        source="RAG",
                        confidence="YÜKSEK" if max_score >= 0.8 else "ORTA",
                        details=f"RAG ile {len(docs)} doküman bulundu (En yüksek skor: {max_score:.2f})"
                    )
        
        except Exception as e:
            print(f"[RAG HATASI] RAG hatası: {e}")
        
        return None
    
    def try_model_response(self, query: str) -> ChatResponse:
        """Try to get response from model's own knowledge"""
        try:
            prompt = f"""Sen yardımcı bir AI asistanısın. Kullanıcının sorusunu genel bilgilerini kullanarak Türkçe olarak yanıtla.
            
Soru: {query}

Yanıt:"""
            
            response = self._generate_model_response(prompt)
            
            return ChatResponse(
                answer=response,
                source="Model",
                confidence="ORTA",
                details="Modelin kendi bilgisi kullanıldı"
            )
        
        except Exception as e:
            print(f"[MODEL HATASI] Model yanıtı hatası: {e}")
            return ChatResponse(
                answer="Üzgünüm, şu anda yanıt üretemiyorum.",
                source="Model",
                confidence="DÜŞÜK",
                details=f"Hata: {str(e)}"
            )
    
    def try_web_search_response(self, query: str) -> ChatResponse:
        """Try to get response from web search"""
        try:
            search_results = search_web(query, max_results=3)
            
            if not search_results:
                # Fallback to model if web search fails
                return self.try_model_response(query)
            
            # Format search results
            context = "\n\n".join([
                f"Başlık: {result['title']}\nİçerik: {result['body']}\nKaynak: {result['href']}"
                for result in search_results
            ])
            
            prompt = f"""Aşağıdaki web arama sonuçlarına dayanarak soruyu Türkçe olarak yanıtla:
            
Soru: {query}

Web Arama Sonuçları:
{context}

Yanıt:"""
            
            response = self._generate_model_response(prompt)
            
            return ChatResponse(
                answer=response,
                source="WebSearch",
                confidence="ORTA",
                details=f"Web araması ile {len(search_results)} sonuç bulundu"
            )
        
        except Exception as e:
            print(f"[WEB HATASI] Web arama hatası: {e}")
            # Final fallback to model
            return self.try_model_response(query)
    
    def _generate_model_response(self, prompt: str, max_tokens: int = 300) -> str:
        """Generate response using the model"""
        messages = [
            {"role": "system", "content": "Sen yardımcı bir AI asistanısın. Her zaman Türkçe yanıtla."},
            {"role": "user", "content": prompt}
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )
        
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True).strip()
    
    def process_query(self, query: str) -> ChatResponse:
        """Process user query with hybrid approach"""
        # Step 1: Try RAG
        rag_response = self.try_rag_response(query)
        if rag_response:
            return rag_response
        
        # Step 2: Try Model Knowledge
        model_response = self.try_model_response(query)
        if model_response and model_response.confidence != "DÜŞÜK":
            return model_response
        
        # Step 3: Try Web Search
        web_response = self.try_web_search_response(query)
        return web_response
    
    def add_to_history(self, query: str, response: ChatResponse):
        """Add conversation to history"""
        self.chat_history.append({
            "query": query,
            "answer": response.answer,
            "source": response.source,
            "timestamp": time.time()
        })
        
        # Keep only last 10 conversations
        if len(self.chat_history) > 10:
            self.chat_history = self.chat_history[-10:]
    
    def clear_history(self):
        """Clear chat history"""
        self.chat_history = []
        print("[TEMIZLE] Sohbet geçmişi temizlendi.")
    
    def display_response(self, query: str, response: ChatResponse):
        """Display the response in a formatted way"""
        print(f"\n{'='*60}")
        print(f"[SORU] Soru: {query}")
        print(f"[KAYNAK] Kaynak: {response.source}")
        print(f"[GUVEN] Güven: {response.confidence}")
        print(f"[DETAY] Detay: {response.details}")
        print(f"\n[YANIT]")
        print(response.answer)
        print(f"{'='*60}\n")
    
    def run(self):
        """Run the hybrid chat system"""
        print("[HOS GELDINIZ] Hibrit Chat Sistemine Hoş Geldiniz!")
        print("Bu sistem sırasıyla RAG, Model Bilgisi ve Web Araması kullanır.")
        print("\nKomutlar:")
        print("  'çıkış' veya 'q' - Çıkış")
        print("  'temizle' - Sohbet geçmişini temizle")
        print("  'geçmiş' - Sohbet geçmişini göster")
        print("-" * 60)
        
        while True:
            try:
                user_input = input("\nSiz: ").strip()
                
                # Check for commands
                if user_input.lower() in ['çıkış', 'q', 'quit', 'exit']:
                    print("[CIKIS] Görüşürüz!")
                    break
                
                if user_input.lower() == 'temizle':
                    self.clear_history()
                    continue
                
                if user_input.lower() == 'geçmiş':
                    self.show_history()
                    continue
                
                if not user_input:
                    continue
                
                # Process the query
                print("\n[ISLENIYOR] İşleniyor...")
                response = self.process_query(user_input)
                
                # Display response
                self.display_response(user_input, response)
                
                # Add to history
                self.add_to_history(user_input, response)
                
            except KeyboardInterrupt:
                print("\n[CIKIS] Görüşürüz!")
                break
            except Exception as e:
                print(f"\n[HATA] Hata: {e}")
    
    def show_history(self):
        """Display chat history"""
        if not self.chat_history:
            print("[BOS] Sohbet geçmişi boş.")
            return
        
        print("\n[GECMIS] Sohbet Geçmişi:")
        print("-" * 60)
        for i, entry in enumerate(self.chat_history, 1):
            print(f"\n{i}. Soru: {entry['query']}")
            print(f"   Kaynak: {entry['source']}")
            print(f"   Yanıt: {entry['answer'][:100]}...")
        print("-" * 60)

def main():
    """Main function to run the hybrid chat system"""
    system = HybridChatSystem()
    system.load_system()
    system.run()

if __name__ == "__main__":
    main()