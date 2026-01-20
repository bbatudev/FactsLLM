import os
import sys
import time
import json
import logging
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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class ChatResponse:
    """Simple response structure for hybrid chat"""
    answer: str
    source: str  # "RAG", "Model", or "WebSearch"
    confidence: str
    details: str
    response_time: float

class HybridChatSystem:
    """Hybrid chat system with RAG, Model Knowledge, and Web Search fallbacks"""
    
    def __init__(self):
        self.rag_resources = None
        self.tokenizer = None
        self.model = None
        self.chat_history = []
        self.rag_threshold = 0.7  # Increased threshold for better RAG quality
        self.max_response_length = 250  # Reduced to prevent long repetitive responses
        
    def load_system(self):
        """Load all required components"""
        print("[YUKLENIYOR] Hibrit Chat Sistemi Yükleniyor...")
        
        # Load RAG system
        try:
            self.rag_resources = load_enhanced_rag_system()
            print("[TAMAMLANDI] RAG Sistemi yüklendi")
            
            # Update thresholds for better quality
            if self.rag_resources and "fallback_config" in self.rag_resources:
                # Lower thresholds to get more relevant content
                self.rag_resources["fallback_config"]["thresholds"]["wiki_min_relevance_score"] = 0.5
                self.rag_resources["fallback_config"]["thresholds"]["teyit_min_relevance_score"] = 0.5
                logger.info("RAG threshold values updated for better content retrieval")
                
        except Exception as e:
            print(f"[UYARI] RAG Sistemi yüklenemedi: {e}")
            self.rag_resources = None
        
        # Load model separately for direct use
        try:
            print("[MODEL] Model yükleniyor...")
            self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_PATH)
            
            # Set padding token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
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
        
        start_time = time.time()
        try:
            # Try Teyit first (Turkish content)
            docs, scores = retrieve_documents_teyit(query, self.rag_resources, k=3)
            source_type = "teyit"
            
            if not docs or not scores:
                # Try Wikipedia as fallback
                docs, scores = retrieve_documents_wiki(query, self.rag_resources, k=3)
                source_type = "wiki"
            
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
                    
                    response_time = time.time() - start_time
                    
                    return ChatResponse(
                        answer=response,
                        source="RAG",
                        confidence="YÜKSEK" if max_score >= 0.8 else "ORTA",
                        details=f"RAG ile {len(docs)} doküman bulundu (En yüksek skor: {max_score:.2f})",
                        response_time=response_time
                    )
        
        except Exception as e:
            logger.error(f"[RAG HATASI] RAG hatası: {e}")
        
        return None
    
    def try_model_response(self, query: str) -> ChatResponse:
        """Try to get response from model's own knowledge"""
        start_time = time.time()
        try:
            prompt = f"""Sen yardımcı bir AI asistanısın. Kullanıcının sorusunu genel bilgilerini kullanarak Türkçe olarak yanıtla.
            
Soru: {query}

Yanıt:"""
            
            response = self._generate_model_response(prompt)
            
            response_time = time.time() - start_time
            
            return ChatResponse(
                answer=response,
                source="Model",
                confidence="ORTA",
                details="Modelin kendi bilgisi kullanıldı",
                response_time=response_time
            )
        
        except Exception as e:
            logger.error(f"[MODEL HATASI] Model yanıtı hatası: {e}")
            response_time = time.time() - start_time
            return ChatResponse(
                answer="Üzgünüm, şu anda yanıt üretemiyorum.",
                source="Model",
                confidence="DÜŞÜK",
                details=f"Hata: {str(e)}",
                response_time=response_time
            )
    
    def try_web_search_response(self, query: str) -> ChatResponse:
        """Try to get response from web search"""
        start_time = time.time()
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
            
            response_time = time.time() - start_time
            
            return ChatResponse(
                answer=response,
                source="WebSearch",
                confidence="ORTA",
                details=f"Web araması ile {len(search_results)} sonuç bulundu",
                response_time=response_time
            )
        
        except Exception as e:
            logger.error(f"[WEB HATASI] Web arama hatası: {e}")
            # Final fallback to model
            return self.try_model_response(query)
    
    def _generate_model_response(self, prompt: str, max_tokens: int = None) -> str:
        """Generate response using the model with improved parameters"""
        if max_tokens is None:
            max_tokens = self.max_response_length
            
        messages = [
            {"role": "system", "content": "Sen yardımcı bir AI asistanısın. Her zaman Türkçe yanıtla. Kısa ve öz cevaplar ver."},
            {"role": "user", "content": prompt}
        ]
        
        input_ids = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt"
        ).to(self.model.device)
        
        # Fixed early_stopping warning by using proper parameters
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
            do_sample=True,
            temperature=0.6,  # Reduced for more focused responses
            top_p=0.85,
            repetition_penalty=1.2,  # Added to reduce repetition
            no_repeat_ngram_size=3,  # Prevent repeating 3-grams
            # Removed early_stopping parameter as it's causing warnings
        )
        
        response = outputs[0][input_ids.shape[-1]:]
        decoded_response = self.tokenizer.decode(response, skip_special_tokens=True).strip()
        
        # Post-process to remove repetition
        return self._clean_response(decoded_response)
    
    def _clean_response(self, response: str) -> str:
        """Clean response to remove repetition and improve quality"""
        # Split into sentences
        sentences = response.split('. ')
        
        # Remove duplicate sentences
        unique_sentences = []
        seen = set()
        for sentence in sentences:
            if sentence and sentence not in seen:
                seen.add(sentence)
                unique_sentences.append(sentence)
        
        # Join back and limit length
        cleaned = '. '.join(unique_sentences)
        
        # Limit to reasonable length
        if len(cleaned) > 500:
            cleaned = cleaned[:500] + "..."
            
        return cleaned
    
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
        print(f"[SURE] Süre: {response.response_time:.2f} saniye")
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
                logger.error(f"\n[HATA] Hata: {e}")
    
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