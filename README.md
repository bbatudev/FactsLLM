# Turkish RAG Fact-Checker

A Retrieval-Augmented Generation (RAG) based fact-checking system for Turkish and English content.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## About The Project

Large Language Models (LLMs) often suffer from hallucinations. This project implements a **Hybrid Verification Engine** to ground the model's responses in verified external sources.

The core model is **Llama 3**, optimized and fine-tuned using **Unsloth** for maximum efficiency and performance. This allows for faster inference and lower memory usage while maintaining high accuracy in reasoning tasks.

## Key Features

*   **Hybrid RAG Architecture**: Combines vector search with logic-based verification.
*   **Smart Routing**: An intelligent router classifies queries to determine if they need fact-checking, web search, or casual conversation.
*   **Chain-of-Thought (CoT) Analysis**: The system acts as an analyst, breaking down claims and cross-referencing them with evidence before issuing a verdict.
*   **Unsloth Optimization**: Uses Unsloth's optimization techniques for the Llama 3 backbone, ensuring high-speed processing.
*   **Bilingual Support**: Natively supports both Turkish and English.

## Technical Architecture & Methodology

The system follows a strict pipeline to ensure accuracy:

1.  **Query Analysis**: An LLM-based router clarifies the user's intent.
2.  **Retrieval**:
    *   **Vector Search (FAISS)**: Performs semantic search on local indices.
    *   **Web Search**: Automatically falls back to DuckDuckGo for real-time information if local data is insufficient.
3.  **Reasoning (CoT)**: Validates the claim against the retrieved evidence using a specialized prompt structure.

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐
│    User     │────▶│    Router    │────▶│      Retrieval       │
│    Query    │     │   (Intent)   │     │ (Vector / Web Search)│
└─────────────┘     └──────────────┘     └──────────────────────┘
                                                    │
                                                    ▼
                                         ┌──────────────────────┐
                                         │   CoT Verification   │
                                         │ (Llama 3 + Unsloth)  │
                                         └──────────────────────┘
```

## Data Sources

The accuracy of this system relies on the following high-quality datasets:

*   **Teyit.org**: Turkey's leading fact-checking platform (primary source for Turkish claims).
*   **Wikipedia**: Used for general knowledge and encyclopedic verification.
*   **FEVER**: A large-scale dateset for Fact Extraction and VERification (used for training the reasoning capabilities).

> **⚠️ IMPORTANT NOTE**: Due to GitHub's file size limits, the **Unsloth adapters**, **Llama 3 model weights**, and **FAISS indices** are NOT included in this repository. You must download them separately to run the system locally.

## License

MIT License - See [LICENSE](LICENSE) for details.

---

# 🇹🇷 Proje Hakkında (Türkçe)

Türkçe ve İngilizce içerik için geliştirilmiş, RAG (İçe Aktarılan Bilgiyle Üretim) tabanlı bir doğrulama sistemi.

## Proje Tanımı

Büyük Dil Modelleri (LLM) bazen yanlış bilgileri doğruymuş gibi sunabilir. Bu proje, modelin cevaplarını doğrulanmış dış kaynaklara dayandırarak bu sorunu çözer.

Projenin temelinde **Unsloth** kütüphanesi ile optimize edilmiş ve fine-tune edilmiş **Llama 3** modeli bulunmaktadır. Unsloth kullanımı, modelin çok daha hızlı çalışmasını ve daha az bellek tüketmesini sağlarken, mantıksal çıkarım yeteneklerini en üst seviyede tutar.

## Temel Özellikler

*   **Hibrit RAG Mimarisi**: Vektör tabanlı aramayı mantıksal doğrulama ile birleştirir.
*   **Akıllı Yönlendirme**: Kullanıcının sohbet mi etmek istediğini yoksa bir iddia mı doğrulamak istediğini otomatik anlar.
*   **Zincirleme Düşünce (CoT) Analizi**: Model bir analist gibi davranarak iddiayı parçalara ayırır, kanıtlarla karşılaştırır ve "Doğru", "Yanlış" veya "Doğrulanamaz" kararı verir.
*   **Unsloth Optimizasyonu**: Llama 3 modeli Unsloth ile hızlandırılmıştır, bu sayede yüksek performanslı çıkarım (inference) yapılır.
*   **İki Dil Desteği**: Hem Türkçe hem de İngilizce sorgularla sorunsuz çalışır.

## Teknik Mimari ve Yöntem

Sistem, doğruluğu sağlamak için şu akışı izler:

1.  **Sorgu Analizi**: Yönlendirici modül, kullanıcı niyetini tespit eder.
2.  **Bilgi Getirme (Retrieval)**:
    *   **Vektör Arama (FAISS)**: Yerel veritabanında anlamsal arama yapar.
    *   **Web Arama**: Yerel veri yetersizse DuckDuckGo üzerinden güncel internet taraması yapar.
3.  **Mantık Yürütme (CoT)**: Llama 3 modeli, bulunan kanıtları iddia ile karşılaştırarak sonuca varır.

```
┌─────────────┐     ┌──────────────┐     ┌──────────────────────┐
│  Kullanıcı  │────▶│ Yönlendirici │────▶│     Veri Arama       │
│   Sorgusu   │     │   (Niyet)    │     │ (Vektör / Web Ara)   │
└─────────────┘     └──────────────┘     └──────────────────────┘
                                                    │
                                                    ▼
                                         ┌──────────────────────┐
                                         │    CoT Doğrulama     │
                                         │  (Llama 3 + Unsloth) │
                                         └──────────────────────┘
```

## Veri Kaynakları

Sistemin başarısı şu yüksek kaliteli veri kaynaklarına dayanmaktadır:

*   **Teyit.org**: Türkiye'nin önde gelen doğrulama platformu (Türkçe iddialar için ana kaynak).
*   **Wikipedia**: Genel kültür ve ansiklopedik doğrulama için kullanılır.
*   **FEVER**: Fact Extraction and VERification veri seti (muhakeme yeteneğinin eğitimi için kullanılmıştır).

> **⚠️ ÖNEMLİ NOT**: GitHub dosya boyutu sınırları nedeniyle, **Unsloth adaptörleri**, **Llama 3 model ağırlıkları** ve **FAISS indeksleri** bu depoda BULUNMAMAKTADIR. Sistemi yerel olarak çalıştırmak için bu dosyaları ayrıca indirmeniz gerekmektedir.

## Lisans

MIT License - Detaylar için [LICENSE](LICENSE) dosyasına bakınız.
