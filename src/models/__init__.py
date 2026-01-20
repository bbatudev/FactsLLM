# -*- coding: utf-8 -*-
"""
Veri Modelleri
==============
Proje genelinde kullanılan data class'lar ve enum'lar.
"""

from dataclasses import dataclass
from enum import Enum
from typing import List


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
