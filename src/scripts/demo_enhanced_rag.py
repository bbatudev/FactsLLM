#!/usr/bin/env python3
"""
Demo script for Enhanced RAG System
Showcases the improved content analysis and intelligent response generation
"""

import sys
import os
import time

# Add project root directory to the path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(project_root)

from src.core.enhanced_rag_chat import (
    load_enhanced_rag_system, 
    process_query_with_enhanced_rag,
    format_enhanced_response
)

def demo_enhanced_rag():
    """Demonstrate the enhanced RAG system with various query types"""
    
    print("Enhanced RAG System Demo")
    print("=" * 60)
    print("This demo showcases:")
    print("- Intelligent content analysis")
    print("- Structured response generation")
    print("- Smart fallback logic")
    print("- Confidence scoring")
    print("=" * 60)
    
    # Load the enhanced RAG system
    print("\nLoading Enhanced RAG system...")
    try:
        rag_system = load_enhanced_rag_system()
        print("Enhanced RAG system loaded successfully!")
    except Exception as e:
        print(f"Failed to load Enhanced RAG system: {e}")
        return
    
    # Demo queries for different scenarios
    demo_queries = [
        {
            "query": "Türkiye'nin başkenti neresidir?",
            "mode": "teyit",
            "description": "Basic factual question (should get high confidence)"
        },
        {
            "query": "Mustafa Kemal Atatürk'ün hayatı hakkında bilgi verir misin?",
            "mode": "teyit", 
            "description": "Historical question (should extract key facts)"
        },
        {
            "query": "What is photosynthesis and how does it work?",
            "mode": "wiki",
            "description": "Scientific explanation (should provide comprehensive answer)"
        },
        {
            "query": "asdfghjklqwertyuiopzxcvbnm123456789",
            "mode": "teyit",
            "description": "Nonsense query (should trigger smart fallback)"
        },
        {
            "query": "What is the quantum entanglement state of a topological insulator?",
            "mode": "wiki",
            "description": "Highly specialized query (should trigger fallback)"
        }
    ]
    
    for i, demo in enumerate(demo_queries, 1):
        print(f"\n{'='*80}")
        print(f"DEMO {i}: {demo['description']}")
        print(f"{'='*80}")
        print(f"Query: {demo['query']}")
        print(f"Mode: {demo['mode'].upper()}")
        
        # Process the query
        start_time = time.time()
        try:
            response = process_query_with_enhanced_rag(
                demo['query'], 
                demo['mode'], 
                rag_system
            )
            end_time = time.time()
            
            # Display the enhanced response
            print(f"\nResponse Time: {end_time - start_time:.2f} seconds")
            print(format_enhanced_response(response))
            
            # Show content analysis details
            analysis = response.content_analysis
            print(f"\nCONTENT ANALYSIS DETAILS:")
            print(f"   • Overall Quality: {analysis.overall_quality:.3f}")
            print(f"   • Factual Accuracy: {analysis.factual_accuracy:.3f}")
            print(f"   • Relevance Score: {analysis.relevance_score:.3f}")
            print(f"   • Source Credibility: {analysis.source_credibility:.3f}")
            print(f"   • Completeness: {analysis.completeness_score:.3f}")
            print(f"   • Answer Directness: {analysis.answer_directness:.3f}")
            print(f"   • Key Facts Found: {len(analysis.key_facts)}")
            print(f"   • Analysis Notes: {analysis.analysis_notes}")
            
        except Exception as e:
            print(f"Error processing query: {e}")
        
        # Wait for user to continue
        if i < len(demo_queries):
            input("\nPress Enter to continue to next demo...")

def interactive_demo():
    """Interactive demo where user can input their own queries"""
    
    print("\nINTERACTIVE MODE")
    print("=" * 60)
    print("Enter your own queries to test the enhanced RAG system!")
    print("Type 'quit' to exit the interactive mode.")
    print("=" * 60)
    
    # Load the enhanced RAG system
    print("\nLoading Enhanced RAG system...")
    try:
        rag_system = load_enhanced_rag_system()
        print("Enhanced RAG system loaded successfully!")
    except Exception as e:
        print(f"Failed to load Enhanced RAG system: {e}")
        return
    
    while True:
        print("\n" + "-" * 60)
        query = input("Enter your question (or 'quit' to exit): ").strip()
        
        if query.lower() in ['quit', 'exit', 'q']:
            print("Thanks for trying the Enhanced RAG system!")
            break
        
        if not query:
            continue
        
        # Choose mode
        while True:
            mode = input("Choose mode (1: Wiki-EN, 2: Teyit-TR): ").strip()
            if mode == '1':
                mode = "wiki"
                break
            elif mode == '2':
                mode = "teyit"
                break
            else:
                print("Please enter 1 for Wiki-EN or 2 for Teyit-TR")
        
        # Process the query
        start_time = time.time()
        try:
            response = process_query_with_enhanced_rag(query, mode, rag_system)
            end_time = time.time()
            
            print(f"\nResponse Time: {end_time - start_time:.2f} seconds")
            print(format_enhanced_response(response))
            
        except Exception as e:
            print(f"Error processing query: {e}")

def main():
    """Main demo function"""
    print("Enhanced RAG System Demonstration")
    print("=" * 80)
    
    # Choose demo type
    while True:
        choice = input("""
Choose demo type:
1. Guided Demo (predefined queries)
2. Interactive Demo (your own queries)
3. Exit

Enter your choice (1-3): """).strip()
        
        if choice == '1':
            demo_enhanced_rag()
            break
        elif choice == '2':
            interactive_demo()
            break
        elif choice == '3':
            print("Goodbye!")
            break
        else:
            print("Please enter 1, 2, or 3")

if __name__ == "__main__":
    main()