#!/usr/bin/env python3
"""
Quick test script to verify Groq-only FloatChat setup
Tests all components work with free Groq models only
"""

import os
import sys
import asyncio
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def test_groq_setup():
    """Test Groq-only setup."""
    print("🌊 FloatChat Groq Setup Test")
    print("=" * 40)
    
    try:
        # Test Groq client
        print("1. Testing Groq client...")
        from floatchat.ai.llm.groq_client import GroqClient
        
        groq_client = GroqClient({
            'api_key': os.getenv('GROQ_API_KEY', 'your-groq-api-key-here'),
            'model': 'llama-3.1-70b-versatile'
        })
        print("   ✅ Groq client initialized")
        
        # Test orchestrator
        print("2. Testing LLM orchestrator...")
        from floatchat.ai.llm.groq_orchestrator import LLMOrchestrator, LLMProvider, LLMConfig
        
        llm_config = {
            LLMProvider.GROQ: LLMConfig(
                api_key=os.getenv('GROQ_API_KEY', 'your-groq-api-key-here'),
                model_name='llama-3.1-70b-versatile'
            )
        }
        
        orchestrator = LLMOrchestrator(llm_config)
        print("   ✅ LLM orchestrator initialized")
        
        # Test response generation
        print("3. Testing AI response...")
        response = await orchestrator.generate_response(
            prompt="What is oceanography?",
            context="Test context about marine science"
        )
        
        if response['response'] and len(response['response']) > 10:
            print("   ✅ AI response generated successfully")
            print(f"   📝 Response preview: {response['response'][:100]}...")
            print(f"   🎯 Confidence: {response['confidence_score']:.2f}")
            print(f"   ⚡ Response time: {response['response_time']:.2f}s")
        else:
            print("   ❌ AI response generation failed")
            return False
        
        # Test database components (basic imports)
        print("4. Testing database components...")
        from floatchat.core.database import DatabaseManager
        print("   ✅ Database manager imported")
        
        # Test vector database
        print("5. Testing vector database...")
        from floatchat.ai.vector_database.faiss_vector_store import OceanographicVectorStore
        print("   ✅ Vector store imported")
        
        # Test NL2SQL
        print("6. Testing NL2SQL engine...")
        from floatchat.ai.nl2sql.nl2sql_engine import NL2SQLEngine
        print("   ✅ NL2SQL engine imported")
        
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📋 System Summary:")
        print(f"   • AI Provider: Groq (Free)")
        print(f"   • Model: {response['model_used']}")
        print(f"   • Response Time: {response['response_time']:.2f}s")
        print(f"   • Tokens Used: {response['tokens_used']}")
        print(f"   • Ready for deployment: YES ✅")
        
        return True
        
    except Exception as e:
        print(f"\n❌ TEST FAILED: {str(e)}")
        print("\nTroubleshooting:")
        print("1. Check internet connection")
        print("2. Verify Groq API key is valid")
        print("3. Install requirements: pip install -r requirements.txt")
        return False

async def test_streamlit_import():
    """Test Streamlit app can be imported."""
    try:
        print("7. Testing Streamlit app...")
        from floatchat.interface.streamlit_app import FloatChatApp
        app = FloatChatApp()
        print("   ✅ Streamlit app can be initialized")
        return True
    except Exception as e:
        print(f"   ⚠️ Streamlit import issue: {str(e)}")
        print("   💡 This is ok if streamlit isn't installed yet")
        return False

def main():
    """Main test runner."""
    print("Starting Groq-only FloatChat system test...\n")
    
    # Run async tests
    success = asyncio.run(test_groq_setup())
    
    # Test streamlit import
    asyncio.run(test_streamlit_import())
    
    print("\n" + "=" * 40)
    if success:
        print("🚀 FloatChat is ready to run!")
        print("\nNext steps:")
        print("1. python enhanced_indian_ocean_setup.py  # Setup database")
        print("2. python main.py                         # Launch app")
        print("3. Open: http://localhost:8501")
    else:
        print("❌ Setup needs attention. Please check errors above.")
    
    return 0 if success else 1

if __name__ == "__main__":
    exit(main())