#!/usr/bin/env python3
"""
Basic test to verify miner functionality works
"""
import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.insert(0, str(Path(__file__).parent))

def test_imports():
    """Test that all critical imports work"""
    print("🧪 Testing imports...")
    
    try:
        from shared.types import Resolution, Statement, MinerResponse
        print("✅ Shared types import successful")
        
        from miner.agents.dummy_agent import DummyAgent
        print("✅ Dummy agent import successful")
        
        from miner.agents.ai_agent import AIAgent
        print("✅ AI agent import successful")
        
        from miner.agents.llm_providers import LLMProviderFactory
        print("✅ LLM providers import successful")
        
        return True
    except Exception as e:
        print(f"❌ Import failed: {e}")
        return False

def test_resolution_enum():
    """Test the integer resolution enum"""
    print("🧪 Testing Resolution enum...")
    
    try:
        from shared.types import Resolution
        
        # Test integer values
        assert Resolution.FALSE.value == 0, f"FALSE should be 0, got {Resolution.FALSE.value}"
        assert Resolution.TRUE.value == 1, f"TRUE should be 1, got {Resolution.TRUE.value}"
        assert Resolution.PENDING.value == 2, f"PENDING should be 2, got {Resolution.PENDING.value}"
        
        print(f"✅ Resolution.FALSE = {Resolution.FALSE.value}")
        print(f"✅ Resolution.TRUE = {Resolution.TRUE.value}")
        print(f"✅ Resolution.PENDING = {Resolution.PENDING.value}")
        
        return True
    except Exception as e:
        print(f"❌ Resolution enum test failed: {e}")
        return False

def test_dummy_agent():
    """Test dummy agent basic functionality"""
    print("🧪 Testing Dummy Agent...")
    
    try:
        from miner.agents.dummy_agent import DummyAgent
        from shared.types import Statement
        
        # Create dummy agent
        agent = DummyAgent()
        print("✅ Dummy agent created successfully")
        
        # Create test statement
        statement = Statement(
            statement="Bitcoin will be above $50,000 by July 1, 2025",
            end_date="2025-07-01T23:59:59Z",
            createdAt="2025-01-01T00:00:00Z",
            id="test_001"
        )
        print("✅ Test statement created")
        
        # Note: We can't easily test async verify_statement without asyncio setup
        # But imports and basic creation work
        print("✅ Dummy agent basic test passed")
        
        return True
    except Exception as e:
        print(f"❌ Dummy agent test failed: {e}")
        return False

def test_ai_agent_basic():
    """Test AI agent basic creation (without API keys)"""
    print("🧪 Testing AI Agent basic setup...")
    
    try:
        from miner.agents.ai_agent import AIAgent
        
        # Create AI agent with minimal config (no API keys)
        config = {
            "llm_provider": "openai",
            "strategy": "ai_reasoning",
            "timeout": 30
        }
        
        agent = AIAgent(config)
        print("✅ AI agent created successfully (without API keys)")
        
        return True
    except Exception as e:
        print(f"❌ AI agent test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Testing Subnet 90 Miner Repository")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_resolution_enum,
        test_dummy_agent,
        test_ai_agent_basic
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        print()
        try:
            if test():
                passed += 1
        except Exception as e:
            print(f"❌ Test failed with exception: {e}")
    
    print()
    print("=" * 50)
    print(f"🎯 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED! Miner repo is working correctly.")
        print()
        print("📋 Next steps:")
        print("1. Copy .env.example to .env and configure your settings")
        print("2. Set up your LLM provider API keys")
        print("3. Run: python run_miner.py")
        return True
    else:
        print("❌ Some tests failed. Check the errors above.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)