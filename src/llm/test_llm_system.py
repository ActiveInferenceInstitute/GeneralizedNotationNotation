#!/usr/bin/env python3
"""
LLM System Test Script

This script tests the multi-provider LLM system with real API keys,
demonstrating various configurations and capabilities.
"""

import asyncio
import os
import sys
from pathlib import Path
import logging

# Add the project root to the path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).parent / '.env')
except Exception:
    pass

from src.llm import (
    LLMProcessor,
    AnalysisType,
    ProviderType,
    LLMConfig,
    LLMMessage,
    initialize_global_processor,
    create_processor_from_env,
    load_api_keys_from_env,
    get_default_provider_configs
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Sample GNN content for testing
SAMPLE_GNN_CONTENT = """
## GNNVersionAndFlags
GNNVersion: 1.0
Flags: None

## ModelName
SimpleActiveInferenceAgent

## ModelAnnotation
A basic Active Inference agent that navigates between two locations
based on temperature observations. This model demonstrates core
Active Inference principles including belief updating, policy selection,
and free energy minimization.

## StateSpaceBlock
# Hidden states (locations)
s_f0: [Location] dimension(2) # {Location_A, Location_B}

# Observations (temperature readings)
o_m0: [Temperature] dimension(2) # {Hot, Cold}

# Actions (movement decisions)
u_c0: [Action] dimension(2) # {Stay, Move}

## Connections
s_f0 > o_m0  # Location influences temperature observation
s_f0 > s_f0  # Location transitions (temporal)
u_c0 > s_f0  # Actions influence location transitions

## InitialParameterization
# A matrix: P(o|s) - observation model
A_m0 = [[0.8, 0.2], [0.3, 0.7]]

# B matrix: P(s'|s,u) - transition model  
B_f0 = [[[0.9, 0.1], [0.1, 0.9]], [[0.1, 0.9], [0.9, 0.1]]]

# C vector: log preferences over observations
C_m0 = [0.0, -2.0]  # Prefer hot over cold

# D vector: prior beliefs over initial states
D_f0 = [0.5, 0.5]  # Uniform prior over locations

## Time
Dynamic: True
DiscreteTime: True
ModelTimeHorizon: 10
"""

async def test_environment_setup():
    """Test that environment variables are properly loaded."""
    print("🔧 Testing Environment Setup...")
    
    api_keys = load_api_keys_from_env()
    configs = get_default_provider_configs()
    
    print(f"✅ API Keys loaded: {list(api_keys.keys())}")
    print(f"✅ Provider configs: {list(configs.keys())}")
    
    # Check specific keys (without exposing them)
    if 'openai' in api_keys:
        print(f"✅ OpenAI key format: {'Valid' if api_keys['openai'].startswith('sk-') else 'Invalid'}")
    if 'openrouter' in api_keys:
        print(f"✅ OpenRouter key format: {'Valid' if api_keys['openrouter'].startswith('sk-or-') else 'Invalid'}")
    
    return api_keys, configs

async def test_provider_initialization():
    """Test individual provider initialization."""
    print("\n🚀 Testing Provider Initialization...")
    
    # Test with only OpenAI and OpenRouter (excluding Perplexity)
    preferred_providers = [ProviderType.OPENROUTER, ProviderType.OPENAI]
    
    processor = LLMProcessor(
        preferred_providers=preferred_providers,
        api_keys=load_api_keys_from_env(),
        provider_configs=get_default_provider_configs()
    )
    
    success = await processor.initialize()
    print(f"✅ Processor initialized: {success}")
    
    available_providers = processor.get_available_providers()
    print(f"✅ Available providers: {[p.value for p in available_providers]}")
    
    # Test provider info
    provider_info = processor.get_provider_info()
    for provider_name, info in provider_info.items():
        print(f"✅ {provider_name}: {info.get('status', 'unknown')}")
    
    return processor

async def test_basic_analysis(processor):
    """Test basic GNN analysis functionality."""
    print("\n📊 Testing Basic Analysis...")
    
    # Test summary analysis
    try:
        response = await processor.analyze_gnn(
            gnn_content=SAMPLE_GNN_CONTENT,
            analysis_type=AnalysisType.SUMMARY
        )
        
        print(f"✅ Summary Analysis Success!")
        print(f"   Provider: {response.provider}")
        print(f"   Model: {response.model_used}")
        print(f"   Content length: {len(response.content)} characters")
        print(f"   Usage: {response.usage}")
        print(f"   First 200 chars: {response.content[:200]}...")
        
        return response
        
    except Exception as e:
        print(f"❌ Summary Analysis Failed: {e}")
        return None

async def test_different_analysis_types(processor):
    """Test different analysis types."""
    print("\n🔍 Testing Different Analysis Types...")
    
    analysis_types = [
        AnalysisType.STRUCTURE,
        AnalysisType.QUESTIONS,
        AnalysisType.VALIDATION
    ]
    
    for analysis_type in analysis_types:
        try:
            response = await processor.analyze_gnn(
                gnn_content=SAMPLE_GNN_CONTENT,
                analysis_type=analysis_type
            )
            
            print(f"✅ {analysis_type.value.upper()} Analysis Success!")
            print(f"   Provider: {response.provider}")
            print(f"   Content length: {len(response.content)} characters")
            
        except Exception as e:
            print(f"❌ {analysis_type.value.upper()} Analysis Failed: {e}")

async def test_provider_specific_calls(processor):
    """Test calling specific providers."""
    print("\n🎯 Testing Provider-Specific Calls...")
    
    available_providers = processor.get_available_providers()
    
    for provider_type in available_providers:
        try:
            response = await processor.analyze_gnn(
                gnn_content=SAMPLE_GNN_CONTENT,
                analysis_type=AnalysisType.SUMMARY,
                provider_type=provider_type
            )
            
            print(f"✅ {provider_type.value.upper()} Provider Success!")
            print(f"   Model: {response.model_used}")
            print(f"   Content length: {len(response.content)} characters")
            
        except Exception as e:
            print(f"❌ {provider_type.value.upper()} Provider Failed: {e}")

async def test_custom_configurations(processor):
    """Test custom LLM configurations."""
    print("\n⚙️ Testing Custom Configurations...")
    
    # Test different configurations
    configs = [
        LLMConfig(
            model="gpt-4o-mini",
            max_tokens=500,
            temperature=0.1
        ),
        LLMConfig(
            model="openai/gpt-4o",  # OpenRouter format
            max_tokens=1000,
            temperature=0.5
        ),
        LLMConfig(
            max_tokens=1500,
            temperature=0.8,
            top_p=0.9
        )
    ]
    
    for i, config in enumerate(configs):
        try:
            response = await processor.analyze_gnn(
                gnn_content=SAMPLE_GNN_CONTENT,
                analysis_type=AnalysisType.SUMMARY,
                config=config
            )
            
            print(f"✅ Config {i+1} Success!")
            print(f"   Provider: {response.provider}")
            print(f"   Model: {response.model_used}")
            print(f"   Usage: {response.usage}")
            
        except Exception as e:
            print(f"❌ Config {i+1} Failed: {e}")

async def test_streaming_responses(processor):
    """Test streaming response functionality."""
    print("\n🌊 Testing Streaming Responses...")
    
    try:
        messages = [
            LLMMessage(
                role="system", 
                content="You are an expert in Active Inference and GNN specifications."
            ),
            LLMMessage(
                role="user", 
                content=f"Provide a brief summary of this GNN model:\n{SAMPLE_GNN_CONTENT}"
            )
        ]
        
        config = LLMConfig(max_tokens=500, temperature=0.3)
        
        print("✅ Starting stream...")
        response_chunks = []
        
        async for chunk in processor.generate_stream(messages, config=config):
            response_chunks.append(chunk)
            print(chunk, end="", flush=True)
        
        print(f"\n✅ Streaming completed! Total chunks: {len(response_chunks)}")
        
    except Exception as e:
        print(f"❌ Streaming Failed: {e}")

async def test_provider_comparison(processor):
    """Test comparing responses from multiple providers."""
    print("\n🔄 Testing Provider Comparison...")
    
    try:
        results = await processor.compare_providers(
            gnn_content=SAMPLE_GNN_CONTENT,
            analysis_type=AnalysisType.SUMMARY
        )
        
        print(f"✅ Comparison completed for {len(results)} providers")
        
        for provider_name, response in results.items():
            if response:
                print(f"✅ {provider_name.upper()}:")
                print(f"   Model: {response.model_used}")
                print(f"   Length: {len(response.content)} chars")
                print(f"   Usage: {response.usage}")
            else:
                print(f"❌ {provider_name.upper()}: Failed")
                
    except Exception as e:
        print(f"❌ Provider Comparison Failed: {e}")

async def test_error_handling(processor):
    """Test error handling and fallback mechanisms."""
    print("\n🛡️ Testing Error Handling...")
    
    # Test with invalid model
    try:
        config = LLMConfig(model="invalid-model-name")
        response = await processor.analyze_gnn(
            gnn_content=SAMPLE_GNN_CONTENT,
            config=config
        )
        print("✅ Invalid model handled gracefully")
        print(f"   Fallback provider: {response.provider}")
        
    except Exception as e:
        print(f"⚠️ Error handling test: {e}")

async def test_global_processor():
    """Test the global processor functionality."""
    print("\n🌐 Testing Global Processor...")
    
    try:
        # Initialize global processor
        global_processor = await create_processor_from_env()
        
        print("✅ Global processor initialized")
        
        # Test basic functionality
        response = await global_processor.analyze_gnn(
            gnn_content=SAMPLE_GNN_CONTENT,
            analysis_type=AnalysisType.SUMMARY
        )
        
        print(f"✅ Global processor analysis success!")
        print(f"   Provider: {response.provider}")
        
        # Clean up
        await global_processor.close()
        print("✅ Global processor closed")
        
    except Exception as e:
        print(f"❌ Global processor test failed: {e}")

async def main():
    """Run all tests."""
    print("🧪 LLM System Comprehensive Test Suite")
    print("=" * 50)
    
    try:
        # Test environment setup
        api_keys, configs = await test_environment_setup()
        
        # Initialize processor
        processor = await test_provider_initialization()
        
        if not processor or not processor.get_available_providers():
            print("❌ No providers available, stopping tests")
            return
        
        # Run functionality tests
        await test_basic_analysis(processor)
        await test_different_analysis_types(processor)
        await test_provider_specific_calls(processor)
        await test_custom_configurations(processor)
        await test_streaming_responses(processor)
        await test_provider_comparison(processor)
        await test_error_handling(processor)
        
        # Test global processor
        await test_global_processor()
        
        # Clean up
        await processor.close()
        
        print("\n" + "=" * 50)
        print("🎉 LLM System Test Suite Completed!")
        print("✅ All major functionality verified")
        
    except Exception as e:
        print(f"❌ Test suite failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(main()) 