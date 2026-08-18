# DSPy Retrieval Guide

> **📋 Document Metadata**  
> **Type**: Technical Guide | **Audience**: Developers, ML Engineers | **Complexity**: Intermediate  
> **Cross-References**: [Modules Reference](dspy_modules_reference.md) | [Agents Guide](dspy_agents_guide.md) | [GNN Integration](dspy_gnn_integration_patterns.md)

## Overview

DSPy provides comprehensive support for Retrieval-Augmented Generation (RAG) through its retrieval modules. This guide covers integration with vector databases, ColBERT, and best practices for building optimizable RAG systems within GNN-based Active Inference architectures.

**Status**: ✅ Production Ready  
**Version**: 1.0

---

## Retrieval-Augmented Generation Fundamentals

RAG enhances LLM outputs by grounding them in relevant retrieved documents:

```
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Query     │ → │  Retriever   │ → │  Documents  │
└─────────────┘     └──────────────┘     └─────────────┘
                                                │
                                                ▼
┌─────────────┐     ┌──────────────┐     ┌─────────────┐
│   Answer    │ ← │  Generator   │ ← │  Context    │
└─────────────┘     └──────────────┘     └─────────────┘
```

---

## Core Retrieval Modules

### dspy.Retrieve

The base retrieval module that interfaces with configured retrieval models.

```python
import dspy

# Configure retrieval model globally
colbert = dspy.ColBERTv2(url='http://colbert-server:8893/api/search')
dspy.configure(rm=colbert)

# Use Retrieve module
retriever = dspy.Retrieve(k=5)
passages = retriever(query="Active Inference generative models")

# Access results
for passage in passages:
    print(passage.long_text)  # Full passage text
    print(passage.score)      # Relevance score
```

**Parameters**:
- `k`: Number of passages to retrieve
- `by_prob`: Whether to weight by probabilities (default: False)

---

### dspy.ColBERTv2

High-performance retrieval using ColBERT's late interaction mechanism.

**What is ColBERT?**

ColBERT (Contextualized Late Interaction over BERT) provides:
- **Multi-vector representations**: Each token gets its own embedding
- **Late interaction**: Similarity computed at retrieval time
- **Efficient indexing**: Pre-computed document embeddings
- **High recall**: Better semantic matching than single-vector methods

```python
# Direct ColBERTv2 usage
colbert = dspy.ColBERTv2(
    url='http://server:port/api/search',
    post_requests=False  # Use GET requests
)

results = colbert(
    query="Expected Free Energy calculation",
    k=10
)

for result in results:
    print(f"Score: {result['score']:.3f}")
    print(f"Text: {result['text'][:200]}...")
```

**Advantages for GNN**:
- Precise semantic matching for technical terminology
- Handles complex queries about Active Inference concepts
- Better retrieval of structured documentation

---

## Vector Database Integrations

### Qdrant

```python
from qdrant_client import QdrantClient

class QdrantRetriever(dspy.Retrieve):
    def __init__(self, collection_name, k=5):
        self.client = QdrantClient("localhost", port=6333)
        self.collection = collection_name
        self.k = k
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def forward(self, query):
        # Encode query
        query_vector = self.encoder.encode(query)
        
        # Search
        results = self.client.search(
            collection_name=self.collection,
            query_vector=query_vector,
            limit=self.k
        )
        
        return [
            dspy.Passage(
                long_text=r.payload['text'],
                score=r.score
            )
            for r in results
        ]

# Usage
qdrant_rm = QdrantRetriever("gnn_documentation", k=5)
dspy.configure(rm=qdrant_rm)
```

### Weaviate

```python
import weaviate

class WeaviateRetriever(dspy.Retrieve):
    def __init__(self, class_name, k=5):
        self.client = weaviate.Client("http://localhost:8080")
        self.class_name = class_name
        self.k = k
    
    def forward(self, query):
        result = (
            self.client.query
            .get(self.class_name, ["content", "title"])
            .with_near_text({"concepts": [query]})
            .with_limit(self.k)
            .with_additional(["certainty"])
            .do()
        )
        
        passages = result['data']['Get'][self.class_name]
        return [
            dspy.Passage(
                long_text=p['content'],
                score=p['_additional']['certainty']
            )
            for p in passages
        ]
```

### Pinecone

```python
from pinecone import Pinecone

class PineconeRetriever(dspy.Retrieve):
    def __init__(self, index_name, k=5):
        pc = Pinecone(api_key="your-api-key")
        self.index = pc.Index(index_name)
        self.k = k
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def forward(self, query):
        query_vector = self.encoder.encode(query).tolist()
        
        results = self.index.query(
            vector=query_vector,
            top_k=self.k,
            include_metadata=True
        )
        
        return [
            dspy.Passage(
                long_text=match['metadata']['text'],
                score=match['score']
            )
            for match in results['matches']
        ]
```

### Milvus

```python
from pymilvus import connections, Collection

class MilvusRetriever(dspy.Retrieve):
    def __init__(self, collection_name, k=5):
        connections.connect("default", host="localhost", port="19530")
        self.collection = Collection(collection_name)
        self.collection.load()
        self.k = k
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
    
    def forward(self, query):
        query_vector = self.encoder.encode(query).tolist()
        
        results = self.collection.search(
            data=[query_vector],
            anns_field="embedding",
            param={"metric_type": "COSINE", "params": {"nprobe": 10}},
            limit=self.k,
            output_fields=["text"]
        )
        
        return [
            dspy.Passage(
                long_text=hit.entity.get('text'),
                score=hit.score
            )
            for hit in results[0]
        ]
```

---

## Building RAG Pipelines

### Basic RAG Module

```python
class SimpleRAG(dspy.Module):
    """Basic retrieval-augmented generation."""
    
    def __init__(self, k=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=k)
        self.generate = dspy.ChainOfThought(
            'context, question -> answer'
        )
    
    def forward(self, question):
        # Retrieve relevant passages
        passages = self.retrieve(question)
        context = "\n\n".join([p.long_text for p in passages])
        
        # Generate answer
        return self.generate(context=context, question=question)
```

### Multi-Hop RAG

For complex questions requiring multiple retrieval steps:

```python
class MultiHopRAG(dspy.Module):
    """Multi-hop retrieval with iterative refinement."""
    
    def __init__(self, num_hops=3, k=5):
        super().__init__()
        self.num_hops = num_hops
        self.k = k
        
        self.retrieve = dspy.Retrieve(k=k)
        self.generate_query = dspy.ChainOfThought(
            'question, context -> search_query'
        )
        self.generate_answer = dspy.ChainOfThought(
            'question, context -> answer'
        )
    
    def forward(self, question):
        context = []
        
        for hop in range(self.num_hops):
            # Generate search query based on current context
            if hop == 0:
                query = question
            else:
                query_result = self.generate_query(
                    question=question,
                    context="\n".join(context)
                )
                query = query_result.search_query
            
            # Retrieve new passages
            passages = self.retrieve(query)
            new_context = [p.long_text for p in passages]
            context.extend(new_context)
        
        # Generate final answer
        return self.generate_answer(
            question=question,
            context="\n\n".join(context)
        )
```

### Self-Reflective RAG

RAG with self-critique and refinement:

```python
class ReflectiveRAG(dspy.Module):
    """RAG with self-reflection and answer refinement."""
    
    def __init__(self, k=5, max_refinements=2):
        super().__init__()
        self.k = k
        self.max_refinements = max_refinements
        
        self.retrieve = dspy.Retrieve(k=k)
        self.generate = dspy.ChainOfThought('context, question -> answer')
        self.critique = dspy.ChainOfThought(
            'question, answer, context -> is_sufficient: bool, missing_info, new_query'
        )
        self.refine = dspy.ChainOfThought(
            'question, previous_answer, new_context -> refined_answer'
        )
    
    def forward(self, question):
        # Initial retrieval and generation
        passages = self.retrieve(question)
        context = "\n\n".join([p.long_text for p in passages])
        answer = self.generate(context=context, question=question).answer
        
        # Iterative refinement
        for _ in range(self.max_refinements):
            critique = self.critique(
                question=question,
                answer=answer,
                context=context
            )
            
            if critique.is_sufficient:
                break
            
            # Retrieve additional context
            new_passages = self.retrieve(critique.new_query)
            new_context = "\n\n".join([p.long_text for p in new_passages])
            context += "\n\n" + new_context
            
            # Refine answer
            answer = self.refine(
                question=question,
                previous_answer=answer,
                new_context=new_context
            ).refined_answer
        
        return dspy.Prediction(answer=answer)
```

---

## GNN Documentation Retrieval

### GNN-Specific Retriever

```python
class GNNDocumentationRetriever(dspy.Module):
    """Specialized retriever for GNN documentation."""
    
    def __init__(self, gnn_index, k=5):
        super().__init__()
        self.gnn_retriever = dspy.ColBERTv2(url=gnn_index)
        self.k = k
        
        self.query_expander = dspy.ChainOfThought(
            'user_query -> expanded_query, gnn_terms: list[str]'
        )
    
    def forward(self, query):
        # Expand query with GNN terminology
        expanded = self.query_expander(user_query=query)
        
        # Search with expanded query
        results = self.gnn_retriever(
            expanded.expanded_query,
            k=self.k
        )
        
        return results


class GNNKnowledgeQA(dspy.Module):
    """Question answering over GNN documentation."""
    
    def __init__(self, gnn_index):
        super().__init__()
        self.retriever = GNNDocumentationRetriever(gnn_index)
        
        self.answer = dspy.ChainOfThought(
            'context, question -> '
            'answer, '
            'gnn_syntax_examples: list[str], '
            'related_concepts: list[str]'
        )
    
    def forward(self, question):
        # Retrieve GNN documentation
        docs = self.retriever(question)
        context = "\n\n".join([d['text'] for d in docs])
        
        # Generate comprehensive answer
        return self.answer(context=context, question=question)
```

### GNN Model Retrieval

Retrieve relevant GNN model examples:

```python
class GNNModelExampleRetriever(dspy.Module):
    """Retrieve similar GNN model examples."""
    
    def __init__(self, model_index):
        super().__init__()
        self.index = model_index
        self.retrieve = dspy.Retrieve(k=3)
        
        self.analyze = dspy.ChainOfThought(
            'user_description -> '
            'model_type, '
            'state_space_type, '
            'search_query'
        )
    
    def forward(self, description):
        # Analyze user's requirements
        analysis = self.analyze(user_description=description)
        
        # Retrieve similar models
        examples = self.retrieve(
            f"{analysis.model_type} {analysis.state_space_type} Active Inference GNN"
        )
        
        return {
            'analysis': analysis,
            'similar_models': examples
        }
```

---

## Optimizing RAG Systems

RAG pipelines can be optimized using DSPy optimizers:

```python
# Define evaluation metric
def rag_metric(example, prediction, trace=None):
    """Evaluate RAG answer quality."""
    # Check if answer contains key information
    key_info_found = any(
        key in prediction.answer.lower()
        for key in example.key_facts
    )
    
    # Check answer relevance
    relevance = len(set(prediction.answer.split()) & 
                   set(example.reference_answer.split())) / \
                len(set(example.reference_answer.split()))
    
    return 0.6 * key_info_found + 0.4 * relevance

# Training data
trainset = [
    dspy.Example(
        question="What is the A matrix in Active Inference?",
        key_facts=["observation", "state", "likelihood"],
        reference_answer="The A matrix represents the likelihood mapping from hidden states to observations"
    ).with_inputs('question')
    for _ in range(20)
]

# Optimize the RAG pipeline
optimizer = dspy.MIPROv2(metric=rag_metric)
optimized_rag = optimizer.compile(
    student=MultiHopRAG(num_hops=2),
    trainset=trainset
)
```

---

## Advanced Patterns

### Hierarchical Retrieval

For large document collections:

```python
class HierarchicalRetriever(dspy.Module):
    """Two-stage retrieval: coarse then fine."""
    
    def __init__(self, coarse_index, fine_index):
        super().__init__()
        self.coarse = dspy.ColBERTv2(url=coarse_index)
        self.fine = dspy.ColBERTv2(url=fine_index)
        
        self.reranker = dspy.ChainOfThought(
            'query, passages -> ranked_passages: list[int]'
        )
    
    def forward(self, query, k=5):
        # Coarse retrieval (document level)
        doc_ids = self.coarse(query, k=20)
        
        # Fine retrieval (passage level within docs)
        passages = []
        for doc_id in doc_ids[:10]:
            doc_passages = self.fine(
                f"{query} document:{doc_id}",
                k=5
            )
            passages.extend(doc_passages)
        
        # Rerank
        ranking = self.reranker(
            query=query,
            passages=[p['text'][:200] for p in passages]
        )
        
        return [passages[i] for i in ranking.ranked_passages[:k]]
```

### Hybrid Retrieval

Combine sparse and dense retrieval:

```python
class HybridRetriever(dspy.Module):
    """Combine BM25 and dense retrieval."""
    
    def __init__(self, bm25_index, dense_index, alpha=0.5):
        super().__init__()
        self.bm25 = BM25Retriever(bm25_index)
        self.dense = dspy.ColBERTv2(url=dense_index)
        self.alpha = alpha
    
    def forward(self, query, k=5):
        # Get results from both
        sparse_results = self.bm25(query, k=k*2)
        dense_results = self.dense(query, k=k*2)
        
        # Combine scores
        combined = {}
        for r in sparse_results:
            combined[r['id']] = {'text': r['text'], 'score': self.alpha * r['score']}
        
        for r in dense_results:
            if r['id'] in combined:
                combined[r['id']]['score'] += (1 - self.alpha) * r['score']
            else:
                combined[r['id']] = {'text': r['text'], 'score': (1 - self.alpha) * r['score']}
        
        # Sort and return top k
        sorted_results = sorted(
            combined.values(),
            key=lambda x: x['score'],
            reverse=True
        )
        
        return sorted_results[:k]
```

---

## Best Practices

### 1. Choose Appropriate k Values

```python
# Start conservative
k_initial = 3

# Increase if answer quality is low
k_optimized = 5

# Use larger k for multi-hop or complex queries
k_complex = 10
```

### 2. Preprocess Documents Effectively

```python
def preprocess_for_retrieval(document):
    """Prepare document for indexing."""
    # Chunk into manageable passages
    chunks = chunk_document(document, max_tokens=512, overlap=50)
    
    # Add metadata
    for chunk in chunks:
        chunk['source'] = document.title
        chunk['section'] = detect_section(chunk)
    
    return chunks
```

### 3. Monitor Retrieval Quality

```python
def evaluate_retrieval(retriever, eval_set):
    """Evaluate retrieval performance."""
    metrics = {'recall@k': [], 'precision@k': [], 'mrr': []}
    
    for example in eval_set:
        results = retriever(example.query)
        result_ids = [r['id'] for r in results]
        
        # Calculate metrics
        relevant = set(example.relevant_ids)
        retrieved = set(result_ids)
        
        metrics['recall@k'].append(
            len(relevant & retrieved) / len(relevant)
        )
        metrics['precision@k'].append(
            len(relevant & retrieved) / len(retrieved)
        )
    
    return {k: sum(v)/len(v) for k, v in metrics.items()}
```

---

## Related Resources

- **[Modules Reference](dspy_modules_reference.md)**: Core module documentation
- **[Agents Guide](dspy_agents_guide.md)**: Tool-using agents with retrieval
- **[Optimizers Guide](dspy_optimizers_guide.md)**: RAG optimization
- **[GNN Integration](dspy_gnn_integration_patterns.md)**: Active Inference retrieval

---

**Status**: ✅ Production Ready  
**Compliance**: GNN documentation standards  
**Maintenance**: Regular updates with new retrieval integrations
