# Results Analysis

## Experiment overview
- Corpus: 6 documents (speech1.txt, speech2.txt, speech3.txt, speech4.txt, speech5.txt, speech6.txt)
- Test set: 25 Q&A pairs (22 answerable, 3 unanswerable)
- Chunk sizes tested: Small (250 chunks, 25 overlap), Medium (550 chunks, 50 overlap), Large (900 chunks, 100 overlap)
- k (retrieval top-k): 3

## Summary results

| Metric | Small | Medium | Large |
|--------|-------|--------|-------|
| Hit Rate | 0.84 | 0.88 | 0.88 |
| MRR | 0.793 | 0.793 | 0.793 |
| Precision@3 | 0.56 | 0.48 | 0.36 |
| ROUGE-L | 0.266 | 0.266 | 0.273 |
| BLEU | 0.065 | 0.068 | 0.076 |
| Cosine Similarity | 0.562 | 0.581 | 0.593 |
| Faithfulness | 0.303 | 0.451 | 0.577 |
| Answer Relevance | 0.721 | 0.731 | 0.701 |
| Correct Refusal Rate | 0.667 | 0.333 | 0.667 |

## Which chunking worked best?
**Medium chunk size (550 tokens)** performed best overall, achieving the highest hit rate (0.88) and strong performance across most metrics. While Large chunks had slightly better faithfulness (0.577) and cosine similarity (0.593), Medium chunks provided the best balance with good retrieval performance and answer quality.

## Retrieval performance
- Hit Rate: Medium and Large both achieved 0.88, significantly better than Small (0.84)
- MRR: Consistent across all chunk sizes at 0.793, indicating similar ranking quality
- Precision@3: Small performed best (0.56), declining with larger chunks (Medium: 0.48, Large: 0.36)

## Answer quality
- ROUGE-L mean: Large chunks slightly ahead (0.273) vs Small/Medium (~0.266)
- BLEU mean: Large chunks best (0.076), followed by Medium (0.068) and Small (0.065)
- Cosine similarity mean: Consistent improvement with chunk size (Small: 0.562, Medium: 0.581, Large: 0.593)
- Faithfulness: Strong correlation with chunk size (Small: 0.303, Medium: 0.451, Large: 0.577)

## Common failure modes
- Missing context due to chunk boundary splits (evident in Small chunks' lower hit rate)
- Hallucinations when LLM lacks direct evidence in retrieved chunks (low faithfulness scores, especially for Small chunks)
- Inconsistent refusal behavior for unanswerable questions (Medium chunks only achieved 33% correct refusal rate)
- Answer relevance decreasing with larger chunks (Large: 0.701 vs Small/Medium: ~0.72+)

## Recommendations
1. Use **Medium chunk size (550 tokens with 50 overlap)** for this corpus as it provides the best balance of retrieval and generation performance.
2. Keep k=3 for retrieval as precision@3 results suggest diminishing returns with larger chunks.
3. Add reranking (cross-encoder) to improve relevance of retrieved chunks, especially for larger chunk sizes.
4. Implement faithfulness filtering using NLI models to reduce hallucinations, particularly important for smaller chunk configurations.
5. Add document-level metadata (speech titles, topics) into embeddings to improve context understanding.
6. Fine-tune refusal detection to improve handling of unanswerable questions (current performance ranges from 33-67%).

## Next steps
- Add an automatic script to produce analysis plots showing the trade-offs between chunk size and different metrics.
- Implement significance testing across chunk sizes to validate performance differences.
- Experiment with hybrid approaches combining multiple chunk sizes.
- Add semantic chunking methods that respect document structure boundaries.

