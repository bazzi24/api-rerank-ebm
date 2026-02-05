# EBM Rerank API - Deployment Guide

## 🎯 What I Optimized

### Changes Made:

1. **Simplified Response Format**
   - Removed `document` field (RAGFlow doesn't use it)
   - Removed `id` field (RAGFlow doesn't use it)
   - Kept only `index` and `relevance_score` (what RAGFlow actually reads)

2. **Better Logging**
   - Shows first few documents in each request
   - Clearer energy/score ranges
   - More concise output

3. **Added Test Endpoint**
   - `/test-rerank` - Test without RAGFlow
   - Useful for debugging

4. **Performance**
   - Removed unnecessary data serialization
   - Faster response times

## 📦 Files Created

1. **`ebm/main_optimized.py`** - Optimized FastAPI application
2. **`diagnose_ragflow.py`** - Diagnostic tool
3. **`DEPLOYMENT_GUIDE.md`** - This file

## 🚀 Deployment Steps

### Step 1: Switch to Optimized Version

```bash
cd /media/bazzi/Bazzi/GitHubBazzi/api-rerank-ebm

# Backup current version
cp ebm/main.py ebm/main_backup.py

# Use optimized version
cp ebm/main_optimized.py ebm/main.py
```

### Step 2: Restart API

```bash
# Stop current server (Ctrl+C)

# Start optimized version
uvicorn ebm.main:app --host 0.0.0.0 --port 8000 --reload
```

You should see:
```
🚀 EBM Reranker API Started
Version: 2.0.0 (Optimized)
Model: sentence-transformers/msmarco-MiniLM-L6-v3
Device: cuda
```

### Step 3: Run Diagnostics

```bash
# Make executable
chmod +x diagnose_ragflow.py

# Run diagnostics
python diagnose_ragflow.py
```

This will check:
- ✅ API health
- ✅ Rerank endpoint
- ✅ RAGFlow configuration
- ✅ RAGFlow backend status

### Step 4: Test API

```bash
# Test the test endpoint
curl -X POST http://127.0.0.1:8000/test-rerank | jq
```

You should see:
```json
{
  "test_query": "What is machine learning?",
  "results": [...],
  "status": "✅ Test passed!"
}
```

### Step 5: Test with RAGFlow

1. Make sure RAGFlow is running
2. Go to a chat with documents
3. Ask a question
4. Watch your API terminal for:
   ```
   🔄 RERANK REQUEST at ...
   ✅ Completed in X.XXXs
   ```

## 🔍 Troubleshooting

### Issue 1: No Rerank Calls in API Logs

**Symptoms:** You query RAGFlow but API shows no activity

**Causes & Solutions:**

1. **No Documents Found**
   - Query doesn't match any documents
   - **Fix:** Ask questions about content you know is in the KB

2. **Empty Knowledge Base**
   - Knowledge base has 0 documents
   - **Fix:** Upload and process documents

3. **Rerank Not Enabled**
   - KB doesn't have reranking enabled
   - **Fix:** KB Settings → Enable Reranking → Select EBM

4. **EBM Not Configured**
   - EBM not added in UI
   - **Fix:** Settings → Model Providers → Add EBM

### Issue 2: API Returns Errors

**Symptoms:** POST /v1/rerank returns 500

**Debug Steps:**

1. Check API logs for error details
2. Verify model file exists: `ls -lh models/ebm_hardneg.pt`
3. Check CUDA availability: `python -c "import torch; print(torch.cuda.is_available())"`
4. Test with curl:
   ```bash
   curl -X POST http://127.0.0.1:8000/v1/rerank \
     -H "Content-Type: application/json" \
     -d '{"query":"test","documents":["doc1","doc2"],"top_n":2}'
   ```

### Issue 3: Slow Performance

**Symptoms:** Reranking takes >2 seconds

**Solutions:**

1. **Use GPU**
   - Check: `python -c "import torch; print(torch.cuda.is_available())"`
   - If False, model runs on CPU (slow)

2. **Reduce Document Length**
   - Model truncates to 512 tokens
   - Pre-truncate documents in RAGFlow

3. **Batch Size**
   - Current implementation processes all docs at once
   - For >100 docs, consider batching

## 📊 Performance Expectations

| Documents | GPU (cuda) | CPU |
|-----------|------------|-----|
| 2-5 docs  | ~50ms      | ~200ms |
| 10 docs   | ~100ms     | ~500ms |
| 50 docs   | ~400ms     | ~2s |
| 100 docs  | ~800ms     | ~4s |

## 🎯 What Makes a "Relevant Question"?

For rerank to trigger, your query must:

1. **Match documents in the KB**
   - If KB has machine learning docs → Ask about ML
   - If KB has recipe docs → Ask about cooking

2. **Retrieve results**
   - RAGFlow's vector search must find documents
   - No results = no rerank

3. **Have reranking enabled**
   - Check KB settings
   - Reranking must be ON
   - EBM must be selected

## 💡 Testing Strategy

### Test 1: Verify API Works

```bash
curl -X POST http://127.0.0.1:8000/test-rerank
```

### Test 2: Verify RAGFlow Can Call API

```bash
# From RAGFlow directory
cd /media/bazzi/Bazzi/GitHubBazzi/ragflow
python /tmp/claude-1000/.../test_like_ragflow.py
```

### Test 3: Verify End-to-End

1. Upload a simple text file to RAGFlow KB
2. Ask a question about that file
3. Check API logs for rerank request

## 📝 Monitoring

### Watch API Logs

```bash
# Terminal 1: API
uvicorn ebm.main:app --host 0.0.0.0 --port 8000

# You'll see each request:
# >>> POST /v1/rerank from 127.0.0.1
# 🔄 RERANK REQUEST at ...
# ✅ Completed in 0.123s
# <<< 200
```

### Watch RAGFlow Logs

```bash
# Terminal 2: RAGFlow
tail -f /media/bazzi/Bazzi/GitHubBazzi/ragflow/logs/ragflow_server.log | grep -i "EBM\|rerank"
```

## ✅ Success Checklist

- [ ] Optimized API running on port 8000
- [ ] `/health` returns healthy
- [ ] `/test-rerank` works
- [ ] `diagnose_ragflow.py` all tests pass
- [ ] RAGFlow backend running
- [ ] EBM configured in RAGFlow UI
- [ ] Knowledge base has documents
- [ ] Documents are processed
- [ ] Reranking enabled for KB
- [ ] EBM selected as rerank model
- [ ] Query matches KB content
- [ ] API logs show rerank requests

## 🎉 Expected Behavior

When everything works:

1. **User asks:** "What is deep learning?"
2. **RAGFlow:**
   - Searches KB
   - Finds 10 documents
   - Calls your API: `POST /v1/rerank`
3. **Your API:**
   - Receives query + 10 documents
   - Computes EBM scores
   - Returns ranked results
4. **RAGFlow:**
   - Uses reranked order
   - Shows best results first

## 🆘 Still Not Working?

Run this complete diagnostic:

```bash
# 1. Check API
curl http://127.0.0.1:8000/health

# 2. Test rerank
curl -X POST http://127.0.0.1:8000/test-rerank

# 3. Check RAGFlow
ps aux | grep ragflow_server

# 4. Check config
grep -A5 "rerank_model" /media/bazzi/Bazzi/GitHubBazzi/ragflow/conf/service_conf.yaml

# 5. Run full diagnostics
python diagnose_ragflow.py
```

Share the output and I can help debug further!

## 📚 Additional Resources

- API Docs: http://127.0.0.1:8000/docs
- Test Endpoint: http://127.0.0.1:8000/test-rerank
- Health Check: http://127.0.0.1:8000/health
