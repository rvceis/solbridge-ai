# 🤗 Hugging Face Spaces Deployment Guide

Deploy SolBridge AI ML Service to Hugging Face Spaces in under 5 minutes!

## Why Hugging Face Spaces?

✅ **FREE unlimited hosting** (persistent, not serverless)  
✅ **2 CPU cores + 16GB RAM** on free tier  
✅ **Prophet & ML libraries pre-installed** (has C compiler)  
✅ **Auto-scaling** infrastructure  
✅ **Public URL** with SSL certificate  
✅ **Git-based deployment** (push to deploy)  
✅ **Build logs & monitoring** built-in  

## Prerequisites

1. **Hugging Face Account** - Create at [huggingface.co](https://huggingface.co/join)
2. **Git installed** - Already on your system
3. **Access token** - Get from [Settings > Access Tokens](https://huggingface.co/settings/tokens)

---

## 🚀 Quick Deploy (3 Steps)

### Step 1: Create a Space on Hugging Face

1. Go to https://huggingface.co/new-space
2. Fill in:
   - **Space name**: `solbridge-ai` (or your choice)
   - **License**: MIT
   - **SDK**: **Docker** ⚠️ (important!)
   - **Hardware**: CPU basic (free)
3. Click **Create Space**

### Step 2: Run Deployment Script

**On Windows (PowerShell):**
```powershell
cd "c:\Users\akash\Desktop\Main EL\solbridge-ai"
.\deploy-huggingface.ps1
```

**On Linux/Mac:**
```bash
cd /path/to/solbridge-ai
chmod +x deploy-huggingface.sh
./deploy-huggingface.sh
```

### Step 3: Authenticate

When prompted for password:
- **Username**: Your Hugging Face username
- **Password**: Your Hugging Face **token** (NOT your password)
  - Get token: https://huggingface.co/settings/tokens
  - Click "New token" → "Write" access → Copy

---

## ⚙️ Manual Deployment (Alternative)

If the script doesn't work:

```powershell
# 1. Copy Hugging Face files
Copy-Item README_HF.md README.md -Force
Copy-Item Dockerfile.hf Dockerfile -Force

# 2. Add Hugging Face remote (replace with YOUR username and space name)
git remote add huggingface https://huggingface.co/spaces/YOUR_USERNAME/solbridge-ai

# 3. Commit
git add README.md Dockerfile
git commit -m "Configure for Hugging Face Spaces"

# 4. Push (use HF token as password)
git push huggingface main --force
```

---

## 📋 After Deployment

### Your Space is Building...

1. Visit your Space: `https://huggingface.co/spaces/YOUR_USERNAME/solbridge-ai`
2. Wait 2-5 minutes for build to complete
3. Look for **"Running"** status (green)

### Test Your API

```bash
# Health check
curl https://YOUR_USERNAME-solbridge-ai.hf.space/health

# Solar forecast
curl -X POST https://YOUR_USERNAME-solbridge-ai.hf.space/api/v1/forecast/solar \
  -H "Content-Type: application/json" \
  -d '{
    "host_id": "H-1",
    "panel_capacity_kw": 5,
    "forecast_hours": 24,
    "historical_data": [],
    "weather_forecast": []
  }'
```

### API Documentation

- **Swagger UI**: `https://YOUR_USERNAME-solbridge-ai.hf.space/docs`
- **ReDoc**: `https://YOUR_USERNAME-solbridge-ai.hf.space/redoc`

---

## 🔧 Configuration

### Environment Variables

Add in Space Settings → Variables:

```bash
ENVIRONMENT=production
LOG_LEVEL=INFO
PORT=7860
```

### Update Your Frontend

Replace Render URL with Hugging Face URL:

```javascript
// Old
const API_URL = "https://solbridge-ai.onrender.com"

// New
const API_URL = "https://YOUR_USERNAME-solbridge-ai.hf.space"
```

---

## 🐛 Troubleshooting

### Build Failed

**Error**: `No space named 'YOUR_USERNAME/solbridge-ai'`
- Make sure you created the Space first on Hugging Face
- Check the Space name matches exactly

**Error**: `Authentication failed`
- Use your Hugging Face **token** as password (not your account password)
- Get token from: https://huggingface.co/settings/tokens

**Error**: `Dockerfile not found`
- Make sure `Dockerfile.hf` exists in your repo
- Run the deployment script, it copies it automatically

### Application Not Starting

1. Check build logs in Space → "Logs" tab
2. Look for errors in startup
3. Common issues:
   - Missing dependencies in `requirements.txt`
   - Port mismatch (should be 7860)
   - Import errors

### Slow Cold Starts

Hugging Face Spaces on free tier can have 10-30 second cold starts if inactive.

**Solutions:**
- Upgrade to Persistent hardware ($0.50/hour when running)
- Keep Space active with periodic pings

---

## 🆚 Comparison: Render vs Hugging Face

| Feature | Render | Hugging Face |
|---------|--------|--------------|
| **Free Tier** | Limited hours | Unlimited |
| **RAM** | 512 MB | 16 GB |
| **CPU** | Shared | 2 cores |
| **C Compiler** | ❌ No | ✅ Yes |
| **Prophet Works** | ❌ No | ✅ Yes |
| **Cold Starts** | 30-60s | 10-30s |
| **Build Time** | 2-3 min | 2-5 min |
| **Auto-scale** | Yes | Yes |
| **Custom Domain** | ✅ Yes | ⚠️ Paid |

**Winner**: Hugging Face for ML applications! 🏆

---

## 📈 Next Steps

### After Successful Deployment:

1. **Update frontend** with new API URL
2. **Test all endpoints** using Swagger UI
3. **Monitor logs** in Hugging Face Space
4. **Enable metrics** (optional)
5. **Add rate limiting** for production

### Upgrade Options:

- **Persistent Hardware** - $0.50/hr, no cold starts
- **GPU** - For deep learning (T4 GPU available)
- **Private Space** - Hide from public listing
- **Custom domain** - Use your own domain

---

## 🎉 Success Checklist

- [x] Created Hugging Face Space
- [x] Ran deployment script
- [x] Authenticated with token
- [x] Build completed successfully
- [x] API responds to health check
- [x] Solar forecast returns predictions
- [x] Updated frontend API URL

---

## 💡 Pro Tips

1. **Keep models in Git** - Hugging Face supports large files (use Git LFS)
2. **Use secrets** - Store API keys in Space settings, not in code
3. **Monitor usage** - Check Space analytics for traffic patterns
4. **Version control** - Each push creates a new build
5. **Test locally** - Build Docker image locally first:
   ```bash
   docker build -f Dockerfile.hf -t solbridge-ai .
   docker run -p 7860:7860 solbridge-ai
   ```

---

## 📞 Support

**Hugging Face Community**: https://discuss.huggingface.co/  
**Documentation**: https://huggingface.co/docs/hub/spaces  
**Status Page**: https://status.huggingface.co/

---

**Happy Deploying!** 🚀

Your ML service will be live at: `https://YOUR_USERNAME-solbridge-ai.hf.space`
