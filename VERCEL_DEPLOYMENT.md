# 🚀 Deploy to Vercel (FREE)

This setup now works entirely on **Vercel free tier** with **Node.js serverless functions**.

## Setup Instructions

### 1️⃣ Get Groq API Key (FREE)
- Go to: https://console.groq.com/keys
- Sign up with email
- Create API key: `gsk_...`
- Copy it 📋

### 2️⃣ Push to GitHub
```bash
cd e:/afcatpyq3
git add -A
git commit -m "ready for Vercel deployment"
git push
```

### 3️⃣ Connect to Vercel
1. Go to: https://vercel.com
2. Sign up with GitHub
3. Click "New Project"
4. Select your `afcatpyq3` repository
5. Click "Import"

### 4️⃣ Add Environment Variable
1. Before deploying, go to **Settings** → **Environment Variables**
2. Add:
   - **Name:** `GROQ_API_KEY`
   - **Value:** `gsk_your_key_here` (paste your actual key)
3. Click "Save"

### 5️⃣ Deploy
- Click "Deploy"
- Wait for deployment to complete ✅

### 6️⃣ Your Site is LIVE!
- Frontend: `https://yourname.vercel.app`
- API: `https://yourname.vercel.app/api/config`

---

## How It Works

📁 **Folder Structure:**
```
afcatpyq3/
├── api/
│   └── config.js          ← Vercel serverless function (Node.js)
├── output/predictions_2026/
│   ├── index.html         ← Main app
│   ├── data.js            ← Question data
│   └── ...
├── vercel.json            ← Vercel config
├── .vercelignore          ← What to ignore
├── .env                   ← Local only (not uploaded)
└── package.json           ← Node.js config
```

🔧 **How API Works:**
1. Frontend requests: `GET /api/config`
2. Vercel runs: `api/config.js` (serverless function)
3. Function reads: `process.env.GROQ_API_KEY` from Vercel settings
4. Returns: `{ groq_api_key: "gsk_..." }`
5. Frontend uses it to call Groq API ✅

---

## Features

✅ **100% on Vercel Free Tier**
✅ **No backend server needed**
✅ **Auto-scaling serverless**
✅ **Instant deployment**
✅ **Custom domain support**

---

## Troubleshooting

**"API key not loading?"**
- Check Vercel Settings → Environment Variables
- Make sure `GROQ_API_KEY` is set
- Redeploy: `git push` → automatic redeploy

**"Page shows 404?"**
- Files should be in `output/predictions_2026/`
- Check vercel.json `outputDirectory` setting

**"AI explanation not working?"**
- Check browser console (F12) for errors
- Verify `GROQ_API_KEY` is correct
- Test at: `/api/config`

---

## Commands

```bash
# Local development
npm run dev                # Runs locally

# Deploy to Vercel
git push                   # Auto-deploys via GitHub

# View logs
vercel logs               # Check deployment logs
```

---

**✨ You now have a production-ready AFCAT prep platform on Vercel! 🎉**
