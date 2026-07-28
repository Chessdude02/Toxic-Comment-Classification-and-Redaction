# Deploying the Web App

This deploys `src/toxicity_web_app.py` (the Flask redaction web app + REST API) behind gunicorn.
It works with **zero manual configuration** — no trained model is required to get a live URL,
because the app automatically falls back to the rule-based heuristic detector
(`src/heuristic_fallback.py`) when no trained model is present. See the root README's
"`enhanced/` Layer" and "Quick Start" sections for what the trained-model path adds on top of this.

**What you'll get on first deploy**: a working URL with the "⚠️ running on heuristic fallback"
banner showing, real (if less accurate) toxicity scoring. Once you train a model and provide it
to the running service (see "Adding a trained model" below), the banner switches to "✅ Trained
model loaded" with no further deploy steps needed.

---

## Option A: Render (recommended — free tier, one blueprint click)

1. Push this repo to GitHub (already done if you're reading this from the repo).
2. Go to [render.com](https://render.com) → sign in → **New +** → **Blueprint**.
3. Connect this GitHub repo and select the branch you want to deploy.
4. Render detects `render.yaml` at the repo root and creates the web service automatically —
   no fields to fill in. Click **Apply**.
5. Wait for the build (a few minutes — it's installing TensorFlow). Render gives you a URL like
   `https://toxic-comment-redaction.onrender.com` once it's live.

That's the entire manual process — one click after connecting the repo.

**Free tier note**: Render's free instances spin down after inactivity and take ~30–60s to wake
back up on the next request. That's expected, not a bug.

**Memory note (confirmed the hard way)**: `toxicity_redactor.py` does `import tensorflow` at
module level, which costs ~600MB of RSS on its own — more than a 512MB free-tier instance's entire
budget, and it used to get imported unconditionally even when falling back to the heuristic
detector. An early deploy of this repo hit exactly that: `Out of memory (used over 512Mi)`, twice,
before ever serving a request. Fixed now — the app checks for trained model files on disk *before*
importing anything from `toxicity_redactor.py`, and skips that import entirely if none are found
(see `_trained_model_files_present()` in `toxicity_web_app.py`). Confirmed footprint: ~73MB RSS
total in heuristic mode, ready in ~2 seconds, vs. ~630MB/~20s before the fix. If you do mount a
real trained model (see below), expect memory usage to go back up substantially once TensorFlow
loads for real — a free tier may not have enough headroom for that case, worth testing before
relying on it.

## Option B: Railway (also free-tier friendly)

1. Go to [railway.app](https://railway.app) → **New Project** → **Deploy from GitHub repo**.
2. Select this repo/branch. Railway auto-detects the root `Dockerfile` and `railway.toml` and
   builds/deploys automatically.
3. Once deployed, go to the service's **Settings → Networking** and click **Generate Domain** to
   get a public URL (Railway doesn't expose one by default).

## Option C: Anything else that runs a Dockerfile or a Procfile

Any platform that can build a Dockerfile (Fly.io, Google Cloud Run, AWS App Runner, Azure
Container Apps, etc.) works with zero changes — just point it at this repo's root `Dockerfile`.
Platforms that expect a `Procfile` instead of Docker (classic Heroku-style buildpacks) will pick up
the one at the repo root.

Manually, from any machine with Docker:

```bash
docker build -t toxic-comment-redaction .
docker run -p 5000:5000 -e PORT=5000 toxic-comment-redaction
# Open http://localhost:5000
```

---

## Adding a trained model (optional, after your first deploy)

None of the options above ship a trained model — training requires the real Jigsaw dataset and
GloVe embeddings, neither of which are checked into the repo (see `.gitignore`), and isn't
something a cloud build step does for you. Once you have a trained model (see the root README's
"Quick Start" → step 4, or `enhanced/README.md` for the BiLSTM path), you need to get these three
files onto the running container at `saved_models/` (paths `src/toxicity_redactor.py` expects):

- `saved_models/demo_toxicity_classifier.h5` (or `.keras`)
- `saved_models/config.pickle`
- `tokenizer.pickle`

How to get them there depends on the platform:
- **Render**: add a persistent disk mounted at `/app/saved_models` (Render dashboard → your
  service → **Disks**), then upload the files via `render ssh` or a one-off deploy step.
- **Railway**: add a volume mounted at `/app/saved_models`, then use `railway run` or the Railway
  CLI's file upload to place the files there.
- **Docker (any platform)**: bind-mount a local directory containing the trained files:
  `docker run -p 5000:5000 -v "$(pwd)/saved_models:/app/saved_models" toxic-comment-redaction`.

Restart the service after adding the files — the app checks for a trained model at startup and
switches to `mode: "trained"` automatically if it finds one.

---

## Health check

Every option above uses `GET /api/health`, which returns `{"mode": "trained" | "heuristic" |
"unavailable", ...}` — that's the fastest way to confirm which mode a live deployment is actually
running in without opening the UI.
