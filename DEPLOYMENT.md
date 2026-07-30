# Deploying the Web App

This deploys `src/toxicity_web_app.py` (the Flask redaction web app + REST API) behind gunicorn.

**As of this commit, a trained model ships in the repo** at `src/saved_models/` (the `enhanced/`
BiLSTM, re-saved to the path the root app looks for — see the root README's note on this), so a
fresh deploy runs in `mode: "trained"` immediately, no training step needed. If those files are
ever removed, the app still works: it falls back to the rule-based heuristic detector
(`src/heuristic_fallback.py`) instead of failing outright.

**The tradeoff, confirmed by directly measuring it, not estimated:**

| Mode | RSS (1 gunicorn worker) | Fits in 512MB free tier? |
|---|---|---|
| Heuristic (no model files present) | ~73MB | Yes, comfortably |
| Trained (model loaded) | ~718MB | **No** |

Loading any real model pulls in TensorFlow, which alone costs ~600MB of RSS before the model
weights are even loaded. There is no lazy-import trick that fixes this once the model is actually
in use — unlike the heuristic-mode fix earlier in this file's history, this is a hard floor on
memory for anything that actually runs the trained classifier. See "Picking a deploy target" below
before you deploy with the shipped model in place.

---

## Picking a deploy target: real predictions vs. free tier

You have three honest options — pick based on whether you want the real trained model live, or a
free URL:

1. **Pay for enough RAM and keep the shipped model.** Render's cheapest paid instance (Starter,
   512MB→a bit more headroom isn't quite enough either in practice — you'd want at least their
   next tier up, or equivalently ~1GB+ RAM elsewhere) comfortably fits ~718MB RSS. This is the only
   option that gets you the real 90.86%/0.938 AUC model live on a public URL.
2. **Deploy on the free tier without the shipped model**, accepting heuristic-only predictions.
   Before deploying, delete or rename `src/saved_models/` and `src/tokenizer.pickle` (or deploy
   from a branch that doesn't have them) — with those gone, `_trained_model_files_present()`
   returns false, TensorFlow is never imported, and you're back to the ~73MB footprint that fits
   free tiers. The live demo still works, just with heuristic rather than trained scoring.
3. **Run it locally / on your own machine** with Docker (see Option C below) where you control the
   RAM — no tier limits to work around.

There isn't a way to get the real trained model under 512MB — that's TensorFlow's baseline import
cost, not something specific to this model or fixable by code changes.

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

## Swapping in a different trained model

A model already ships in the repo (see above), but if you retrain and want to replace it —
e.g. after actually training the root Transformer notebook (needs the real Jigsaw dataset + GloVe
embeddings, neither checked into the repo, see `.gitignore`) — put these three files at the paths
`src/toxicity_redactor.py`'s `load_pretrained_model()` looks for by default:

- `src/saved_models/demo_toxicity_classifier.h5` (or `.keras`)
- `src/saved_models/config.pickle` (needs at least `label_columns`, `threshold`, `max_len`)
- `src/tokenizer.pickle` (note: NOT under `saved_models/` — that's `load_pretrained_model()`'s
  existing default, not a typo)

For a container already running, you'd need to get the new files onto it and restart:
- **Render**: add a persistent disk mounted at `/app/src/saved_models`, upload via `render ssh` or
  a one-off deploy step.
- **Railway**: add a volume mounted at `/app/src/saved_models`, use `railway run` or the CLI's file
  upload.
- **Docker (any platform)**: bind-mount a local directory: `docker run -p 5000:5000 -v
  "$(pwd)/saved_models:/app/src/saved_models" toxic-comment-redaction`.

Simplest in practice, though: commit the new model files to the repo (same as this one was) and
redeploy — the app checks for a trained model at startup and picks up whatever's present.

---

## Health check

Every option above uses `GET /api/health`, which returns `{"mode": "trained" | "heuristic" |
"unavailable", ...}` — that's the fastest way to confirm which mode a live deployment is actually
running in without opening the UI.
