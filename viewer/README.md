# FireComp research viewer

Three parts: offline export → FastAPI backend → Cesium frontend.

## 1. Export (GPU)

Models and thresholds are listed in `firecomp/viewer/models.json`.

```bash
python -m firecomp.viewer.export smoke                   # 8 samples, quick check
python -m firecomp.viewer.export inputs                  # input layers for the test set
python -m firecomp.viewer.export preds --model unetpp    # one model (morph, unet, unetpp, vit)
python -m firecomp.viewer.export collect                 # merge into data/viewer/
```

## 2. Backend

From the repo root (`pip install -e .[viewer]`). Do not use `--reload` — it
watches `data/` and can get OOM-killed.

```bash
python -m uvicorn viewer.backend.main:app --host 127.0.0.1 --port 8000
```

## 3. Frontend

Node ≥ 20 and pnpm.

```bash
cd viewer/frontend
pnpm install
pnpm exec vite --host 127.0.0.1
```

Open http://127.0.0.1:5173 — `/api` is proxied to the backend. Use `127.0.0.1`,
not `localhost` (Vite may bind IPv6-only). If 5173 is taken, Vite picks the next
free port.

URL state: `/?event=1234&sample=1234_0_0_2021-09-19&model=unetpp&layer=confusion`
