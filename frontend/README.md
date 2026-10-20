# checkpoint UI

Dark React/Vite UI for the checkpoint daemon. It talks to the local API and
receives live updates over `/ws`.

## Develop

```powershell
cd frontend
npm ci
npm run dev
```

`vite.config.ts` proxies `/api`, `/v1`, and `/ws` to `http://127.0.0.1:4040`.
Start the daemon separately:

```powershell
cd backend
uv run checkpoint web --no-open
```

## Build

```powershell
cd frontend
npm run build
```

The build is written to `../backend/checkpoint/ui_dist/` and is committed so
`uvx checkpoint web` (or an installed wheel) serves the UI without Node.

## End-to-end tests

```powershell
cd frontend
npx playwright install chromium   # first run only
npm run test:e2e
```

The Playwright config starts a seeded daemon from `e2e/serve.py` on port 4173.
To point the tests at an already-running server instead:

```powershell
$env:CHECKPOINT_E2E_EXTERNAL = "1"
$env:CHECKPOINT_E2E_URL = "http://127.0.0.1:4173"
npm run test:e2e
```

The seed cassette includes `<script>window.__xss=1</script>` in a prompt. The
smoke test asserts that the payload is rendered as text and never executes.
