# Deployment and local fallback

The application is Python Streamlit. Static/Cloudflare Worker Sites hosting cannot execute this server, so deployment targets Streamlit Community Cloud. Hosting status will be updated after the live app is verified.

The [project repository](https://github.com/Tanish520/financial-time-series-mining) is public, on branch `main`. It was created privately and made public with the team's explicit authorization on 19 November 2026 so Streamlit's existing GitHub connection could access this course project. Original course PDFs, raw research caches, environments and secrets are excluded from Git.

The initial full release commit `10c28665c063637387f15ba65b4f89c13befeb00` passed [GitHub Actions CI](https://github.com/Tanish520/financial-time-series-mining/actions/runs/44382806463). After a hosted import-path correction, a fresh local verification passed all 273 tests with warnings treated as errors and found no broken dependencies.

## Streamlit Community Cloud

The complete app runs from the versioned display bundle; raw research caches and secrets are not required.

1. Open https://share.streamlit.io and use the GitHub account that owns the project repository.
2. Create an app from the project repository and its published branch.
3. Main file: `app/streamlit_app.py`.
4. In advanced settings select Python **3.13**, matching the pinned requirements. Do not silently use a different version.
5. Deploy and wait for a successful app load. Test all five views, portfolio/stock and cost selectors, and the cached Latest page. Use an explicit refresh to test the provider boundary separately.

GitHub is connected to the `tanish520` Streamlit workspace. A verified live URL will be recorded here after publication. No deployment success or hosted visual check is inferred from preparing these files.

## Local fallback

```bash
python3.13 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

Open http://127.0.0.1:8501. The default bundle is app/artifacts/full_walkforward. To inspect a different verified bundle, set ADM_RESULTS_DIR before launching the server.

The local loopback address is for this workstation; it is not a phone-accessible or published URL. Test a phone browser with the hosted URL. Browser device emulation is a layout check, not proof of testing a physical phone.

## Container option

```bash
docker build -t financial-pattern-mining .
docker run --rm -p 8501:8501 financial-pattern-mining
```

The Dockerfile packages only the app, source and pinned environment. It provides a local health check at /_stcore/health. The container build was not run in this session because the Docker daemon was unavailable; the native Python app remains the verified runtime.

## Release verification

```bash
.venv/bin/python -m pip check
.venv/bin/python -m pytest -q -W error
```

The GitHub Actions workflow repeats these checks on Python 3.13. Local passes and remote CI results are distinct; check the workflow result after pushing.

Never put credentials in the repository. .env files and .streamlit/secrets.toml are ignored. Latest Signals uses a public provider request; it does not require a personal API key. Core pages continue to use saved results when the provider is unavailable.
