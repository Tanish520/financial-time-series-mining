# Deployment and local fallback

The application is deployed on Streamlit Community Cloud:

**Live app:** https://financial-time-series-mining-jeksedhe7pc74gmdqekycu.streamlit.app/

All five hosted views loaded the real ten-stock, 44-fold result bundle on 19 November 2026 (India time). The served Streamlit backend's health endpoint returned HTTP 200 with body `ok`.

The [project repository](https://github.com/Tanish520/financial-time-series-mining) is public, on branch `main`. It was created privately and made public with the team's explicit authorization on 19 November 2026 so Streamlit's existing GitHub connection could access this course project. Original course PDFs, raw research caches, environments and secrets are excluded from Git.

After a hosted import-path correction, release commit `e97e6c23155ba1ff1ed4f62dd02a718c4ef4d4d3` passed [GitHub Actions CI](https://github.com/Tanish520/financial-time-series-mining/actions/runs/44389933145): all 273 tests and dependency checks on both Python 3.13 and 3.14. A fresh local verification also passed all 273 tests with warnings treated as errors and found no broken dependencies.

The frozen research runs use Python 3.13.12 and retain their original software provenance in the saved manifest. Hosting logs report Python 3.14.10; the platform also replaced pinned PyArrow 26.0.1 with its managed PyArrow 24.0.0. The hosted app displays the frozen results rather than recomputing the study. Both Python versions pass CI with the pinned requirements; actual hosted interaction checks cover the platform-managed display environment separately.

## Streamlit Community Cloud

The complete app runs from the versioned display bundle; raw research caches and secrets are not required.

1. Open https://share.streamlit.io and use the GitHub account that owns the project repository.
2. Create an app from the project repository and its published branch.
3. Main file: `app/streamlit_app.py`.
4. For a new deployment, select Python **3.13** to match the frozen research environment. This existing deployment used the host's default **3.14**, which is also checked in CI; see the exact runtime distinction above.
5. Deploy and wait for a successful app load. Test all five views, portfolio/stock and cost selectors, and the cached Latest page. Use an explicit refresh to test the provider boundary separately.

GitHub is connected to the `tanish520` Streamlit workspace. The app runs from branch `main`, entry point `app/streamlit_app.py`. The initial hosted import-path failure was reproduced in an isolated local process, covered by a failing regression, fixed and automatically updated through the GitHub push.

## Verified hosted checks

| View | Observed result |
| --- | --- |
| Rules Explorer | Final frozen model, real training rules and fitting/target dates loaded. |
| Tuning & Drift | 44 eligible folds, zero fallbacks, and window counts 27/10/7. |
| Backtest | Portfolio at 0.1% cost: -13.30% tuned, +0.04% fixed, +45.57% holding and +44.32% uncosted NIFTY. Changing cost to 0.5% showed -43.78% tuned. Selecting TCS at 0.5% showed -37.46% tuned and 76 trades, matching saved stock metrics. All seven metric rows and four chart series rendered; the legend uses two columns and date labels include years. |
| Asymmetry | U precision/base rate 38.08%/36.20%, D 42.85%/41.12%; lifts 1.052/1.042, with real counts. |
| Latest Signals | Cached ten-stock signals loaded with observation 17 November and model cutoff 27 October. Explicit live refresh rejected incomplete 18 November prices and displayed a dated stale-cache fallback with its fetch timestamp. No fabricated current signal was substituted. |

Provider availability and valid completed bars cannot be guaranteed by hosting. The demonstrated refresh fallback is intentional. Core saved results remain usable. Browser emulation was used for a 390-pixel layout check during development; testing on a physical phone is not claimed.

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

The GitHub Actions workflow repeats these checks on Python 3.13 and 3.14. Local passes and remote CI results are distinct; check the workflow result after pushing.

Never put credentials in the repository. .env files and .streamlit/secrets.toml are ignored. Latest Signals uses a public provider request; it does not require a personal API key. Core pages continue to use saved results when the provider is unavailable.
