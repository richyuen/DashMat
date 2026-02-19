# Theme Store Migration Checklist

Use this when migrating callback theme reads from `theme-store` to `mantine-provider.forceColorScheme`.

## 1. Locate current usage

```powershell
rg -n -F 'theme-store' app.py pages
```

Split by purpose:

```powershell
rg -n -F 'State("theme-store", "data")' app.py pages
rg -n -F 'Input("theme-store", "data")' app.py pages
rg -n -F 'Output("theme-store", "data")' app.py pages
```

## 2. Safe replacement target

Only replace callback theme reads like:

```python
State("theme-store", "data")
```

with:

```python
State("mantine-provider", "forceColorScheme")
```

Keep as `State` (not `Input`) if you want to avoid extra expensive callback reruns on each theme toggle.

## 3. Verify replacements

```powershell
rg -n -F 'State("mantine-provider", "forceColorScheme")' app.py pages
rg -n -F 'State("theme-store", "data")' app.py pages
```

For a partial migration (page callbacks only), verify specific files:

```powershell
rg -n -F 'theme-store' pages/analyticstool.py pages/portopt.py
```

## 4. Validate app still compiles/runs

```powershell
conda run -n dashmat python -m py_compile app.py pages/analyticstool.py pages/portopt.py
conda run -n dashmat python app.py --debug
```

## 5. Before removing `theme-store` entirely

Confirm there are no remaining `Input/State/Output` references to `theme-store`:

```powershell
rg -n -F 'theme-store' app.py pages utils
```

Then replace toggle wiring in `app.py` to read/write `mantine-provider.forceColorScheme` directly.
