# ColorSchemeToggle Migration (app.py)

This guide explains how to migrate `app.py` from menu-click theme toggling to `dmc.ColorSchemeToggle` in production.

## Goal

- Use Dash Mantine Components `ColorSchemeToggle` (DMC 2.6+).
- Let Mantine manage color scheme state.
- Remove app-level manual toggle callbacks.
- Optionally remove `theme-store` entirely.

## Prerequisites

- `dash-mantine-components>=2.6.0` in `requirements.txt`.
- `dmc.pre_render_color_scheme()` already called in `app.py` (good for no-flash startup).

## Recommended Target State

- `MantineProvider` uses `defaultColorScheme="light"` (not `forceColorScheme`).
- Header includes a `dmc.ColorSchemeToggle(...)`.
- No callback needed to toggle dark/light.
- `theme-store` removed after all pages stop using it.

## Step-by-Step (app.py)

1. Add icon import for toggle visuals:

```python
from dash_iconify import DashIconify
```

2. Replace provider mode control:

- Remove:

```python
_provider_kwargs["forceColorScheme"] = "light"
```

- Add:

```python
_provider_kwargs["defaultColorScheme"] = "light"
```

3. Add `ColorSchemeToggle` to header and remove menu toggle item:

- Add near the Menu button:

```python
dmc.ColorSchemeToggle(
    id="global-color-scheme-toggle",
    lightIcon=DashIconify(icon="tabler:sun", width=18),
    darkIcon=DashIconify(icon="tabler:moon", width=18),
    variant="outline",
    size="lg",
    color="gray",
)
```

- Remove old menu item:

```python
dmc.MenuItem("Toggle Dark Mode", id="global-menu-toggle-dark-mode")
```

4. Remove manual theme callbacks in `app.py`:

- Remove callback: `theme-store -> mantine-provider.forceColorScheme`
- Remove callback: `global-menu-toggle-dark-mode -> theme-store`

5. Remove `theme-store` from layout only after all remaining usage is gone:

- Remove:

```python
dcc.Store(id="theme-store", data="light", storage_type="local")
```

## Transitional Option (if some pages still read theme-store)

If you are not ready to remove `theme-store`, keep it temporarily and sync it from `ColorSchemeToggle`:

```python
app.clientside_callback(
    "function(scheme){ return (scheme === 'dark' || scheme === 'light') ? scheme : window.dash_clientside.no_update; }",
    Output("theme-store", "data"),
    Input("global-color-scheme-toggle", "computedColorScheme"),
)
```

This lets old callbacks keep working while the UI uses `ColorSchemeToggle`.

## Validation Commands

```powershell
rg -n -F 'theme-store' app.py pages utils
rg -n -F 'forceColorScheme' app.py
rg -n -F 'ColorSchemeToggle' app.py
conda run -n dashmat python -m py_compile app.py pages/analyticstool.py pages/portopt.py
conda run -n dashmat python app.py --debug
```

## Expected Result

- Theme toggles from `ColorSchemeToggle` without custom toggle logic.
- Persisted scheme handled by Mantine/DMC.
- Chart callbacks can read `State("mantine-provider", "forceColorScheme")` directly.
