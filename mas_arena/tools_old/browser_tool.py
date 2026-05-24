# coding: utf-8
import base64
import subprocess
from pathlib import Path
from typing import List, Optional
import sys
import importlib
import asyncio
import os
from langchain.tools import StructuredTool
from mas_arena.tools_old.base import ToolFactory

BROWSER = "browser"


class _BrowserPool:
    """Process-level singleton: one Chromium process, bounded context pool.

    All concurrent workers share the same browser process.  The pool size caps
    simultaneous page sessions so a large benchmark concurrency (e.g. 13) does
    not spawn 13 independent Chromium processes.

    Pool size is controlled by MAS_ARENA_BROWSER_POOL_SIZE (default 4).
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._init_lock = asyncio.Lock()
        self._playwright = None
        self._browser = None
        self._pw_cm = None
        self._initialized = False
        pool_size = int(os.getenv("MAS_ARENA_BROWSER_POOL_SIZE", "4"))
        self._semaphore = asyncio.Semaphore(pool_size)

    async def _ensure_browser(self) -> None:
        if self._initialized:
            return
        async with self._init_lock:
            if self._initialized:
                return
            from playwright.async_api import async_playwright
            self._pw_cm = async_playwright()
            self._playwright = await self._pw_cm.__aenter__()
            disable_security_args = [
                '--disable-web-security',
                '--disable-site-isolation-trials',
                '--disable-features=IsolateOrigins,site-per-process',
            ]
            args = [
                '--no-sandbox', '--disable-crash-reporter',
                '--disable-blink-features=AutomationControlled',
                '--disable-infobars', '--disable-background-timer-throttling',
                '--disable-popup-blocking', '--disable-backgrounding-occluded-windows',
                '--disable-renderer-backgrounding', '--disable-window-activation',
                '--disable-focus-on-load', '--no-first-run', '--no-default-browser-check',
                '--no-startup-window', '--window-position=0,0', '--window-size=1280,720',
            ] + disable_security_args
            self._browser = await self._playwright.chromium.launch(headless=True, slow_mo=0, args=args)
            self._initialized = True

    async def acquire_context(self):
        """Wait for a slot (up to MAS_ARENA_BROWSER_ACQUIRE_TIMEOUT_SECONDS), then return a fresh context+page."""
        await self._ensure_browser()
        acquire_timeout = float(os.getenv("MAS_ARENA_BROWSER_ACQUIRE_TIMEOUT_SECONDS", "300"))
        try:
            await asyncio.wait_for(self._semaphore.acquire(), timeout=acquire_timeout)
        except asyncio.TimeoutError:
            raise RuntimeError(
                f"Timed out waiting for a browser context slot after {acquire_timeout:.0f}s. "
                f"Consider increasing MAS_ARENA_BROWSER_POOL_SIZE (current: "
                f"{int(os.getenv('MAS_ARENA_BROWSER_POOL_SIZE', '4'))}) or "
                f"MAS_ARENA_BROWSER_ACQUIRE_TIMEOUT_SECONDS."
            )
        from playwright.async_api import ViewportSize
        context = await self._browser.new_context(
            viewport=ViewportSize(width=1280, height=720),
            java_script_enabled=True,
            bypass_csp=True,
            ignore_https_errors=True,
            device_scale_factor=1,
        )
        page = await context.new_page()
        return context, page

    async def release_context(self, context) -> None:
        """Close the context and release the semaphore slot."""
        try:
            await context.close()
        except Exception:
            pass
        finally:
            self._semaphore.release()

    async def shutdown(self) -> None:
        if not self._initialized:
            return
        try:
            await self._browser.close()
        except Exception:
            pass
        try:
            await self._pw_cm.__aexit__(None, None, None)
        except Exception:
            pass
        self._initialized = False


# Module-level singleton — created lazily on first use.
_pool: Optional[_BrowserPool] = None
_pool_create_lock = asyncio.Lock()


async def _get_pool() -> _BrowserPool:
    global _pool
    if _pool is not None:
        return _pool
    async with _pool_create_lock:
        if _pool is None:
            _pool = _BrowserPool()
        return _pool


def import_and_install(package_name: str):
    """Tries to import a package, and if it fails, attempts to install it and then import it again."""
    try:
        # Check if the package is available first
        spec = importlib.util.find_spec(package_name)
        if spec is not None:
            return importlib.import_module(package_name)
        else:
            raise ImportError(f"No module named '{package_name}'")
    except ImportError:
        # Installing packages at runtime inside benchmark runs is slow and brittle.
        # Keep the helper for optional/manual use, but do not auto-install by default.
        print(f"Package '{package_name}' not found.")
        try:
            # Use subprocess.run to capture both stdout and stderr
            process = subprocess.run(
                [sys.executable, "-m", "pip", "install", package_name],
                capture_output=True,
                text=True,
                check=True
            )
            print(f"Successfully installed '{package_name}'.")
            importlib.invalidate_caches()
            return importlib.import_module(package_name)
        except subprocess.CalledProcessError as e:
            # Now we can inspect both e.stdout and e.stderr
            stdout_details = e.stdout.strip() if e.stdout else "No stdout output."
            stderr_details = e.stderr.strip() if e.stderr else "No stderr output."
            print(f"--- pip install failed ---")
            print(f"Failed to install '{package_name}' via pip. Error details below:")
            print("--- STDOUT ---")
            print(stdout_details)
            print("--- STDERR ---")
            print(stderr_details)
            print(f"--- End of pip error ---")
            # Don't re-raise, just return None so Browser init can decide what to do
            # or re-raise if strictly required. For now, let's re-raise to be safe.
            raise ImportError(f"Could not install {package_name}") from e
        except Exception as e:
            print(f"An unexpected error occurred during the installation of '{package_name}': {e}")
            raise ImportError(f"Could not import or install {package_name}") from e

class Browser:
    """Thin session wrapper backed by the shared _BrowserPool.

    Each instance borrows one context+page slot from the pool on first use and
    returns it on close().  The pool caps the total number of live Chromium
    contexts across all concurrent workers.
    """

    def __init__(self, **kwargs) -> None:
        self.initialized = False
        self._context = None
        self.page = None
        self.record_trace = kwargs.get("enable_recording", False)

        try:
            import playwright  # noqa: F401
        except Exception as e:
            raise ImportError(
                "Playwright is required for the browser tool. "
                "Please install it in your environment (e.g., `uv sync` / `pip install playwright`). "
                f"Import error: {e}"
            )

        auto_install = str(os.getenv("MAS_ARENA_AUTO_INSTALL_PLAYWRIGHT", "")).strip().lower() in {"1", "true", "yes"}
        if auto_install:
            print("Checking/installing Playwright browser binaries...")
            try:
                subprocess.check_call(
                    [sys.executable, "-m", "playwright", "install"],
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.PIPE,
                )
            except Exception as e:
                print(f"Warning: Failed to install playwright browsers. Error: {e}")

    async def init(self) -> None:
        if self.initialized:
            return
        pool = await _get_pool()
        self._context, self.page = await pool.acquire_context()
        if self.record_trace:
            await self._context.tracing.start(screenshots=True, snapshots=True)
        self.initialized = True

    async def navigate(self, url: str) -> str:
        """Navigate to a URL."""
        if not self.initialized: await self.init()
        try:
            timeout_ms = int(os.getenv("MAS_ARENA_BROWSER_GOTO_TIMEOUT_MS", "20000"))
            await self.page.goto(url, timeout=timeout_ms, wait_until="domcontentloaded")
            return f"Navigated to {url}"
        except Exception as e:
            return (
                f"Failed to navigate to {url}: {e}. "
                "If this looks like a missing Playwright browser binary, run `playwright install` once "
                "outside the benchmark (or set MAS_ARENA_AUTO_INSTALL_PLAYWRIGHT=1)."
            )

    async def get_page_content(self, clean=True) -> str:
        """
        Get the text content of the current page.
        Args:
            clean: Whether to run a cleaning script to remove irrelevant content.
        """
        if not self.initialized: await self.init()
        evaluate_timeout_ms = int(os.getenv("MAS_ARENA_BROWSER_EVALUATE_TIMEOUT_MS", "15000"))
        try:
            if clean:
                # A simple script to remove common clutter like nav, footer, scripts, styles
                js_script = """() => {
                    const doc = document.cloneNode(true);
                    doc.querySelectorAll('nav, footer, script, style, aside, [role="navigation"], [role="banner"], [role="contentinfo"]').forEach(el => el.remove());
                    return doc.body.innerText;
                }"""
                return await self.page.evaluate(js_script, timeout=evaluate_timeout_ms)
            else:
                return await self.page.inner_text('body', timeout=evaluate_timeout_ms)
        except Exception as e:
            return f"Failed to get page content: {e}"

    async def get_current_url(self) -> str:
        if not self.initialized: await self.init()
        """Get the current URL."""
        return self.page.url

    async def screenshot(self, full_page: bool = False) -> str:
        """Returns a base64 encoded screenshot of the current page."""
        if not self.initialized: await self.init()
        try:
            await self.page.bring_to_front()
            self.page.wait_for_load_state(timeout=2000)
        except:
            pass

        screenshot_timeout_ms = int(os.getenv("MAS_ARENA_BROWSER_SCREENSHOT_TIMEOUT_MS", "30000"))
        screenshot = await self.page.screenshot(
            full_page=full_page,
            animations='disabled',
            timeout=screenshot_timeout_ms
        )
        screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
        return screenshot_base64

    async def close(self) -> None:
        if not self.initialized:
            return
        if self.record_trace:
            self.save_trace("trace.zip")
        pool = await _get_pool()
        await pool.release_context(self._context)
        self._context = None
        self.page = None
        self.initialized = False

    def save_trace(self, trace_path: str | Path) -> None:
        self._context.tracing.stop(path=trace_path)


@ToolFactory.register(name=BROWSER, desc="A tool for browsing the web.")
class BrowserTool:
    def __init__(self):
        self.browser = None
        try:
            self.browser = Browser()
        except Exception as e:
            print(f"Error: tool browser load failed - {e}")
            raise

    def get_tools(self) -> List[StructuredTool]:
        if not self.browser:
            return []
            
        return [
            StructuredTool.from_function(
                func=self.browser.navigate,
                name="navigate_to_url",
                description="Navigate to a specific URL."
            ),
            StructuredTool.from_function(
                func=self.browser.get_page_content,
                name="get_page_content",
                description="Get the text content of the current web page, optionally cleaning it."
            ),
            StructuredTool.from_function(
                func=self.browser.get_current_url,
                name="get_current_url",
                description="Get the current URL of the browser."
            ),
            StructuredTool.from_function(
                func=self.browser.screenshot,
                name="take_screenshot",
                description="Take a screenshot of the current page."
            ),
            StructuredTool.from_function(
                func=self.browser.close,
                name="close_browser",
                description="Close the browser."
            )
        ]

    def __del__(self):
        # Browser.close() is async; calling it from __del__ (sync) would silently no-op.
        # Cleanup is handled by the pool's release_context when Browser.close() is awaited
        # normally, or by pool.shutdown() at process exit.
        pass