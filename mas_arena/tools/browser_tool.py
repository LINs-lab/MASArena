# coding: utf-8
import base64
import subprocess
from pathlib import Path
from typing import List
import sys
import importlib

from langchain.tools import StructuredTool

from mas_arena.tools.base import ToolFactory

BROWSER = "browser"

def import_and_install(package_name: str):
    """Tries to import a package, and if it fails, attempts to install it and then import it again."""
    try:
        return __import__(package_name)
    except ImportError:
        print(f"Package '{package_name}' not found. Attempting to install...")
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
            return __import__(package_name)
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
            raise ImportError(f"Could not install {package_name}") from e
        except Exception as e:
            print(f"An unexpected error occurred during the installation of '{package_name}': {e}")
            raise ImportError(f"Could not import or install {package_name}") from e

class Browser:
    def __init__(self, **kwargs) -> None:
        self.initialized = False
        
        # Ensure playwright is available
        import_and_install("playwright")

        # Ensure browser binaries are installed
        print("Checking/installing Playwright browser binaries...")
        try:
            # Using sys.executable to ensure we use the correct playwright
            subprocess.check_call([sys.executable, "-m", "playwright", "install"], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        except Exception as e:
            print(f"Warning: Failed to install playwright browsers, but continuing. Error: {e}")

        self._finish = False
        self.record_trace = kwargs.get("enable_recording", False)
        self.sleep_after_init = kwargs.get("sleep_after_init", False)

        self.init()

    def init(self) -> None:
        from playwright.sync_api import sync_playwright

        if self.initialized:
            return

        self.context_manager = sync_playwright()
        self.playwright = self.context_manager.start()
        self.browser = self._create_browser()
        self.context = self._create_browser_context()

        if self.record_trace:
            self.context.tracing.start(screenshots=True, snapshots=True)

        self.page = self.context.new_page()
        self.initialized = True

    def _create_browser(self):
        browse_name = "chromium"
        browse = getattr(self.playwright, browse_name)
        headless = True
        slow_mo = 0
        disable_security_args = ['--disable-web-security', '--disable-site-isolation-trials', '--disable-features=IsolateOrigins,site-per-process']
        args = ['--no-sandbox', '--disable-crash-reporter', '--disable-blink-features=AutomationControlled', '--disable-infobars', '--disable-background-timer-throttling', '--disable-popup-blocking', '--disable-backgrounding-occluded-windows', '--disable-renderer-backgrounding', '--disable-window-activation', '--disable-focus-on-load', '--no-first-run', '--no-default-browser-check', '--no-startup-window', '--window-position=0,0', '--window-size=1280,720'] + disable_security_args
        browser = browse.launch(
            headless=headless,
            slow_mo=slow_mo,
            args=args,
        )
        return browser

    def _create_browser_context(self):
        from playwright.sync_api import ViewportSize

        browser = self.browser
        viewport_size = ViewportSize(width=1280, height=720)
        disable_security = True

        context = browser.new_context(viewport=viewport_size,
                                      no_viewport=False,
                                      java_script_enabled=True,
                                      bypass_csp=disable_security,
                                      ignore_https_errors=disable_security,
                                      device_scale_factor=1)
        return context

    def navigate(self, url: str) -> str:
        """Navigate to a URL."""
        try:
            self.page.goto(url)
            return f"Navigated to {url}"
        except Exception as e:
            return f"Failed to navigate to {url}: {e}"

    def get_page_content(self, clean=True) -> str:
        """
        Get the text content of the current page.
        Args:
            clean: Whether to run a cleaning script to remove irrelevant content.
        """
        try:
            if clean:
                # A simple script to remove common clutter like nav, footer, scripts, styles
                js_script = """() => {
                    const doc = document.cloneNode(true);
                    doc.querySelectorAll('nav, footer, script, style, aside, [role="navigation"], [role="banner"], [role="contentinfo"]').forEach(el => el.remove());
                    return doc.body.innerText;
                }"""
                return self.page.evaluate(js_script)
            else:
                return self.page.inner_text('body')
        except Exception as e:
            return f"Failed to get page content: {e}"

    def get_current_url(self) -> str:
        """Get the current URL."""
        return self.page.url

    def screenshot(self, full_page: bool = False) -> str:
        """Returns a base64 encoded screenshot of the current page."""
        try:
            self.page.bring_to_front()
            self.page.wait_for_load_state(timeout=2000)
        except:
            pass

        screenshot = self.page.screenshot(
            full_page=full_page,
            animations='disabled',
            timeout=600000
        )
        screenshot_base64 = base64.b64encode(screenshot).decode('utf-8')
        return screenshot_base64

    def close(self) -> None:
        if not self.initialized:
            return
        if self.record_trace:
            self.save_trace("trace.zip")

        self.page.close()
        self.context.close()
        self.browser.close()
        if hasattr(self, 'context_manager') and self.context_manager:
            self.context_manager.stop()
        self.initialized = False

    def save_trace(self, trace_path: str | Path) -> None:
        self.context.tracing.stop(path=trace_path)


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
        if self.browser:
            self.browser.close() 