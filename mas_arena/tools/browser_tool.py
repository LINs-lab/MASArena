import os
from smolagents import Tool
from dotenv import load_dotenv

load_dotenv()

class BrowserTool(Tool):
    """
    Wrapper for the old BrowserTool logic.
    """
    def __init__(self, model=None, text_limit=1000):
        self.name = "browser"
        self.description = "A tool for browsing the web, navigating to URLs, and extracting content."
        self.inputs = {
            "action": {
                "type": "string",
                "description": "The action to perform: 'navigate', 'get_content', 'get_url', 'screenshot', 'close'.",
                "enum": ["navigate", "get_content", "get_url", "screenshot", "close"]
            },
            "url": {
                "type": "string",
                "description": "The URL to navigate to (required for 'navigate' action).",
                "nullable": True
            }
        }
        self.output_type = "string"
        
        super().__init__()
        
        # Try importing from tools_old
        try:
            from mas_arena.tools_old.browser_tool import Browser
            self.browser_instance = Browser()
            self.available = True
        except ImportError as e:
            print(f"Failed to import old BrowserTool: {e}")
            self.available = False
            self.browser_instance = None
        except Exception as e:
            print(f"Failed to initialize old BrowserTool: {e}")
            self.available = False
            self.browser_instance = None

    def forward(self, action: str = None, url: str = None, **kwargs) -> str:
        # Re-import strictly required modules for sandbox execution context
        # But self.browser_instance is an instance attribute, so we rely on it being available.
        
        # Handle cases where action is implicit or passed as a keyword argument mixed with others
        if action is None:
            if url:
                action = "navigate"
            elif kwargs.get("action"):
                action = kwargs["action"]
            else:
                return "Error: 'action' argument is required for browser tool (e.g., action='navigate', action='get_content')."
        
        if not self.available or not self.browser_instance:
            # Try initializing again if it failed or wasn't available (e.g. inside sandbox)
            try:
                from mas_arena.tools_old.browser_tool import Browser
                self.browser_instance = Browser()
                self.available = True
            except ImportError:
                 return "Browser tool is not available (failed to load)."
            except Exception as e:
                return f"Failed to initialize BrowserTool: {str(e)}"

        try:
            if action == "navigate":
                if not url:
                    return "Error: URL is required for navigation."
                return self.browser_instance.navigate(url)
            elif action == "get_content":
                return self.browser_instance.get_page_content()
            elif action == "get_url":
                return self.browser_instance.get_current_url()
            elif action == "screenshot":
                return self.browser_instance.screenshot()
            elif action == "close":
                self.browser_instance.close()
                return "Browser closed."
            else:
                return f"Unknown browser action: {action}"
        except Exception as e:
            return f"Error executing browser action '{action}': {str(e)}"

    def __del__(self):
        if hasattr(self, 'browser_instance') and self.browser_instance:
            try:
                self.browser_instance.close()
            except:
                pass
