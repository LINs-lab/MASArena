"""
Browser Process Cleanup Utility

This module provides functions to clean up browser processes that might be left running
after benchmark tests complete. It's particularly useful for ensuring that Selenium
WebDriver instances and Playwright browser processes are properly terminated.
"""

import os
import signal
import subprocess
import sys
import logging
import platform
import time
import re

logger = logging.getLogger(__name__)

def find_mcp_browser_processes():
    """Find browser processes specifically launched by MCP tools.
    
    This function looks for Chrome/Chromium/Playwright processes that were launched
    by MCP tools, identified by specific command line arguments or parent processes.
    
    Returns:
        List of process IDs (PIDs) for MCP-related browser processes
    """
    mcp_pids = []
    
    try:
        if platform.system() == "Windows":
            # Windows - use wmic to get process info with command lines
            try:
                # Look for Chrome processes with MCP-related command line arguments
                cmd = "wmic process where \"name='chrome.exe' or name='chromium.exe' or name='chromedriver.exe'\" get processid,commandline"
                output = subprocess.check_output(cmd, shell=True).decode()
                
                for line in output.splitlines()[1:]:  # Skip header
                    if any(marker in line.lower() for marker in ['mcp', 'browseruse-tmp', 'masarena']):
                        # Extract PID from the line
                        match = re.search(r'(\d+)\s*$', line)
                        if match:
                            try:
                                mcp_pids.append(int(match.group(1)))
                            except ValueError:
                                pass
                
                # Look for Playwright processes
                cmd = "wmic process where \"commandline like '%playwright%'\" get processid,commandline"
                output = subprocess.check_output(cmd, shell=True).decode()
                
                for line in output.splitlines()[1:]:  # Skip header
                    if line.strip():
                        match = re.search(r'(\d+)\s*$', line)
                        if match:
                            try:
                                mcp_pids.append(int(match.group(1)))
                            except ValueError:
                                pass
            except subprocess.CalledProcessError:
                logger.error("Failed to run wmic command")
                
        else:
            # macOS/Linux - use ps with more detailed output
            try:
                # Look for Chrome/Chromium processes with MCP markers
                cmd = "ps -eo pid,args | grep -i 'chrome\\|chromium\\|chromedriver\\|playwright\\|browseruse' | grep -v grep"
                output = subprocess.check_output(cmd, shell=True).decode()
                
                for line in output.splitlines():
                    # Only include processes related to MCP
                    if any(marker in line.lower() for marker in ['mcp', 'browseruse-tmp', 'masarena']):
                        parts = line.strip().split()
                        if len(parts) > 0:
                            try:
                                pid = int(parts[0])
                                # Skip the current process or its parent
                                if pid != os.getpid() and pid != os.getppid():
                                    mcp_pids.append(pid)
                            except ValueError:
                                pass
            except subprocess.CalledProcessError:
                # grep returns non-zero exit code when no matches are found
                pass
                
            # Also look specifically for Playwright and browseruse processes
            try:
                cmd = "ps -eo pid,args | grep -i 'playwright\\|browseruse' | grep -v grep"
                output = subprocess.check_output(cmd, shell=True).decode()
                
                for line in output.splitlines():
                    parts = line.strip().split()
                    if len(parts) > 0:
                        try:
                            pid = int(parts[0])
                            # Skip the current process or its parent
                            if pid != os.getpid() and pid != os.getppid():
                                mcp_pids.append(pid)
                        except ValueError:
                            pass
            except subprocess.CalledProcessError:
                # grep returns non-zero exit code when no matches are found
                pass
    
    except Exception as e:
        logger.error(f"Error finding MCP browser processes: {e}")
        
    return mcp_pids

def find_chrome_processes():
    """Find all Chrome/Chromium browser processes.
    
    Returns:
        List of process IDs (PIDs) for Chrome/Chromium processes
    """
    chrome_pids = []
    
    try:
        if platform.system() == "Windows":
            # Windows - use tasklist
            output = subprocess.check_output("tasklist /FI \"IMAGENAME eq chrome.exe\" /FO CSV", shell=True).decode()
            for line in output.splitlines()[1:]:  # Skip header
                if "chrome.exe" in line:
                    parts = line.split(",")
                    if len(parts) > 1:
                        pid_part = parts[1].strip('"')
                        try:
                            chrome_pids.append(int(pid_part))
                        except ValueError:
                            pass
            
            # Check for Chromium
            output = subprocess.check_output("tasklist /FI \"IMAGENAME eq chromium.exe\" /FO CSV", shell=True).decode()
            for line in output.splitlines()[1:]:  # Skip header
                if "chromium.exe" in line:
                    parts = line.split(",")
                    if len(parts) > 1:
                        pid_part = parts[1].strip('"')
                        try:
                            chrome_pids.append(int(pid_part))
                        except ValueError:
                            pass
                            
            # Also check for chromedriver
            output = subprocess.check_output("tasklist /FI \"IMAGENAME eq chromedriver.exe\" /FO CSV", shell=True).decode()
            for line in output.splitlines()[1:]:  # Skip header
                if "chromedriver.exe" in line:
                    parts = line.split(",")
                    if len(parts) > 1:
                        pid_part = parts[1].strip('"')
                        try:
                            chrome_pids.append(int(pid_part))
                        except ValueError:
                            pass
        else:
            # macOS/Linux - use ps and grep, but exclude the grep process itself
            try:
                # Look for Chrome/Chromium processes
                output = subprocess.check_output("ps -A | grep -i 'chrome\|chromium' | grep -v grep", shell=True).decode()
                for line in output.splitlines():
                    parts = line.strip().split()
                    if len(parts) > 0:
                        try:
                            pid = int(parts[0])
                            # Skip the current process or its parent
                            if pid != os.getpid() and pid != os.getppid():
                                chrome_pids.append(pid)
                        except ValueError:
                            pass
            except subprocess.CalledProcessError:
                # grep returns non-zero exit code when no matches are found
                pass
                
            try:
                # Look for chromedriver processes
                output = subprocess.check_output("ps -A | grep -i chromedriver | grep -v grep", shell=True).decode()
                for line in output.splitlines():
                    parts = line.strip().split()
                    if len(parts) > 0:
                        try:
                            pid = int(parts[0])
                            # Skip the current process or its parent
                            if pid != os.getpid() and pid != os.getppid():
                                chrome_pids.append(pid)
                        except ValueError:
                            pass
            except subprocess.CalledProcessError:
                # grep returns non-zero exit code when no matches are found
                pass
                
            # Also look for Playwright browser processes
            try:
                output = subprocess.check_output("ps -A | grep -i playwright | grep -v grep", shell=True).decode()
                for line in output.splitlines():
                    parts = line.strip().split()
                    if len(parts) > 0:
                        try:
                            pid = int(parts[0])
                            # Skip the current process or its parent
                            if pid != os.getpid() and pid != os.getppid():
                                chrome_pids.append(pid)
                        except ValueError:
                            pass
            except subprocess.CalledProcessError:
                # grep returns non-zero exit code when no matches are found
                pass
                
    except Exception as e:
        logger.error(f"Error finding Chrome processes: {e}")
        
    return chrome_pids

def kill_process(pid, force=False):
    """Kill a process by PID.
    
    Args:
        pid: Process ID to kill
        force: If True, use SIGKILL (force kill) immediately
        
    Returns:
        bool: True if process was killed successfully, False otherwise
    """
    try:
        if platform.system() == "Windows":
            subprocess.call(['taskkill', '/F', '/PID', str(pid)])
        else:
            if force:
                # Force kill immediately
                os.kill(pid, signal.SIGKILL)
            else:
                # Try graceful termination first
                try:
                    os.kill(pid, signal.SIGTERM)
                    # Give it a moment to terminate gracefully
                    time.sleep(0.5)
                    # Check if still running
                    try:
                        os.kill(pid, 0)  # This will raise an exception if process is gone
                        # Process still running, force kill
                        os.kill(pid, signal.SIGKILL)
                    except OSError:
                        # Process already terminated
                        pass
                except OSError:
                    # If SIGTERM fails, try SIGKILL
                    try:
                        os.kill(pid, signal.SIGKILL)
                    except OSError as e:
                        if e.errno != 3:  # No such process
                            raise
        
        # Verify the process is actually gone
        time.sleep(0.2)  # Short delay to ensure OS has processed the kill signal
        try:
            if platform.system() == "Windows":
                # Check if process still exists on Windows
                subprocess.check_call(['tasklist', '/FI', f'PID eq {pid}'], 
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                # If we get here, process still exists
                return False
            else:
                # On Unix, try sending signal 0 which doesn't actually send a signal
                # but performs error checking
                os.kill(pid, 0)
                # If we get here, process still exists
                return False
        except (subprocess.CalledProcessError, OSError):
            # Process is gone
            return True
            
        return True
    except Exception as e:
        logger.error(f"Error killing process {pid}: {e}")
        return False

def cleanup_browser_processes(verbose=True, force=False, cleanup_temp=True, mcp_only=True):
    """Clean up browser processes that might be left running.
    
    Args:
        verbose: Whether to print status messages
        force: If True, use SIGKILL (force kill) immediately
        cleanup_temp: If True, also clean up temporary browser directories
        mcp_only: If True, only clean up MCP-related browser processes (default),
                  otherwise clean up all Chrome/Chromium processes
        
    Returns:
        int: Number of processes killed
    """
    if verbose:
        if mcp_only:
            print("Checking for MCP-related browser processes to clean up...")
        else:
            print("Checking for all browser processes to clean up...")
    
    # Choose which browser processes to clean up
    if mcp_only:
        browser_pids = find_mcp_browser_processes()
    else:
        browser_pids = find_chrome_processes()
    
    if not browser_pids:
        if verbose:
            print("No browser processes found.")
    else:    
        if verbose:
            print(f"Found {len(browser_pids)} browser processes to clean up: {browser_pids}")
    
    killed_count = 0
    for pid in browser_pids:
        if kill_process(pid, force=force):
            killed_count += 1
            if verbose:
                print(f"Successfully killed browser process {pid}")
        else:
            # Try with force=True if the first attempt failed
            if kill_process(pid, force=True):
                killed_count += 1
                if verbose:
                    print(f"Successfully force-killed browser process {pid}")
            else:
                if verbose:
                    print(f"Failed to kill browser process {pid}")
    
    if killed_count > 0 and verbose:
        print(f"Cleaned up {killed_count} browser processes")
    
    # Clean up temporary browser directories
    if cleanup_temp:
        temp_dirs_cleaned = cleanup_browser_temp_dirs(verbose=verbose)
        if temp_dirs_cleaned > 0 and verbose:
            print(f"Cleaned up {temp_dirs_cleaned} temporary browser directories")
            
    return killed_count


def cleanup_browser_temp_dirs(verbose=True):
    """Clean up temporary browser directories.
    
    Args:
        verbose: Whether to print status messages
        
    Returns:
        int: Number of directories cleaned
    """
    import glob
    import shutil
    
    temp_patterns = [
        "/tmp/browseruse-tmp*",
        "/tmp/playwright*",
        "/var/folders/*/T/browseruse-tmp*",
        "/var/folders/*/T/playwright*"
    ]
    
    if platform.system() == "Windows":
        temp_patterns = [
            os.path.join(os.environ.get("TEMP", "C:\\Windows\\Temp"), "browseruse-tmp*"),
            os.path.join(os.environ.get("TEMP", "C:\\Windows\\Temp"), "playwright*")
        ]
    
    cleaned_count = 0
    
    for pattern in temp_patterns:
        try:
            for path in glob.glob(pattern):
                if os.path.exists(path):
                    try:
                        if os.path.isdir(path):
                            if verbose:
                                print(f"Removing temporary directory: {path}")
                            shutil.rmtree(path, ignore_errors=True)
                        else:
                            if verbose:
                                print(f"Removing temporary file: {path}")
                            os.unlink(path)
                        cleaned_count += 1
                    except Exception as e:
                        if verbose:
                            print(f"Failed to remove {path}: {e}")
        except Exception as e:
            if verbose:
                print(f"Error searching for temporary files with pattern {pattern}: {e}")
    
    return cleaned_count

def kill_mcp_browser_processes(verbose=True):
    """Kill all MCP-related browser processes using pkill (Unix) or taskkill (Windows).
    
    This is a more aggressive approach that uses system commands to kill processes
    by name pattern rather than by PID. Useful for killing parent processes that
    might be spawning new browser instances.
    
    Args:
        verbose: Whether to print status messages
        
    Returns:
        bool: True if the command executed successfully
    """
    try:
        if platform.system() == "Windows":
            # Windows - use taskkill with filter
            if verbose:
                print("Killing MCP browser processes with taskkill...")
                
            # Kill Chrome processes with MCP-related command line arguments
            cmd = "taskkill /F /IM chrome.exe /FI \"WINDOWTITLE eq *mcp*\""
            subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Kill Playwright processes
            cmd = "taskkill /F /IM playwright.exe"
            subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Kill node processes related to Playwright
            cmd = "taskkill /F /IM node.exe /FI \"WINDOWTITLE eq *playwright*\""
            subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        else:
            # Unix - use pkill with pattern matching
            if verbose:
                print("Killing MCP browser processes with pkill...")
                
            # Kill all processes with 'playwright' or 'browseruse' in their command line
            cmd = "pkill -9 -f \"playwright|browseruse\""
            subprocess.call(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
        return True
    except Exception as e:
        if verbose:
            print(f"Error killing MCP browser processes: {e}")
        return False

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Clean up browser processes")
    parser.add_argument("--all", action="store_true", help="Clean up all Chrome/Chromium processes, not just MCP-related ones")
    parser.add_argument("--force", action="store_true", help="Force kill processes immediately")
    parser.add_argument("--no-temp", action="store_false", dest="cleanup_temp", help="Don't clean up temporary directories")
    parser.add_argument("--pkill", action="store_true", help="Use pkill/taskkill for more aggressive process termination")
    parser.add_argument("-q", "--quiet", action="store_true", help="Suppress verbose output")
    
    args = parser.parse_args()
    
    if args.pkill:
        # Use more aggressive approach
        kill_mcp_browser_processes(verbose=not args.quiet)
        
    # Clean up processes
    cleanup_browser_processes(
        verbose=not args.quiet,
        force=args.force,
        cleanup_temp=args.cleanup_temp,
        mcp_only=not args.all
    )