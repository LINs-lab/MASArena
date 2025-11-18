#!/usr/bin/env python3
import json
import sys
import os
import subprocess
import tempfile
import traceback
import shutil
from typing import Dict, Any, List, Optional

"""
Git Server - Communicates with client via stdio
Supported commands:
- clone: Clone repository
- checkout: Switch branch or commit
- apply_diff: Apply diff/patch
- commit: Commit changes
- push: Push changes to remote repository
- pull: Pull changes from remote repository
"""

def log_error(message: str) -> None:
    """Log error message to stderr"""
    print(f"[ERROR] {message}", file=sys.stderr)
    sys.stderr.flush()

def send_response(status: str, data: Dict[str, Any]) -> None:
    """Send response to stdout"""
    response = {"status": status, "data": data}
    print(json.dumps(response))
    sys.stdout.flush()

def run_command(cmd: List[str], cwd: Optional[str] = None, env: Optional[Dict[str, str]] = None,
                timeout: Optional[int] = None) -> Dict[str, Any]:
    """Execute shell command and return results"""
    try:
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        
        process = subprocess.run(
            cmd,
            cwd=cwd,
            env=process_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            timeout=timeout
        )
        
        result = {
            "returncode": process.returncode,
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip()
        }
        
        if process.returncode != 0:
            result["error"] = f"Command failed with code {process.returncode}"
        
        return result
    except subprocess.TimeoutExpired:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": f"Command timed out after {timeout} seconds",
            "error": f"Command timed out after {timeout} seconds",
            "timeout": True
        }
    except Exception as e:
        return {
            "returncode": -1,
            "stdout": "",
            "stderr": str(e),
            "error": f"Exception running command: {str(e)}"
        }

def handle_clone(args: Dict[str, Any]) -> None:
    """Handle clone repository request"""
    repo_url = args.get("repo_url")
    target_dir = args.get("target_dir")
    shallow = args.get("shallow", False)  # Whether to perform shallow clone
    depth = args.get("depth", 1)  # Shallow clone depth
    branch = args.get("branch")  # Specific branch
    timeout = args.get("timeout", 600)  # Timeout in seconds
    force = args.get("force", False)  # Whether to force overwrite existing directory
    
    if not repo_url or not target_dir:
        send_response("error", {"message": "Missing required parameters: repo_url and target_dir"})
        return
    
    try:
        # Check if target directory already exists
        if os.path.exists(target_dir):
            if force:
                try:
                    shutil.rmtree(target_dir)
                except Exception as e:
                    send_response("error", {
                        "message": f"Failed to remove existing directory: {str(e)}",
                        "error": str(e)
                    })
                    return
            else:
                send_response("error", {
                    "message": f"Target directory already exists: {target_dir}. Use force=true to overwrite."
                })
                return
        
        # Build clone command
        clone_cmd = ["git", "clone"]
        
        if shallow:
            clone_cmd.extend(["--depth", str(depth)])
        
        if branch:
            clone_cmd.extend(["--branch", branch])
        
        clone_cmd.extend([repo_url, target_dir])
        
        # Execute clone command
        result = run_command(clone_cmd, timeout=timeout)
        
        if result["returncode"] != 0:
            send_response("error", {
                "message": f"Failed to clone repository: {result.get('stderr')}",
                "details": result
            })
            return
        
        # Get information about cloned repository
        git_log_cmd = ["git", "log", "-1", "--pretty=format:%h %s"]
        log_result = run_command(git_log_cmd, cwd=target_dir)
        
        if log_result["returncode"] == 0:
            current_commit = log_result["stdout"]
        else:
            current_commit = "Unknown"
        
        send_response("success", {
            "message": f"Repository cloned successfully to {target_dir}",
            "target_dir": target_dir,
            "current_commit": current_commit
        })
        
    except Exception as e:
        send_response("error", {
            "message": f"Error cloning repository: {str(e)}",
            "error": str(e),
            "traceback": traceback.format_exc()
        })

def handle_checkout(args: Dict[str, Any]) -> None:
    """Handle branch or commit checkout request"""
    target_dir = args.get("target_dir")
    commit_or_branch = args.get("commit_or_branch")
    create_branch = args.get("create_branch", False)  # Whether to create new branch
    
    if not target_dir or not commit_or_branch:
        send_response("error", {
            "message": "Missing required parameters: target_dir and commit_or_branch"
        })
        return
    
    try:
        if not os.path.exists(target_dir):
            send_response("error", {
                "message": f"Repository directory does not exist: {target_dir}"
            })
            return
        
        # Build checkout command
        checkout_cmd = ["git", "checkout"]
        
        if create_branch:
            checkout_cmd.append("-b")
        
        checkout_cmd.append(commit_or_branch)
        
        # Execute checkout command
        result = run_command(checkout_cmd, cwd=target_dir)
        
        if result["returncode"] != 0:
            send_response("error", {
                "message": f"Failed to checkout: {result.get('stderr')}",
                "details": result
            })
            return
        
        # Get current branch/commit information
        git_log_cmd = ["git", "log", "-1", "--pretty=format:%h %s"]
        log_result = run_command(git_log_cmd, cwd=target_dir)
        
        if log_result["returncode"] == 0:
            current_commit = log_result["stdout"]
        else:
            current_commit = "Unknown"
        
        send_response("success", {
            "message": f"Successfully checked out {commit_or_branch}",
            "current_commit": current_commit
        })
        
    except Exception as e:
        send_response("error", {
            "message": f"Error during checkout: {str(e)}",
            "error": str(e),
            "traceback": traceback.format_exc()
        })

def handle_apply_diff(args: Dict[str, Any]) -> None:
    """Handle apply diff/patch request"""
    target_dir = args.get("target_dir")
    diff_content = args.get("diff_content")
    diff_file = args.get("diff_file")
    check_only = args.get("check_only", False)  # Only check if patch can be applied, don't actually apply
    
    if not target_dir or (not diff_content and not diff_file):
        send_response("error", {
            "message": "Missing required parameters: target_dir and either diff_content or diff_file"
        })
        return
    
    try:
        if not os.path.exists(target_dir):
            send_response("error", {
                "message": f"Repository directory does not exist: {target_dir}"
            })
            return
        
        # If diff content provided, create temporary file
        temp_file = None
        if diff_content:
            fd, temp_file = tempfile.mkstemp(suffix=".patch")
            with os.fdopen(fd, 'w') as f:
                f.write(diff_content)
            diff_to_apply = temp_file
        else:
            diff_to_apply = diff_file
        
        try:
            # Build git apply command
            apply_cmd = ["git", "apply"]
            
            if check_only:
                apply_cmd.append("--check")
            
            apply_cmd.append(diff_to_apply)
            
            # Execute apply patch command
            result = run_command(apply_cmd, cwd=target_dir)
            
            if result["returncode"] != 0:
                send_response("error", {
                    "message": f"Failed to apply patch: {result.get('stderr')}",
                    "details": result
                })
                return
            
            if check_only:
                send_response("success", {
                    "message": "Patch can be applied cleanly"
                })
            else:
                # Get changed file list
                status_cmd = ["git", "status", "--porcelain"]
                status_result = run_command(status_cmd, cwd=target_dir)
                
                changed_files = []
                if status_result["returncode"] == 0 and status_result["stdout"]:
                    for line in status_result["stdout"].splitlines():
                        if line.strip():
                            status_code = line[:2].strip()
                            file_path = line[3:].strip()
                            changed_files.append({"status": status_code, "file": file_path})
                
                send_response("success", {
                    "message": "Patch applied successfully",
                    "changed_files": changed_files
                })
                
        finally:
            # Clean up temporary file
            if temp_file and os.path.exists(temp_file):
                os.unlink(temp_file)
                
    except Exception as e:
        send_response("error", {
            "message": f"Error applying patch: {str(e)}",
            "error": str(e),
            "traceback": traceback.format_exc()
        })

def handle_commit(args: Dict[str, Any]) -> None:
    """Handle commit changes request"""
    target_dir = args.get("target_dir")
    message = args.get("message", "Automated commit via Git Server")
    add_all = args.get("add_all", True)  # Whether to add all changes
    author = args.get("author")  # Optional author information
    
    if not target_dir:
        send_response("error", {"message": "Missing required parameter: target_dir"})
        return
    
    try:
        if not os.path.exists(target_dir):
            send_response("error", {
                "message": f"Repository directory does not exist: {target_dir}"
            })
            return
        
        # Add changes
        if add_all:
            add_cmd = ["git", "add", "-A"]
            add_result = run_command(add_cmd, cwd=target_dir)
            
            if add_result["returncode"] != 0:
                send_response("error", {
                    "message": f"Failed to stage changes: {add_result.get('stderr')}",
                    "details": add_result
                })
                return
        
        # Build commit command
        commit_cmd = ["git", "commit", "-m", message]
        
        if author:
            commit_cmd.extend(["--author", author])
        
        # Execute commit command
        result = run_command(commit_cmd, cwd=target_dir)
        
        if result["returncode"] != 0:
            # Commit failed, no changes
            if "nothing to commit" in result.get("stderr", "") or "nothing to commit" in result.get("stdout", ""):
                send_response("success", {
                    "message": "Nothing to commit, working tree clean",
                    "no_changes": True
                })
                return
            else:
                send_response("error", {
                    "message": f"Failed to commit changes: {result.get('stderr')}",
                    "details": result
                })
                return
        
        # Get current commit information
        git_log_cmd = ["git", "log", "-1", "--pretty=format:%h %s"]
        log_result = run_command(git_log_cmd, cwd=target_dir)
        
        if log_result["returncode"] == 0:
            current_commit = log_result["stdout"]
        else:
            current_commit = "Unknown"
        
        send_response("success", {
            "message": "Changes committed successfully",
            "current_commit": current_commit
        })
        
    except Exception as e:
        send_response("error", {
            "message": f"Error committing changes: {str(e)}",
            "error": str(e),
            "traceback": traceback.format_exc()
        })

def main():
    """Main function, handle input requests"""
    try:
        # Read JSON input
        request = json.loads(input())
        
        command = request.get("command")
        args = request.get("args", {})
        
        # Dispatch to corresponding handler function based on command
        if command == "clone":
            handle_clone(args)
        elif command == "checkout":
            handle_checkout(args)
        elif command == "apply_diff":
            handle_apply_diff(args)
        elif command == "commit":
            handle_commit(args)
        else:
            send_response("error", {"message": f"Unknown command: {command}"})
            
    except json.JSONDecodeError:
        send_response("error", {"message": "Invalid JSON input"})
    except Exception as e:
        send_response("error", {
            "message": f"Unexpected error: {str(e)}",
            "error": str(e),
            "traceback": traceback.format_exc()
        })
        log_error(traceback.format_exc())

if __name__ == "__main__":
    main() 