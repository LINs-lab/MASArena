#!/usr/bin/env python3
import json
import sys
import os
import subprocess
import tempfile
import time
import traceback
import shutil
import re
from typing import Dict, Any, List, Optional

"""
SWE-bench Evaluator Server - Communicates with client via stdio
Supported commands:
- evaluate_patch: Evaluate if a patch successfully fixes the issue
- setup_repository: Set up and prepare a code repository for evaluation
- run_test: Run a specific test
- verify_fix: Verify if a fix is successful
- get_evaluation_results: Get evaluation results
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
                timeout: Optional[int] = None, shell: bool = False) -> Dict[str, Any]:
    """Execute shell command and return results"""
    try:
        if shell and isinstance(cmd, list):
            cmd = " ".join(cmd)
        
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        
        process = subprocess.run(
            cmd,
            shell=shell,
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

def handle_setup_repository(args: Dict[str, Any]) -> None:
    """Handle repository setup request"""
    repo_url = args.get("repo_url")
    repo_path = args.get("repo_path")
    commit_hash = args.get("commit_hash")
    force_clean = args.get("force_clean", False)
    setup_command = args.get("setup_command")  # Optional setup command, e.g., installing dependencies
    setup_timeout = args.get("setup_timeout", 600)  # Setup command timeout in seconds
    
    if not repo_url or not repo_path:
        send_response("error", {"message": "Missing required parameters: repo_url and repo_path"})
        return
    
    try:
        # If directory exists and force clean is set
        if os.path.exists(repo_path) and force_clean:
            shutil.rmtree(repo_path)
        
        # Create or clone repository
        if not os.path.exists(repo_path):
            # Clone repository
            clone_cmd = ["git", "clone", repo_url, repo_path]
            clone_result = run_command(clone_cmd)
            
            if clone_result["returncode"] != 0:
                send_response("error", {
                    "message": f"Failed to clone repository: {clone_result.get('stderr')}",
                    "details": clone_result
                })
                return
        
        # Checkout specific commit
        if commit_hash:
            checkout_cmd = ["git", "checkout", commit_hash]
            checkout_result = run_command(checkout_cmd, cwd=repo_path)
            
            if checkout_result["returncode"] != 0:
                send_response("error", {
                    "message": f"Failed to checkout commit {commit_hash}: {checkout_result.get('stderr')}",
                    "details": checkout_result
                })
                return
        
        # Run setup command (if provided)
        if setup_command:
            setup_result = run_command(
                setup_command,
                cwd=repo_path,
                timeout=setup_timeout,
                shell=True
            )
            
            if setup_result["returncode"] != 0:
                send_response("error", {
                    "message": f"Repository setup command failed: {setup_result.get('stderr')}",
                    "details": setup_result
                })
                return
            
            setup_output = {
                "setup_stdout": setup_result["stdout"],
                "setup_stderr": setup_result["stderr"]
            }
        else:
            setup_output = {}
        
        # Get current commit information
        git_log_cmd = ["git", "log", "-1", "--pretty=format:%h %s"]
        log_result = run_command(git_log_cmd, cwd=repo_path)
        
        if log_result["returncode"] == 0:
            current_commit_info = log_result["stdout"]
        else:
            current_commit_info = "Unknown"
        
        send_response("success", {
            "message": f"Repository setup completed successfully at {repo_path}",
            "repo_path": repo_path,
            "current_commit": current_commit_info,
            **setup_output
        })
        
    except Exception as e:
        send_response("error", {
            "message": f"Error setting up repository: {str(e)}",
            "traceback": traceback.format_exc()
        })

def handle_evaluate_patch(args: Dict[str, Any]) -> None:
    """Handle patch evaluation request"""
    repo_path = args.get("repo_path")
    patch_content = args.get("patch_content")
    patch_file = args.get("patch_file")
    test_command = args.get("test_command")
    test_timeout = args.get("test_timeout", 300)
    setup_command = args.get("setup_command")  # Setup command after applying patch
    setup_timeout = args.get("setup_timeout", 600)
    
    if not repo_path or (not patch_content and not patch_file) or not test_command:
        send_response("error", {
            "message": "Missing required parameters: repo_path, patch (content or file), and test_command"
        })
        return
    
    try:
        # Save initial repository state
        initial_state_cmd = ["git", "rev-parse", "HEAD"]
        initial_state = run_command(initial_state_cmd, cwd=repo_path)
        
        if initial_state["returncode"] != 0:
            send_response("error", {
                "message": f"Failed to get initial repository state: {initial_state.get('stderr')}",
                "details": initial_state
            })
            return
        
        initial_commit = initial_state["stdout"]
        
        # Create temporary patch file (if content provided)
        temp_patch_file = None
        if patch_content:
            fd, temp_patch_file = tempfile.mkstemp(suffix=".patch")
            with os.fdopen(fd, 'w') as f:
                f.write(patch_content)
            patch_to_apply = temp_patch_file
        else:
            patch_to_apply = patch_file
        
        try:
            # Apply patch
            apply_cmd = ["git", "apply", "--check", patch_to_apply]
            check_result = run_command(apply_cmd, cwd=repo_path)
            
            if check_result["returncode"] != 0:
                send_response("error", {
                    "message": f"Patch validation failed: {check_result.get('stderr')}",
                    "details": check_result
                })
                return
            
            # Actually apply patch
            apply_cmd = ["git", "apply", patch_to_apply]
            apply_result = run_command(apply_cmd, cwd=repo_path)
            
            if apply_result["returncode"] != 0:
                send_response("error", {
                    "message": f"Failed to apply patch: {apply_result.get('stderr')}",
                    "details": apply_result
                })
                return
            
            # Run setup command after applying patch (if provided)
            if setup_command:
                setup_result = run_command(
                    setup_command,
                    cwd=repo_path,
                    timeout=setup_timeout,
                    shell=True
                )
                
                if setup_result["returncode"] != 0:
                    send_response("error", {
                        "message": f"Post-patch setup command failed: {setup_result.get('stderr')}",
                        "details": setup_result
                    })
                    return
            
            # Run test command
            start_time = time.time()
            test_result = run_command(
                test_command,
                cwd=repo_path,
                timeout=test_timeout,
                shell=True
            )
            test_duration = time.time() - start_time
            
            # Collect results
            result = {
                "test_passed": test_result["returncode"] == 0,
                "test_duration": test_duration,
                "test_stdout": test_result["stdout"],
                "test_stderr": test_result["stderr"],
                "test_exit_code": test_result["returncode"]
            }
            
            # Check if timed out
            if "timeout" in test_result and test_result["timeout"]:
                result["test_timeout"] = True
            
            # If test passed, mark result as successful
            if result["test_passed"]:
                # Calculate modified files and lines
                diff_cmd = ["git", "diff", "--stat"]
                diff_result = run_command(diff_cmd, cwd=repo_path)
                
                if diff_result["returncode"] == 0:
                    result["diff_stats"] = diff_result["stdout"]
                    
                    # Parse modified files and lines
                    modified_files = 0
                    modified_lines = 0
                    
                    for line in diff_result["stdout"].splitlines():
                        if "|" in line:
                            modified_files += 1
                            # Try to extract modified lines
                            match = re.search(r"(\d+) [+-]", line)
                            if match:
                                modified_lines += int(match.group(1))
                    
                    result["modified_files"] = modified_files
                    result["modified_lines"] = modified_lines
                
                send_response("success", {
                    "message": "Patch evaluation completed successfully, test passed",
                    "successful": True,
                    "evaluation_result": result
                })
            else:
                send_response("success", {
                    "message": "Patch evaluation completed, test failed",
                    "successful": False,
                    "evaluation_result": result
                })
        
        finally:
            # Clean temporary files
            if temp_patch_file and os.path.exists(temp_patch_file):
                os.unlink(temp_patch_file)
            
            # Reset repository state
            reset_cmd = ["git", "reset", "--hard", initial_commit]
            run_command(reset_cmd, cwd=repo_path)
            
            # Clean untracked files
            clean_cmd = ["git", "clean", "-fd"]
            run_command(clean_cmd, cwd=repo_path)
    
    except Exception as e:
        send_response("error", {
            "message": f"Error evaluating patch: {str(e)}",
            "traceback": traceback.format_exc()
        })
        
        # Try to reset repository state
        try:
            if 'initial_commit' in locals():
                reset_cmd = ["git", "reset", "--hard", initial_commit]
                run_command(reset_cmd, cwd=repo_path)
                
                clean_cmd = ["git", "clean", "-fd"]
                run_command(clean_cmd, cwd=repo_path)
        except:
            pass

def handle_run_test(args: Dict[str, Any]) -> None:
    """Handle run test request"""
    repo_path = args.get("repo_path")
    test_command = args.get("test_command")
    test_timeout = args.get("test_timeout", 300)
    test_name = args.get("test_name", "unspecified")
    
    if not repo_path or not test_command:
        send_response("error", {"message": "Missing required parameters: repo_path and test_command"})
        return
    
    try:
        # Run test command
        start_time = time.time()
        test_result = run_command(
            test_command,
            cwd=repo_path,
            timeout=test_timeout,
            shell=True
        )
        test_duration = time.time() - start_time
        
        # Collect results
        result = {
            "test_name": test_name,
            "test_passed": test_result["returncode"] == 0,
            "test_duration": test_duration,
            "test_stdout": test_result["stdout"],
            "test_stderr": test_result["stderr"],
            "test_exit_code": test_result["returncode"]
        }
        
        # Check if timed out
        if "timeout" in test_result and test_result["timeout"]:
            result["test_timeout"] = True
        
        if result["test_passed"]:
            send_response("success", {
                "message": f"Test '{test_name}' completed successfully and passed",
                "successful": True,
                "test_result": result
            })
        else:
            send_response("success", {
                "message": f"Test '{test_name}' completed but failed",
                "successful": False,
                "test_result": result
            })
    
    except Exception as e:
        send_response("error", {
            "message": f"Error running test: {str(e)}",
            "traceback": traceback.format_exc()
        })

def handle_verify_fix(args: Dict[str, Any]) -> None:
    """Handle verify fix request"""
    repo_path = args.get("repo_path")
    patch_content = args.get("patch_content")
    patch_file = args.get("patch_file")
    test_commands = args.get("test_commands", [])  # Can be single command or command list
    test_timeout = args.get("test_timeout", 300)
    setup_command = args.get("setup_command")
    setup_timeout = args.get("setup_timeout", 600)
    
    if not repo_path or (not patch_content and not patch_file) or not test_commands:
        send_response("error", {
            "message": "Missing required parameters: repo_path, patch (content or file), and test_commands"
        })
        return
    
    # Ensure test_commands is a list
    if isinstance(test_commands, str):
        test_commands = [test_commands]
    
    try:
        # Save initial repository state
        initial_state_cmd = ["git", "rev-parse", "HEAD"]
        initial_state = run_command(initial_state_cmd, cwd=repo_path)
        
        if initial_state["returncode"] != 0:
            send_response("error", {
                "message": f"Failed to get initial repository state: {initial_state.get('stderr')}",
                "details": initial_state
            })
            return
        
        initial_commit = initial_state["stdout"]
        
        # Create temporary patch file (if content provided)
        temp_patch_file = None
        if patch_content:
            fd, temp_patch_file = tempfile.mkstemp(suffix=".patch")
            with os.fdopen(fd, 'w') as f:
                f.write(patch_content)
            patch_to_apply = temp_patch_file
        else:
            patch_to_apply = patch_file
        
        try:
            # Apply patch
            apply_cmd = ["git", "apply", patch_to_apply]
            apply_result = run_command(apply_cmd, cwd=repo_path)
            
            if apply_result["returncode"] != 0:
                # Try using --reject option
                apply_reject_cmd = ["git", "apply", "--reject", patch_to_apply]
                apply_reject_result = run_command(apply_reject_cmd, cwd=repo_path)
                
                if apply_reject_result["returncode"] != 0:
                    send_response("error", {
                        "message": f"Failed to apply patch: {apply_result.get('stderr')}",
                        "details": apply_result,
                        "reject_details": apply_reject_result
                    })
                    return
                
                # Check rejected files
                reject_files = []
                for root, dirs, files in os.walk(repo_path):
                    for file in files:
                        if file.endswith(".rej"):
                            reject_files.append(os.path.join(root, file))
                
                if reject_files:
                    send_response("error", {
                        "message": f"Patch applied partially with rejects: {len(reject_files)} files had conflicts",
                        "reject_files": reject_files
                    })
                    return
            
            # Run setup command after applying patch (if provided)
            if setup_command:
                setup_result = run_command(
                    setup_command,
                    cwd=repo_path,
                    timeout=setup_timeout,
                    shell=True
                )
                
                if setup_result["returncode"] != 0:
                    send_response("error", {
                        "message": f"Post-patch setup command failed: {setup_result.get('stderr')}",
                        "details": setup_result
                    })
                    return
            
            # Run all test commands
            test_results = []
            all_tests_passed = True
            
            for i, test_cmd in enumerate(test_commands):
                test_name = f"test-{i+1}" if len(test_commands) > 1 else "test"
                start_time = time.time()
                
                test_result = run_command(
                    test_cmd,
                    cwd=repo_path,
                    timeout=test_timeout,
                    shell=True
                )
                
                test_duration = time.time() - start_time
                test_passed = test_result["returncode"] == 0
                
                if not test_passed:
                    all_tests_passed = False
                
                result = {
                    "test_name": test_name,
                    "test_command": test_cmd,
                    "test_passed": test_passed,
                    "test_duration": test_duration,
                    "test_stdout": test_result["stdout"],
                    "test_stderr": test_result["stderr"],
                    "test_exit_code": test_result["returncode"]
                }
                
                # Check if timed out
                if "timeout" in test_result and test_result["timeout"]:
                    result["test_timeout"] = True
                
                test_results.append(result)
            
            # If all tests passed, calculate modified statistics
            if all_tests_passed:
                # Calculate modified files and lines
                diff_cmd = ["git", "diff", "--stat"]
                diff_result = run_command(diff_cmd, cwd=repo_path)
                
                diff_stats = None
                modified_files = 0
                modified_lines = 0
                
                if diff_result["returncode"] == 0:
                    diff_stats = diff_result["stdout"]
                    
                    # Parse modified files and lines
                    for line in diff_result["stdout"].splitlines():
                        if "|" in line:
                            modified_files += 1
                            # Try to extract modified lines
                            match = re.search(r"(\d+) [+-]", line)
                            if match:
                                modified_lines += int(match.group(1))
            
            # Collect all results
            verify_result = {
                "all_tests_passed": all_tests_passed,
                "test_results": test_results,
                "test_count": len(test_results),
                "passed_test_count": sum(1 for r in test_results if r["test_passed"])
            }
            
            if all_tests_passed and diff_stats:
                verify_result["diff_stats"] = diff_stats
                verify_result["modified_files"] = modified_files
                verify_result["modified_lines"] = modified_lines
            
            if all_tests_passed:
                send_response("success", {
                    "message": "Verification completed successfully, all tests passed",
                    "successful": True,
                    "verification_result": verify_result
                })
            else:
                send_response("success", {
                    "message": "Verification completed, some tests failed",
                    "successful": False,
                    "verification_result": verify_result
                })
        
        finally:
            # Clean temporary files
            if temp_patch_file and os.path.exists(temp_patch_file):
                os.unlink(temp_patch_file)
            
            # Reset repository state
            reset_cmd = ["git", "reset", "--hard", initial_commit]
            run_command(reset_cmd, cwd=repo_path)
            
            # Clean untracked files and rejected files
            clean_cmd = ["git", "clean", "-fd"]
            run_command(clean_cmd, cwd=repo_path)
    
    except Exception as e:
        send_response("error", {
            "message": f"Error verifying fix: {str(e)}",
            "traceback": traceback.format_exc()
        })
        
        # Try to reset repository state
        try:
            if 'initial_commit' in locals():
                reset_cmd = ["git", "reset", "--hard", initial_commit]
                run_command(reset_cmd, cwd=repo_path)
                
                clean_cmd = ["git", "clean", "-fd"]
                run_command(clean_cmd, cwd=repo_path)
        except:
            pass

def handle_get_evaluation_results(args: Dict[str, Any]) -> None:
    """Handle get evaluation results request"""
    # evaluation_id = args.get("evaluation_id")
    
    # This is just a placeholder, in a real application, you might need to retrieve evaluation results from persistent storage
    # Since MCP is stateless, this function might need to connect to a database or read from a file
    # Here, we simply return an error indicating that this functionality is not implemented
    
    send_response("error", {
        "message": "The get_evaluation_results command is not implemented in this stateless MCP server. Results should be managed by the client."
    })

def main():
    """Main function - Handle requests from stdin"""
    handlers = {
        "setup_repository": handle_setup_repository,
        "evaluate_patch": handle_evaluate_patch,
        "run_test": handle_run_test,
        "verify_fix": handle_verify_fix,
        "get_evaluation_results": handle_get_evaluation_results
    }
    
    for line in sys.stdin:
        try:
            # Parse JSON request
            request = json.loads(line.strip())
            command = request.get("command")
            args = request.get("args", {})
            
            if command not in handlers:
                send_response("error", {"message": f"Unknown command: {command}"})
                continue
            
            # Call corresponding handler function
            handlers[command](args)
            
        except json.JSONDecodeError:
            send_response("error", {"message": f"Invalid JSON: {line.strip()}"})
        except Exception as e:
            tb = traceback.format_exc()
            send_response("error", {
                "message": f"Error processing request: {str(e)}",
                "traceback": tb
            })

if __name__ == "__main__":
    main() 