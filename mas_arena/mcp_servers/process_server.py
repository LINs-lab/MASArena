#!/usr/bin/env python3
import json
import sys
import os
import subprocess
import time
import threading
import traceback
import shlex
from typing import Dict, Any, List

"""
Process Execution Server - Communicates with client via stdio
Supported commands:
- run_command: Execute command and wait for completion
- start_process: Start a long-running process
- stop_process: Stop/terminate a process
- get_process_status: Get process status
- list_processes: List all running processes
"""

# Store running processes
running_processes = {}
process_lock = threading.Lock()
next_process_id = 1

def log_error(message: str) -> None:
    """Log error message to stderr"""
    print(f"[ERROR] {message}", file=sys.stderr)
    sys.stderr.flush()

def send_response(status: str, data: Dict[str, Any]) -> None:
    """Send response to stdout"""
    response = {"status": status, "data": data}
    print(json.dumps(response))
    sys.stdout.flush()

def process_output_reader(process, process_id: int, stream_name: str, buffer: List[str], max_lines: int = 1000):
    """Thread function to read process output stream (stdout/stderr)"""
    for line in iter(process.stdout.readline if stream_name == "stdout" else process.stderr.readline, ""):
        if not line:  # Empty line indicates EOF
            break
        line = line.rstrip('\n')
        with process_lock:
            buffer.append(line)
            # Limit buffer size
            if len(buffer) > max_lines:
                buffer.pop(0)
    
    # When the output stream closes, record this information
    with process_lock:
        if process_id in running_processes:
            running_processes[process_id][f"{stream_name}_closed"] = True

def handle_run_command(args: Dict[str, Any]) -> None:
    """Handle run command request"""
    command = args.get("command")
    shell = args.get("shell", False)
    cwd = args.get("cwd")
    env = args.get("env")
    timeout = args.get("timeout")  # seconds, None indicates no timeout
    
    if not command:
        send_response("error", {"message": "Missing required parameter: command"})
        return
    
    try:
        # Prepare environment variables
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        
        # Start process
        start_time = time.time()
        
        if shell:
            # Execute in shell, command can be a string
            process = subprocess.run(
                command,
                shell=True,
                cwd=cwd,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                timeout=timeout
            )
        else:
            # Execute directly, command should be a list, if string then split
            if isinstance(command, str):
                command = shlex.split(command)
                
            process = subprocess.run(
                command,
                shell=False,
                cwd=cwd,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                timeout=timeout
            )
        
        end_time = time.time()
        duration = end_time - start_time
        
        result = {
            "returncode": process.returncode,
            "stdout": process.stdout.strip(),
            "stderr": process.stderr.strip(),
            "duration": duration,
            "command": command,
            "cwd": cwd
        }
        
        send_response("success", result)
        
    except subprocess.TimeoutExpired:
        send_response("error", {
            "message": f"Command timed out after {timeout} seconds",
            "timeout": True,
            "command": command
        })
    except FileNotFoundError:
        send_response("error", {
            "message": f"Command not found: {command}",
            "command": command
        })
    except Exception as e:
        send_response("error", {
            "message": f"Error executing command: {str(e)}",
            "command": command,
            "error": str(e)
        })

def handle_start_process(args: Dict[str, Any]) -> None:
    """Handle start long-running process request"""
    global next_process_id
    
    command = args.get("command")
    shell = args.get("shell", False)
    cwd = args.get("cwd")
    env = args.get("env")
    process_name = args.get("name")
    
    if not command:
        send_response("error", {"message": "Missing required parameter: command"})
        return
    
    try:
        # Prepare environment variables
        process_env = os.environ.copy()
        if env:
            process_env.update(env)
        
        if shell:
            # Execute in shell
            process = subprocess.Popen(
                command,
                shell=True,
                cwd=cwd,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                bufsize=1,  # Line buffering
                universal_newlines=True
            )
        else:
            # Execute directly
            if isinstance(command, str):
                command = shlex.split(command)
                
            process = subprocess.Popen(
                command,
                shell=False,
                cwd=cwd,
                env=process_env,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding='utf-8',
                bufsize=1,  # Line buffering
                universal_newlines=True
            )
        
        # Assign process ID
        with process_lock:
            process_id = next_process_id
            next_process_id += 1
            
            # Store process information
            stdout_buffer = []
            stderr_buffer = []
            
            running_processes[process_id] = {
                "process": process,
                "command": command,
                "cwd": cwd,
                "pid": process.pid,
                "name": process_name or f"process-{process_id}",
                "start_time": time.time(),
                "stdout_buffer": stdout_buffer,
                "stderr_buffer": stderr_buffer,
                "stdout_closed": False,
                "stderr_closed": False,
                "exit_code": None
            }
        
        # Start threads to monitor process output
        stdout_thread = threading.Thread(
            target=process_output_reader,
            args=(process, process_id, "stdout", stdout_buffer),
            daemon=True
        )
        stderr_thread = threading.Thread(
            target=process_output_reader,
            args=(process, process_id, "stderr", stderr_buffer),
            daemon=True
        )
        
        stdout_thread.start()
        stderr_thread.start()
        
        # Return process ID
        send_response("success", {
            "process_id": process_id,
            "pid": process.pid,
            "name": process_name or f"process-{process_id}",
            "command": command,
            "start_time": time.time()
        })
        
    except FileNotFoundError:
        send_response("error", {
            "message": f"Command not found: {command}",
            "command": command
        })
    except Exception as e:
        send_response("error", {
            "message": f"Error starting process: {str(e)}",
            "command": command,
            "error": str(e)
        })

def handle_stop_process(args: Dict[str, Any]) -> None:
    """Handle stop process request"""
    process_id = args.get("process_id")
    force = args.get("force", False)
    timeout = args.get("timeout", 5)  # Wait for process termination timeout (seconds)
    
    if process_id is None:
        send_response("error", {"message": "Missing required parameter: process_id"})
        return
    
    with process_lock:
        if process_id not in running_processes:
            send_response("error", {"message": f"Process ID {process_id} not found"})
            return
        
        process_info = running_processes[process_id]
        process = process_info["process"]
    
    try:
        if process.poll() is not None:
            # Process already terminated
            exit_code = process.returncode
            send_response("success", {
                "message": f"Process {process_id} was already terminated",
                "process_id": process_id,
                "exit_code": exit_code,
                "already_terminated": True
            })
            
            # Update process information
            with process_lock:
                if process_id in running_processes:
                    running_processes[process_id]["exit_code"] = exit_code
            
            return
        
        # Send termination signal
        if force:
            # SIGKILL - Force termination
            process.kill()
        else:
            # SIGTERM - Request normal termination
            process.terminate()
        
        # Wait for process termination
        try:
            exit_code = process.wait(timeout=timeout)
            
            # Update process information
            with process_lock:
                if process_id in running_processes:
                    running_processes[process_id]["exit_code"] = exit_code
            
            send_response("success", {
                "message": f"Process {process_id} terminated successfully",
                "process_id": process_id,
                "exit_code": exit_code,
                "force": force
            })
        except subprocess.TimeoutExpired:
            # If timeout and no force termination, try to force terminate
            if not force:
                process.kill()
                try:
                    exit_code = process.wait(timeout=2)
                    
                    # Update process information
                    with process_lock:
                        if process_id in running_processes:
                            running_processes[process_id]["exit_code"] = exit_code
                    
                    send_response("success", {
                        "message": f"Process {process_id} forcefully terminated after timeout",
                        "process_id": process_id,
                        "exit_code": exit_code,
                        "force": True,
                        "timeout_occurred": True
                    })
                except subprocess.TimeoutExpired:
                    send_response("error", {
                        "message": f"Failed to terminate process {process_id} even with SIGKILL",
                        "process_id": process_id
                    })
            else:
                send_response("error", {
                    "message": f"Failed to terminate process {process_id} within timeout",
                    "process_id": process_id,
                    "force": force
                })
    except Exception as e:
        send_response("error", {
            "message": f"Error stopping process {process_id}: {str(e)}",
            "process_id": process_id,
            "error": str(e)
        })

def handle_get_process_status(args: Dict[str, Any]) -> None:
    """Handle get process status request"""
    process_id = args.get("process_id")
    include_output = args.get("include_output", True)
    max_output_lines = args.get("max_output_lines", 100)
    
    if process_id is None:
        send_response("error", {"message": "Missing required parameter: process_id"})
        return
    
    with process_lock:
        if process_id not in running_processes:
            send_response("error", {"message": f"Process ID {process_id} not found"})
            return
        
        process_info = running_processes[process_id]
        process = process_info["process"]
    
    try:
        # Check process status
        exit_code = process.poll()
        running = exit_code is None
        
        # Update process information
        with process_lock:
            if process_id in running_processes:
                running_processes[process_id]["exit_code"] = exit_code
        
        # Prepare response
        response = {
            "process_id": process_id,
            "pid": process_info["pid"],
            "name": process_info["name"],
            "command": process_info["command"],
            "cwd": process_info["cwd"],
            "start_time": process_info["start_time"],
            "running": running,
            "exit_code": exit_code,
            "elapsed_time": time.time() - process_info["start_time"]
        }
        
        # Include output if needed
        if include_output:
            with process_lock:
                stdout_buffer = process_info["stdout_buffer"][-max_output_lines:] if max_output_lines > 0 else process_info["stdout_buffer"]
                stderr_buffer = process_info["stderr_buffer"][-max_output_lines:] if max_output_lines > 0 else process_info["stderr_buffer"]
                
                response["stdout"] = "\n".join(stdout_buffer)
                response["stderr"] = "\n".join(stderr_buffer)
                response["stdout_line_count"] = len(process_info["stdout_buffer"])
                response["stderr_line_count"] = len(process_info["stderr_buffer"])
                response["stdout_closed"] = process_info["stdout_closed"]
                response["stderr_closed"] = process_info["stderr_closed"]
        
        send_response("success", response)
        
    except Exception as e:
        send_response("error", {
            "message": f"Error getting status for process {process_id}: {str(e)}",
            "process_id": process_id,
            "error": str(e)
        })

def handle_list_processes(args: Dict[str, Any]) -> None:
    """Handle list all processes request"""
    # Check dead processes and update status
    with process_lock:
        for process_id, process_info in list(running_processes.items()):
            process = process_info["process"]
            exit_code = process.poll()
            
            if exit_code is not None and process_info["exit_code"] is None:
                running_processes[process_id]["exit_code"] = exit_code
    
    # Prepare response
    process_list = []
    with process_lock:
        for process_id, process_info in running_processes.items():
            process_list.append({
                "process_id": process_id,
                "pid": process_info["pid"],
                "name": process_info["name"],
                "command": process_info["command"],
                "start_time": process_info["start_time"],
                "running": process_info["exit_code"] is None,
                "exit_code": process_info["exit_code"],
                "elapsed_time": time.time() - process_info["start_time"]
            })
    
    send_response("success", {
        "processes": process_list,
        "count": len(process_list)
    })

def cleanup_terminated_processes(args: Dict[str, Any]) -> None:
    """Clean up terminated processes"""
    older_than = args.get("older_than", 3600)  # Default clean up processes terminated more than 1 hour ago
    
    to_remove = []
    current_time = time.time()
    
    with process_lock:
        for process_id, process_info in running_processes.items():
            # Check if process is terminated
            if process_info["exit_code"] is not None:
                # Check if terminated enough time
                if current_time - process_info["start_time"] > older_than:
                    to_remove.append(process_id)
    
    # Remove terminated processes from dictionary
    for process_id in to_remove:
        with process_lock:
            running_processes.pop(process_id, None)
    
    send_response("success", {
        "processes_removed": to_remove,
        "count": len(to_remove)
    })

def main():
    """Main function - Handle requests from stdin"""
    handlers = {
        "run_command": handle_run_command,
        "start_process": handle_start_process,
        "stop_process": handle_stop_process,
        "get_process_status": handle_get_process_status,
        "list_processes": handle_list_processes,
        "cleanup_terminated_processes": cleanup_terminated_processes
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