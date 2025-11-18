#!/usr/bin/env python3
import json
import sys
import os
import subprocess
import shutil
import venv
import traceback
from typing import Dict, Any, List, Optional

"""
Environment Management Server - Communicates with client via stdio
Supported commands:
- create_venv: Create Python virtual environment
- run_in_venv: Execute command in virtual environment
- install_packages: Install dependencies in virtual environment
- check_docker: Check if Docker is available
- run_in_docker: Execute command in Docker container
- build_docker_image: Build Docker image
- get_system_info: Get system information
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

def handle_create_venv(args: Dict[str, Any]) -> None:
    """Handle create Python virtual environment request"""
    venv_path = args.get("venv_path")
    python_exe = args.get("python_exe")  # Optional, use specific Python interpreter
    with_pip = args.get("with_pip", True)
    system_site_packages = args.get("system_site_packages", False)
    clear = args.get("clear", False)  # If exists, whether to clear
    upgrade_deps = args.get("upgrade_deps", False)
    
    if not venv_path:
        send_response("error", {"message": "Missing required parameter: venv_path"})
        return
    
    try:
        # Check if already exists
        if os.path.exists(venv_path):
            if clear:
                # If clear is set, delete existing environment
                shutil.rmtree(venv_path)
            else:
                send_response("error", {
                    "message": f"Virtual environment already exists at {venv_path}. Use clear=true to recreate."
                })
                return
        
        # Ensure parent directory exists
        os.makedirs(os.path.dirname(os.path.abspath(venv_path)), exist_ok=True)
        
        if python_exe:
            # Create environment with specified Python interpreter
            cmd = [python_exe, "-m", "venv"]
            if with_pip:
                cmd.append("--with-pip")
            if system_site_packages:
                cmd.append("--system-site-packages")
            if upgrade_deps:
                cmd.append("--upgrade-deps")
            cmd.append(venv_path)
            
            result = run_command(cmd)
            
            if result["returncode"] != 0:
                send_response("error", {
                    "message": f"Failed to create virtual environment: {result['stderr']}",
                    "details": result
                })
                return
        else:
            # Use venv module to create directly
            try:
                builder = venv.EnvBuilder(
                    system_site_packages=system_site_packages,
                    clear=clear,
                    with_pip=with_pip,
                    upgrade_deps=upgrade_deps
                )
                builder.create(venv_path)
            except Exception as e:
                send_response("error", {
                    "message": f"Failed to create virtual environment: {str(e)}"
                })
                return
        
        # Get Python path in virtual environment
        if os.name == 'nt':  # Windows
            venv_python = os.path.join(venv_path, "Scripts", "python.exe")
        else:  # Unix-like
            venv_python = os.path.join(venv_path, "bin", "python")
        
        # Check if pip is available
        pip_check_cmd = [venv_python, "-m", "pip", "--version"]
        pip_result = run_command(pip_check_cmd)
        pip_available = pip_result["returncode"] == 0
        
        # Check virtual environment information
        version_cmd = [venv_python, "--version"]
        version_result = run_command(version_cmd)
        python_version = version_result["stdout"] if version_result["returncode"] == 0 else "Unknown"
        
        send_response("success", {
            "message": f"Virtual environment created at {venv_path}",
            "venv_path": venv_path,
            "python_path": venv_python,
            "python_version": python_version,
            "pip_available": pip_available
        })
    except Exception as e:
        send_response("error", {
            "message": f"Error creating virtual environment: {str(e)}"
        })

def get_venv_python(venv_path: str) -> str:
    """Get Python interpreter path in virtual environment"""
    if os.name == 'nt':  # Windows
        return os.path.join(venv_path, "Scripts", "python.exe")
    else:  # Unix-like
        return os.path.join(venv_path, "bin", "python")

def handle_run_in_venv(args: Dict[str, Any]) -> None:
    """Handle execute command in virtual environment request"""
    venv_path = args.get("venv_path")
    command = args.get("command")
    cwd = args.get("cwd")
    timeout = args.get("timeout")
    env = args.get("env", {})
    
    if not venv_path or not command:
        send_response("error", {"message": "Missing required parameters: venv_path and command"})
        return
    
    if not os.path.exists(venv_path):
        send_response("error", {"message": f"Virtual environment not found at {venv_path}"})
        return
    
    try:
        # Get Python interpreter path in virtual environment
        venv_python = get_venv_python(venv_path)
        
        if not os.path.exists(venv_python):
            send_response("error", {
                "message": f"Python interpreter not found in virtual environment: {venv_python}"
            })
            return
        
        # Prepare environment variables
        process_env = os.environ.copy()
        if os.name == 'nt':  # Windows
            process_env["PATH"] = f"{os.path.join(venv_path, 'Scripts')}{os.pathsep}{process_env['PATH']}"
        else:  # Unix-like
            process_env["PATH"] = f"{os.path.join(venv_path, 'bin')}{os.pathsep}{process_env['PATH']}"
        
        process_env.update(env)
        
        # If command is a list, ensure first element is python interpreter path
        if isinstance(command, list):
            if command[0] == "python":
                command[0] = venv_python
        else:
            # For string commands, try to replace leading python command
            if command.startswith("python "):
                command = f"{venv_python} {command[7:]}"
        
        # Execute command
        result = run_command(
            command,
            cwd=cwd,
            env=process_env,
            timeout=timeout,
            shell=isinstance(command, str)
        )
        
        if result["returncode"] == 0:
            send_response("success", {
                "message": "Command executed successfully in virtual environment",
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "command": command,
                "venv_path": venv_path
            })
        else:
            send_response("error", {
                "message": f"Command failed in virtual environment: {result.get('error', result['stderr'])}",
                "details": result,
                "command": command,
                "venv_path": venv_path
            })
    except Exception as e:
        send_response("error", {
            "message": f"Error running command in virtual environment: {str(e)}",
            "command": command,
            "venv_path": venv_path
        })

def handle_install_packages(args: Dict[str, Any]) -> None:
    """Handle install dependencies in virtual environment request"""
    venv_path = args.get("venv_path")
    packages = args.get("packages", [])
    requirements_file = args.get("requirements_file")
    upgrade = args.get("upgrade", False)
    index_url = args.get("index_url")
    extra_index_url = args.get("extra_index_url")
    no_cache = args.get("no_cache", False)
    timeout = args.get("timeout")
    
    if not venv_path:
        send_response("error", {"message": "Missing required parameter: venv_path"})
        return
    
    if not packages and not requirements_file:
        send_response("error", {"message": "Either packages or requirements_file must be provided"})
        return
    
    if not os.path.exists(venv_path):
        send_response("error", {"message": f"Virtual environment not found at {venv_path}"})
        return
    
    try:
        # Get virtual environment pip path
        if os.name == 'nt':  # Windows
            pip_path = os.path.join(venv_path, "Scripts", "pip.exe")
        else:  # Unix-like
            pip_path = os.path.join(venv_path, "bin", "pip")
        
        if not os.path.exists(pip_path):
            venv_python = get_venv_python(venv_path)
            pip_path = f"{venv_python} -m pip"
        
        # Build pip command
        if isinstance(pip_path, str) and " " in pip_path:
            # If pip_path contains spaces (like "python -m pip"), use shell mode
            cmd = pip_path + " install"
            shell = True
        else:
            cmd = [pip_path, "install"]
            shell = False
        
        # Add options
        if upgrade:
            if shell:
                cmd += " --upgrade"
            else:
                cmd.append("--upgrade")
        
        if index_url:
            if shell:
                cmd += f" --index-url {index_url}"
            else:
                cmd.extend(["--index-url", index_url])
        
        if extra_index_url:
            if shell:
                cmd += f" --extra-index-url {extra_index_url}"
            else:
                cmd.extend(["--extra-index-url", extra_index_url])
        
        if no_cache:
            if shell:
                cmd += " --no-cache-dir"
            else:
                cmd.append("--no-cache-dir")
        
        # Add packages or requirements file
        if requirements_file:
            if shell:
                cmd += f" -r {requirements_file}"
            else:
                cmd.extend(["-r", requirements_file])
        
        if packages:
            if shell:
                cmd += " " + " ".join(packages)
            else:
                cmd.extend(packages)
        
        # Execute command
        result = run_command(cmd, timeout=timeout, shell=shell)
        
        if result["returncode"] == 0:
            send_response("success", {
                "message": "Packages installed successfully in virtual environment",
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "packages": packages,
                "requirements_file": requirements_file,
                "venv_path": venv_path
            })
        else:
            send_response("error", {
                "message": f"Failed to install packages in virtual environment: {result.get('error', result['stderr'])}",
                "details": result,
                "packages": packages,
                "requirements_file": requirements_file,
                "venv_path": venv_path
            })
    except Exception as e:
        send_response("error", {
            "message": f"Error installing packages in virtual environment: {str(e)}",
            "packages": packages,
            "requirements_file": requirements_file,
            "venv_path": venv_path
        })

def handle_check_docker(args: Dict[str, Any]) -> None:
    """Handle check if Docker is available request"""
    try:
        # Check if docker command is available
        result = run_command(["docker", "--version"])
        
        docker_available = result["returncode"] == 0
        docker_version = result["stdout"] if docker_available else None
        
        if docker_available:
            # Check if docker daemon is running
            info_result = run_command(["docker", "info"])
            docker_running = info_result["returncode"] == 0
            docker_info = info_result["stdout"] if docker_running else None
            
            if docker_running:
                # Check if able to pull image (optional, as this might be time-consuming)
                if args.get("check_pull", False):
                    pull_result = run_command(["docker", "pull", "hello-world"], timeout=30)
                    can_pull = pull_result["returncode"] == 0
                else:
                    can_pull = None
                
                send_response("success", {
                    "available": True,
                    "running": True,
                    "version": docker_version,
                    "can_pull": can_pull,
                    "info": docker_info
                })
            else:
                send_response("success", {
                    "available": True,
                    "running": False,
                    "version": docker_version,
                    "error": info_result["stderr"]
                })
        else:
            send_response("success", {
                "available": False,
                "running": False,
                "error": result["stderr"]
            })
    except Exception as e:
        send_response("error", {
            "message": f"Error checking Docker: {str(e)}"
        })

def handle_run_in_docker(args: Dict[str, Any]) -> None:
    """Handle execute command in Docker container request"""
    image = args.get("image")
    command = args.get("command")
    volumes = args.get("volumes", [])  # Format: ["/host/path:/container/path", ...]
    env = args.get("env", {})  # Format: {"VAR_NAME": "value", ...}
    network = args.get("network")
    ports = args.get("ports", [])  # Format: ["host_port:container_port", ...]
    working_dir = args.get("working_dir")
    user = args.get("user")
    rm = args.get("rm", True)  # Delete container after running
    timeout = args.get("timeout")
    
    if not image or not command:
        send_response("error", {"message": "Missing required parameters: image and command"})
        return
    
    try:
        # Build docker run command
        cmd = ["docker", "run"]
        
        if rm:
            cmd.append("--rm")
        
        for volume in volumes:
            cmd.extend(["-v", volume])
        
        for key, value in env.items():
            cmd.extend(["-e", f"{key}={value}"])
        
        if network:
            cmd.extend(["--network", network])
        
        for port in ports:
            cmd.extend(["-p", port])
        
        if working_dir:
            cmd.extend(["-w", working_dir])
        
        if user:
            cmd.extend(["-u", user])
        
        cmd.append(image)
        
        # Add command to execute inside container
        if isinstance(command, list):
            cmd.extend(command)
        else:
            # If command is string, use shell mode to execute
            cmd.extend(["sh", "-c", command])
        
        # Execute command
        result = run_command(cmd, timeout=timeout)
        
        if result["returncode"] == 0:
            send_response("success", {
                "message": "Command executed successfully in Docker container",
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "image": image,
                "command": command
            })
        else:
            send_response("error", {
                "message": f"Command failed in Docker container: {result.get('error', result['stderr'])}",
                "details": result,
                "image": image,
                "command": command
            })
    except Exception as e:
        send_response("error", {
            "message": f"Error running command in Docker container: {str(e)}",
            "image": image,
            "command": command
        })

def handle_build_docker_image(args: Dict[str, Any]) -> None:
    """Handle build Docker image request"""
    dockerfile_path = args.get("dockerfile_path")
    context_path = args.get("context_path")  # Docker build context path
    tag = args.get("tag")  # Image tag
    build_args = args.get("build_args", {})  # Build arguments
    no_cache = args.get("no_cache", False)
    pull = args.get("pull", False)  # Always try to pull newer base image
    timeout = args.get("timeout")
    
    if not dockerfile_path or not tag:
        send_response("error", {"message": "Missing required parameters: dockerfile_path and tag"})
        return
    
    if not os.path.exists(dockerfile_path):
        send_response("error", {"message": f"Dockerfile not found at {dockerfile_path}"})
        return
    
    if not context_path:
        # Default to use Dockerfile directory as context
        context_path = os.path.dirname(os.path.abspath(dockerfile_path))
    
    if not os.path.exists(context_path):
        send_response("error", {"message": f"Context path not found: {context_path}"})
        return
    
    try:
        # Build docker build command
        cmd = ["docker", "build"]
        
        cmd.extend(["-f", dockerfile_path])
        cmd.extend(["-t", tag])
        
        for key, value in build_args.items():
            cmd.extend(["--build-arg", f"{key}={value}"])
        
        if no_cache:
            cmd.append("--no-cache")
        
        if pull:
            cmd.append("--pull")
        
        cmd.append(context_path)
        
        # Execute command
        result = run_command(cmd, timeout=timeout)
        
        if result["returncode"] == 0:
            send_response("success", {
                "message": f"Docker image built successfully: {tag}",
                "stdout": result["stdout"],
                "stderr": result["stderr"],
                "tag": tag,
                "dockerfile_path": dockerfile_path,
                "context_path": context_path
            })
        else:
            send_response("error", {
                "message": f"Failed to build Docker image: {result.get('error', result['stderr'])}",
                "details": result,
                "tag": tag,
                "dockerfile_path": dockerfile_path,
                "context_path": context_path
            })
    except Exception as e:
        send_response("error", {
            "message": f"Error building Docker image: {str(e)}",
            "tag": tag,
            "dockerfile_path": dockerfile_path,
            "context_path": context_path
        })

def handle_get_system_info(args: Dict[str, Any]) -> None:
    """Handle get system information request"""
    try:
        import platform
        import sys
        
        # Get system information
        system_info = {
            "platform": platform.platform(),
            "system": platform.system(),
            "release": platform.release(),
            "version": platform.version(),
            "architecture": platform.architecture()[0],
            "machine": platform.machine(),
            "processor": platform.processor(),
            "python_version": platform.python_version(),
            "python_implementation": platform.python_implementation(),
            "python_path": sys.executable
        }
        
        # Get CPU information
        try:
            if platform.system() == "Linux":
                cpu_info_cmd = ["lscpu"]
                cpu_result = run_command(cpu_info_cmd)
                if cpu_result["returncode"] == 0:
                    system_info["cpu_info"] = cpu_result["stdout"]
            elif platform.system() == "Darwin":  # macOS
                cpu_info_cmd = ["sysctl", "-n", "machdep.cpu.brand_string"]
                cpu_result = run_command(cpu_info_cmd)
                if cpu_result["returncode"] == 0:
                    system_info["cpu_info"] = cpu_result["stdout"]
            elif platform.system() == "Windows":
                cpu_info_cmd = ["wmic", "cpu", "get", "name"]
                cpu_result = run_command(cpu_info_cmd)
                if cpu_result["returncode"] == 0:
                    system_info["cpu_info"] = cpu_result["stdout"].splitlines()[1].strip()
        except Exception:
            system_info["cpu_info"] = "Unknown"
        
        # Get memory information
        try:
            if platform.system() == "Linux":
                mem_info_cmd = ["free", "-h"]
                mem_result = run_command(mem_info_cmd)
                if mem_result["returncode"] == 0:
                    system_info["memory_info"] = mem_result["stdout"]
            elif platform.system() == "Darwin":  # macOS
                mem_info_cmd = ["sysctl", "-n", "hw.memsize"]
                mem_result = run_command(mem_info_cmd)
                if mem_result["returncode"] == 0:
                    try:
                        mem_bytes = int(mem_result["stdout"])
                        mem_gb = mem_bytes / (1024**3)
                        system_info["memory_info"] = f"Total Memory: {mem_gb:.2f} GB"
                    except Exception:
                        system_info["memory_info"] = mem_result["stdout"]
            elif platform.system() == "Windows":
                mem_info_cmd = ["wmic", "OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize"]
                mem_result = run_command(mem_info_cmd)
                if mem_result["returncode"] == 0:
                    system_info["memory_info"] = mem_result["stdout"]
        except Exception:
            system_info["memory_info"] = "Unknown"
        
        # Get Docker status
        try:
            docker_version_cmd = ["docker", "--version"]
            docker_result = run_command(docker_version_cmd)
            system_info["docker_available"] = docker_result["returncode"] == 0
            if system_info["docker_available"]:
                system_info["docker_version"] = docker_result["stdout"]
        except Exception:
            system_info["docker_available"] = False
        
        # Get Python package information
        try:
            pip_list_cmd = [sys.executable, "-m", "pip", "list"]
            pip_result = run_command(pip_list_cmd)
            if pip_result["returncode"] == 0:
                system_info["python_packages"] = pip_result["stdout"]
        except Exception:
            system_info["python_packages"] = "Unknown"
        
        send_response("success", system_info)
    except Exception as e:
        send_response("error", {
            "message": f"Error getting system information: {str(e)}"
        })

def main():
    """Main function - Handle requests from stdin"""
    handlers = {
        "create_venv": handle_create_venv,
        "run_in_venv": handle_run_in_venv,
        "install_packages": handle_install_packages,
        "check_docker": handle_check_docker,
        "run_in_docker": handle_run_in_docker,
        "build_docker_image": handle_build_docker_image,
        "get_system_info": handle_get_system_info
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