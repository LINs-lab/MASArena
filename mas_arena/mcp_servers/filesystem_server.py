#!/usr/bin/env python3
import json
import sys
import os
import shutil
import glob
import traceback
from typing import Dict, Any

"""
Filesystem Operations Server - Communicates with client via stdio
Supported commands:
- read_file: Read file contents
- write_file: Write content to file
- append_file: Append content to file
- delete_file: Delete file
- copy_file: Copy file
- list_dir: List directory contents
- make_dir: Create directory
- delete_dir: Delete directory
- glob_files: Find files using glob pattern
- get_file_info: Get file information (size, modification time, etc.)
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

def handle_read_file(args: Dict[str, Any]) -> None:
    """Handle read file request"""
    file_path = args.get("file_path")
    encoding = args.get("encoding", "utf-8")
    offset = args.get("offset", 0)
    limit = args.get("limit", -1)  # -1 means read all
    binary_mode = args.get("binary_mode", False)
    
    if not file_path:
        send_response("error", {"message": "Missing required parameter: file_path"})
        return
    
    if not os.path.exists(file_path):
        send_response("error", {"message": f"File not found: {file_path}"})
        return
    
    if not os.path.isfile(file_path):
        send_response("error", {"message": f"Path is not a file: {file_path}"})
        return
    
    try:
        mode = "rb" if binary_mode else "r"
        encoding_arg = {} if binary_mode else {"encoding": encoding}
        
        with open(file_path, mode, **encoding_arg) as f:
            if offset > 0:
                if binary_mode:
                    f.seek(offset)
                else:
                    # In text mode, read line by line until offset is reached
                    for _ in range(offset):
                        f.readline()
            
            if binary_mode:
                if limit > 0:
                    content = f.read(limit)
                else:
                    content = f.read()
                # Binary data needs to be encoded as Base64
                import base64
                content = base64.b64encode(content).decode('ascii')
                is_base64 = True
            else:
                if limit > 0:
                    lines = []
                    for _ in range(limit):
                        line = f.readline()
                        if not line:
                            break
                        lines.append(line)
                    content = "".join(lines)
                else:
                    content = f.read()
                is_base64 = False
        
        file_size = os.path.getsize(file_path)
        send_response("success", {
            "content": content,
            "is_base64": is_base64,
            "file_size": file_size,
            "encoding": encoding if not binary_mode else None,
            "offset": offset,
            "limit": limit
        })
    except UnicodeDecodeError as e:
        send_response("error", {
            "message": f"Encoding error: {str(e)}. File may not be in the specified encoding ({encoding})."
        })
    except Exception as e:
        send_response("error", {"message": f"Error reading file: {str(e)}"})

def handle_write_file(args: Dict[str, Any]) -> None:
    """Handle write file request"""
    file_path = args.get("file_path")
    content = args.get("content")
    encoding = args.get("encoding", "utf-8")
    binary_mode = args.get("binary_mode", False)
    is_base64 = args.get("is_base64", False)
    
    if not file_path or content is None:
        send_response("error", {"message": "Missing required parameters: file_path and content"})
        return
    
    # Ensure target directory exists
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    try:
        mode = "wb" if binary_mode else "w"
        encoding_arg = {} if binary_mode else {"encoding": encoding}
        
        # If it's Base64 encoded binary data, decode first
        if is_base64:
            import base64
            decoded_content = base64.b64decode(content)
        else:
            decoded_content = content
        
        with open(file_path, mode, **encoding_arg) as f:
            if binary_mode or is_base64:
                f.write(decoded_content)
            else:
                f.write(content)
        
        send_response("success", {
            "message": f"File written successfully: {file_path}",
            "bytes_written": len(decoded_content) if is_base64 else len(content.encode(encoding) if not binary_mode else content)
        })
    except Exception as e:
        send_response("error", {"message": f"Error writing file: {str(e)}"})

def handle_append_file(args: Dict[str, Any]) -> None:
    """Handle append to file request"""
    file_path = args.get("file_path")
    content = args.get("content")
    encoding = args.get("encoding", "utf-8")
    binary_mode = args.get("binary_mode", False)
    is_base64 = args.get("is_base64", False)
    
    if not file_path or content is None:
        send_response("error", {"message": "Missing required parameters: file_path and content"})
        return
    
    # Ensure target directory exists
    os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
    
    try:
        mode = "ab" if binary_mode else "a"
        encoding_arg = {} if binary_mode else {"encoding": encoding}
        
        # If it's Base64 encoded binary data, decode first
        if is_base64:
            import base64
            decoded_content = base64.b64decode(content)
        else:
            decoded_content = content
        
        with open(file_path, mode, **encoding_arg) as f:
            if binary_mode or is_base64:
                f.write(decoded_content)
            else:
                f.write(content)
        
        send_response("success", {
            "message": f"Content appended to file: {file_path}",
            "bytes_written": len(decoded_content) if is_base64 else len(content.encode(encoding) if not binary_mode else content)
        })
    except Exception as e:
        send_response("error", {"message": f"Error appending to file: {str(e)}"})

def handle_delete_file(args: Dict[str, Any]) -> None:
    """Handle delete file request"""
    file_path = args.get("file_path")
    
    if not file_path:
        send_response("error", {"message": "Missing required parameter: file_path"})
        return
    
    if not os.path.exists(file_path):
        send_response("success", {"message": f"File does not exist: {file_path}", "already_deleted": True})
        return
    
    if not os.path.isfile(file_path):
        send_response("error", {"message": f"Path is not a file: {file_path}"})
        return
    
    try:
        os.remove(file_path)
        send_response("success", {"message": f"File deleted: {file_path}"})
    except Exception as e:
        send_response("error", {"message": f"Error deleting file: {str(e)}"})

def handle_copy_file(args: Dict[str, Any]) -> None:
    """Handle copy file request"""
    source_path = args.get("source_path")
    dest_path = args.get("dest_path")
    overwrite = args.get("overwrite", False)
    
    if not source_path or not dest_path:
        send_response("error", {"message": "Missing required parameters: source_path and dest_path"})
        return
    
    if not os.path.exists(source_path):
        send_response("error", {"message": f"Source file not found: {source_path}"})
        return
    
    if not os.path.isfile(source_path):
        send_response("error", {"message": f"Source path is not a file: {source_path}"})
        return
    
    if os.path.exists(dest_path) and not overwrite:
        send_response("error", {"message": f"Destination file already exists: {dest_path}. Use overwrite=true to replace it."})
        return
    
    try:
        # Ensure target directory exists
        os.makedirs(os.path.dirname(os.path.abspath(dest_path)), exist_ok=True)
        
        shutil.copy2(source_path, dest_path)
        send_response("success", {
            "message": f"File copied from {source_path} to {dest_path}",
            "source_size": os.path.getsize(source_path),
            "dest_size": os.path.getsize(dest_path)
        })
    except Exception as e:
        send_response("error", {"message": f"Error copying file: {str(e)}"})

def handle_list_dir(args: Dict[str, Any]) -> None:
    """Handle list directory contents request"""
    dir_path = args.get("dir_path")
    include_hidden = args.get("include_hidden", False)
    recursive = args.get("recursive", False)
    
    if not dir_path:
        send_response("error", {"message": "Missing required parameter: dir_path"})
        return
    
    if not os.path.exists(dir_path):
        send_response("error", {"message": f"Directory not found: {dir_path}"})
        return
    
    if not os.path.isdir(dir_path):
        send_response("error", {"message": f"Path is not a directory: {dir_path}"})
        return
    
    try:
        result = []
        
        if recursive:
            for root, dirs, files in os.walk(dir_path):
                # Skip hidden directories
                if not include_hidden:
                    dirs[:] = [d for d in dirs if not d.startswith('.')]
                
                for file in files:
                    if include_hidden or not file.startswith('.'):
                        full_path = os.path.join(root, file)
                        rel_path = os.path.relpath(full_path, dir_path)
                        result.append({
                            "name": file,
                            "path": rel_path,
                            "type": "file",
                            "size": os.path.getsize(full_path),
                            "modified": os.path.getmtime(full_path)
                        })
                
                for dir_name in dirs:
                    full_path = os.path.join(root, dir_name)
                    rel_path = os.path.relpath(full_path, dir_path)
                    result.append({
                        "name": dir_name,
                        "path": rel_path,
                        "type": "directory",
                        "modified": os.path.getmtime(full_path)
                    })
        else:
            items = os.listdir(dir_path)
            for item in items:
                if include_hidden or not item.startswith('.'):
                    full_path = os.path.join(dir_path, item)
                    item_info = {
                        "name": item,
                        "path": item,
                        "type": "directory" if os.path.isdir(full_path) else "file",
                        "modified": os.path.getmtime(full_path)
                    }
                    if item_info["type"] == "file":
                        item_info["size"] = os.path.getsize(full_path)
                    result.append(item_info)
        
        send_response("success", {
            "items": result,
            "count": len(result),
            "dir_path": dir_path
        })
    except Exception as e:
        send_response("error", {"message": f"Error listing directory: {str(e)}"})

def handle_make_dir(args: Dict[str, Any]) -> None:
    """Handle create directory request"""
    dir_path = args.get("dir_path")
    create_parents = args.get("create_parents", True)
    
    if not dir_path:
        send_response("error", {"message": "Missing required parameter: dir_path"})
        return
    
    if os.path.exists(dir_path):
        if os.path.isdir(dir_path):
            send_response("success", {"message": f"Directory already exists: {dir_path}", "already_exists": True})
        else:
            send_response("error", {"message": f"Path exists but is not a directory: {dir_path}"})
        return
    
    try:
        if create_parents:
            os.makedirs(dir_path)
        else:
            os.mkdir(dir_path)
        send_response("success", {"message": f"Directory created: {dir_path}"})
    except Exception as e:
        send_response("error", {"message": f"Error creating directory: {str(e)}"})

def handle_delete_dir(args: Dict[str, Any]) -> None:
    """Handle delete directory request"""
    dir_path = args.get("dir_path")
    recursive = args.get("recursive", False)
    
    if not dir_path:
        send_response("error", {"message": "Missing required parameter: dir_path"})
        return
    
    if not os.path.exists(dir_path):
        send_response("success", {"message": f"Directory does not exist: {dir_path}", "already_deleted": True})
        return
    
    if not os.path.isdir(dir_path):
        send_response("error", {"message": f"Path is not a directory: {dir_path}"})
        return
    
    try:
        if recursive:
            shutil.rmtree(dir_path)
        else:
            os.rmdir(dir_path)
        send_response("success", {"message": f"Directory deleted: {dir_path}"})
    except OSError as e:
        if "Directory not empty" in str(e):
            send_response("error", {"message": f"Directory not empty: {dir_path}. Use recursive=true to delete non-empty directories."})
        else:
            send_response("error", {"message": f"Error deleting directory: {str(e)}"})
    except Exception as e:
        send_response("error", {"message": f"Error deleting directory: {str(e)}"})

def handle_glob_files(args: Dict[str, Any]) -> None:
    """Handle glob files request"""
    pattern = args.get("pattern")
    root_dir = args.get("root_dir", ".")
    recursive = args.get("recursive", True)
    include_dirs = args.get("include_dirs", True)
    
    if not pattern:
        send_response("error", {"message": "Missing required parameter: pattern"})
        return
    
    try:
        if recursive:
            glob_pattern = os.path.join(root_dir, "**", pattern)
            matched_files = glob.glob(glob_pattern, recursive=True)
        else:
            glob_pattern = os.path.join(root_dir, pattern)
            matched_files = glob.glob(glob_pattern)
        
        result = []
        for path in matched_files:
            is_dir = os.path.isdir(path)
            if is_dir and not include_dirs:
                continue
            
            file_info = {
                "path": path,
                "name": os.path.basename(path),
                "type": "directory" if is_dir else "file",
                "modified": os.path.getmtime(path)
            }
            
            if not is_dir:
                file_info["size"] = os.path.getsize(path)
            
            result.append(file_info)
        
        send_response("success", {
            "matches": result,
            "count": len(result),
            "pattern": glob_pattern
        })
    except Exception as e:
        send_response("error", {"message": f"Error globbing files: {str(e)}"})

def handle_get_file_info(args: Dict[str, Any]) -> None:
    """Handle get file info request"""
    file_path = args.get("file_path")
    
    if not file_path:
        send_response("error", {"message": "Missing required parameter: file_path"})
        return
    
    if not os.path.exists(file_path):
        send_response("error", {"message": f"File or directory not found: {file_path}"})
        return
    
    try:
        stat_info = os.stat(file_path)
        result = {
            "path": file_path,
            "name": os.path.basename(file_path),
            "exists": True,
            "type": "directory" if os.path.isdir(file_path) else "file",
            "size": stat_info.st_size,
            "created": stat_info.st_ctime,
            "modified": stat_info.st_mtime,
            "accessed": stat_info.st_atime,
            "mode": stat_info.st_mode
        }
        
        # Check read, write, execute permissions
        result["readable"] = os.access(file_path, os.R_OK)
        result["writable"] = os.access(file_path, os.W_OK)
        result["executable"] = os.access(file_path, os.X_OK)
        
        send_response("success", result)
    except Exception as e:
        send_response("error", {"message": f"Error getting file info: {str(e)}"})

def main():
    """Main function - Handle requests from stdin"""
    handlers = {
        "read_file": handle_read_file,
        "write_file": handle_write_file,
        "append_file": handle_append_file,
        "delete_file": handle_delete_file,
        "copy_file": handle_copy_file,
        "list_dir": handle_list_dir,
        "make_dir": handle_make_dir,
        "delete_dir": handle_delete_dir,
        "glob_files": handle_glob_files,
        "get_file_info": handle_get_file_info
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