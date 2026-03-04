import datetime
import subprocess
import os
from typing import Optional

def run_date():
    """Return current system time"""
    return datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def run_shell(cmd: str):
    """Run a shell command (allowlisted)"""
    allowlist = {
        "date": ["date"],
        "uname -a": ["uname", "-a"],
        "uptime": ["uptime"],
    }
    if cmd not in allowlist:
        return f"Error: Command '{cmd}' not allowed."

    try:
        result = subprocess.check_output(allowlist[cmd], shell=False, text=True, stderr=subprocess.STDOUT)
        return result.strip()
    except subprocess.CalledProcessError as e:
        return f"Error executing command: {e.output}"

def read_file(path: str) -> str:
    """Read contents of a file"""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except FileNotFoundError:
        return f"Error: File not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error reading file: {str(e)}"

def write_file(path: str, content: str) -> str:
    """Write content to a file (creates or overwrites)"""
    try:
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Successfully wrote to {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error writing file: {str(e)}"

def list_dir(path: str = ".") -> str:
    """List files in a directory"""
    try:
        items = os.listdir(path)
        return "\n".join(items)
    except FileNotFoundError:
        return f"Error: Directory not found: {path}"
    except PermissionError:
        return f"Error: Permission denied: {path}"
    except Exception as e:
        return f"Error listing directory: {str(e)}"

AVAILABLE_TOOLS = {
    "run_date": run_date,
    "run_shell": run_shell,
    "read_file": read_file,
    "write_file": write_file,
    "list_dir": list_dir,
}

TOOL_DEFS = [
    {
        "type": "function",
        "function": {
            "name": "run_date",
            "description": "Return current system time",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_shell",
            "description": "Run an allowed shell command",
            "parameters": {
                "type": "object",
                "properties": {
                    "cmd": {
                        "type": "string",
                        "enum": ["date", "uname -a", "uptime"],
                        "description": "The command to run"
                    }
                },
                "required": ["cmd"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read contents of a file from the filesystem",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Full path to the file to read"
                    }
                },
                "required": ["path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "write_file",
            "description": "Write content to a file (creates new file or overwrites existing)",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Full path to the file to write"
                    },
                    "content": {
                        "type": "string",
                        "description": "Content to write to the file"
                    }
                },
                "required": ["path", "content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_dir",
            "description": "List files and directories in a folder",
            "parameters": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Directory path to list (default: current directory)"
                    }
                },
                "required": []
            }
        }
    }
]
