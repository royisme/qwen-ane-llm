import datetime
import subprocess

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

AVAILABLE_TOOLS = {
    "run_date": run_date,
    "run_shell": run_shell
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
    }
]
