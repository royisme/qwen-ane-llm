import subprocess
import json
import os
import re
from typing import List, Dict, Any, Tuple


class ANEAdapter:
    def __init__(self, binary_path: str, model_dir: str):
        self.binary_path = binary_path
        self.model_dir = model_dir

    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.6, max_tokens: int = 0) -> Tuple[str, Dict[str, Any]]:
        cmd = [
            self.binary_path,
            "generate",
            "--model", self.model_dir,
            "--json-messages", json.dumps(messages),
            "--temp", str(temperature),
        ]
        if max_tokens > 0:
            cmd.extend(["--max-tokens", str(max_tokens)])

        # Run process
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        try:
            stdout, stderr = process.communicate(timeout=300)
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = process.communicate()
            raise TimeoutError("qwen-ane-llm generation timed out after 300 seconds")

        if process.returncode != 0:
            raise Exception(f"qwen-ane-llm failed with exit code {process.returncode}: {stderr}")

        # Parse generated text and stats
        # The generated text is before the stats
        # Stats are in format: JSON_STATS: {...}
        lines = stdout.splitlines()
        generated_text_lines = []
        stats = {}

        for line in lines:
            if line.startswith("JSON_STATS: "):
                try:
                    stats = json.loads(line[len("JSON_STATS: "):])
                except json.JSONDecodeError:
                    pass
            else:
                generated_text_lines.append(line)

        # Filter out the "==========" markers if they appeared in stdout
        text = "\n".join(generated_text_lines).strip()
        text = re.sub(r"^==========\n?", "", text, flags=re.MULTILINE)
        text = re.sub(r"\n?==========$", "", text, flags=re.MULTILINE)
        text = text.strip()

        return text, stats
