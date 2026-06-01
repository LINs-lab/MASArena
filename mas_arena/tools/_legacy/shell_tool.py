# coding: utf-8
import subprocess
import threading
import queue
from typing import List

from langchain.tools import StructuredTool

from mas_arena.tools.base import ToolFactory

SHELL = "shell"

class ShellSession:
    def __init__(self):
        self.process = subprocess.Popen(
            ['/bin/bash'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        self.output_queue = queue.Queue()
        self.is_reading = True

        # Start a thread to read stdout/stderr without blocking
        self.stdout_thread = threading.Thread(target=self._read_output, args=(self.process.stdout,))
        self.stderr_thread = threading.Thread(target=self._read_output, args=(self.process.stderr,))
        self.stdout_thread.daemon = True
        self.stderr_thread.daemon = True
        self.stdout_thread.start()
        self.stderr_thread.start()

    def _read_output(self, pipe):
        while self.is_reading:
            try:
                line = pipe.readline()
                if line:
                    self.output_queue.put(line)
                else:
                    break
            except Exception:
                break

    def run(self, command: str, timeout: int = 5) -> str:
        """Execute a shell command in the persistent session."""
        if self.process.poll() is not None:
            return "Error: Shell session has been terminated."
            
        # Add a newline to execute the command
        self.process.stdin.write(command + '\n')
        self.process.stdin.flush()

        # Unique marker to signal end of output
        end_marker = "END_OF_COMMAND_OUTPUT"
        self.process.stdin.write(f'echo {end_marker}\n')
        self.process.stdin.flush()

        output_lines = []
        try:
            while True:
                line = self.output_queue.get(timeout=timeout)
                if end_marker in line:
                    break
                if line.strip() != f'echo {end_marker}':
                    output_lines.append(line)
        except queue.Empty:
            return "Error: Command timed out. No output received."
        
        return "".join(output_lines)

    def close(self):
        """Close the shell session."""
        self.is_reading = False
        if self.process.poll() is None:
            self.process.stdin.write('exit\n')
            self.process.stdin.flush()
            self.process.terminate()
            self.process.wait(timeout=2)
        
        self.stdout_thread.join(timeout=1)
        self.stderr_thread.join(timeout=1)

@ToolFactory.register(name=SHELL, desc="A tool for executing shell commands in a persistent session.")
class ShellTool:
    def __init__(self):
        self.session = ShellSession()

    def get_tools(self) -> List[StructuredTool]:
        return [
            StructuredTool.from_function(
                func=self.session.run,
                name="run_in_shell",
                description="Run a command in the persistent shell session.",
            )
        ]

    def __del__(self):
        if hasattr(self, 'session'):
            self.session.close() 