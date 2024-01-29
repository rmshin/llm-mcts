from datetime import datetime
import os, shutil
from typing import Optional, Dict, Any
import subprocess
import re

from utils import process_vcd
import pydantic
from enum import Enum
import outlines


class VerilogStatus(Enum):
    SUCCESS = "success"
    SYNTAX_ERROR = "syntax_error"
    COMPILE_ERROR = "compile_error"
    RUNTIME_ERROR = "runtime_error"
    TIMEOUT = "timeout"
    OTHER = "other"


class VerilogExecution(pydantic.BaseModel):
    """
    A class to represent the execution of a verilog program.
    """

    status: VerilogStatus
    stdout: str
    stderr: str
    passed: bool
    pass_rate: float
    result: Optional[Dict]
    df_err: Optional[Any]
    code: str


@outlines.vectorize
def evaluate_code(task_id, completion, problem) -> VerilogExecution:
    """
    Evaluate the code for a given task_id and completion.
    problem is a dict with many keys including "prompt", "description", "task_id", etc..
    """
    verilog_test = problem["test"] + "\n" + problem["prompt"] + "\n" + completion

    # Create a timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    directory = f"/tmp/{task_id}_{timestamp}"

    # Create the directory
    if not os.path.exists(directory):
        os.makedirs(directory)

    # Write the Verilog test file in the new directory
    file_path = os.path.join(directory, f"{task_id}.sv")
    with open(file_path, "w") as f:
        f.write(verilog_test)

    # Prepare the compilation and simulation commands
    compile_cmd = f"iverilog -Wall -Winfloop -Wno-timescale -g2012 -s tb -o {task_id}.vvp {task_id}.sv"
    simulation_cmd = f"vvp -n {task_id}.vvp"
    # HACK: use trap to terminate hanging vvp process upon python subprocess timeout,
    # otherwise simply killing the subprocess won't cleanup the child and overload cpu
    full_cmd = f"trap 'kill -TERM $child_pid' SIGINT; cd {directory} && {compile_cmd}; {simulation_cmd} & child_pid=$!; wait $child_pid"

    try:
        # Execute the commands and capture the output
        process = subprocess.Popen(
            full_cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        try:
            stdout, stderr = process.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            process.send_signal(2)
            stdout, stderr = process.communicate()
            status = VerilogStatus.TIMEOUT

        # Decode the outputs from byte to string
        stdout = stdout.decode("utf-8")
        stderr = stderr.decode("utf-8")

        match = re.search(r"Mismatches: ([0-9]*) in ([0-9]*) samples", stdout)
        if "syntax error" in stderr:
            status = VerilogStatus.SYNTAX_ERROR
        elif len(stderr.strip()) > 0:
            status = VerilogStatus.COMPILE_ERROR
        elif match:
            incor, tot = [int(i) for i in match.groups()]
            if incor == 0:
                status = VerilogStatus.SUCCESS
            else:
                status = VerilogStatus.RUNTIME_ERROR
        elif status != VerilogStatus.TIMEOUT:
            raise ValueError("Unexpected error")

        if os.path.exists(f"{directory}/wave.vcd"):
            df_err = process_vcd(f"{directory}/wave.vcd")
        else:
            df_err = None

        if status == VerilogStatus.SUCCESS:
            pass_rate = 1.0
        elif status == VerilogStatus.RUNTIME_ERROR:
            pass_rate = 1.0 - incor / tot
        elif status == VerilogStatus.COMPILE_ERROR:
            pass_rate = -0.2
        elif status == VerilogStatus.SYNTAX_ERROR:
            pass_rate = -0.4
        else:
            pass_rate = 0.0

        out = VerilogExecution(
            status=status,
            stdout=stdout,
            stderr=stderr,
            passed=status == VerilogStatus.SUCCESS,
            pass_rate=pass_rate,
            result=None,
            df_err=df_err,
            code=completion,
        )

        return out
    except Exception as e:
        # Handle exceptions
        raise ValueError(f"Error while executing the code: {e}")
    finally:
        # Cleanup /tmp directory
        shutil.rmtree(directory)
