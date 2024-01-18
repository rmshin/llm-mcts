import re
import os
import pandas as pd
from vcdvcd import VCDVCD


def process_vcd(wave_vcd_file: str) -> pd.DataFrame:
    """
    Process a VCD file to extract error messages and generate a summary CSV file.

    Args:
    wave_vcd_file (str): Path to the VCD file to be processed.

    Returns:
    DataFrame: A Pandas DataFrame containing the summarized error messages.
    """
    # Extract directory from the VCD file path
    directory = os.path.dirname(wave_vcd_file)

    # Convert VCD to text format and read it
    text_file_path = os.path.join(directory, "wave.txt")
    os.system(f"vcdcat {wave_vcd_file} > {text_file_path}")

    # Process the VCD file using vcdvcd
    vcd_data = VCDVCD(wave_vcd_file)
    stimclk = vcd_data["tb.stim1.clk"]
    # print(f"Aliases to stim clk are {stimclk.references}")

    # Read and process the text file for DataFrame
    with open(text_file_path, "r") as f:
        data = f.read()
        lines = data.strip().split("\n")

    # Extract headers
    headers = []
    for line in lines:
        if not line.strip():
            break
        elif line[0].isdigit():
            headers.append(line.split()[-1].replace("tb.", ""))

    # Extract data rows
    rows = []
    while lines:
        line = lines.pop(0)
        if line.startswith("="):
            break

    while lines:
        line = lines.pop(0)
        if line[0].isdigit():
            rows.append(line.split())

    df = pd.DataFrame(rows, columns=headers)

    # Duplicate columns for each reference of stim1.clk
    ignored_signals = [
        i.replace("tb.", "") for i in stimclk.references if i != "tb.stim1.clk"
    ]
    for ref in ignored_signals:
        if ref not in df.columns:
            df[ref] = df["stim1.clk"]

    csv_file_path = os.path.join(directory, "wave.csv")
    df.to_csv(csv_file_path, index=False)

    # Process DataFrame for error messages
    df = df[df["tb_mismatch"] == "1"]
    df = df.drop("tb_mismatch", axis=1)
    df = df.drop("stim1.clk", axis=1)
    df = df.drop_duplicates()

    return df


def extract_signals(prompt: str) -> tuple[list[str], list[str]]:
    # Separate the prompt into lines and remove comments
    lines = prompt.split("\n")
    lines = [line.split("//")[0] for line in lines]  # Remove comments
    prompt_no_comments = " ".join(lines)

    # Regular expression patterns for inputs and outputs
    # Updated to handle 'reg', 'logic', and 'wire' types
    type_pattern = r"(?:reg|logic|wire)?\s+"
    input_pattern = rf"input\s+{type_pattern}(?:\[(\d+:\d+)\]\s+)?(\w+)"
    output_pattern = rf"output\s+{type_pattern}(?:\[(\d+:\d+)\]\s+)?(\w+)"

    # Extracting signals using regular expressions
    input_signals = re.findall(input_pattern, prompt_no_comments)
    output_signals = re.findall(output_pattern, prompt_no_comments)

    # Formatting the signals with or without bit-width
    input_signals = [
        f"{signal}[{bits}]" if bits else signal for bits, signal in input_signals
    ]
    output_signals = [
        f"{signal}[{bits}]" if bits else signal for bits, signal in output_signals
    ]

    return input_signals, output_signals


def extract_mismatch_messages(
    df: pd.DataFrame, input_signals: list[str], output_signals: list[str]
) -> list[str]:
    """
    Extracts and constructs error messages for each mismatch found between the device under test
    outputs (`_dut`) and the reference outputs (`_ref`) in the provided DataFrame.

    The function sorts the output to display mismatches ('[mismatch]') first, followed by
    correct matches ('[ok]'), for each set of input signals.

    Parameters:
    df (pandas.DataFrame): DataFrame containing simulation data.
    input_signals (list of str): Names of input signals.
    output_signals (list of str): Names of output signals.

    Returns:
    list of str: Constructed error messages for each mismatch.

    Example usage:
    error_messages = extract_mismatch_messages(df_err, input_signals, output_signals)
    for message in error_messages:
        print(message)
    """
    output_signals_vec = [sig.split("[") for sig in output_signals]
    all_error_messages = []

    for index, row in df.iterrows():
        mismatches = []
        oks = []
        for signal_parts in output_signals_vec:
            sig = signal_parts[0]
            idx = signal_parts[1].rstrip("]") if len(signal_parts) > 1 else None
            expected = row[f"{sig}_ref[{idx}]"] if idx else row[f"{sig}_ref"]
            actual = row[f"{sig}_dut[{idx}]"] if idx else row[f"{sig}_dut"]

            if expected != actual:
                mismatches.append(
                    f"[mismatch] {sig} - expected = {expected}, got = {actual}"
                )
            else:
                oks.append(f"[ok] {sig} = {expected}")

        if mismatches or oks:
            ip_sig_values = ", ".join(f"{sig} = {row[sig]}" for sig in input_signals)
            combined_messages = mismatches + oks
            all_error_messages.append(
                f"For inputs: {ip_sig_values}\n    "
                + "\n    ".join(combined_messages)
                + "\n"
            )

    return all_error_messages
