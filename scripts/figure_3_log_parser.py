import re
import sys
import json

def parse_log(logfile):
    data = {}
    current_algo = ""
    current_workload = ""

    with open(logfile, 'r') as f:
        for line in f:
            exec_match = re.search(r'Executing (\w+-Seq-CPU-Profile) with workload: (\w+)', line)
            if exec_match:
                current_algo = exec_match.group(1)
                current_workload = exec_match.group(2)
                if current_algo not in data:
                    data[current_algo] = {}
                data[current_algo][current_workload] = {}

            profiling_match = re.search(r'\w+-Seq-CPU-Profile (\w+(?: \w+)*) Time: ([\d.]+) ms', line)
            if profiling_match and current_algo and current_workload:
                operation = profiling_match.group(1)
                time = float(profiling_match.group(2))
                data[current_algo][current_workload][operation] = time

    return data

if __name__ == "__main__":
    logfile = sys.argv[1]
    parsed_data = parse_log(logfile)
    print(json.dumps(parsed_data, indent=2))
