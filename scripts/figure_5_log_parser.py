import re
import sys
import json

def parse_log(logfile):
    data = {}
    current_task = None
    size = None
    pattern_exec = re.compile(r'Executing SmpInitEngine(\S+) with workload: Solo, size: (\d+)')
    pattern_init = re.compile(r'Initialize (\w+) Profiling Info:')
    pattern_time = re.compile(r'(Kernel|Host-Device Transfer|Total Init|Init) Time: ([\d.]+) ms')

    with open(logfile, 'r') as f:
        for line in f:
            exec_match = pattern_exec.search(line)
            if exec_match:
                engine, size = exec_match.group(1), int(exec_match.group(2))
                current_task = f"{engine}_{size}"
                if current_task not in data:
                    data[current_task] = {'engine': engine, 'size': size}

            init_match = pattern_init.search(line)
            if init_match and current_task:
                init_type = init_match.group(1)
                data[current_task]['init_type'] = init_type

            time_match = pattern_time.search(line)
            if time_match and current_task:
                time_label, time_value = time_match.group(1), float(time_match.group(2))
                data[current_task][time_label] = time_value

    return data

if __name__ == "__main__":
    logfile = sys.argv[1]
    parsed_data = parse_log(logfile)
    print(json.dumps(parsed_data, indent=2))