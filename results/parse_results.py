import csv
import re
import os

def parse_file(file_path):
    with open(file_path, 'r') as f:
        content = f.read()

    executions = []
    # Split by execution (1, 2, 3, 4, 5 at the beginning of lines)
    # Each block starts with a number on a line by itself.
    blocks = re.split(r'\n(?=\d+\n)', '\n' + content)
    
    for block in blocks:
        block = block.strip()
        if not block:
            continue
            
        lines = block.split('\n')
        execution_id_str = lines[0].strip()
        if not execution_id_str.isdigit():
            continue
        execution_id = int(execution_id_str)
        # print(f"Processing Execution {execution_id}")
        
        train_data = {}
        eval_data = {}

        # Use a more global search for rounds and values
        # Find train losses
        # Pattern: \d+: {'train_loss': '3.8216e+00'}
        train_entries = re.findall(r"(\d+):\s*\{\s*'train_loss':\s*'([^']+)'\s*\}", block)
        for round_id, loss in train_entries:
            train_data[int(round_id)] = float(loss)
            
        # Find evaluate metrics
        # Pattern: \d+: {'eval_acc': '4.7236e-01', 'eval_loss': '1.3222e+00'}
        # This needs to be careful because both acc and loss are present
        eval_entries = re.findall(r"(\d+):\s*\{\s*'eval_acc':\s*'([^']+)'.*?'eval_loss':\s*'([^']+)'\s*\}", block)
        for round_id, acc, loss in eval_entries:
            eval_data[int(round_id)] = (float(acc), float(loss))
            
        for round_id in range(1, 11):
            train_loss = train_data.get(round_id, None)
            eval_acc, eval_loss = eval_data.get(round_id, (None, None))
            executions.append({
                'execution': execution_id,
                'round': round_id,
                'train_loss': train_loss,
                'eval_loss': eval_loss,
                'eval_acc': eval_acc
            })
            
    return executions

def save_to_csv(data, output_path):
    if not data:
        print(f"No data found for {output_path}")
        return
    keys = data[0].keys()
    with open(output_path, 'w', newline='') as f:
        dict_writer = csv.DictWriter(f, fieldnames=keys)
        dict_writer.writeheader()
        dict_writer.writerows(data)
    print(f"Saved to {output_path}")

if __name__ == "__main__":
    results_dir = 'results'
    # Find all raw-*.txt files in the results directory
    files_to_parse = [f for f in os.listdir(results_dir) if f.startswith('raw-') and f.endswith('.txt')]
    
    for filename in files_to_parse:
        input_path = os.path.join(results_dir, filename)
        output_filename = filename.replace('raw-', '').replace('.txt', '.csv')
        output_path = os.path.join(results_dir, output_filename)
        
        if os.path.exists(input_path):
            data = parse_file(input_path)
            save_to_csv(data, output_path)
        else:
            print(f"File not found: {input_path}")
