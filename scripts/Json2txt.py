import json, argparse, os, ipdb

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--file_path', type=str, default=None)
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--output_path', type=str, default=None)
    args = parser.parse_args()

    if args.exp_name in args.file_path:
        pass
    else:
        print("Inconsistent paths and experimental resultson")
    
    with open(args.file_path, 'r') as json_file:
        data = json.load(json_file)
    
    tasks = ['detection', 'selection', 'generation']
    for task in tasks:
        with open(args.output_path, 'a') as txt_output:
            content = '\t'.join([task] + list(data[task].keys()))
            txt_output.write(content + '\n')

            content = '\t'.join([args.exp_name] +  [str(format(value, '.4f')) for value in data[task].values()])
            txt_output.write(content + '\n' + '\n' )



if __name__ == "__main__":
    main()
