import argparse
import os
import json


from detect_video import detect

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Detect 3D Human Pose')
    parser.add_argument('--video', type=str, help='specify the path of input video')
    parser.add_argument('--output', type=str, default="")
    args = parser.parse_args()
    detect_result = detect(args.video)
    if not args.output:
        output_path = os.path.splitext(args.video)[0] + '.json'
    else:
        output_path = args.output
    if not os.path.exists(os.path.split(output_path)[0]):
        os.makedirs(os.path.split(output_path)[0])
    json.dump(detect_result, open(output_path, 'w'), indent=2)
    print(f"output path: {output_path}")
