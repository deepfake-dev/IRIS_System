import os
import sys
import ast
from os.path import isfile, join
from metadata_scanner import analyze_media, Decision
from vlm_classifier import llm_classify_video
from deepfake_detector import DeepfakeDetector

def process_file(file):
    print("Current File: ", file)

    result = analyze_media(file)

    if (result.isAIGenerated == Decision.YES):
        print("Verdict:", result.reason)
    elif (result.isAIGenerated == Decision.MAYBE):
        print(f"{result.reason}")
        verdict = llm_classify_video(file, result.reason)
        verdict, reason = verdict

        if verdict in ['Animated', 'Recording', 'Generated']:
            print("Verdict:", verdict)
        else:
            print(f"LLM Can't Decide [{verdict} | {reason}]. Let's try our Classifier")
            detector = DeepfakeDetector(onnx_path="models/provenance/deepfake_detector_model.onnx")
            final_verdict = detector.predict(file)
            print(f"Overall Verdict:        {'🛑 FAKE' if final_verdict['is_fake'] else '✅ REAL'}")
            print(f"Highest Fake Spike:     {final_verdict['max_confidence']:.4f} (Used for final verdict)")
            print(f"Average AI Confidence:  {final_verdict['average_confidence']:.4f} (For reference only)")

def process_dir(folder_path):
    folder_path = "C:/Users/owen/Desktop/thesis/vids_to_test"
    print("STARTING ANALYSIS!!!")
    for f in os.listdir(folder_path):
        if isfile(fil:=join(folder_path, f)):
            file = fil.replace("\\", "/")
            process_file(file)
            print("----------------------------------------\n\n")

def main():
    arg = None
    if len(sys.argv) == 2:
        arg = sys.argv[1]
    
    if not arg:
        print("Please specify a file or folder to process.")
        return 
    
    if not os.path.exists(arg):
        print("The specified path doesn't exist, or is not a file.")
        return

    if os.path.isdir(arg):
        process_dir(arg)
    elif os.path.isfile(arg):
        process_file(arg)
    else:
        print("File doesn't exist.")

if __name__ == '__main__':
    main()