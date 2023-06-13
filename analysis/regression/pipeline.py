from train import train
from argparse import ArgumentParser


def save_log(file, str):
    pipeline_log = open(file, "a")
    pipeline_log.write(str)
    pipeline_log.close()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--model_name")
    parser.add_argument("--size")
    parser.add_argument("--cropped")
    parser.add_argument("--device")
    parser.add_argument("--labels")
    args = parser.parse_args()

    for i in range(1):
        args.model_name, args.size, args.cropped, args.device, args.labels = "resnet18", "512", False, 0, "wind"
        train_log = train(args)
        save_log("pipeline_logs.txt", "training session " + str(i*3) + " : " + str(args) + " " + train_log + "\n")

        args.model_name, args.size, args.cropped, args.device, args.labels = "resnet50", "224", "False", 0, "wind"
        train_log = train(args)
        save_log("pipeline_logs.txt", "training session " + str(i*3 +1) + " : " + str(args) + " " + train_log + "\n")

        args.model_name, args.size, args.cropped, args.device, args.labels = "resnet18", "224", "True", 0, "wind"
        train_log = train(args)
        save_log("pipeline_logs.txt", "training session " + str(i*3 +2) + " : " + str(args) + " " + train_log + "\n")
