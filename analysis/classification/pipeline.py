from train import train
from argparse import ArgumentParser


def save_log(file, str):
    """Write brief logs for every training of the pipeline"""
    pipeline_log = open(file, "a")
    pipeline_log.write(str)
    pipeline_log.close()

if __name__ == "__main__":
    """Pipeline which directly call the train function of the train.py file"""
    parser = ArgumentParser()
    parser.add_argument("--model_name")
    parser.add_argument("--size")
    parser.add_argument("--cropped")
    parser.add_argument("--device")
    args = parser.parse_args()

    for i in range(5):
        for model in ["resnet18", "vgg", "vit"]:
            if model != "vit":
                args.model_name, args.size, args.cropped, args.device = model, "512", False, 0
                train_log = train(args)
                save_log("pipeline_logs.txt", "training session " + str(i*3) + " : " + str(args) + " " + train_log + "\n")

            args.model_name, args.size, args.cropped, args.device = model, "224", "False", 0
            train_log = train(args)
            save_log("pipeline_logs.txt", "training session " + str(i*3 +1) + " : " + str(args) + " " + train_log + "\n")

            args.model_name, args.size, args.cropped, args.device = model, "224", "True", 0
            train_log = train(args)
            save_log("pipeline_logs.txt", "training session " + str(i*3 +2) + " : " + str(args) + " " + train_log + "\n")
