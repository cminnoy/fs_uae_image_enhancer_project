import argparse
import onnx

def main() -> None:
    parser = argparse.ArgumentParser(description="Print an ONNX model graph.")
    parser.add_argument("model", nargs="?", default="model.onnx", help="Path to the ONNX model file")
    args = parser.parse_args()

    model = onnx.load_model(args.model)
    print(onnx.helper.printable_graph(model.graph))

if __name__ == "__main__":
    main()

