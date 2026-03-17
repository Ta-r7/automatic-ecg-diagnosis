# Filename: model_inspection.py
# pip install tensorflow scikit-learn scipy numpy
# Run this FIRST before any other script.
# It inspects the pretrained Ribeiro et al. model and prints all layer details.

import sys
import io


def check_python_version():
    version = sys.version
    major, minor = sys.version_info.major, sys.version_info.minor
    print(f"Python version: {version}")
    if major < 3:
        print("WARNING: Python 2.x detected. Python 3.x is required.")
        return "WARNING"
    print("Python version OK.")
    return "OK"


def check_tensorflow():
    try:
        import tensorflow as tf
        version = tf.__version__
        print(f"TensorFlow version: {version}")
        major, minor = int(version.split('.')[0]), int(version.split('.')[1])
        if major < 2 or (major == 2 and minor < 4):
            print("=" * 60)
            print("WARNING: TensorFlow version is below 2.4.")
            print("Compatibility issues may occur when loading the model.")
            print("Consider: pip install tensorflow>=2.4")
            print("=" * 60)
            return "WARNING"
        print("TensorFlow version OK.")
        return "OK"
    except ImportError:
        print("ERROR: TensorFlow is not installed.")
        print("Install with: pip install tensorflow")
        return "MISSING"


def check_packages():
    packages = {
        'sklearn': 'scikit-learn',
        'scipy': 'scipy',
        'numpy': 'numpy',
    }
    all_ok = True
    for import_name, pip_name in packages.items():
        try:
            mod = __import__(import_name)
            version = getattr(mod, '__version__', 'unknown')
            print(f"{pip_name} version: {version} - OK")
        except ImportError:
            print(f"ERROR: {pip_name} is not installed.")
            print(f"Install with: pip install {pip_name}")
            all_ok = False
    return "OK" if all_ok else "MISSING"


def load_and_inspect_model():
    import tensorflow as tf

    model_path = 'model.hdf5'
    print(f"\nLoading model from: {model_path}")
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        print("Model loaded successfully.\n")
    except Exception as e:
        print(f"ERROR: Failed to load model: {e}")
        print("Likely causes:")
        print("  - Wrong TensorFlow version (model was saved with TF 2.2)")
        print("  - Corrupt or incomplete model.hdf5 file")
        print("  - Missing custom objects (try compile=False)")
        return "FAILED"

    # Print model summary to console and capture it for file
    print("=" * 60)
    print("MODEL SUMMARY")
    print("=" * 60)
    model.summary()

    # Capture summary to string for saving to file
    summary_buffer = io.StringIO()
    model.summary(print_fn=lambda x: summary_buffer.write(x + '\n'))
    summary_text = summary_buffer.getvalue()

    # Print every layer with index, name, type, output shape
    print("\n" + "=" * 60)
    print("DETAILED LAYER LIST")
    print("=" * 60)
    detailed_lines = []
    for i, layer in enumerate(model.layers):
        layer_type = type(layer).__name__
        try:
            output_shape = layer.output_shape
        except AttributeError:
            output_shape = "unknown"
        line = f"[{i:3d}] {layer.name:40s} | {layer_type:25s} | {output_shape}"
        print(line)
        detailed_lines.append(line)

    total_layers = len(model.layers)

    # Input layer
    input_layer = model.layers[0]
    print(f"\n{'=' * 60}")
    print("KEY LAYER INFORMATION")
    print(f"{'=' * 60}")
    print(f"Input layer name:                   {input_layer.name}")
    print(f"Input layer index:                  0")

    # Final output layer
    output_layer = model.layers[-1]
    output_index = total_layers - 1
    print(f"Final output layer name:            {output_layer.name}")
    print(f"Final output layer index:           {output_index}")
    print(f"Final output layer type:            {type(output_layer).__name__}")

    # Layer just before final output
    before_output_layer = model.layers[-2]
    before_output_index = total_layers - 2
    print(f"Layer before output name:           {before_output_layer.name}")
    print(f"Layer before output index:          {before_output_index}")
    print(f"Layer before output type:           {type(before_output_layer).__name__}")

    print(f"Total number of layers:             {total_layers}")

    # Save full summary to file
    with open('model_summary.txt', 'w', encoding='utf-8') as f:
        f.write("MODEL SUMMARY\n")
        f.write("=" * 60 + "\n")
        f.write(summary_text)
        f.write("\n\nDETAILED LAYER LIST\n")
        f.write("=" * 60 + "\n")
        for line in detailed_lines:
            f.write(line + "\n")
        f.write(f"\nInput layer name:                   {input_layer.name}\n")
        f.write(f"Final output layer name:            {output_layer.name}\n")
        f.write(f"Final output layer index:           {output_index}\n")
        f.write(f"Layer before output name:           {before_output_layer.name}\n")
        f.write(f"Layer before output index:          {before_output_index}\n")
        f.write(f"Total number of layers:             {total_layers}\n")

    print(f"\nFull summary saved to: model_summary.txt")
    return "OK"


def main():
    print("=" * 60)
    print("ECG MODEL INSPECTION SCRIPT")
    print("=" * 60)

    # Step 1: Python version
    print("\n--- Python Version Check ---")
    python_status = check_python_version()

    # Step 2: TensorFlow version
    print("\n--- TensorFlow Version Check ---")
    tf_status = check_tensorflow()

    # Step 3: Other packages
    print("\n--- Package Checks ---")
    pkg_status = check_packages()

    # Step 4: Load and inspect model
    if tf_status != "MISSING":
        print("\n--- Model Inspection ---")
        model_status = load_and_inspect_model()
    else:
        model_status = "FAILED"
        print("\nSkipping model inspection because TensorFlow is not available.")

    # Final checklist
    print("\n" + "=" * 60)
    print("CHECKLIST SUMMARY")
    print("=" * 60)
    print(f"  Python version:    {python_status}")
    print(f"  TensorFlow:        {tf_status}")
    print(f"  All packages:      {pkg_status}")
    print(f"  Model loaded:      {model_status}")
    print("=" * 60)

    if all(s == "OK" for s in [python_status, tf_status, pkg_status, model_status]):
        print("All checks passed. Ready to proceed.")
    else:
        print("Some checks failed or have warnings. Review above output.")


if __name__ == '__main__':
    main()
