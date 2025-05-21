import logging
from pathlib import Path
import numpy

# Attempt to import JAX and Matplotlib, but allow them to be optional
try:
    import jax.numpy as jnp
    JAX_AVAILABLE = True
except ImportError:
    jnp = None
    JAX_AVAILABLE = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.colors as mcolors
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    plt = None
    mcolors = None
    MATPLOTLIB_AVAILABLE = False

logger = logging.getLogger(__name__)

def plot_tensor_output(tensor_data, output_file_path: Path, title: str = "Tensor Output", verbose: bool = False):
    """
    Visualizes tensor data and saves it to a file.
    - 0D (scalar): Saves as text.
    - 1D: Saves as a line plot.
    - 2D: Saves as a heatmap.
    - >2D: Saves raw data to a text file and logs a message.

    Args:
        tensor_data: The tensor data (JAX array or NumPy array).
        output_file_path: The base Path object for the output file (e.g., Path("output/diagram_result")).
                          Extensions will be added automatically (_scalar.txt, _plot.png, etc.).
        title (str): A title for the plot or output.
        verbose (bool): If True, sets logger to DEBUG level for this function.
    """
    if verbose:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if JAX_AVAILABLE and isinstance(tensor_data, jnp.ndarray):
        logger.debug("Converting JAX array to NumPy array for visualization.")
        try:
            numpy_tensor = numpy.array(tensor_data)
        except Exception as e:
            logger.error(f"Failed to convert JAX array to NumPy array: {e}")
            # Try to save raw JAX array representation if conversion fails
            raw_data_path = output_file_path.with_suffix(".jax_raw.txt")
            try:
                with open(raw_data_path, 'w', encoding='utf-8') as f:
                    f.write(f"{title} (JAX Array - conversion failed):\n{tensor_data!r}")
                logger.info(f"Saved raw JAX array representation to {raw_data_path}")
            except Exception as save_e:
                logger.error(f"Could not even save raw JAX representation: {save_e}")
            return
    elif isinstance(tensor_data, numpy.ndarray):
        numpy_tensor = tensor_data
    else:
        logger.warning(f"Tensor data is not a JAX or NumPy array (type: {type(tensor_data)}). Attempting to convert to NumPy array.")
        try:
            numpy_tensor = numpy.array(tensor_data)
        except Exception as e:
            logger.error(f"Failed to convert tensor_data of type {type(tensor_data)} to NumPy array: {e}. Cannot visualize.")
            # Try to save raw representation if conversion fails
            raw_data_path = output_file_path.with_suffix(".raw_repr.txt")
            try:
                with open(raw_data_path, 'w', encoding='utf-8') as f:
                    f.write(f"{title} (Unknown Type - conversion failed):\n{tensor_data!r}")
                logger.info(f"Saved raw representation to {raw_data_path}")
            except Exception as save_e:
                logger.error(f"Could not even save raw representation: {save_e}")
            return

    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    ndims = numpy_tensor.ndim
    array_shape = numpy_tensor.shape

    logger.debug(f"Processing tensor for '{title}': {ndims}D, shape {array_shape}")

    if ndims == 0: # Scalar
        file_path = output_file_path.with_name(output_file_path.name + "_scalar.txt")
        try:
            scalar_value = numpy_tensor.item()
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(f"{title}: {scalar_value}\n")
                f.write(f"Shape: {array_shape}\n")
            logger.info(f"Saved scalar output to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save scalar output for '{title}' to {file_path}: {e}")

    elif MATPLOTLIB_AVAILABLE:
        if ndims == 1:
            file_path = output_file_path.with_name(output_file_path.name + "_plot.png")
            try:
                plt.figure()
                plt.plot(numpy_tensor)
                plt.title(f"{title} (1D Plot)")
                plt.xlabel("Index")
                plt.ylabel("Value")
                plt.grid(True)
                plt.savefig(file_path)
                plt.close()
                logger.info(f"Saved 1D plot to {file_path}")
            except Exception as e:
                logger.error(f"Failed to generate 1D plot for '{title}' to {file_path}: {e}", exc_info=True)
                # Fallback to saving raw data
                raw_data_path = output_file_path.with_name(output_file_path.name + "_1d_raw.txt")
                try:
                    numpy.savetxt(raw_data_path, numpy_tensor, header=f"{title}\nShape: {array_shape}", encoding='utf-8')
                    logger.info(f"Saved 1D raw data to {raw_data_path} as fallback.")
                except Exception as save_e:
                    logger.error(f"Failed to save 1D raw data for '{title}' to {raw_data_path}: {save_e}")


        elif ndims == 2:
            file_path = output_file_path.with_name(output_file_path.name + "_heatmap.png")
            try:
                plt.figure(figsize=(max(6, array_shape[1] // 2), max(4, array_shape[0] // 2))) # Adjust size
                # Choose a suitable colormap, e.g., 'viridis' or 'coolwarm' for signed data
                cmap = 'viridis'
                # If data might be centered around zero (e.g. weights, diffs), use a diverging colormap
                if numpy.min(numpy_tensor) < 0 and numpy.max(numpy_tensor) > 0:
                    cmap = 'coolwarm'
                    norm = mcolors.TwoSlopeNorm(vmin=numpy.min(numpy_tensor), vcenter=0, vmax=numpy.max(numpy_tensor))
                    img = plt.imshow(numpy_tensor, aspect='auto', cmap=cmap, norm=norm)
                else:
                    img = plt.imshow(numpy_tensor, aspect='auto', cmap=cmap)
                
                plt.title(f"{title} (2D Heatmap)")
                plt.xlabel("Column Index")
                plt.ylabel("Row Index")
                plt.colorbar(img, label="Value")
                plt.savefig(file_path)
                plt.close()
                logger.info(f"Saved 2D heatmap to {file_path}")
            except Exception as e:
                logger.error(f"Failed to generate 2D heatmap for '{title}' to {file_path}: {e}", exc_info=True)
                 # Fallback to saving raw data
                raw_data_path = output_file_path.with_name(output_file_path.name + "_2d_raw.txt")
                try:
                    numpy.savetxt(raw_data_path, numpy_tensor, header=f"{title}\nShape: {array_shape}", encoding='utf-8', fmt='%.18e')
                    logger.info(f"Saved 2D raw data to {raw_data_path} as fallback.")
                except Exception as save_e:
                    logger.error(f"Failed to save 2D raw data for '{title}' to {raw_data_path}: {save_e}")


        else: # ndims > 2
            logger.info(f"Tensor '{title}' is {ndims}D. Saving raw data instead of plotting directly.")
            file_path = output_file_path.with_name(output_file_path.name + f"_{ndims}d_raw.txt")
            try:
                # Reshape to 2D for savetxt: (dim1, dim2*dim3*...)
                reshaped_tensor = numpy_tensor.reshape(array_shape[0], -1)
                numpy.savetxt(file_path, reshaped_tensor, header=f"{title}\nOriginal Shape: {array_shape}", encoding='utf-8', fmt='%.18e')
                logger.info(f"Saved {ndims}D raw data to {file_path}")
            except Exception as e:
                logger.error(f"Failed to save {ndims}D raw data for '{title}' to {file_path}: {e}")

    elif not MATPLOTLIB_AVAILABLE and ndims > 0: # Matplotlib not available, but data is not scalar
        logger.warning(f"Matplotlib not available. Saving raw data for '{title}' ({ndims}D).")
        file_path = output_file_path.with_name(output_file_path.name + f"_{ndims}d_raw_no_mpl.txt")
        try:
            if ndims == 1:
                numpy.savetxt(file_path, numpy_tensor, header=f"{title}\nShape: {array_shape}", encoding='utf-8')
            else: # >= 2D
                reshaped_tensor = numpy_tensor.reshape(array_shape[0], -1) if ndims > 1 else numpy_tensor
                numpy.savetxt(file_path, reshaped_tensor, header=f"{title}\nOriginal Shape: {array_shape}", encoding='utf-8', fmt='%.18e')
            logger.info(f"Saved raw data (no matplotlib) to {file_path}")
        except Exception as e:
            logger.error(f"Failed to save raw data (no matplotlib) for '{title}' to {file_path}: {e}")

if __name__ == '__main__':
    # Basic logging for standalone testing
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Create a dummy output directory for testing
    test_output_dir = Path("output/discopy_translator_module_test_visuals")
    test_output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"--- Running visualize_jax_output.py Standalone Tests ---")
    logger.info(f"Matplotlib available: {MATPLOTLIB_AVAILABLE}")
    logger.info(f"JAX available: {JAX_AVAILABLE}")

    # Test cases
    test_data = {
        "scalar": numpy.array(42.0),
        "1d_small": numpy.array([1.0, 1.5, 1.0, 0.5, 0.0, 0.2]),
        "1d_large": numpy.linspace(-5, 5, 100),
        "2d_small": numpy.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]),
        "2d_random": numpy.random.rand(10, 15) * 10 - 5, # With negative values for coolwarm
        "3d_data": numpy.random.rand(3, 4, 5),
        "4d_data": numpy.arange(2*3*2*2).reshape((2,3,2,2))
    }

    if JAX_AVAILABLE:
        test_data["jax_scalar"] = jnp.array(101.1)
        test_data["jax_1d"] = jnp.array([10.0, -2.0, 5.5, 0.0])
        test_data["jax_2d_identity"] = jnp.eye(4, dtype=jnp.float32)
        test_data["jax_3d_zeros"] = jnp.zeros((2,2,2), dtype=jnp.float16)


    for name, data in test_data.items():
        logger.info(f"Testing with: {name} (Type: {type(data)}, Shape: {getattr(data, 'shape', 'N/A')})")
        output_path = test_output_dir / name
        plot_tensor_output(data, output_path, title=f"Test: {name}", verbose=True)
        print("-" * 20)

    # Test with non-array data
    logger.info("Testing with non-array data (list):")
    plot_tensor_output([1,2,3,4], test_output_dir / "list_data", title="Test: Python List", verbose=True)
    print("-" * 20)
    
    logger.info("Testing with non-array data (int):")
    plot_tensor_output(12345, test_output_dir / "int_data", title="Test: Python Int", verbose=True)
    print("-" * 20)


    logger.info(f"--- Standalone Tests Finished. Check '{test_output_dir}' directory. ---") 